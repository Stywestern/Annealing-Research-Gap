"""Step 3: sampler back-ends acting on the *programmed* (scaled, quantised, noisy) Ising model.

Both return an int8 array (num_reads, n) of +/-1 spins, columns ordered like IsingArrays.labels.

  SAFreezeout : simulated annealing whose final inverse temperature is the physical freeze-out
                beta,  beta = (B(s*)/2) / kT.   Cheap; no quantum-inspired dynamics.
  SVMCSampler : spin-vector Monte Carlo (Shin, Smith, Smolin, Vazirani 2014). Each qubit is a planar
                rotor angle theta in [0, pi]; the schedule A(s), B(s) drives a Metropolis anneal
                with  E = -A/2 sum sin(theta_i) + B/2 [sum h_i cos(theta_i) + sum J_ij cos(theta_i) cos(theta_j)].
                Updates are vectorised over reads and over graph-colour classes.
  SQASampler  : simulated quantum annealing (path-integral Monte Carlo, Trotter slices). Quantum
                (tunnelling, quantum correlations) but not real-time dynamics. See its docstring.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp

from solvers.simulated_quantum_annealing.s1_hardware import KB_OVER_H_GHZ_PER_K, Schedule, beta_eff
from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays, energies as ising_energies

def _colour_blocks(ar: IsingArrays):
    """Groups of qubits with no coupler between members: a whole group can be updated at once."""
    g = nx.Graph()
    g.add_nodes_from(range(len(ar.labels)))
    g.add_edges_from(zip(ar.rows.tolist(), ar.cols.tolist()))
    groups = {}
    for v, c in nx.greedy_color(g, strategy="largest_first").items():
        groups.setdefault(c, []).append(v)
    return [np.array(b) for b in groups.values()]

class SAFreezeout:
    """Classical thermal sampler of the programmed Ising model at the freeze-out inverse temperature
        beta_eff = (B(s*)/2) / kT      (units: 1 / scaled Ising energy).
    This is the classical 'freeze-out' picture: no transverse field is simulated (A(s) is not used), so it is
    the idealised thermal limit of the annealer. Real devices show an effective temperature higher than the
    physical one, a freeze-out point that depends on the instance, and beta_eff that fluctuates between
    programming cycles. freeze_s=0.6 and temperature_mk=12 are UNCALIBRATED placeholders.

    One call of sample() is treated as ONE programming cycle (this is how EmulatedQPU calls it): beta_eff is
    drawn once per call and shared by all reads of that call.

    method = "anneal" : neal SA from beta_hot to beta_eff over num_sweeps (the original behaviour). It equals a
                        Boltzmann sample at beta_eff only if the schedule is long enough to equilibrate.
    method = "pt"     : parallel tempering with pt_replicas replicas on a geometric ladder from beta_hot' to
                        beta_eff; each read is an independent chain and returns its coldest replica. Targets
                        the Boltzmann distribution at beta_eff directly. Check it with check_equilibrium().

    beta_eff source (first match):
      beta_override           explicit beta_eff in 1/scaled-energy (e.g. from an effective-temperature fit)
      freeze_s                a float s*, a callable anneal_time_us -> s*, or a table [(anneal_time_us, s*), ...]
                              interpolated linearly in log(time); the last two need anneal_time_us
    beta_sigma_rel: relative log-normal spread of beta_eff across programming cycles (0 = none). Calibration
                    parameter; no default is assumed.
    """

    METHODS = ("anneal", "pt")

    def __init__(self, schedule: Schedule, temperature_mk: float = 12.0, freeze_s=0.6,
                 num_sweeps: int = 1000, beta_hot: float = 0.1, use_auto_beta: bool = False, *,
                 method: str = "anneal", beta_override: float | None = None,
                 anneal_time_us: float | None = None, beta_sigma_rel: float = 0.0,
                 pt_replicas: int = 24, pt_sweeps: int | None = None, pt_swap_every: int = 1):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        if method == "pt" and use_auto_beta:
            raise ValueError("use_auto_beta only applies to method='anneal'")
        if beta_sigma_rel < 0:
            raise ValueError("beta_sigma_rel must be >= 0")
        if pt_replicas < 2:
            raise ValueError("pt_replicas must be >= 2")
        self.schedule, self.temperature_mk = schedule, temperature_mk
        self.freeze_s, self.num_sweeps, self.beta_hot = freeze_s, num_sweeps, beta_hot
        self.use_auto_beta = use_auto_beta
        self.method, self.beta_override = method, beta_override
        self.anneal_time_us, self.beta_sigma_rel = anneal_time_us, float(beta_sigma_rel)
        self.pt_replicas = int(pt_replicas)
        self.pt_sweeps = int(num_sweeps if pt_sweeps is None else pt_sweeps)
        self.pt_swap_every = max(1, int(pt_swap_every))
        self._freeze_kind, self._freeze = self._parse_freeze(freeze_s)
        self.last_info, self._beta_history = {}, []

    # ------------------------------------------------------------------ beta_eff
    @staticmethod
    def _parse_freeze(freeze_s):
        if callable(freeze_s):
            return "callable", freeze_s
        arr = np.asarray(freeze_s, dtype=float)
        if arr.ndim == 0:
            return "const", float(arr)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
            arr = arr[np.argsort(arr[:, 0])]
            if np.any(arr[:, 0] <= 0):
                raise ValueError("anneal times in the freeze_s table must be > 0 (microseconds)")
            return "table", arr
        raise ValueError("freeze_s must be a float, a callable(anneal_time_us), or a [(time_us, s*), ...] table")

    def freeze_point(self, anneal_time_us: float | None = None) -> float:
        t = self.anneal_time_us if anneal_time_us is None else anneal_time_us
        if self._freeze_kind == "const":
            s = self._freeze
        else:
            if t is None:
                raise ValueError("freeze_s depends on the anneal time: pass anneal_time_us (constructor or sample())")
            if self._freeze_kind == "callable":
                s = float(self._freeze(t))
            else:
                s = float(np.interp(np.log(t), np.log(self._freeze[:, 0]), self._freeze[:, 1]))
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"freeze point s*={s} outside [0, 1]")
        return s

    def beta_final(self, anneal_time_us: float | None = None) -> float:
        """Nominal beta_eff (no cycle-to-cycle fluctuation) in 1 / scaled Ising energy."""
        if self.beta_override is not None:
            return float(self.beta_override)
        _, B = self.schedule.at(self.freeze_point(anneal_time_us))
        return beta_eff(B, self.temperature_mk)

    # ------------------------------------------------------------------ sampling
    def sample(self, ar: IsingArrays, num_reads: int, seed: int,
               anneal_time_us: float | None = None) -> np.ndarray:
        n = len(ar.labels)
        if n == 0 or num_reads <= 0:
            return np.zeros((max(int(num_reads), 0), n), dtype=np.int8)
        t_us = self.anneal_time_us if anneal_time_us is None else anneal_time_us
        auto = self.method == "anneal" and self.use_auto_beta
        b_nom = None if auto else self.beta_final(t_us)
        b_used = b_nom
        if b_nom is not None and self.beta_sigma_rel > 0:
            z = np.random.default_rng([int(seed), 7919]).standard_normal()   # own stream: sampler rng untouched
            b_used = b_nom * float(np.exp(self.beta_sigma_rel * z))

        if self.method == "pt":
            spins, extra = self._sample_pt(ar, int(num_reads), seed, b_used)
        else:
            spins, extra = self._sample_anneal(ar, int(num_reads), seed, b_used)

        en = ising_energies(ar, spins)
        self.last_info = {
            "method": self.method, "num_reads": int(num_reads), "n_spins": n,
            "beta_nominal": b_nom, "beta_used": b_used, "anneal_time_us": t_us,
            "freeze_s_used": (None if (auto or self.beta_override is not None) else self.freeze_point(t_us)),
            "mean_energy": float(en.mean()), "min_energy": float(en.min()), **extra}
        if b_used is not None and len(self._beta_history) < 100000:
            self._beta_history.append(float(b_used))
        return spins

    def _sample_anneal(self, ar, R, seed, beta):
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            from neal import SimulatedAnnealingSampler
        beta_range = None if self.use_auto_beta else (min(self.beta_hot, beta / 10.0), beta)
        res = SimulatedAnnealingSampler().sample(ar.to_bqm(), num_reads=R, num_sweeps=self.num_sweeps,
                                                 beta_range=beta_range, seed=int(seed))
        pos = {v: i for i, v in enumerate(res.variables)}
        spins = np.asarray(res.record.sample)[:, [pos[v] for v in ar.labels]].astype(np.int8)
        return spins, {"num_sweeps": self.num_sweeps, "beta_hot_used": None if beta_range is None else beta_range[0]}

    def _sample_pt(self, ar, R, seed, beta):
        rng = np.random.default_rng(int(seed))
        n, K = len(ar.labels), self.pt_replicas
        betas = np.geomspace(min(self.beta_hot, beta / 10.0), beta, K)            # hot ... cold
        beta3 = betas.astype(np.float32)[None, :, None]
        Jm = sp.coo_matrix((np.r_[ar.J, ar.J], (np.r_[ar.rows, ar.cols], np.r_[ar.cols, ar.rows])),
                           shape=(n, n)).tocsr().astype(np.float32)
        blocks = _colour_blocks(ar)
        Jrows = [Jm[b] for b in blocks]
        hcol = ar.h.astype(np.float32)
        hb = [hcol[b][:, None, None] for b in blocks]

        S = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, K, R))   # infinite-temperature start
        swap_acc, swap_try = np.zeros(K - 1), np.zeros(K - 1)
        for t in range(self.pt_sweeps):
            for b, Jb, h in zip(blocks, Jrows, hb):                                 # Metropolis, per replica beta
                F = (Jb @ S.reshape(n, K * R)).reshape(len(b), K, R) + h
                cur = S[b]
                dE = -2.0 * cur * F
                accept = rng.random(dE.shape, dtype=np.float32) < np.exp(-np.clip(beta3 * dE, 0.0, 50.0))
                S[b] = np.where(accept, -cur, cur)
            if (t + 1) % self.pt_swap_every == 0:                                   # replica exchange
                E = (hcol[:, None, None] * S).sum(axis=0) \
                    + 0.5 * (S * (Jm @ S.reshape(n, K * R)).reshape(n, K, R)).sum(axis=0)       # (K, R)
                ks = np.arange(((t + 1) // self.pt_swap_every) % 2, K - 1, 2)
                if ks.size:
                    x = (betas[ks + 1] - betas[ks])[:, None] * (E[ks + 1] - E[ks])
                    acc = rng.random(x.shape) < np.exp(np.clip(x, -50.0, 50.0))     # min(1, exp(x))
                    a, c = S[:, ks, :], S[:, ks + 1, :]
                    m = acc[None, :, :]
                    S[:, ks, :] = np.where(m, c, a)
                    S[:, ks + 1, :] = np.where(m, a, c)
                    swap_acc[ks] += acc.mean(axis=1)
                    swap_try[ks] += 1
        spins = np.where(S[:, -1, :] > 0.0, 1, -1).T.astype(np.int8)
        rate = np.divide(swap_acc, swap_try, out=np.zeros_like(swap_acc), where=swap_try > 0)
        return spins, {"pt_replicas": K, "pt_sweeps": self.pt_sweeps,
                       "pt_beta_ladder": [float(x) for x in betas],
                       "pt_swap_acceptance": [float(x) for x in rate],
                       "pt_swap_acceptance_min": float(rate.min())}

    # ------------------------------------------------------------------ diagnostics
    def thermal_resolution(self, ar: IsingArrays, beta: float | None = None) -> dict:
        """Thermal energy 1/beta in the SCALED units of this Ising model, compared with its coefficients.
        Coefficients far below kT are effectively random in the output. Compare kT_scaled with the DAC step."""
        b = self.beta_final() if beta is None else float(beta)
        kT = 1.0 / b
        aJ, aH = np.abs(ar.J[np.abs(ar.J) > 0]), np.abs(ar.h[np.abs(ar.h) > 0])
        return {"beta": b, "kT_scaled": kT,
                "J_nonzero": int(aJ.size), "J_median_abs": float(np.median(aJ)) if aJ.size else None,
                "J_min_abs": float(aJ.min()) if aJ.size else None,
                "J_below_kT_pct": float(100.0 * np.mean(aJ < kT)) if aJ.size else None,
                "h_nonzero": int(aH.size), "h_below_kT_pct": float(100.0 * np.mean(aH < kT)) if aH.size else None,
                "kT_over_max_abs_J": float(kT / aJ.max()) if aJ.size else None}

    def check_equilibrium(self, ar: IsingArrays, num_reads: int = 2000, seed: int = 0,
                          max_spins: int = 18, n_bins: int = 30) -> dict:
        """Compares this sampler (beta fluctuation switched off) with the EXACT Boltzmann distribution at
        beta_final() on a small model (full enumeration, so len(ar.labels) <= max_spins). Reports mean energy
        with a z-score, ground-state probability, and the total-variation distance between the energy
        histograms next to its expected statistical noise floor. Only validates the settings used here."""
        n = len(ar.labels)
        if n > max_spins:
            raise ValueError(f"{n} spins > max_spins={max_spins}; use a smaller instance")
        saved, self.beta_sigma_rel = self.beta_sigma_rel, 0.0
        try:
            beta = self.beta_final()
            spins = self.sample(ar, num_reads, seed)
        finally:
            self.beta_sigma_rel = saved
        idx = np.arange(2 ** n, dtype=np.uint32)
        bits = np.arange(n, dtype=np.uint32)
        states = ((idx[:, None] >> bits[None, :]) & 1).astype(np.float32) * 2.0 - 1.0
        Jd = np.zeros((n, n), dtype=np.float32)
        np.add.at(Jd, (ar.rows, ar.cols), ar.J.astype(np.float32))
        E = (states @ ar.h.astype(np.float32) + np.sum((states @ Jd) * states, axis=1)).astype(np.float64)
        p = np.exp(-beta * (E - E.min()))
        p /= p.sum()
        sidx = ((spins > 0).astype(np.uint32) << bits[None, :]).sum(axis=1)
        Es = E[sidx]
        mean_ex, std_ex = float(np.sum(p * E)), float(np.sqrt(np.sum(p * E ** 2) - np.sum(p * E) ** 2))
        gs = E <= E.min() + 1e-9
        edges = np.linspace(E.min(), E.max(), n_bins + 1)
        pe, _ = np.histogram(E, bins=edges, weights=p)
        pm = np.histogram(Es, bins=edges)[0] / len(Es)
        return {"n_spins": n, "num_reads": int(num_reads), "beta": float(beta),
                "mean_energy_exact": mean_ex, "mean_energy_sampled": float(Es.mean()),
                "mean_energy_z": float((Es.mean() - mean_ex) / max(std_ex / np.sqrt(len(Es)), 1e-300)),
                "p_ground_exact": float(p[gs].sum()), "p_ground_sampled": float(np.mean(Es <= E.min() + 1e-9)),
                "tv_energy_hist": float(0.5 * np.abs(pe - pm).sum()),
                "tv_noise_floor": float(0.5 * np.sum(np.sqrt(2.0 * pe * (1.0 - pe) / (np.pi * len(Es))))),
                "method": self.method}

    # ------------------------------------------------------------------ record keeping
    def describe(self) -> dict:
        """Plain JSON-safe description of this sampler and what it has done so far, for result files."""
        if self._freeze_kind == "const":
            fz = self._freeze
        elif self._freeze_kind == "table":
            fz = {"table_time_us_to_s": self._freeze.tolist()}
        else:
            fz = "callable"
        d = {"class": "SAFreezeout", "method": self.method, "temperature_mk": self.temperature_mk,
             "freeze_s": fz, "anneal_time_us": self.anneal_time_us, "beta_override": self.beta_override,
             "use_auto_beta": self.use_auto_beta, "beta_hot": self.beta_hot, "num_sweeps": self.num_sweeps,
             "beta_sigma_rel": self.beta_sigma_rel,
             "beta_nominal": (None if (self.use_auto_beta and self.method == "anneal") else self.beta_final()),
             "freeze_point_s": (None if (self.beta_override is not None or self._freeze_kind != "const"
                                         ) else self._freeze),
             "schedule": (self.schedule.provenance() if hasattr(self.schedule, "provenance")
                          else {"source": getattr(self.schedule, "source", "unknown")}),
             "uncalibrated_defaults": (self.beta_override is None and self._freeze_kind == "const"
                                       and self._freeze == 0.6 and self.temperature_mk == 12.0)}
        if self.method == "pt":
            d["pt"] = {"replicas": self.pt_replicas, "sweeps": self.pt_sweeps, "swap_every": self.pt_swap_every}
        if self._beta_history:
            h = np.asarray(self._beta_history)
            d["runs"] = {"programming_cycles": int(h.size), "beta_used_mean": float(h.mean()),
                         "beta_used_std": float(h.std()), "beta_used_min": float(h.min()),
                         "beta_used_max": float(h.max())}
        d["last_call"] = self.last_info
        return d


class SVMCSampler:
    def __init__(self, schedule: Schedule, temperature_mk: float = 12.0, num_sweeps: int = 1000):
        """num_sweeps is a stand-in for annealing time (calibration parameter)."""
        self.schedule, self.temperature_mk, self.num_sweeps = schedule, temperature_mk, num_sweeps

    def sample(self, ar: IsingArrays, num_reads: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n, R = len(ar.labels), num_reads
        kT = KB_OVER_H_GHZ_PER_K * self.temperature_mk * 1e-3     # GHz

        # symmetric sparse coupling matrix and graph-colour blocks (no two nodes in a block are adjacent,
        # so a whole block can be Metropolis-updated simultaneously without violating detailed balance)
        Jm = sp.coo_matrix((np.r_[ar.J, ar.J], (np.r_[ar.rows, ar.cols], np.r_[ar.cols, ar.rows])),
                           shape=(n, n)).tocsr()
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(zip(ar.rows.tolist(), ar.cols.tolist()))
        colour = nx.greedy_color(g, strategy="largest_first")
        blocks = {}
        for v, c in colour.items():
            blocks.setdefault(c, []).append(v)
        blocks = [np.array(b) for b in blocks.values()]
        Jrows = [Jm[b] for b in blocks]
        hb = [ar.h[b][:, None] for b in blocks]

        theta = np.full((n, R), np.pi / 2)                        # start in the transverse-field ground state
        for t in range(self.num_sweeps):
            A, B = self.schedule.at((t + 0.5) / self.num_sweeps)
            for b, Jb, h in zip(blocks, Jrows, hb):
                c = np.cos(theta)
                field = h + Jb @ c                                # (|b|, R): h_i + sum_j J_ij cos(theta_j)
                old = theta[b]
                new = rng.uniform(0.0, np.pi, old.shape)
                dE = (-0.5 * A * (np.sin(new) - np.sin(old))
                      + 0.5 * B * field * (np.cos(new) - np.cos(old)))
                accept = (dE <= 0.0) | (rng.random(dE.shape) < np.exp(-np.clip(dE / kT, 0.0, 700.0)))
                theta[b] = np.where(accept, new, old)
        return np.where(np.cos(theta) > 0.0, 1, -1).T.astype(np.int8)


class SQASampler:
    """Simulated quantum annealing = path-integral Monte Carlo of the transverse-field Ising model
    that the annealer implements:

        H(s) = -(A(s)/2) sum_i sigma^x_i + (B(s)/2) [ sum_i h_i sigma^z_i + sum_ij J_ij sigma^z_i sigma^z_j ]

    Suzuki-Trotter maps the quantum system at temperature T onto P coupled classical replicas
    ("imaginary-time slices") of the problem, with an inter-slice ferromagnetic coupling
        K = 0.5 * ln coth( A/(2 kT P) ),
    and each slice weighted by exp(-(B/2) H_ising / (kT P)).  Spins fluctuate along the slices, which is
    how tunnelling and quantum correlations enter (the SVMC rotor model has neither).

    What this is:  a genuine quantum model (samples the quantum Boltzmann distribution at every point of
                   the schedule, with the real A(s), B(s) and temperature); scales ~linearly in qubits.
    What it is not: real-time Schrodinger/Lindblad dynamics. Monte Carlo sweeps are a stand-in for
                   anneal time (num_sweeps is a calibration parameter), and there is no open-system
                   decoherence/relaxation model beyond the thermal bath in the sampling.
    Accuracy knob: Trotter error. Need P large enough that (B/2)/(kT P) is not >> 1 near freeze-out;
                   test convergence by doubling trotter_slices on a small instance.
    Returns slice 0 of each replica stack (no post-selection on the best slice).
    Cost ~ num_sweeps * trotter_slices * n_qubits * num_reads.
    """

    def __init__(self, schedule: Schedule, temperature_mk: float = 12.0, num_sweeps: int = 300,
                 trotter_slices: int = 32, global_moves: bool = True):
        if trotter_slices % 2:
            raise ValueError("trotter_slices must be even (slices are updated in a checkerboard)")
        self.schedule, self.temperature_mk = schedule, temperature_mk
        self.num_sweeps, self.P, self.global_moves = num_sweeps, trotter_slices, global_moves

    def sample(self, ar: IsingArrays, num_reads: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n, R, P = len(ar.labels), num_reads, self.P
        beta = 1.0 / (KB_OVER_H_GHZ_PER_K * self.temperature_mk * 1e-3)         # 1/GHz

        Jm = sp.coo_matrix((np.r_[ar.J, ar.J], (np.r_[ar.rows, ar.cols], np.r_[ar.cols, ar.rows])),
                           shape=(n, n)).tocsr().astype(np.float32)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(zip(ar.rows.tolist(), ar.cols.tolist()))
        colour = nx.greedy_color(g, strategy="largest_first")
        groups = {}
        for v, c in colour.items():
            groups.setdefault(c, []).append(v)
        blocks = [np.array(b) for b in groups.values()]
        Jrows = [Jm[b] for b in blocks]
        hb = [ar.h[b].astype(np.float32)[:, None, None] for b in blocks]

        s = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, P, R))   # (qubit, slice, read)
        parities = [np.arange(p, P, 2) for p in (0, 1)]

        for t in range(self.num_sweeps):
            A, B = self.schedule.at((t + 0.5) / self.num_sweeps)
            x = max(beta * 0.5 * A / P, 1e-9)
            K = 0.5 * np.log(1.0 / np.tanh(x))          # inter-slice coupling (>0, ferromagnetic)
            w = beta * 0.5 * B / P                       # weight of the problem energy per slice

            # local single-spin flips; even then odd slices so simultaneously updated slices never neighbour
            for ks in parities:
                kp, kn = (ks - 1) % P, (ks + 1) % P
                for b, Jb, h in zip(blocks, Jrows, hb):
                    F = (Jb @ s[:, ks, :].reshape(n, -1)).reshape(len(b), len(ks), R) + h
                    cur = s[b[:, None], ks[None, :]]
                    nn = s[b[:, None], kp[None, :]] + s[b[:, None], kn[None, :]]
                    dS = 2.0 * cur * (K * nn - w * F)    # change in Trotter action if this spin flips
                    accept = rng.random(dS.shape, dtype=np.float32) < np.exp(-np.clip(dS, 0.0, 50.0))
                    s[b[:, None], ks[None, :]] = np.where(accept, -cur, cur)

            # global world-line flips (one qubit, all slices): inter-slice term unchanged
            if self.global_moves:
                for b, Jb, h in zip(blocks, Jrows, hb):
                    F = (Jb @ s.reshape(n, -1)).reshape(len(b), P, R) + h
                    cur = s[b]
                    dS = (-2.0 * w * cur * F).sum(axis=1)                     # (|b|, R)
                    accept = rng.random(dS.shape, dtype=np.float32) < np.exp(-np.clip(dS, 0.0, 50.0))
                    s[b] = np.where(accept[:, None, :], -cur, cur)

        return np.where(s[:, 0, :] > 0.0, 1, -1).T.astype(np.int8)


def make_sampler(kind: str, schedule: Schedule, temperature_mk: float, **kw):
    if kind == "sa":
        return SAFreezeout(schedule, temperature_mk, **kw)
    if kind == "svmc":
        return SVMCSampler(schedule, temperature_mk, **kw)
    if kind == "sqa":
        return SQASampler(schedule, temperature_mk, **kw)
    raise ValueError(f"unknown sampler {kind!r}; use 'sa', 'svmc' or 'sqa'")


# =========================================================================
# Execution Block
# =========================================================================
if __name__ == "__main__":
    import time
    import io
    import dimod
    import numpy as np
    
    try:
        from homemade_grids.small_grids import case3_low_gen
        from solvers.qubo_formulator import QuboFormulator
        from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
        from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays, scale_to_hardware, quantise, add_ice_noise
        from solvers.simulated_quantum_annealing.s3_samplers import make_sampler
        from dwave.embedding import embed_bqm
        from dwave.embedding.chain_strength import uniform_torque_compensation
    except ImportError as e:
        print(f"[-] Import error: {e}. Ensure script is run from project root.")
        exit(1)

    print("=== Testing All Sampler Backends on case3_low_gen ===")

    # 1. Formulation
    print("\n[1] Formulating Grid (DC-PTDF, 10 MW precision)...")
    net = case3_low_gen()
    if not net.ext_grid.empty: 
        net.ext_grid['min_p_mw'] = 0.0
        
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0)
    logical_bqm, _ = formulator._formulate_qubo(net)
    logical_spin = logical_bqm.change_vartype(dimod.SPIN, inplace=False)

    # 2. Minor-Embedding & Chain Diagnostics
    print("\n[2] Minor-Embedding to Z12 & Analyzing Chains...")
    hw = Hardware.ideal_zephyr(m=12, t=4)
    emb_results = find_embeddings(logical_spin, hw, methods=["minorminer"], seeds=[0], verbose=False)
    _, embedding = select_best(emb_results)
    
    chain_lengths = [len(c) for c in embedding.values()]
    max_chain = max(chain_lengths) if chain_lengths else 0
    mean_chain = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0
    
    # 1. Find the absolute maximum energy coefficient in your logical problem
    max_logical_weight = max(
        max(abs(v) for v in logical_spin.linear.values()) if logical_spin.linear else 0,
        max(abs(v) for v in logical_spin.quadratic.values()) if logical_spin.quadratic else 0
    )

    # 2. Define a relative multiplier (usually between 0.5 and 2.0)
    # A multiplier of 1.0 means the chains are exactly as strong as the heaviest grid constraint.
    relative_chain_multiplier = 1.0 
    capped_cs = max_logical_weight * relative_chain_multiplier

    embedded_bqm = embed_bqm(logical_spin, embedding, hw.graph, chain_strength=capped_cs)
    ar_ideal = IsingArrays.from_bqm(embedded_bqm)

    print(f"    -> Max Chain Length: {max_chain} physical qubits")
    print(f"    -> Mean Chain Length: {mean_chain:.2f} physical qubits")
    print(f"    -> Raw Artificial Chain Strength (J_chain): {capped_cs:,.2f}")

    # 3. Analog Hardware Mapping
    print("\n[3] Mapping to Analog Hardware (5-bit DAC, 0.01 ICE)...")
    ar_scaled, scale_factor = scale_to_hardware(ar_ideal, hw.h_range, hw.extended_j_range)
    ar_quant = quantise(ar_scaled, hw.h_range, hw.extended_j_range, bits=5)
    
    rng = np.random.default_rng(42)
    ice_sigma = 0.01
    ar_physical = add_ice_noise(ar_quant, sigma_h=ice_sigma, sigma_j=ice_sigma, rng=rng)

    # Calculate what happened to the chains during down-scaling
    scaled_cs = capped_cs * scale_factor
    print(f"    -> Global Scale Factor applied: {scale_factor:e}")
    print(f"    -> Scaled Chain Strength: {scaled_cs:.5f} (Physical Limit: 1.0)")
    print(f"    -> ICE Noise Floor (sigma): {ice_sigma}")
    
    if scaled_cs < ice_sigma:
        print("    -> [!] CRITICAL WARNING: Scaled chain strength is below the analog noise floor.")
        print("    -> [!] The magnetic glue holding chains together is weaker than the ambient heat.")
    elif scaled_cs < 0.0625: # Half a step size on a 5-bit DAC for [-1, 1]
        print("    -> [!] CRITICAL WARNING: Scaled chain strength is erased by 5-bit DAC quantization.")

    # 4. Helper for Chain Breaks
    def get_chain_break_fraction(samples, emb, labels):
        pos = {v: i for i, v in enumerate(labels)}
        total_chains = sum(1 for c in emb.values() if len(c) > 1)
        if total_chains == 0: 
            return 0.0
            
        broken_chains = np.zeros(samples.shape[0])
        for chain in emb.values():
            if len(chain) > 1:
                idxs = [pos[q] for q in chain]
                chain_spins = samples[:, idxs]
                # Chain breaks if the physical qubits do not all perfectly align (+1 or -1)
                is_broken = np.abs(chain_spins.sum(axis=1)) != len(chain)
                broken_chains += is_broken
                
        return (broken_chains / total_chains).mean()

    # 5. Define Competing Schedules
    # A. The smooth mathematical placeholder
    sched_placeholder = Schedule.placeholder()
    
    # B. A synthesized physical QPU curve (A decays fast, B spikes exponentially at the end)
    sched_real = Schedule.from_csv("solvers/simulated_quantum_annealing/standart_annealing_schedule_Ad2Sys1.csv")
    
    schedules = {
        "Math Placeholder": sched_placeholder,
        "Simulated Real QPU": sched_real
    }

    # 6. Execute and Benchmark
    temp_mk = 12.0  
    num_reads = 300
    num_sweeps = 5000

    print("\n" + "="*85)
    print(f"{'Schedule':<20} | {'Engine':<6} | {'Time (s)':<8} | {'Unique States':<13} | {'Chain Breaks (%)'}")
    print("="*85)
    
    for sched_name, schedule in schedules.items():
        for kind in ["sa", "svmc", "sqa"]:
            kwargs = {"num_sweeps": num_sweeps}
            if kind == "sqa":
                kwargs["trotter_slices"] = 16 
                
            sampler = make_sampler(kind, schedule, temp_mk, **kwargs)

            t0 = time.time()
            samples = sampler.sample(ar_physical, num_reads=num_reads, seed=42)
            t_exec = time.time() - t0

            cbf = get_chain_break_fraction(samples, embedding, ar_physical.labels) * 100
            unique_states = len(np.unique(samples, axis=0))

            print(f"{sched_name:<20} | {kind.upper():<6} | {t_exec:<8.4f} | {unique_states:<13} | {cbf:.2f}%")
        
        # Print a separator between schedule blocks
        if sched_name == "Math Placeholder":
            print("-" * 85)
            
    print("="*85 + "\n")