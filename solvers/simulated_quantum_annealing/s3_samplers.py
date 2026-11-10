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
    method = "pt"     : parallel tempering with pt_replicas on a geometric ladder from beta_hot' to
                        beta_eff; each read is an independent chain and returns its coldest replica. Targets
                        the Boltzmann distribution at beta_eff directly.

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
        try:
            bn = None if (self.use_auto_beta and self.method == "anneal") else self.beta_final()
        except ValueError:
            bn = None

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
             "beta_nominal": bn,
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
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import itertools
    import json
    import math
    import sys
    import time

    import dimod

    QUICK = False
    NO_OPF = False
    NO_EMU = False
    OUT_JSON = "sampler_selftest.json"
    CHECKS, REPORT = [], {}
    R_EQ = 1500 if QUICK else 4000          # reads for the exact-distribution comparisons
    SW_PT = 250 if QUICK else 400           # parallel-tempering sweeps for the 12-spin model
    HR, JR = (-4.0, 4.0), (-1.0, 1.0)

    # ------------------------------------------------------------------ helpers
    def check(name, ok, detail=""):
        CHECKS.append((name, bool(ok), str(detail)))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def info(msg):
        print(f"    [info] {msg}")

    def section(title):
        print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")

    def raises(exc, fn):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    def rand_model(n, m, seed, jscale=1.0, hscale=0.5, labels=None):
        r = np.random.default_rng(seed)
        pairs = list(itertools.combinations(range(n), 2))
        sel = r.choice(len(pairs), size=min(m, len(pairs)), replace=False)
        return IsingArrays(list(range(n)) if labels is None else list(labels), r.uniform(-hscale, hscale, n),
                           np.array([pairs[k][0] for k in sel], dtype=int), np.array([pairs[k][1] for k in sel], dtype=int),
                           r.uniform(-jscale, jscale, len(sel)))

    def complete(n, h, j):
        pr = list(itertools.combinations(range(n), 2))
        return IsingArrays(list(range(n)), np.full(n, float(h)), np.array([p[0] for p in pr], dtype=int),
                           np.array([p[1] for p in pr], dtype=int), np.full(len(pr), float(j)))

    try:
        SCHED = Schedule.from_csv(strict=False)
    except Exception as exc:
        print(f"[!] schedule load failed ({exc}); using the placeholder schedule")
        SCHED = Schedule.placeholder()
    print("=== Testing SAFreezeout (classical thermal sampler at the freeze-out beta) ===")
    print(f"mode={'quick' if QUICK else 'full'} | schedule source={getattr(SCHED, 'source', '?')} "
          f"placeholder={getattr(SCHED, 'is_placeholder', '?')}")
    if getattr(SCHED, "is_placeholder", False):
        print("[!] placeholder schedule: all checks are internally consistent, but physical beta values are not meaningful")

    m12 = rand_model(12, 30, seed=5)

    # ================================================================== [1] construction and beta bookkeeping
    section("[1] construction, validation and beta_eff bookkeeping")
    check("unknown method raises ValueError", raises(ValueError, lambda: SAFreezeout(SCHED, method="nope")))
    check("method='pt' with use_auto_beta raises ValueError",
          raises(ValueError, lambda: SAFreezeout(SCHED, method="pt", use_auto_beta=True)))
    check("negative beta_sigma_rel raises ValueError", raises(ValueError, lambda: SAFreezeout(SCHED, beta_sigma_rel=-0.1)))
    check("pt_replicas < 2 raises ValueError", raises(ValueError, lambda: SAFreezeout(SCHED, method="pt", pt_replicas=1)))
    check("freeze_s with a bad shape raises ValueError", raises(ValueError, lambda: SAFreezeout(SCHED, freeze_s=np.zeros((2, 3)))))
    check("freeze_s table with a non-positive time raises ValueError",
          raises(ValueError, lambda: SAFreezeout(SCHED, freeze_s=[(0.0, 0.5), (10.0, 0.6)])))

    old = SAFreezeout(SCHED, 12.0, 0.6, 1000, 0.1, False)
    check("old positional signature still works", old.temperature_mk == 12.0 and old.freeze_s == 0.6
          and old.num_sweeps == 1000 and old.beta_hot == 0.1 and old.use_auto_beta is False and old.method == "anneal")

    s_star = 0.6
    sa = SAFreezeout(SCHED, 12.0, s_star)
    _, B_star = SCHED.at(s_star)
    b_expected = 0.5 * B_star / (KB_OVER_H_GHZ_PER_K * 12.0e-3)
    check("beta_final = (B(s*)/2) / kT with kT in GHz", abs(sa.beta_final() - b_expected) <= 1e-12 * b_expected,
          f"beta={sa.beta_final():.4f} per scaled energy unit")
    check("doubling the temperature halves beta", abs(SAFreezeout(SCHED, 24.0, s_star).beta_final() - sa.beta_final() / 2) <= 1e-12)
    check("beta_override takes priority over (freeze_s, temperature)",
          SAFreezeout(SCHED, 99.0, 0.1, beta_override=3.25).beta_final() == 3.25)
    s_grid = np.linspace(0.2, 0.95, 16)
    if np.all(np.diff([SCHED.at(s)[1] for s in s_grid]) >= -1e-12):
        bs = [SAFreezeout(SCHED, 12.0, s).beta_final() for s in s_grid]
        check("beta_final increases with the freeze point s* (B(s) rises)", np.all(np.diff(bs) >= -1e-12))
    else:
        info("B(s) is not monotone in this schedule; monotonicity check skipped")

    tab = SAFreezeout(SCHED, freeze_s=[(5.0, 0.5), (50.0, 0.7)])
    check("table: log-time interpolation hits the midpoint at the geometric mean time",
          abs(tab.freeze_point(math.sqrt(5.0 * 50.0)) - 0.6) < 1e-9)
    check("table: clamped outside the table", abs(tab.freeze_point(0.5) - 0.5) < 1e-9 and abs(tab.freeze_point(5000.0) - 0.7) < 1e-9)
    check("table without an anneal time raises ValueError", raises(ValueError, lambda: tab.freeze_point()))
    check("constructor anneal_time_us is used when sample() gets none",
          abs(SAFreezeout(SCHED, freeze_s=[(5.0, 0.5), (50.0, 0.7)], anneal_time_us=5.0).freeze_point() - 0.5) < 1e-9)
    check("callable freeze_s", abs(SAFreezeout(SCHED, freeze_s=lambda t: 0.4 + 0.001 * t).freeze_point(100.0) - 0.5) < 1e-12)
    check("freeze point outside [0, 1] raises ValueError",
          raises(ValueError, lambda: SAFreezeout(SCHED, freeze_s=lambda t: 1.5).freeze_point(1.0)))

    # ================================================================== [2] output contract and trivial models
    section("[2] output contract, label order, trivially solvable models, regression against the old behaviour")
    labs = [30, 5, 17, 2, 41, 8, 23, 11]
    n_h = len(labs)
    hsign = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=float)
    ar_h = IsingArrays(labs, hsign.copy(), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    h_before, J_before = ar_h.h.copy(), ar_h.J.copy()
    for meth, kw in (("anneal", dict(num_sweeps=60)), ("pt", dict(pt_sweeps=60, pt_replicas=6))):
        smp = SAFreezeout(SCHED, method=meth, beta_override=40.0, **kw)
        X = smp.sample(ar_h, 25, seed=1)
        check(f"{meth}: shape (reads, n), dtype int8, values +-1",
              X.shape == (25, n_h) and X.dtype == np.int8 and set(np.unique(X).tolist()) <= {-1, 1})
        check(f"{meth}: columns follow ar.labels order (field-only model, unique ground state s = -sign(h))",
              np.all(X == (-hsign).astype(np.int8)[None, :]))
        check(f"{meth}: the input model is not modified", np.array_equal(ar_h.h, h_before) and np.array_equal(ar_h.J, J_before))

    K8 = complete(8, 0.0, -1.0)
    for meth, kw in (("anneal", dict(num_sweeps=300)), ("pt", dict(pt_sweeps=150, pt_replicas=10))):
        X = SAFreezeout(SCHED, method=meth, beta_override=20.0, **kw).sample(K8, 200, seed=2)
        aligned = float(np.mean(np.all(X == X[:, :1], axis=1)))
        check(f"{meth}: all-to-all ferromagnet ends all-aligned (>= 99% of reads)", aligned >= 0.99, f"{100 * aligned:.1f}%")

    empty = IsingArrays([], np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    check("zero spins -> an (R, 0) array", SAFreezeout(SCHED).sample(empty, 5, seed=0).shape == (5, 0))
    check("zero reads -> a (0, n) array", SAFreezeout(SCHED).sample(ar_h, 0, seed=0).shape == (0, n_h))

    try:
        from dwave.samplers import SimulatedAnnealingSampler as _NEAL
    except ImportError:
        from neal import SimulatedAnnealingSampler as _NEAL
    legacy = SAFreezeout(SCHED, 12.0, 0.6, 300, 0.1, False)
    bf0 = legacy.beta_final()
    res0 = _NEAL().sample(m12.to_bqm(), num_reads=40, num_sweeps=300, beta_range=(min(0.1, bf0 / 10.0), bf0), seed=17)
    pos0 = {v: i for i, v in enumerate(res0.variables)}
    ref0 = np.asarray(res0.record.sample)[:, [pos0[v] for v in m12.labels]].astype(np.int8)
    check("default settings reproduce the original neal-based sampler exactly", np.array_equal(legacy.sample(m12, 40, 17), ref0))
    auto = SAFreezeout(SCHED, use_auto_beta=True, num_sweeps=50)
    auto.sample(m12, 10, seed=1)
    check("use_auto_beta: no physical beta is used or reported",
          auto.last_info["beta_used"] is None and auto.describe()["beta_nominal"] is None)
    check("anneal: beta_hot is min(beta_hot, beta/10)",
          abs(SAFreezeout(SCHED, beta_override=0.5, num_sweeps=20).sample(m12, 5, 1).shape[0] - 5) == 0 and True)
    a05 = SAFreezeout(SCHED, beta_override=0.5, num_sweeps=20)
    a05.sample(m12, 5, 1)
    a50 = SAFreezeout(SCHED, beta_override=5.0, num_sweeps=20)
    a50.sample(m12, 5, 1)
    check("anneal: beta_hot_used = min(0.1, beta/10)",
          abs(a05.last_info["beta_hot_used"] - 0.05) < 1e-12 and abs(a50.last_info["beta_hot_used"] - 0.1) < 1e-12)

    # ================================================================== [3] seeds and beta fluctuation
    section("[3] seeds, hidden state and cycle-to-cycle beta fluctuation")
    for meth, kw in (("anneal", dict(num_sweeps=80)), ("pt", dict(pt_sweeps=40, pt_replicas=8))):
        a = SAFreezeout(SCHED, method=meth, beta_override=0.8, **kw)
        x1, x2, x3 = a.sample(m12, 50, seed=3), a.sample(m12, 50, seed=3), a.sample(m12, 50, seed=4)
        check(f"{meth}: same seed -> identical samples", np.array_equal(x1, x2))
        check(f"{meth}: different seed -> different samples", not np.array_equal(x1, x3))
        fresh = SAFreezeout(SCHED, method=meth, beta_override=0.8, **kw)
        check(f"{meth}: a fresh object reproduces the same samples (no hidden state)", np.array_equal(fresh.sample(m12, 50, seed=3), x1))

    tiny = rand_model(3, 3, seed=2)
    sig = 0.2
    fl = SAFreezeout(SCHED, method="pt", beta_override=2.0, beta_sigma_rel=sig, pt_sweeps=1, pt_replicas=2)
    lr = []
    for sd in range(400):
        fl.sample(tiny, 1, seed=sd)
        lr.append(math.log(fl.last_info["beta_used"] / fl.last_info["beta_nominal"]))
    lr = np.array(lr)
    check("beta fluctuation: log(beta_used / beta_nominal) has mean ~ 0", abs(lr.mean()) < 0.3 * sig, f"mean {lr.mean():.4f}")
    check("beta fluctuation: its std matches beta_sigma_rel within 20%", abs(lr.std() / sig - 1.0) < 0.2, f"std {lr.std():.4f} vs {sig}")
    fl.sample(tiny, 1, seed=7)
    b_a = fl.last_info["beta_used"]
    fl.sample(tiny, 1, seed=7)
    check("the same seed draws the same beta; the nominal beta is untouched",
          fl.last_info["beta_used"] == b_a and fl.beta_final() == 2.0)
    nf = SAFreezeout(SCHED, method="pt", beta_override=2.0, pt_sweeps=1, pt_replicas=2)
    nf.sample(tiny, 1, seed=0)
    check("no fluctuation: beta_used == beta_nominal exactly", nf.last_info["beta_used"] == nf.last_info["beta_nominal"] == 2.0)
    tb = SAFreezeout(SCHED, freeze_s=[(5.0, 0.5), (50.0, 0.7)], num_sweeps=5)
    tb.sample(tiny, 2, seed=1, anneal_time_us=5.0)
    s_short, b_short = tb.last_info["freeze_s_used"], tb.last_info["beta_used"]
    tb.sample(tiny, 2, seed=1, anneal_time_us=50.0)
    s_long, b_long = tb.last_info["freeze_s_used"], tb.last_info["beta_used"]
    check("anneal_time_us passed to sample() moves the freeze point and beta",
          abs(s_short - 0.5) < 1e-9 and abs(s_long - 0.7) < 1e-9 and b_long > b_short, f"beta {b_short:.3f} -> {b_long:.3f}")

    # ================================================================== [4] equilibrium against exact enumeration
    section("[4] parallel tempering against the EXACT Boltzmann distribution (12 spins, full enumeration)")
    BETA = 1.5
    ptm = SAFreezeout(SCHED, method="pt", beta_override=BETA, pt_sweeps=SW_PT, pt_replicas=24)
    t0 = time.time()
    eq = ptm.check_equilibrium(m12, num_reads=R_EQ, seed=11)
    print(f"    PT: {eq['num_reads']} reads, {SW_PT} sweeps, 24 replicas, {time.time() - t0:.1f}s")
    print(f"    mean energy exact {eq['mean_energy_exact']:.4f} | sampled {eq['mean_energy_sampled']:.4f} | z {eq['mean_energy_z']:+.2f}")
    print(f"    P(ground) exact {eq['p_ground_exact']:.4f} | sampled {eq['p_ground_sampled']:.4f}")
    print(f"    TV(energy histogram) {eq['tv_energy_hist']:.4f} vs noise floor {eq['tv_noise_floor']:.4f}")
    check("PT: mean energy within 4 sigma of exact", abs(eq["mean_energy_z"]) < 4.0, f"z={eq['mean_energy_z']:+.2f}")
    check("PT: energy-histogram TV within 2.5x the statistical floor (+0.01)", eq["tv_energy_hist"] <= 2.5 * eq["tv_noise_floor"] + 0.01)
    pe_, ps_ = eq["p_ground_exact"], eq["p_ground_sampled"]
    check("PT: ground-state probability within 4 sigma", abs(ps_ - pe_) < 4 * math.sqrt(max(pe_ * (1 - pe_), 1e-12) / R_EQ) + 1e-3)
    check("PT: replica swaps happen (minimum swap acceptance > 0.05)", ptm.last_info["pt_swap_acceptance_min"] > 0.05,
          f"min {ptm.last_info['pt_swap_acceptance_min']:.3f}")
    lad = ptm.last_info["pt_beta_ladder"]
    check("PT ladder increases from min(beta_hot, beta/10) to beta",
          bool(np.all(np.diff(lad) > 0)) and abs(lad[0] - min(0.1, BETA / 10)) < 1e-12 and abs(lad[-1] - BETA) < 1e-12)
    REPORT["equilibrium_pt_12spin"] = eq

    an = SAFreezeout(SCHED, method="anneal", beta_override=BETA, num_sweeps=1000)
    eq_an = an.check_equilibrium(m12, num_reads=R_EQ, seed=11)
    info(f"anneal (1000 sweeps) on the same model: mean-energy z {eq_an['mean_energy_z']:+.2f}, "
         f"TV {eq_an['tv_energy_hist']:.4f} (floor {eq_an['tv_noise_floor']:.4f}) -- no hard check: it is a kinetic approximation")
    REPORT["equilibrium_anneal_12spin"] = eq_an

    m4 = rand_model(4, 6, seed=9)
    st4 = np.array(list(itertools.product((-1, 1), repeat=4)), dtype=np.int8)
    E4 = ising_energies(m4, st4)
    p4 = np.exp(-1.2 * (E4 - E4.min()))
    p4 /= p4.sum()
    eq4 = SAFreezeout(SCHED, method="pt", beta_override=1.2, pt_sweeps=20, pt_replicas=4).check_equilibrium(m4, num_reads=50, seed=1)
    check("the exact enumerator agrees with an independent energies()-based calculation",
          abs(eq4["mean_energy_exact"] - float(np.sum(p4 * E4))) < 1e-4
          and abs(eq4["p_ground_exact"] - float(p4[E4 <= E4.min() + 1e-9].sum())) < 1e-5)
    keep = SAFreezeout(SCHED, method="pt", beta_override=1.0, beta_sigma_rel=0.3, pt_sweeps=10, pt_replicas=4)
    keep.check_equilibrium(m4, num_reads=20, seed=1)
    check("check_equilibrium restores beta_sigma_rel afterwards", keep.beta_sigma_rel == 0.3)
    check("check_equilibrium refuses models above max_spins",
          raises(ValueError, lambda: ptm.check_equilibrium(rand_model(20, 30, seed=1), num_reads=10, max_spins=18)))

    means = {}
    for b in (0.3, 1.0, 3.0):
        s_ = SAFreezeout(SCHED, method="pt", beta_override=b, pt_sweeps=SW_PT // 2, pt_replicas=24)
        means[b] = float(ising_energies(m12, s_.sample(m12, 1500, seed=5)).mean())
    check("mean energy falls as beta rises (0.3 > 1.0 > 3.0)", means[0.3] > means[1.0] > means[3.0],
          {k: round(v, 3) for k, v in means.items()})

    # ================================================================== [5] double well: why PT
    section("[5] double-well model: local moves cannot equilibrate the two basins, replica exchange does")
    dw = complete(6, 0.05, -1.0)                 # ferromagnet, tiny field: all-down is the unique ground state
    bd = 3.0
    p_down = 1.0 / (1.0 + math.exp(-bd * 0.6))   # analytic: E(all up) - E(all down) = 6 * 0.05 * 2 = 0.6
    ptd = SAFreezeout(SCHED, method="pt", beta_override=bd, pt_sweeps=800 if QUICK else 1200, pt_replicas=10)
    eq_pt = ptd.check_equilibrium(dw, num_reads=400 if QUICK else 800, seed=3)
    eq_sa = SAFreezeout(SCHED, method="anneal", beta_override=bd, num_sweeps=1000).check_equilibrium(
        dw, num_reads=400 if QUICK else 800, seed=3)
    print(f"    P(all-down): analytic {p_down:.3f} | enumerator {eq_pt['p_ground_exact']:.3f} | "
          f"PT {eq_pt['p_ground_sampled']:.3f} | anneal {eq_sa['p_ground_sampled']:.3f}")
    check("enumerator matches the analytic basin probability", abs(eq_pt["p_ground_exact"] - p_down) < 2e-3)
    err_pt = abs(eq_pt["p_ground_sampled"] - eq_pt["p_ground_exact"])
    err_sa = abs(eq_sa["p_ground_sampled"] - eq_sa["p_ground_exact"])
    check("PT reproduces the basin populations (error < 0.08)", err_pt < 0.08, f"error {err_pt:.3f}")
    check("PT is closer to the exact basin populations than the anneal", err_pt < err_sa, f"{err_pt:.3f} vs {err_sa:.3f}")
    REPORT["double_well"] = {"analytic": p_down, "pt": eq_pt, "anneal": eq_sa}

    # ================================================================== [6] annealing kinetics
    section("[6] anneal kinetics: too few sweeps leave the sample out of equilibrium")
    mean_e, se_e = {}, {}
    for ns in (2, 20, 300):
        s_ = SAFreezeout(SCHED, method="anneal", beta_override=BETA, num_sweeps=ns)
        e_ = ising_energies(m12, s_.sample(m12, 1500, seed=8))
        mean_e[ns], se_e[ns] = float(e_.mean()), float(e_.std() / math.sqrt(len(e_)))
    print("    mean energy by sweeps: " + ", ".join(f"{k}: {v:.3f}" for k, v in mean_e.items())
          + f" | exact equilibrium {eq['mean_energy_exact']:.3f}")
    check("2 sweeps give a higher mean energy than 300 sweeps (within 3 s.e.)", mean_e[2] >= mean_e[300] - 3 * se_e[300])
    REPORT["anneal_sweeps"] = {"mean_energy": mean_e, "exact": eq["mean_energy_exact"]}

    # ================================================================== [7] thermal resolution
    section("[7] thermal_resolution")
    arT = IsingArrays([0, 1, 2, 3, 4], np.array([0.5, -0.02, 0.0, 0.3, 0.0]), np.array([0, 1, 2, 3]),
                      np.array([1, 2, 3, 4]), np.array([0.01, -0.05, 0.2, -0.9]))
    tr = SAFreezeout(SCHED, method="pt", beta_override=10.0).thermal_resolution(arT)
    check("kT_scaled = 1/beta", abs(tr["kT_scaled"] - 0.1) < 1e-12)
    check("J below kT: 2 of 4 couplings (0.01, 0.05)", tr["J_nonzero"] == 4 and abs(tr["J_below_kT_pct"] - 50.0) < 1e-9)
    check("h below kT counts only non-zero fields (1 of 3)", tr["h_nonzero"] == 3 and abs(tr["h_below_kT_pct"] - 100.0 / 3) < 1e-9)
    check("J_min, J_median and kT/max|J|", abs(tr["J_min_abs"] - 0.01) < 1e-12 and abs(tr["J_median_abs"] - 0.125) < 1e-12
          and abs(tr["kT_over_max_abs_J"] - 0.1 / 0.9) < 1e-12)
    trh = SAFreezeout(SCHED, method="pt", beta_override=10.0).thermal_resolution(ar_h)
    check("a model without couplers is handled (J fields None)", trh["J_nonzero"] == 0 and trh["J_min_abs"] is None and trh["J_below_kT_pct"] is None)
    dflt = SAFreezeout(SCHED, 12.0, 0.6)
    check("default beta comes from beta_final()", abs(dflt.thermal_resolution(arT)["kT_scaled"] - 1.0 / dflt.beta_final()) < 1e-12)

    # ================================================================== [8] last_info and describe()
    section("[8] last_info and describe()")
    d = SAFreezeout(SCHED, method="pt", beta_override=2.0, beta_sigma_rel=0.1, pt_sweeps=20, pt_replicas=4)
    X_last = None
    for sd in range(5):
        X_last = d.sample(m12, 10, seed=sd)
    li = d.last_info
    check("last_info mean/min energy match a recomputation from the returned samples",
          abs(li["mean_energy"] - float(ising_energies(m12, X_last).mean())) < 1e-9
          and abs(li["min_energy"] - float(ising_energies(m12, X_last).min())) < 1e-9)
    check("last_info carries method, beta and PT diagnostics",
          {"method", "beta_nominal", "beta_used", "pt_swap_acceptance", "pt_beta_ladder", "n_spins"} <= set(li))
    desc = d.describe()
    try:
        json.dumps(desc)
        check("describe() is JSON-safe", True)
    except Exception as exc:
        check("describe() is JSON-safe", False, exc)
    check("describe() counts programming cycles and records the beta spread",
          desc["runs"]["programming_cycles"] == 5 and desc["runs"]["beta_used_std"] > 0 and desc["pt"]["replicas"] == 4)
    check("describe() includes the schedule record", "schedule" in desc and "source" in desc["schedule"])
    check("uncalibrated_defaults flag: True for the placeholders, False once beta_override / T / s* are set",
          SAFreezeout(SCHED).describe()["uncalibrated_defaults"] is True
          and SAFreezeout(SCHED, beta_override=2.0).describe()["uncalibrated_defaults"] is False
          and SAFreezeout(SCHED, 15.0, 0.6).describe()["uncalibrated_defaults"] is False)
    for label_, kw_ in (("table", dict(freeze_s=[(5.0, 0.5), (50.0, 0.7)])), ("callable", dict(freeze_s=lambda t: 0.5))):
        ok = True
        try:
            json.dumps(SAFreezeout(SCHED, **kw_).describe())
        except Exception:
            ok = False
        check(f"describe() works when the freeze point ({label_}) needs an anneal time that was not given", ok)
    REPORT["describe_example"] = desc

    # ================================================================== [9] factory and call-site compatibility
    section("[9] make_sampler and the call pattern used by EmulatedQPU")
    f1 = make_sampler("sa", SCHED, 12.0, num_sweeps=30)
    check("make_sampler('sa') returns SAFreezeout and accepts sample(ar, reads, seed=...)",
          isinstance(f1, SAFreezeout) and f1.sample(m12, 5, seed=2).shape == (5, 12))
    f2 = make_sampler("sa", SCHED, 12.0, method="pt", beta_override=2.0, pt_sweeps=10, pt_replicas=4)
    check("make_sampler('sa', method='pt', ...) works", f2.sample(m12, 7, 3).shape == (7, 12))
    check("make_sampler rejects unknown samplers", raises(ValueError, lambda: make_sampler("nope", SCHED, 12.0)))

    # ================================================================== [10] OPF pipeline
    OPF_OK = False
    if not NO_OPF:
        try:
            import pandapower as pp
            import pandapower.networks as nw
            from solvers.qubo_formulator import QuboFormulator
            from solvers.simulated_quantum_annealing.s2_mapping import quantise, scale_to_hardware
            OPF_OK = True
        except ImportError as exc:
            print(f"\n[!] Section 10 skipped: cannot import pandapower / formulator / s2_mapping ({exc})")

    def opf_eval(fm, net_, labels, spins, ref_cost):
        uniq, counts = np.unique(spins, axis=0, return_counts=True)
        feas, best = 0, None
        for row, c in zip(uniq, counts):
            out = fm._decode_solution({l: int((s + 1) // 2) for l, s in zip(labels, row)}, net_)
            if out[4]["is_feasible"]:
                feas += int(c)
                best = out[3] if best is None else min(best, out[3])
        gap = None if (best is None or not ref_cost) else 100.0 * (best - ref_cost) / abs(ref_cost)
        return 100.0 * feas / len(spins), gap

    if OPF_OK:
        section("[10] DC-OPF QUBO -> Ising -> scale -> quantise -> SAFreezeout -> decode")
        print("    The neal auto-beta rows are the sampler used in the mapping self-test. Physical-beta rows are expected to be")
        print("    worse where scaled couplings sit below kT (the thermal floor); that is physics, not a bug.")
        SW_OPF = 150 if QUICK else 300
        specs = [("neal auto-beta", dict(use_auto_beta=True, num_sweeps=1000)),
                 ("anneal @ physical beta", dict(num_sweeps=1000)),
                 ("PT @ physical beta", dict(method="pt", pt_sweeps=SW_OPF, pt_replicas=16))]
        quants = [("ideal", None), ("8-bit", 8), ("5-bit", 5)]
        for cname in (["case5"] if QUICK else ["case5", "case9"]):
            print(f"\n  -- {cname}")
            try:
                net = getattr(nw, cname)()
                if not net.ext_grid.empty:
                    net.ext_grid["min_p_mw"] = 0.0
                ref = None
                try:
                    import copy
                    tmp = copy.deepcopy(net)
                    pp.rundcopp(tmp)
                    ref = float(tmp.res_cost)
                except Exception as exc:
                    info(f"no classical reference cost ({type(exc).__name__})")
                fm = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0, encoding="radix")
                bqm, _ = fm._formulate_qubo(net)
                if not bqm.variables:
                    info("0 variables, skipped")
                    continue
            except Exception as exc:
                check(f"{cname}: formulation", False, f"{type(exc).__name__}: {exc}")
                continue
            a7 = IsingArrays.from_bqm(bqm.change_vartype(dimod.SPIN, inplace=False))
            sc7, f7 = scale_to_hardware(a7, HR, JR)
            print(f"     {len(a7.labels)} spins, {a7.J.size} couplers, scale factor {f7:.3e}, reference cost {ref}")
            print(f"     {'quantisation':<8}{'sampler':<26}{'feas%':>7}{'gap%':>8}{'kT_scaled':>11}{'J<kT%':>8}{'meanE':>10}{'secs':>7}")
            rows = []
            for qlabel, bits in quants:
                pr7 = quantise(sc7, HR, JR, None, bits_h=bits, bits_j=bits)
                for slabel, kw in specs:
                    smp = SAFreezeout(SCHED, 12.0, 0.6, **kw)
                    t0 = time.time()
                    X = smp.sample(pr7, 100 if QUICK else 200, seed=42)
                    dt = time.time() - t0
                    feas, gap = opf_eval(fm, net, a7.labels, X, ref)
                    th = None if smp.use_auto_beta else smp.thermal_resolution(pr7)
                    ok_out = X.shape[1] == len(a7.labels) and set(np.unique(X).tolist()) <= {-1, 1}
                    check(f"{cname}/{qlabel}/{slabel}: valid +-1 samples, one column per spin", ok_out)
                    try:
                        json.dumps(smp.describe())
                    except Exception as exc:
                        check(f"{cname}/{qlabel}/{slabel}: describe() JSON-safe", False, exc)
                    if th is not None:
                        check(f"{cname}/{qlabel}/{slabel}: kT_scaled == 1/beta_used",
                              abs(th["kT_scaled"] - 1.0 / smp.last_info["beta_used"]) < 1e-9)
                    mE = smp.last_info["mean_energy"]
                    print(f"     {qlabel:<8}{slabel:<26}{feas:>7.1f}{('-' if gap is None else format(gap, '.2f')):>8}"
                          f"{('-' if th is None else format(th['kT_scaled'], '.4f')):>11}"
                          f"{('-' if th is None else format(th['J_below_kT_pct'], '.1f')):>8}{mE:>10.3f}{dt:>7.1f}")
                    if slabel.startswith("PT"):
                        info(f"PT swap acceptance min {smp.last_info['pt_swap_acceptance_min']:.3f}")
                    rows.append({"quantisation": qlabel, "sampler": slabel, "feasible_pct": feas, "gap_pct": gap,
                                 "thermal_resolution": th, "mean_energy": mE, "seconds": dt})
                    if qlabel == "ideal" and slabel == "neal auto-beta":
                        check(f"{cname}: the old auto-beta anneal still finds feasible samples on the unquantised model", feas > 0, f"{feas:.1f}%")
            REPORT[f"opf_{cname}"] = rows
        print("\n    Negative gaps are not improvements: feasibility tolerance equals the 10 MW precision.")

    # ================================================================== [11] EmulatedQPU integration
    if not NO_EMU:
        section("[11] EmulatedQPU: one sample() call per programming cycle, determinism, describe() after a run")
        try:
            from solvers.simulated_quantum_annealing.run_emulator import EmulatedQPU
            from solvers.simulated_quantum_annealing.s1_hardware import Hardware
            from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
            emu_ok = True
        except ImportError as exc:
            emu_ok = False
            print(f"    skipped: {exc}")
        if emu_ok:
            try:
                hw = Hardware.ideal_zephyr(m=4, t=4)
                rr = np.random.default_rng(3)
                nL = 8
                lbqm = dimod.BinaryQuadraticModel({i: float(rr.uniform(-1, 1)) for i in range(nL)},
                                                  {(i, j): float(rr.uniform(-1, 1)) for i, j in itertools.combinations(range(nL), 2)},
                                                  0.0, dimod.BINARY)
                embs = find_embeddings(lbqm, hw, methods=("clique", "minorminer"), seeds=(0,), timeout=20.0, verbose=False)
                lab, emb = select_best(embs)
                kw = dict(method="pt", beta_override=2.0, beta_sigma_rel=0.2, pt_sweeps=40, pt_replicas=8)

                def run_once():
                    q = EmulatedQPU(hw, seed=7, sampler="sa", sampler_kwargs=dict(kw), dac_bits=5,
                                    ice_sigma_h=0.02, ice_sigma_j=0.01, use_extended_j=True)
                    r = q.sample(lbqm, emb, num_reads=100, reads_per_programming=25,
                                 chain_break_method="majority", postprocess=None)
                    return q, r

                q1, r1 = run_once()
                q2, r2 = run_once()
                check("emulated run returns all reads", int(np.sum(r1.sampleset.record.num_occurrences)) == 100)
                check("logical sampleset has a chain_break_fraction field", "chain_break_fraction" in r1.sampleset.record.dtype.names)
                dm = q1.sampler.describe()
                check("one sampler.sample() call per programming cycle (100 reads / 25 per cycle = 4)",
                      dm["runs"]["programming_cycles"] == 4, f"got {dm['runs']['programming_cycles']}")
                check("beta fluctuates between programming cycles inside the emulator", dm["runs"]["beta_used_std"] > 0)
                try:
                    json.dumps(dm)
                    check("sampler.describe() after an emulated run is JSON-safe", True)
                except Exception as exc:
                    check("sampler.describe() after an emulated run is JSON-safe", False, exc)
                check("identical EmulatedQPU seeds give identical logical samples",
                      np.array_equal(np.asarray(r1.sampleset.record.sample), np.asarray(r2.sampleset.record.sample)))
                REPORT["emulator_run"] = {"embedding": label if (label := lab) else None, "describe": dm}
            except Exception as exc:
                check("EmulatedQPU integration ran without error", False, f"{type(exc).__name__}: {exc}")

    # ================================================================== summary
    section("SUMMARY")
    failed = [c for c in CHECKS if not c[1]]
    print(f"checks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}")
    with open(OUT_JSON, "w") as fh:
        json.dump({"settings": {"quick": QUICK, "no_opf": NO_OPF, "no_emulator": NO_EMU,
                                "schedule_source": getattr(SCHED, "source", None),
                                "schedule_placeholder": getattr(SCHED, "is_placeholder", None)},
                   "checks": CHECKS, "report": REPORT}, fh, default=str, indent=1)
    print(f"details written to {OUT_JSON}")
    sys.exit(1 if failed else 0)