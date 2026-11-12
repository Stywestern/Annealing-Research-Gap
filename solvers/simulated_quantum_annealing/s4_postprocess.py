"""Greedy steepest-descent postprocessing (numpy, vectorised over reads), with full reporting.

For each sample, repeatedly flip the single spin whose flip lowers the Ising energy the most, until no
single flip helps (a one-flip local minimum). Optionally followed by two-flip moves on coupled pairs.
Same idea as D-Wave's SteepestDescentSolver (dwave-greedy), reimplemented so that sample order, tolerances and
reporting stay under our control.

Ising energy convention: E = sum_i h_i s_i + sum_(ij) J_ij s_i s_j (each coupler once).
Flipping spin i changes E by -2 * s_i * (h_i + sum_j J_ij s_j); gain = s_i * field_i > 0 means E drops.
Flipping the coupled pair (i, j) together: gain_ij = g_i + g_j - 2 J_ij s_i s_j (> 0 means E drops, by 2*gain_ij).

REPORTING -- every call records its settings and what it did:
  postprocess_report()   all engines: settings, cumulative statistics and the last call (JSON-safe dict)
  last_report(label)     the last call (most recent overall if label is None)
  reset_reports()        clear the statistics
  configure(**kw)        change the default settings used by steepest_descent() / descend_sampleset(); this also
                         starts fresh reports. get_config() shows the current defaults.
  descend_sampleset() also stores its call report in sampleset.info["postprocess_descent"].
Report labels: steepest_descent() uses "steepest_descent" (EmulatedQPU calls it on the PHYSICAL embedded model);
descend_sampleset() uses "logical_sampleset".
"""
from __future__ import annotations

import copy
import time

import dimod
import numpy as np
import scipy.sparse as sp

from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays, energies as ising_energies

_TIE_BREAKS = ("first", "random")


def _energies_chunked(ar: IsingArrays, X: np.ndarray, chunk: int = 64) -> np.ndarray:
    out = np.empty(X.shape[0])
    for a in range(0, X.shape[0], chunk):
        out[a:a + chunk] = ising_energies(ar, X[a:a + chunk])
    return out


class GreedyDescent:
    """Steepest descent engine. One object = one set of settings + cumulative statistics.

    tol_rel      : improvement threshold relative to the model's scale, max_i(|h_i| + sum_j |J_ij|). A move counts
                   only if its gain exceeds tol_rel * scale, so floating-point noise on large coefficients (for
                   example chain strengths ~1e5) cannot create spurious flips between exactly degenerate states.
    tol_abs      : if given, replaces tol_rel * scale with this absolute threshold.
    max_steps    : step cap per one-flip pass (default 10 * n); reads that still have an improving flip are
                   reported as not converged.
    tie_break    : "first" (deterministic: lowest index among the best moves) or "random" (random choice among
                   moves within 0.25 * tolerance of the best; seeded by `seed` and the call number).
    two_flip     : after the one-flip minimum, also apply the best improving 2-flip move on coupled pairs, then
                   re-descend, for up to two_flip_rounds rounds. Off by default.
    pair_chunk   : reads processed at a time in the pair search (memory = couplers x pair_chunk floats).
    refresh_every: the local fields are updated incrementally and recomputed exactly every this many steps,
                   and always before declaring convergence.
    """

    def __init__(self, tol_rel: float = 1e-9, tol_abs: float | None = None, max_steps: int | None = None,
                 tie_break: str = "first", seed: int = 0, two_flip: bool = False, two_flip_rounds: int = 20,
                 pair_chunk: int = 32, refresh_every: int = 64, label: str = "steepest_descent"):
        if tie_break not in _TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {_TIE_BREAKS}, got {tie_break!r}")
        if tol_rel < 0 or (tol_abs is not None and tol_abs < 0):
            raise ValueError("tolerances must be >= 0")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1 (or None for the default 10 * n)")
        if two_flip_rounds < 1 or pair_chunk < 1 or refresh_every < 1:
            raise ValueError("two_flip_rounds, pair_chunk and refresh_every must be >= 1")
        self.tol_rel, self.tol_abs, self.max_steps = float(tol_rel), tol_abs, max_steps
        self.tie_break, self.seed, self.two_flip = tie_break, int(seed), bool(two_flip)
        self.two_flip_rounds, self.pair_chunk, self.refresh_every = int(two_flip_rounds), int(pair_chunk), int(refresh_every)
        self.label = label
        self.reset()

    # ------------------------------------------------------------------ bookkeeping
    def reset(self) -> None:
        self._calls = 0
        self.last_info: dict = {}
        self._cum = {"calls": 0, "reads": 0, "flips": 0, "pair_moves": 0, "reads_changed": 0,
                     "reads_not_converged": 0, "energy_drop_sum": 0.0, "seconds": 0.0}

    def settings(self) -> dict:
        return {"tol_rel": self.tol_rel, "tol_abs": self.tol_abs, "max_steps": self.max_steps,
                "tie_break": self.tie_break, "seed": self.seed, "two_flip": self.two_flip,
                "two_flip_rounds": self.two_flip_rounds, "pair_chunk": self.pair_chunk,
                "refresh_every": self.refresh_every}

    def describe(self) -> dict:
        """JSON-safe: settings, cumulative statistics over all calls so far, and the last call."""
        c = self._cum
        cum = dict(c)
        cum["energy_drop_mean_per_read"] = (c["energy_drop_sum"] / c["reads"]) if c["reads"] else None
        cum["reads_changed_pct"] = (100.0 * c["reads_changed"] / c["reads"]) if c["reads"] else None
        return {"class": "GreedyDescent", "label": self.label, "settings": self.settings(),
                "cumulative": cum, "last_call": copy.deepcopy(self.last_info)}

    def _finish(self, info: dict) -> None:
        self.last_info = info
        c = self._cum
        c["calls"] += 1
        c["reads"] += info["n_reads"]
        c["flips"] += info["flips"]
        c["pair_moves"] += info["pair_moves"]
        c["reads_changed"] += info["reads_changed"]
        c["reads_not_converged"] += info["reads_not_converged"]
        c["energy_drop_sum"] += info["energy_drop_mean"] * info["n_reads"]
        c["seconds"] += info["seconds"]

    # ------------------------------------------------------------------ moves
    def _one_flip(self, s, Jm, Jcsc, h, tol, cap, rng):
        """In-place steepest descent on s (n, R). Returns (flips, steps_per_read)."""
        n, R = s.shape
        steps = np.zeros(R, dtype=np.int64)
        flips = 0
        field = h[:, None] + Jm @ s
        fresh = True
        for step in range(cap):
            if step and step % self.refresh_every == 0:
                field = h[:, None] + Jm @ s
                fresh = True
            gain = s * field                                    # > 0 where flipping lowers the energy
            gmax = gain.max(axis=0)
            active = gmax > tol
            if not active.any():
                if fresh:
                    break                                       # converged on exact fields
                field = h[:, None] + Jm @ s                     # incremental fields may have drifted: re-check
                fresh = True
                continue
            if self.tie_break == "first":
                i = gain.argmax(axis=0)
            else:
                near_best = gain >= (gmax - 0.25 * tol)[None, :]
                i = np.where(near_best, rng.random(gain.shape), -1.0).argmax(axis=0)
            ca = np.nonzero(active)[0]
            ia = i[ca]
            old = s[ia, ca]
            s[ia, ca] = -old
            upd = Jcsc[:, ia] @ sp.diags(-2.0 * old)            # field change of every neighbour: -2 J_ji s_i(old)
            field[:, ca] += upd.toarray()
            steps[ca] += 1
            flips += int(ca.size)
            fresh = False
        return flips, steps

    def _pair_pass(self, s, Jm, h, rows, cols, Jv, tol) -> int:
        """Apply, for each read, the best improving 2-flip move over COUPLED pairs (at a one-flip minimum an
        uncoupled pair cannot improve). In place. Returns the number of reads changed."""
        n, R = s.shape
        g = s * (h[:, None] + Jm @ s)
        applied = 0
        for r0 in range(0, R, self.pair_chunk):
            sl = slice(r0, min(R, r0 + self.pair_chunk))
            gs, ss = g[:, sl], s[:, sl]
            gain2 = gs[rows] + gs[cols] - 2.0 * Jv[:, None] * ss[rows] * ss[cols]       # (couplers, k)
            e = gain2.argmax(axis=0)
            k = np.arange(gain2.shape[1])
            idx = np.nonzero(gain2[e, k] > tol)[0]
            if idx.size:
                rr = r0 + idx
                s[rows[e[idx]], rr] *= -1.0
                s[cols[e[idx]], rr] *= -1.0
                applied += int(idx.size)
        return applied

    # ------------------------------------------------------------------ main entry
    def run(self, ar: IsingArrays, samples: np.ndarray, max_steps: int | None = None, tol_abs: float | None = None):
        """samples: (num_reads, n) of +/-1 in ar.labels order. Returns (new_samples int8, total_flips, info).
        max_steps / tol_abs given here override the engine settings for this call only (and are reported)."""
        t0 = time.perf_counter()
        X = np.asarray(samples)
        if X.ndim != 2:
            raise ValueError("samples must be 2-D (num_reads, n)")
        n, R = len(ar.labels), X.shape[0]
        if X.shape[1] != n:
            raise ValueError(f"samples have {X.shape[1]} columns but the model has {n} spins")
        self._calls += 1
        cap_setting = max_steps if max_steps is not None else self.max_steps
        tol_setting = tol_abs if tol_abs is not None else self.tol_abs
        base = {"label": self.label, "call": self._calls, "n_reads": int(R), "n_spins": int(n),
                "settings": {**self.settings(), "max_steps": cap_setting, "tol_abs": tol_setting}}
        if n == 0 or R == 0:
            info = {**base, "tol_used": 0.0, "scale": 0.0, "max_steps_used": 0, "flips": 0, "pair_moves": 0,
                    "two_flip_rounds_used": 0, "two_flip_converged": None, "reads_changed": 0, "reads_changed_pct": 0.0,
                    "steps_mean": 0.0, "steps_max": 0, "reads_not_converged": 0, "one_flip_local_minima_pct": 100.0,
                    "energy_before_mean": 0.0, "energy_before_min": 0.0, "energy_after_mean": 0.0,
                    "energy_after_min": 0.0, "energy_drop_mean": 0.0, "energy_increased_reads": 0,
                    "hamming_mean": 0.0, "hamming_mean_frac": 0.0, "hamming_max": 0,
                    "seconds": time.perf_counter() - t0}
            self._finish(info)
            return X.astype(np.int8), 0, info
        if not np.all((X == 1) | (X == -1)):
            raise ValueError("samples must be +-1 spins (convert BINARY samples first)")

        Jm = sp.coo_matrix((np.r_[ar.J, ar.J], (np.r_[ar.rows, ar.cols], np.r_[ar.cols, ar.rows])),
                           shape=(n, n)).tocsr()
        Jcsc = Jm.tocsc()
        h = ar.h.astype(np.float64)
        scale = float(np.max(np.abs(h) + np.asarray(abs(Jm).sum(axis=1)).ravel()))   # bound on any |field|
        tol = float(tol_setting) if tol_setting is not None else self.tol_rel * scale
        cap = int(cap_setting) if cap_setting is not None else 10 * n
        rng = np.random.default_rng([self.seed, self._calls])

        e0 = _energies_chunked(ar, X)
        s = X.T.astype(np.float64)                                                  # (n, R), a copy
        s_init = s.copy()
        flips, steps = self._one_flip(s, Jm, Jcsc, h, tol, cap, rng)

        pair_moves, rounds_used, pair_converged = 0, 0, None
        if self.two_flip and ar.rows.size:
            pair_converged = False
            for _ in range(self.two_flip_rounds):
                applied = self._pair_pass(s, Jm, h, ar.rows, ar.cols, ar.J.astype(np.float64), tol)
                rounds_used += 1
                if applied == 0:
                    pair_converged = True
                    break
                pair_moves += applied
                f2, st2 = self._one_flip(s, Jm, Jcsc, h, tol, cap, rng)
                flips += 2 * applied + f2
                steps += st2

        gmax_final = (s * (h[:, None] + Jm @ s)).max(axis=0)                        # exact final check
        unconverged = gmax_final > tol
        out = s.T.astype(np.int8)
        e1 = _energies_chunked(ar, out)
        ham = (s != s_init).sum(axis=0)
        e_tol = 1e-9 * max(1.0, float(np.abs(e0).max()))
        info = {**base,
                "tol_used": tol, "scale": scale, "max_steps_used": cap,
                "flips": int(flips), "pair_moves": int(pair_moves),
                "two_flip_rounds_used": int(rounds_used), "two_flip_converged": pair_converged,
                "reads_changed": int((ham > 0).sum()), "reads_changed_pct": float(100.0 * np.mean(ham > 0)),
                "steps_mean": float(steps.mean()), "steps_max": int(steps.max()),
                "reads_not_converged": int(unconverged.sum()),
                "one_flip_local_minima_pct": float(100.0 * np.mean(~unconverged)),
                "energy_before_mean": float(e0.mean()), "energy_before_min": float(e0.min()),
                "energy_after_mean": float(e1.mean()), "energy_after_min": float(e1.min()),
                "energy_drop_mean": float((e0 - e1).mean()),
                "energy_increased_reads": int((e1 > e0 + e_tol).sum()),          # sanity: must be 0
                "hamming_mean": float(ham.mean()), "hamming_mean_frac": float(ham.mean() / n),
                "hamming_max": int(ham.max()), "seconds": time.perf_counter() - t0}
        self._finish(info)
        return out, int(flips), info


# ---------------------------------------------------------------------- module-level interface
_DEFAULTS = dict(tol_rel=1e-9, tol_abs=None, max_steps=None, tie_break="first", seed=0,
                 two_flip=False, two_flip_rounds=20, pair_chunk=32, refresh_every=64)
_CONFIG = dict(_DEFAULTS)
_ENGINES: dict = {}
_LAST_LABEL = None


def configure(**kw) -> dict:
    """Change the default settings used by steepest_descent() and descend_sampleset(), for example
    configure(two_flip=True) or configure(tie_break='random', seed=3). Starts fresh reports. Returns the config."""
    unknown = set(kw) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}; available: {sorted(_DEFAULTS)}")
    trial = {**_CONFIG, **kw}
    GreedyDescent(**trial)                      # validates the combination before it is applied
    _CONFIG.update(kw)
    _ENGINES.clear()
    return dict(_CONFIG)


def get_config() -> dict:
    return dict(_CONFIG)


def _engine(label: str) -> GreedyDescent:
    e = _ENGINES.get(label)
    if e is None:
        e = _ENGINES[label] = GreedyDescent(label=label, **_CONFIG)
    return e


def postprocess_report() -> dict:
    """JSON-safe report of every engine used so far: settings, cumulative statistics and the last call."""
    return {"config": get_config(), "engines": {k: e.describe() for k, e in _ENGINES.items()}}


def last_report(label: str | None = None) -> dict:
    lab = label if label is not None else _LAST_LABEL
    e = _ENGINES.get(lab)
    return copy.deepcopy(e.last_info) if e is not None else {}


def reset_reports() -> None:
    for e in _ENGINES.values():
        e.reset()


def steepest_descent(ar: IsingArrays, samples: np.ndarray, max_steps: int | None = None, tol: float | None = None,
                     label: str = "steepest_descent"):
    """samples: (num_reads, n) of +/-1 in ar.labels order. Returns (new_samples int8, total_flips).
    tol, if given, is an ABSOLUTE improvement threshold (the default is relative to the model's scale, see
    GreedyDescent). The call's full report is available from last_report(label) / postprocess_report()."""
    global _LAST_LABEL
    new, flips, _ = _engine(label).run(ar, samples, max_steps=max_steps, tol_abs=tol)
    _LAST_LABEL = label
    return new, flips


def descend_sampleset(sampleset: dimod.SampleSet, spin_bqm: dimod.BinaryQuadraticModel,
                      label: str = "logical_sampleset"):
    """Steepest descent on an already-unembedded LOGICAL SPIN sampleset. Keeps num_occurrences, extra vectors
    (e.g. chain_break_fraction, which still describes the samples BEFORE this descent) and sampleset.info,
    adds the call report as info['postprocess_descent'], recomputes energies. Returns (sampleset, total_flips)."""
    global _LAST_LABEL
    if spin_bqm.vartype is not dimod.SPIN or sampleset.vartype is not dimod.SPIN:
        raise ValueError("descend_sampleset needs a SPIN sampleset and a SPIN bqm")
    ar = IsingArrays.from_bqm(spin_bqm)
    pos = {v: i for i, v in enumerate(sampleset.variables)}
    arr = np.asarray(sampleset.record.sample)[:, [pos[v] for v in ar.labels]]
    new, flips, info = _engine(label).run(ar, arr)
    _LAST_LABEL = label
    energies = spin_bqm.energies((new, ar.labels))
    extra = {k: sampleset.record[k] for k in sampleset.record.dtype.names
             if k not in ("sample", "energy", "num_occurrences")}
    meta = dict(sampleset.info)
    meta["postprocess_descent"] = copy.deepcopy(info)
    out = dimod.SampleSet.from_samples((new, ar.labels), energy=energies, vartype=dimod.SPIN,
                                       num_occurrences=sampleset.record.num_occurrences, info=meta, **extra)
    return out, flips


# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import itertools
    import json
    import math
    import sys

    QUICK = False
    NO_OPF = False
    NO_EMU = False
    NO_REF = False
    OUT_JSON = "postprocessing_selftest.json"
    CHECKS, REPORT = [], {}
    R_SYN = 150 if QUICK else 400
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

    def rand_model(n, m, seed, jscale=1.0, hscale=0.5):
        r = np.random.default_rng(seed)
        pairs = list(itertools.combinations(range(n), 2))
        sel = r.choice(len(pairs), size=min(m, len(pairs)), replace=False)
        return IsingArrays(list(range(n)), r.uniform(-hscale, hscale, n),
                           np.array([pairs[k][0] for k in sel], dtype=int), np.array([pairs[k][1] for k in sel], dtype=int),
                           r.uniform(-jscale, jscale, len(sel)))

    def rand_spins(R, n, seed):
        return np.random.default_rng(seed).choice(np.array([-1, 1], dtype=np.int8), size=(R, n))

    def dense_sym(ar):
        S = np.zeros((len(ar.labels), len(ar.labels)))
        np.add.at(S, (ar.rows, ar.cols), ar.J)
        np.add.at(S, (ar.cols, ar.rows), ar.J)
        return S

    def model_scale(ar):
        return float(np.max(np.abs(ar.h) + np.abs(dense_sym(ar)).sum(axis=1)))

    def gains(ar, X):
        Xf = X.astype(float)
        return Xf * (ar.h[None, :] + Xf @ dense_sym(ar))

    def is_one_flip_min(ar, X, rel=1e-8):
        """Independent check (dense algebra): no single flip lowers the energy by more than rel * scale."""
        return gains(ar, X).max(axis=1) <= rel * model_scale(ar)

    def naive_descent(ar, X, tol):
        """Plain per-read reference implementation: recompute all gains, flip the best one, repeat."""
        S, h = dense_sym(ar), ar.h
        out = X.astype(float).copy()
        for r in range(out.shape[0]):
            s = out[r]
            for _ in range(10 * len(ar.labels)):
                g = s * (h + S @ s)
                i = int(np.argmax(g))
                if g[i] <= tol:
                    break
                s[i] = -s[i]
        return out.astype(np.int8)

    def exhaustive_min(ar, X, tol, pairs=False):
        """True where no single flip (and, if pairs, no coupled-pair flip) lowers the energy by more than tol."""
        E = ising_energies(ar, X)
        ok = np.ones(X.shape[0], dtype=bool)
        moves = [(i,) for i in range(X.shape[1])]
        if pairs:
            moves += [(int(a), int(b)) for a, b in zip(ar.rows, ar.cols)]
        for mv in moves:
            Y = X.copy()
            for i in mv:
                Y[:, i] = -Y[:, i]
            ok &= ising_energies(ar, Y) >= E - tol
        return ok

    def zlow(a, b):
        """z-score for 'mean(a) is lower than mean(b)'; positive supports it."""
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) + 1e-300
        return float((b.mean() - a.mean()) / se)

    def neal_spins(a, reads, seed, sweeps=1000):
        try:
            from dwave.samplers import SimulatedAnnealingSampler as _SA
        except ImportError:
            try:
                from neal import SimulatedAnnealingSampler as _SA
            except ImportError:
                return None
        res = _SA().sample(a.to_bqm(), num_reads=reads, num_sweeps=sweeps, seed=int(seed))
        pos = {v: i for i, v in enumerate(res.variables)}
        return np.asarray(res.record.sample)[:, [pos[v] for v in a.labels]].astype(np.int8)

    print("=== Testing the postprocessing module (steepest descent, optional two-flip, reporting) ===")
    print(f"mode={'quick' if QUICK else 'full'} | opf={'off' if NO_OPF else 'on'} | emulator={'off' if NO_EMU else 'on'}")

    # ================================================================== [1] construction
    section("[1] construction and validation")
    check("unknown tie_break raises ValueError", raises(ValueError, lambda: GreedyDescent(tie_break="x")))
    check("negative tolerance raises ValueError", raises(ValueError, lambda: GreedyDescent(tol_rel=-1.0)))
    check("max_steps < 1 raises ValueError", raises(ValueError, lambda: GreedyDescent(max_steps=0)))
    check("two_flip_rounds < 1 raises ValueError", raises(ValueError, lambda: GreedyDescent(two_flip_rounds=0)))
    eng0 = GreedyDescent(tol_rel=1e-7, max_steps=50, tie_break="random", seed=4, two_flip=True, two_flip_rounds=3,
                         pair_chunk=8, refresh_every=5)
    st = eng0.settings()
    check("settings() round-trips the constructor arguments", st["tol_rel"] == 1e-7 and st["max_steps"] == 50
          and st["tie_break"] == "random" and st["seed"] == 4 and st["two_flip"] is True
          and st["two_flip_rounds"] == 3 and st["pair_chunk"] == 8 and st["refresh_every"] == 5)
    try:
        json.dumps(GreedyDescent().describe())
        check("describe() of a fresh engine is JSON-safe", True)
    except Exception as exc:
        check("describe() of a fresh engine is JSON-safe", False, exc)

    # ================================================================== [2] basic correctness
    section("[2] basic correctness of one-flip steepest descent")
    m30 = rand_model(30, 100, seed=1)
    X30 = rand_spins(80, 30, seed=2)
    X30_copy = X30.copy()
    eng = GreedyDescent()
    out, flips, inf = eng.run(m30, X30)
    check("output: shape (reads, n), dtype int8, values +-1",
          out.shape == X30.shape and out.dtype == np.int8 and set(np.unique(out).tolist()) <= {-1, 1})
    check("the input array is not modified", np.array_equal(X30, X30_copy))
    e_in, e_out = ising_energies(m30, X30), ising_energies(m30, out)
    check("energy never increases, read by read", bool(np.all(e_out <= e_in + 1e-9)))
    check("report counter energy_increased_reads == 0", inf["energy_increased_reads"] == 0)
    check("every output is a one-flip local minimum (independent dense check)", bool(is_one_flip_min(m30, out).all()))
    check("report: reads_not_converged == 0 and one_flip_local_minima_pct == 100",
          inf["reads_not_converged"] == 0 and abs(inf["one_flip_local_minima_pct"] - 100.0) < 1e-12)
    check("report energies match a recomputation",
          abs(inf["energy_before_mean"] - e_in.mean()) < 1e-9 and abs(inf["energy_after_mean"] - e_out.mean()) < 1e-9
          and abs(inf["energy_after_min"] - e_out.min()) < 1e-9 and abs(inf["energy_drop_mean"] - (e_in - e_out).mean()) < 1e-9)
    ham = (out != X30).sum(axis=1)
    check("report Hamming statistics match a recomputation",
          abs(inf["hamming_mean"] - ham.mean()) < 1e-12 and inf["hamming_max"] == int(ham.max())
          and abs(inf["hamming_mean_frac"] - ham.mean() / 30) < 1e-12 and inf["reads_changed"] == int((ham > 0).sum()))
    check("flips returned equals the number of spins that changed when each is flipped at most once", flips >= int(ham.sum()) and flips == inf["flips"],
          f"flips={flips}, hamming total={int(ham.sum())}")
    out2, flips2, _ = eng.run(m30, out)
    check("idempotent: descending a local minimum changes nothing", flips2 == 0 and np.array_equal(out2, out))
    check("tol_used == tol_rel * model scale", abs(inf["tol_used"] - 1e-9 * model_scale(m30)) <= 1e-12 * model_scale(m30) + 1e-18,
          f"scale {inf['scale']:.4g}")

    labs = [30, 5, 17, 2, 41]
    hh = np.array([1.0, -1.0, 2.0, -0.5, 3.0])
    arF = IsingArrays(labs, hh, np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    oF, _, _ = GreedyDescent().run(arF, np.ones((6, 5), dtype=np.int8))
    check("field-only model: columns follow ar.labels (result is -sign(h) in column order)", np.all(oF == (-np.sign(hh)).astype(np.int8)[None, :]))

    empty = IsingArrays([], np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    o0, f0, i0 = GreedyDescent().run(empty, np.zeros((5, 0), dtype=np.int8))
    check("zero spins -> (R, 0) array and zero flips", o0.shape == (5, 0) and f0 == 0)
    o1, f1, i1 = GreedyDescent().run(m30, np.zeros((0, 30), dtype=np.int8))
    check("zero reads -> (0, n) array and zero flips", o1.shape == (0, 30) and f1 == 0 and i1["n_reads"] == 0)
    check("BINARY (0/1) samples are rejected", raises(ValueError, lambda: GreedyDescent().run(m30, np.random.default_rng(1).integers(0, 2, (4, 30)))))
    check("wrong column count is rejected", raises(ValueError, lambda: GreedyDescent().run(m30, rand_spins(4, 29, 1))))
    check("1-D samples are rejected", raises(ValueError, lambda: GreedyDescent().run(m30, np.ones(30, dtype=np.int8))))
    check("module-level steepest_descent agrees with the engine (same output, same flip count)",
          (lambda r: np.array_equal(r[0], out) and r[1] == flips)(steepest_descent(m30, X30)))

    # ================================================================== [3] reference implementation + exhaustive
    section("[3] against a naive reference implementation and exhaustive neighbourhood checks")
    mR = rand_model(40, 200, seed=7)
    XR = rand_spins(100, 40, seed=8)
    tolR = 1e-9 * model_scale(mR)
    ref_out = naive_descent(mR, XR, tolR)
    o_exact, _, _ = GreedyDescent(tol_abs=tolR, refresh_every=1).run(mR, XR)
    check("refresh_every=1 reproduces the naive reference exactly", np.array_equal(o_exact, ref_out),
          f"{100.0 * np.mean(np.all(o_exact == ref_out, axis=1)):.1f}% of reads identical")
    o_inc, _, _ = GreedyDescent(tol_abs=tolR, refresh_every=64).run(mR, XR)
    frac_inc = float(np.mean(np.all(o_inc == o_exact, axis=1)))
    check("incremental fields (refresh_every=64) agree with exact fields on >= 99% of reads", frac_inc >= 0.99, f"{100 * frac_inc:.1f}%")
    m12 = rand_model(12, 40, seed=9)
    X12 = rand_spins(60, 12, seed=10)
    o12, _, _ = GreedyDescent().run(m12, X12)
    check("exhaustive: no single flip improves any output (12 spins)", bool(exhaustive_min(m12, o12, 1e-8 * model_scale(m12)).all()))
    check("exhaustive: every output has energy <= its start", bool(np.all(ising_energies(m12, o12) <= ising_energies(m12, X12) + 1e-9)))
    import time as _t
    big = rand_model(400, 2400, seed=11)
    Xbig = rand_spins(150, 400, seed=12)
    t0 = _t.perf_counter(); GreedyDescent().run(big, Xbig); t_eng = _t.perf_counter() - t0
    t0 = _t.perf_counter(); naive_descent(big, Xbig[:30], 1e-9 * model_scale(big)); t_naive = _t.perf_counter() - t0
    info(f"timing, 400 spins / 2400 couplers: engine {t_eng:.2f}s for 150 reads; naive reference {t_naive:.2f}s for only 30 reads")

    # ================================================================== [4] ties, tolerance, caps
    section("[4] degenerate states, tolerance and step cap")
    n16 = 16
    pr16 = list(itertools.combinations(range(n16), 2))
    big_scale = 1.0e5
    ferro = IsingArrays(list(range(n16)), np.zeros(n16), np.array([p[0] for p in pr16]), np.array([p[1] for p in pr16]),
                        np.full(len(pr16), -big_scale))
    oFe, fFe, iFe = GreedyDescent().run(ferro, np.ones((20, n16), dtype=np.int8))
    check("aligned ferromagnet with coefficients ~1e5: zero flips (no spurious moves from roundoff)", fFe == 0 and iFe["reads_not_converged"] == 0)
    rg = np.random.default_rng(13)
    glass = IsingArrays(list(range(n16)), np.zeros(n16), ferro.rows, ferro.cols, big_scale * rg.choice([-1.0, 1.0], len(pr16)))
    Xg0 = rand_spins(100, n16, seed=14)
    oG, fG, iG = GreedyDescent().run(glass, Xg0)
    check("highly degenerate +-1e5 model: all reads converge, no cycling, no energy increase",
          iG["reads_not_converged"] == 0 and iG["energy_increased_reads"] == 0 and iG["steps_max"] < 10 * n16,
          f"flips {fG}, max steps {iG['steps_max']}")
    check("degenerate model: outputs are one-flip local minima", bool(is_one_flip_min(glass, oG).all()))
    _, fTiny, iTiny = GreedyDescent(tol_abs=1e-14).run(glass, Xg0)
    info(f"same model with an absolute tolerance of 1e-14: flips {fTiny} (default tolerance: {fG}), not converged {iTiny['reads_not_converged']}")
    o_cap, _, i_cap = GreedyDescent(max_steps=1).run(m30, X30)
    check("max_steps=1: not-converged reads are reported", i_cap["reads_not_converged"] > 0 and i_cap["max_steps_used"] == 1,
          f"{i_cap['reads_not_converged']} of {i_cap['n_reads']}")
    o_cap2, _ = steepest_descent(m30, X30, max_steps=1)
    check("per-call max_steps override takes effect and is reported", last_report()["max_steps_used"] == 1
          and last_report()["settings"]["max_steps"] == 1)
    o_hi, f_hi, i_hi = GreedyDescent().run(m30, X30, tol_abs=1e9)
    check("absolute tolerance override: huge tolerance -> no flips; tol_used reported", f_hi == 0 and i_hi["tol_used"] == 1e9
          and np.array_equal(o_hi, X30))

    # ================================================================== [5] pair moves
    section("[5] two-flip moves on coupled pairs")
    a2 = IsingArrays([0, 1], np.array([1.0, 1.0]), np.array([0]), np.array([1]), np.array([-1.5]))
    X2 = np.array([[1, 1]], dtype=np.int8)
    o_a, f_a, _ = GreedyDescent().run(a2, X2)
    o_b, f_b, i_b = GreedyDescent(two_flip=True).run(a2, X2)
    check("hand example: (+,+) is a one-flip minimum (energy 0.5)", np.array_equal(o_a, X2) and f_a == 0
          and abs(float(ising_energies(a2, X2)[0]) - 0.5) < 1e-12)
    check("hand example: the pair move reaches (-,-) with energy -3.5", np.array_equal(o_b, np.array([[-1, -1]])) and i_b["pair_moves"] == 1
          and abs(float(ising_energies(a2, o_b)[0]) + 3.5) < 1e-12)
    m14 = rand_model(14, 60, seed=15)
    s14 = rand_spins(30, 14, seed=16)
    S14 = dense_sym(m14)
    g14 = gains(m14, s14)
    E14 = ising_energies(m14, s14)
    maxdev = 0.0
    for a_, b_, J_ in zip(m14.rows, m14.cols, m14.J):
        Y = s14.copy(); Y[:, a_] = -Y[:, a_]; Y[:, b_] = -Y[:, b_]
        dE = ising_energies(m14, Y) - E14
        pred = -2.0 * (g14[:, a_] + g14[:, b_] - 2.0 * J_ * s14[:, a_] * s14[:, b_])
        maxdev = max(maxdev, float(np.abs(dE - pred).max()))
    check("pair-gain formula g_i + g_j - 2 J_ij s_i s_j matches the true energy change of the double flip", maxdev < 1e-9, f"max deviation {maxdev:.2e}")
    X14 = rand_spins(60, 14, seed=17)
    o1f, _, _ = GreedyDescent().run(m14, X14)
    o2f, _, i2f = GreedyDescent(two_flip=True).run(m14, X14)
    tol14 = 1e-8 * model_scale(m14)
    check("two_flip: outputs are minima for all single flips AND all coupled-pair flips (exhaustive)", bool(exhaustive_min(m14, o2f, tol14, pairs=True).all()))
    check("two_flip: final energy <= one-flip-only energy, read by read",
          bool(np.all(ising_energies(m14, o2f) <= ising_energies(m14, o1f) + 1e-9)))
    check("two_flip converged within its rounds", i2f["two_flip_converged"] is True, f"rounds {i2f['two_flip_rounds_used']}, pair moves {i2f['pair_moves']}")
    w = np.array([1, 2, 4, 8, 16, 1, 2, 4, 8, 16], dtype=float)
    L = 25.0
    pq = dimod.BinaryQuadraticModel({i: float(w[i] ** 2 - 2 * L * w[i]) for i in range(10)},
                                    {(i, j): float(2 * w[i] * w[j]) for i in range(10) for j in range(i + 1, 10)}, L ** 2, dimod.BINARY)
    ps = pq.change_vartype(dimod.SPIN, inplace=False)
    par = IsingArrays.from_bqm(ps)
    Xp = rand_spins(600, 10, seed=18)
    ob1, _, _ = GreedyDescent().run(par, Xp)
    ob2, _, ib2 = GreedyDescent(two_flip=True).run(par, Xp)
    sat1 = float(np.mean(ps.energies((ob1, par.labels)) < 1e-6))
    sat2 = float(np.mean(ps.energies((ob2, par.labels)) < 1e-6))
    print(f"    penalty model (sum of binary-weighted bits = 25, constraint-like): reads satisfying it: one-flip {100 * sat1:.1f}% | + two-flip {100 * sat2:.1f}%")
    check("penalty model: two-flip never satisfies fewer reads than one-flip", sat2 >= sat1 - 1e-12)
    REPORT["pair_moves_penalty_model"] = {"one_flip_satisfied_pct": 100 * sat1, "two_flip_satisfied_pct": 100 * sat2}

    # ================================================================== [6] tie-breaking
    section("[6] tie-breaking")
    ring = IsingArrays(list(range(20)), np.zeros(20), np.arange(20), (np.arange(20) + 1) % 20, np.ones(20))   # antiferromagnetic ring
    X_tie = np.ones((40, 20), dtype=np.int8)
    o_first, _, _ = GreedyDescent(tie_break="first").run(ring, X_tie)
    o_rand, _, _ = GreedyDescent(tie_break="random", seed=1).run(ring, X_tie)
    o_rand2, _, _ = GreedyDescent(tie_break="random", seed=1).run(ring, X_tie)
    o_rand3, _, _ = GreedyDescent(tie_break="random", seed=2).run(ring, X_tie)
    check("first: all identical starts end in one identical state", len(np.unique(o_first, axis=0)) == 1)
    check("random: identical starts spread over several minima", len(np.unique(o_rand, axis=0)) > 1, f"{len(np.unique(o_rand, axis=0))} distinct")
    check("random: reproducible for the same seed, different for another seed", np.array_equal(o_rand, o_rand2) and not np.array_equal(o_rand, o_rand3))
    check("random: outputs are still one-flip local minima", bool(is_one_flip_min(ring, o_rand).all()))

    # ================================================================== [7] against dwave-greedy
    section("[7] against Ocean's SteepestDescentSolver (same starting states)")
    if NO_REF:
        info("skipped (--no-greedy-compare)")
    else:
        try:
            from dwave.samplers import SteepestDescentSolver
            mg = rand_model(40, 200, seed=21)
            Xg = rand_spins(120, 40, seed=22)
            ours, _, _ = GreedyDescent().run(mg, Xg)
            ref_ss = SteepestDescentSolver().sample(mg.to_bqm(), initial_states=(Xg, mg.labels))
            pos = {v: i for i, v in enumerate(ref_ss.variables)}
            theirs = np.asarray(ref_ss.record.sample)[:, [pos[v] for v in mg.labels]].astype(np.int8)
            same = float(np.mean(np.all(ours == theirs, axis=1)))
            e_o, e_t = ising_energies(mg, ours), ising_energies(mg, theirs)
            check("dwave-greedy outputs are one-flip minima by our independent check", bool(is_one_flip_min(mg, theirs).all()))
            check("same final state on >= 90% of reads, or mean energy equal to 1e-6 relative",
                  same >= 0.9 or abs(e_o.mean() - e_t.mean()) <= 1e-6 * abs(e_t.mean()),
                  f"identical {100 * same:.1f}%, mean energy ours {e_o.mean():.5f} vs dwave-greedy {e_t.mean():.5f}")
            REPORT["dwave_greedy_compare"] = {"identical_pct": 100 * same, "mean_energy_ours": float(e_o.mean()), "mean_energy_dwave": float(e_t.mean())}
        except ImportError:
            info("dwave.samplers.SteepestDescentSolver not available; skipped")
        except Exception as exc:
            info(f"comparison could not run ({type(exc).__name__}: {exc}); skipped")

    # ================================================================== [8] configure + reports
    section("[8] module-level configuration and reporting")
    DEF = dict(_DEFAULTS)
    check("configuration starts at the defaults", get_config() == DEF)
    reset_reports()
    Xa, Xb = rand_spins(20, 30, seed=31), rand_spins(35, 30, seed=32)
    steepest_descent(m30, Xa)
    steepest_descent(m30, Xb)
    rep = postprocess_report()
    cum = rep["engines"]["steepest_descent"]["cumulative"]
    check("cumulative counters add up over calls", cum["calls"] == 2 and cum["reads"] == 55, f"calls {cum['calls']}, reads {cum['reads']}")
    check("last_call describes the second call", rep["engines"]["steepest_descent"]["last_call"]["call"] == 2
          and rep["engines"]["steepest_descent"]["last_call"]["n_reads"] == 35)
    lr = last_report()
    lr["n_reads"] = -1
    check("last_report() returns a copy", last_report()["n_reads"] == 35)
    steepest_descent(m30, Xa, label="custom")
    check("a custom label creates its own engine", "custom" in postprocess_report()["engines"] and last_report()["label"] == "custom")
    try:
        json.dumps(postprocess_report())
        check("postprocess_report() is JSON-safe", True)
    except Exception as exc:
        check("postprocess_report() is JSON-safe", False, exc)
    reset_reports()
    check("reset_reports() zeroes the statistics", all(e["cumulative"]["calls"] == 0 and e["cumulative"]["reads"] == 0
                                                     for e in postprocess_report()["engines"].values()))
    check("configure() rejects unknown settings", raises(ValueError, lambda: configure(nonsense=1)))
    check("configure() rejects an invalid value and leaves the config unchanged",
          raises(ValueError, lambda: configure(tie_break="nope")) and get_config() == DEF)
    configure(two_flip=True)
    check("configure(two_flip=True) changes the defaults", get_config()["two_flip"] is True)
    steepest_descent(m30, Xa)
    check("new engines pick the configuration up and report it", postprocess_report()["engines"]["steepest_descent"]["settings"]["two_flip"] is True
          and last_report()["settings"]["two_flip"] is True)
    configure(**DEF)
    check("configuration restored to the defaults", get_config() == DEF)

    # ================================================================== [9] descend_sampleset
    section("[9] descend_sampleset on a logical SPIN sampleset")
    lab12 = [f"v{i}" for i in range(12)]
    rr = np.random.default_rng(41)
    sbqm = dimod.BinaryQuadraticModel.from_ising({l: float(v) for l, v in zip(lab12, rr.uniform(-0.5, 0.5, 12))},
                                                 {(lab12[i], lab12[j]): float(x) for (i, j), x in
                                                  zip(itertools.combinations(range(12), 2), rr.uniform(-1, 1, 66))}, offset=0.75)
    perm_labels = lab12[::-1]
    arr = rand_spins(50, 12, seed=42)
    occ = np.random.default_rng(43).integers(1, 5, 50)
    cbf = np.random.default_rng(44).random(50)
    ss = dimod.SampleSet.from_samples((arr, perm_labels), energy=sbqm.energies((arr, perm_labels)), vartype=dimod.SPIN,
                                      num_occurrences=occ, info={"keep_me": 7}, chain_break_fraction=cbf)
    reset_reports()
    ss2, fl = descend_sampleset(ss, sbqm)
    out_labels = list(ss2.variables)
    check("recomputed energies match the BQM (offset included)", np.allclose(ss2.record.energy, sbqm.energies((ss2.record.sample, out_labels))))
    check("energies do not increase", bool(np.all(ss2.record.energy <= ss.record.energy + 1e-9)))
    check("num_occurrences preserved", np.array_equal(ss2.record.num_occurrences, occ))
    check("extra vectors preserved (chain_break_fraction)", np.array_equal(ss2.record["chain_break_fraction"], cbf))
    check("sampleset.info preserved and the call report added",
          ss2.info.get("keep_me") == 7 and "postprocess_descent" in ss2.info and "settings" in ss2.info["postprocess_descent"]
          and ss2.info["postprocess_descent"]["label"] == "logical_sampleset")
    arL = IsingArrays.from_bqm(sbqm)
    posL = {v: i for i, v in enumerate(out_labels)}
    check("every output row is a one-flip local minimum of the logical model",
          bool(is_one_flip_min(arL, np.asarray(ss2.record.sample)[:, [posL[v] for v in arL.labels]]).all()))
    check("a BINARY bqm is rejected", raises(ValueError, lambda: descend_sampleset(ss, sbqm.change_vartype(dimod.BINARY, inplace=False))))
    check("a BINARY sampleset is rejected", raises(ValueError, lambda: descend_sampleset(ss.change_vartype(dimod.BINARY, inplace=False), sbqm)))
    check("the report for this call is available under 'logical_sampleset'", last_report("logical_sampleset").get("n_reads") == 50)

    # ================================================================== [10] does the sampler do the work?
    SAMPLER_OK = False
    try:
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
        from solvers.simulated_quantum_annealing.s3_samplers import SAFreezeout
        SCHED = Schedule.from_csv(strict=False)
        SAMPLER_OK = True
    except Exception as exc:
        print(f"\n[!] Sections 10-12 need the sampler / hardware modules ({type(exc).__name__}: {exc}); skipped")

    if SAMPLER_OK:
        section("[10] the sampler does the work: sampler samples + descent vs random starts + descent (60-spin glass)")
        a60 = rand_model(60, 240, seed=51, jscale=1.0, hscale=0.3)
        R = R_SYN
        BETA = 4.0
        sw = 120 if QUICK else 300
        rnd = rand_spins(R, 60, seed=52)
        e_rnd = ising_energies(a60, rnd)
        eng_r, eng_s = GreedyDescent(label="rand_start"), GreedyDescent(label="sampler_start")
        rd, _, i_rd = eng_r.run(a60, rnd)
        sx = SAFreezeout(SCHED, beta_override=BETA, num_sweeps=sw).sample(a60, R, seed=53)
        e_sx = ising_energies(a60, sx)
        sd, _, i_sd = eng_s.run(a60, sx)
        e_rd, e_sd = ising_energies(a60, rd), ising_energies(a60, sd)
        nref = neal_spins(a60, 40, 54, sweeps=3000)
        e_ref = float(ising_energies(a60, nref).min()) if nref is not None else float(min(e_sd.min(), e_rd.min()))
        print(f"    reference energy (best of long SA): {e_ref:.4f}")
        print(f"    {'population':<26}{'mean E':>10}{'min E':>10}{'resid%':>9}{'reads changed%':>16}{'mean Hamming':>14}")
        for name, e_, i_ in (("random states", e_rnd, None), ("random + descent", e_rd, i_rd), ("sampler (raw)", e_sx, None), ("sampler + descent", e_sd, i_sd)):
            res_pct = 100.0 * (e_.mean() - e_ref) / abs(e_ref)
            print(f"    {name:<26}{e_.mean():>10.3f}{e_.min():>10.3f}{res_pct:>9.2f}"
                  f"{('-' if i_ is None else format(i_['reads_changed_pct'], '.1f')):>16}{('-' if i_ is None else format(i_['hamming_mean'], '.2f')):>14}")
        z1, z2 = zlow(e_sx, e_rnd), zlow(e_sd, e_rd)
        check("sampler output has far lower energy than random states (z > 3)", z1 > 3.0, f"z={z1:.1f}")
        check("sampler + descent beats random + descent (z > 3): the annealer, not the descent, supplies the quality", z2 > 3.0, f"z={z2:.1f}")
        check("descent moves sampler output less than it moves random starts (mean Hamming distance)", i_sd["hamming_mean"] < i_rd["hamming_mean"],
              f"{i_sd['hamming_mean']:.2f} vs {i_rd['hamming_mean']:.2f}")
        check("fewer sampler reads needed descent than random reads", i_sd["reads_changed_pct"] < i_rd["reads_changed_pct"],
              f"{i_sd['reads_changed_pct']:.1f}% vs {i_rd['reads_changed_pct']:.1f}%")
        check("descent never raised the energy of a sampler read", i_sd["energy_increased_reads"] == 0)
        em = {b: ising_energies(a60, SAFreezeout(SCHED, beta_override=b, num_sweeps=sw).sample(a60, R, seed=55)) for b in (0.5, 2.0, 6.0)}
        check("colder sampler -> lower raw mean energy (beta 0.5 > 2 > 6)", em[0.5].mean() > em[2.0].mean() > em[6.0].mean(),
              {k: round(float(v.mean()), 2) for k, v in em.items()})
        es = {ns: ising_energies(a60, SAFreezeout(SCHED, beta_override=BETA, num_sweeps=ns).sample(a60, R, seed=56)) for ns in (2, 30, sw)}
        check("more sweeps -> lower raw mean energy (2 > 30 > max)", es[2].mean() > es[30].mean() >= es[sw].mean() - 3 * es[sw].std() / math.sqrt(R),
              {k: round(float(v.mean()), 2) for k, v in es.items()})
        REPORT["synthetic_60spin"] = {"beta": BETA, "ref_energy": e_ref, "mean_energy": {"random": float(e_rnd.mean()), "random_descent": float(e_rd.mean()),
                                      "sampler": float(e_sx.mean()), "sampler_descent": float(e_sd.mean())},
                                      "hamming_mean": {"random_descent": i_rd["hamming_mean"], "sampler_descent": i_sd["hamming_mean"]},
                                      "z_sampler_descent_vs_random_descent": z2}

    # ================================================================== [11] OPF pipeline
    OPF_OK = False
    if SAMPLER_OK and not NO_OPF:
        try:
            import copy as _copy
            import pandapower as pp
            import pandapower.networks as nw
            from solvers.qubo_formulator import QuboFormulator
            from solvers.simulated_quantum_annealing.s2_mapping import quantise, scale_to_hardware
            OPF_OK = True
        except ImportError as exc:
            print(f"\n[!] Sections 11-12 (OPF parts) skipped: cannot import pandapower / formulator / s2_mapping ({exc})")

    def opf_eval(fm, net_, labels, spins, ref_cost):
        uniq, counts = np.unique(spins, axis=0, return_counts=True)
        feas, best = 0, None
        for row, c in zip(uniq, counts):
            o = fm._decode_solution({l: int((s + 1) // 2) for l, s in zip(labels, row)}, net_)
            if o[4]["is_feasible"]:
                feas += int(c)
                best = o[3] if best is None else min(best, o[3])
        gap = None if (best is None or not ref_cost) else 100.0 * (best - ref_cost) / abs(ref_cost)
        return 100.0 * feas / len(spins), gap

    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    if OPF_OK:
        section("[11] DC-OPF QUBO: programmed model -> sampler -> descent on the IDEAL model -> decode")
        print("    Descent runs on the ideal (unscaled, unquantised) Hamiltonian, as postprocessing does on a real device.")
        for cname in (["case5"] if QUICK else ["case5", "case9"]):
            print(f"\n  -- {cname}")
            try:
                net = getattr(nw, cname)()
                if not net.ext_grid.empty:
                    net.ext_grid["min_p_mw"] = 0.0
                ref = None
                try:
                    tmp = _copy.deepcopy(net)
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
            Rr = 120 if QUICK else 300
            print(f"     {len(a7.labels)} spins, {a7.J.size} couplers, scale factor {f7:.3e}, reference cost {ref}, reads {Rr}")
            print(f"     {'programming':<12}{'population':<22}{'feas%':>7}{'gap%':>8}{'mean E':>12}{'changed%':>10}{'Hamming':>9}")
            rows = []
            for qlabel, bits in (("ideal", None), ("8-bit", 8)):
                pr7 = quantise(sc7, HR, JR, None, bits_h=bits, bits_j=bits)
                rndO = rand_spins(Rr, len(a7.labels), seed=61)
                e_r0 = ising_energies(a7, rndO)
                rdO, _, i_r = GreedyDescent(label="opf_random").run(a7, rndO)
                sxO = SAFreezeout(SCHED, 12.0, 0.6, num_sweeps=300).sample(pr7, Rr, seed=62)
                sdO, _, i_s = GreedyDescent(label="opf_sampler").run(a7, sxO)
                nO = neal_spins(a7, Rr, 63)
                pops = [("random + descent", rdO, i_r), ("sampler (raw)", sxO, None), ("sampler + descent", sdO, i_s)]
                if nO is not None:
                    pops.append(("neal SA (reference)", nO, None))
                stats = {}
                for pname, spins, inf_ in pops:
                    feas, gap = opf_eval(fm, net, a7.labels, spins, ref)
                    eI = ising_energies(a7, spins)
                    stats[pname] = (feas, gap, eI)
                    print(f"     {qlabel:<12}{pname:<22}{feas:>7.1f}{fmt(gap, '.2f'):>8}{eI.mean():>12.3f}"
                          f"{fmt(None if inf_ is None else inf_['reads_changed_pct'], '.1f'):>10}{fmt(None if inf_ is None else inf_['hamming_mean'], '.2f'):>9}")
                    rows.append({"programming": qlabel, "population": pname, "feasible_pct": feas, "gap_pct": gap, "mean_ideal_energy": float(eI.mean())})
                check(f"{cname}/{qlabel}: valid +-1 outputs, one column per spin", sdO.shape == (Rr, len(a7.labels)) and set(np.unique(sdO).tolist()) <= {-1, 1})
                check(f"{cname}/{qlabel}: descent never raised the energy of a sampler read", i_s["energy_increased_reads"] == 0)
                zz = zlow(stats["sampler + descent"][2], stats["random + descent"][2])
                check(f"{cname}/{qlabel}: sampler + descent has lower ideal energy than random + descent (z > 3)", zz > 3.0, f"z={zz:.1f}")
                check(f"{cname}/{qlabel}: sampler + descent is at least as feasible as random + descent",
                      stats["sampler + descent"][0] >= stats["random + descent"][0] - 1e-9,
                      f"{stats['sampler + descent'][0]:.1f}% vs {stats['random + descent'][0]:.1f}%")
                check(f"{cname}/{qlabel}: descent moves the sampler's reads less than random starts", i_s["hamming_mean"] < i_r["hamming_mean"],
                      f"{i_s['hamming_mean']:.2f} vs {i_r['hamming_mean']:.2f}")
                info(f"{qlabel}: descent changed feasibility of the sampler output from {stats['sampler (raw)'][0]:.1f}% to {stats['sampler + descent'][0]:.1f}%")
            REPORT[f"opf_{cname}"] = rows
        print("\n    Negative gaps are not improvements: the feasibility tolerance equals the 10 MW precision.")

    # ================================================================== [12] EmulatedQPU integration
    if SAMPLER_OK and not NO_EMU:
        section("[12] EmulatedQPU: postprocess = None / physical / logical / both (reports read from the imported module copy)")
        try:
            import solvers.simulated_quantum_annealing.s4_postprocess as PPM
            from solvers.simulated_quantum_annealing.emulated_qpu_solver import EmulatedQPU
            from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
            emu_ok = True
        except ImportError as exc:
            emu_ok = False
            print(f"    skipped: {exc}")

        if emu_ok:
            def run_emu(pp_, bqm_, emb_, hw_, reads, per, kw, seed=7):
                PPM.reset_reports()
                q = EmulatedQPU(hw_, seed=seed, sampler="sa", sampler_kwargs=dict(kw), dac_bits=8,
                                ice_sigma_h=0.01, ice_sigma_j=0.005, use_extended_j=True)
                r = q.sample(bqm_, emb_, num_reads=reads, reads_per_programming=per, chain_break_method="majority", postprocess=pp_)
                return r, PPM.postprocess_report()

            def ncalls(rep_, lab_):
                return rep_["engines"].get(lab_, {}).get("cumulative", {}).get("calls", 0)

            try:
                hw = Hardware.ideal_zephyr(m=4, t=4)
                rr2 = np.random.default_rng(3)
                nL = 8
                lbqm = dimod.BinaryQuadraticModel({i: float(rr2.uniform(-1, 1)) for i in range(nL)},
                                                  {(i, j): float(rr2.uniform(-1, 1)) for i, j in itertools.combinations(range(nL), 2)}, 0.0, dimod.BINARY)
                embs = find_embeddings(lbqm, hw, methods=("clique", "minorminer"), seeds=(0,), timeout=20.0, verbose=False)
                elab, emb = select_best(embs)
                print(f"    synthetic 8-variable dense problem, embedding {elab}")
                kw = dict(num_sweeps=150 if QUICK else 300)
                res = {pp_: run_emu(pp_, lbqm, emb, hw, 120, 40, kw) for pp_ in (None, "physical", "logical", "both")}
                for pp_, (r_, rep_) in res.items():
                    fl = r_.info["postprocess_flips"]
                    e_ = r_.sampleset.record.energy
                    cb = float(np.mean(r_.sampleset.record["chain_break_fraction"]))
                    print(f"    postprocess={str(pp_):<9} mean logical E {e_.mean():>9.4f} | min {e_.min():>9.4f} | chain-break fraction {cb:.4f} | flips (phys, log) {fl}")
                check("calls recorded per stage: None -> (0, 0)", ncalls(res[None][1], "steepest_descent") == 0 and ncalls(res[None][1], "logical_sampleset") == 0)
                check("calls recorded per stage: physical -> (1, 0)", ncalls(res["physical"][1], "steepest_descent") == 1 and ncalls(res["physical"][1], "logical_sampleset") == 0)
                check("calls recorded per stage: logical -> (0, 1)", ncalls(res["logical"][1], "steepest_descent") == 0 and ncalls(res["logical"][1], "logical_sampleset") == 1)
                check("calls recorded per stage: both -> (1, 1)", ncalls(res["both"][1], "steepest_descent") == 1 and ncalls(res["both"][1], "logical_sampleset") == 1)
                check("EmulationResult.info flips agree with the stages used",
                      res[None][0].info["postprocess_flips"] == (0, 0) and res["logical"][0].info["postprocess_flips"][0] == 0
                      and res["physical"][0].info["postprocess_flips"][1] == 0)
                check("the report carries the settings that produced the result",
                      set(res["both"][1]["engines"]["steepest_descent"]["settings"]) == set(PPM.get_config())
                      and res["both"][1]["engines"]["steepest_descent"]["last_call"]["n_reads"] == 120)
                try:
                    json.dumps(res["both"][1])
                    check("the emulator-run report is JSON-safe", True)
                except Exception as exc:
                    check("the emulator-run report is JSON-safe", False, exc)
                eN, eL = res[None][0].sampleset.record.energy, res["logical"][0].sampleset.record.energy
                if len(eN) == len(eL):
                    check("logical descent lowers (or keeps) the energy of every read relative to no postprocessing", bool(np.all(eL <= eN + 1e-9)),
                          f"mean {eN.mean():.4f} -> {eL.mean():.4f}")
                else:
                    check("same number of rows with and without postprocessing", False, f"{len(eN)} vs {len(eL)}")
                eB = res["both"][0].sampleset.record.energy
                check("'both' mean logical energy is no worse than no postprocessing (within 3 s.e.)", eB.mean() <= eN.mean() + 3 * eN.std() / math.sqrt(len(eN)),
                      f"{eB.mean():.4f} vs {eN.mean():.4f}")
                cbN = float(np.mean(res[None][0].sampleset.record["chain_break_fraction"]))
                cbP = float(np.mean(res["physical"][0].sampleset.record["chain_break_fraction"]))
                check("physical descent does not increase the reported chain-break fraction (it repairs chains)", cbP <= cbN + 0.02, f"{cbN:.4f} -> {cbP:.4f}")
                info(f"the reported fraction after physical descent ({cbP:.4f}) understates the raw device value ({cbN:.4f}); record the pre-descent value when reporting")
                again = run_emu("both", lbqm, emb, hw, 120, 40, kw)[0]
                check("identical seeds reproduce the postprocessed result exactly",
                      np.array_equal(np.asarray(again.sampleset.record.sample), np.asarray(res["both"][0].sampleset.record.sample)))
                REPORT["emulator_synthetic"] = {str(k): {"mean_E": float(v[0].sampleset.record.energy.mean()),
                                                         "cb": float(np.mean(v[0].sampleset.record["chain_break_fraction"])),
                                                         "report": v[1]} for k, v in res.items()}
            except Exception as exc:
                check("EmulatedQPU synthetic integration ran without error", False, f"{type(exc).__name__}: {exc}")

            if OPF_OK:
                try:
                    print("\n  -- OPF case5 through the emulator, all four postprocess settings, versus a random-start baseline")
                    net5 = nw.case5()
                    if not net5.ext_grid.empty:
                        net5.ext_grid["min_p_mw"] = 0.0
                    tmp5 = _copy.deepcopy(net5)
                    pp.rundcopp(tmp5)
                    ref5 = float(tmp5.res_cost)
                    fm5 = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0, encoding="radix")
                    bqm5, _ = fm5._formulate_qubo(net5)
                    hw12 = Hardware.ideal_zephyr(m=12, t=4)
                    embs5 = find_embeddings(bqm5, hw12, methods=("clique", "minorminer"), seeds=(0,), timeout=30.0, verbose=False)
                    if not embs5:
                        print("     no embedding found, skipped")
                    else:
                        lab5, emb5 = select_best(embs5)
                        reads5 = 100 if QUICK else 200
                        res5 = {pp_: run_emu(pp_, bqm5, emb5, hw12, reads5, 50, dict(num_sweeps=300)) for pp_ in (None, "physical", "logical", "both")}
                        sbq5 = bqm5.change_vartype(dimod.SPIN, inplace=False)
                        a5 = IsingArrays.from_bqm(sbq5)
                        base_in = rand_spins(reads5, len(a5.labels), seed=71)
                        base_out, _, base_inf = GreedyDescent(label="opf_emu_baseline").run(a5, base_in)
                        e_base = sbq5.energies((base_out, a5.labels))
                        fe_b, gp_b = opf_eval(fm5, net5, a5.labels, base_out, ref5)
                        print(f"     embedding {lab5}: {embs5[lab5]['stats']['physical']} physical qubits for {len(a5.labels)} logical variables")
                        print(f"     {'postprocess':<22}{'feas%':>7}{'gap%':>8}{'mean logical E':>16}{'chain breaks':>14}{'flips (phys, log)':>20}")
                        table = {}
                        for pp_, (r_, rep_) in res5.items():
                            ss_ = r_.sampleset
                            labs_ = list(ss_.variables)
                            spins_ = 2 * np.asarray(ss_.record.sample).astype(np.int8) - 1
                            fe, gp = opf_eval(fm5, net5, labs_, spins_, ref5)
                            cb = float(np.mean(ss_.record["chain_break_fraction"]))
                            table[pp_] = (fe, gp, ss_.record.energy)
                            print(f"     {str(pp_):<22}{fe:>7.1f}{fmt(gp, '.2f'):>8}{ss_.record.energy.mean():>16.3f}{cb:>14.4f}{str(r_.info['postprocess_flips']):>20}")
                        print(f"     {'random + descent (base)':<22}{fe_b:>7.1f}{fmt(gp_b, '.2f'):>8}{e_base.mean():>16.3f}{'-':>14}{'-':>20}")
                        eN5, eL5 = res5[None][0].sampleset.record.energy, res5["logical"][0].sampleset.record.energy
                        if len(eN5) == len(eL5):
                            check("OPF/emulator: logical descent lowers (or keeps) the energy of every read", bool(np.all(eL5 <= eN5 + 1e-9)),
                                  f"mean {eN5.mean():.3f} -> {eL5.mean():.3f}")
                        z5 = zlow(table["logical"][2], e_base)
                        check("OPF/emulator: emulator output + logical descent has lower logical energy than random + descent (z > 3)", z5 > 3.0, f"z={z5:.1f}")
                        check("OPF/emulator: emulator output + logical descent is at least as feasible as random + descent",
                              table["logical"][0] >= fe_b - 1e-9, f"{table['logical'][0]:.1f}% vs {fe_b:.1f}%")
                        for pp_ in ("physical", "both"):
                            check(f"OPF/emulator: postprocess='{pp_}' is at least as feasible as the random-start baseline", table[pp_][0] >= fe_b - 1e-9,
                                  f"{table[pp_][0]:.1f}% vs {fe_b:.1f}%")
                        info("rows with 'None' show what the emulated annealer alone delivers; the gain from each postprocessing stage is the difference to that row")
                        REPORT["emulator_opf_case5"] = {"embedding": lab5, "baseline_feasible_pct": fe_b,
                                                        "rows": {str(k): {"feasible_pct": v[0], "gap_pct": v[1], "mean_logical_energy": float(v[2].mean())} for k, v in table.items()},
                                                        "report_both": res5["both"][1]}
                except Exception as exc:
                    check("OPF emulator integration ran without error", False, f"{type(exc).__name__}: {exc}")

    # ================================================================== summary
    section("SUMMARY")
    configure(**dict(_DEFAULTS))
    failed = [c for c in CHECKS if not c[1]]
    print(f"checks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}")
    with open(OUT_JSON, "w") as fh:
        json.dump({"settings": {"quick": QUICK, "no_opf": NO_OPF, "no_emulator": NO_EMU, "no_greedy_compare": NO_REF,
                                "defaults": dict(_DEFAULTS)}, "checks": CHECKS, "report": REPORT}, fh, default=str, indent=1)
    print(f"details written to {OUT_JSON}")
    sys.exit(1 if failed else 0)