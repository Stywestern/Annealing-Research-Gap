"""Step 2: what happens between "embedded Ising model" and "what the annealer is actually programmed with".

  scale_to_hardware : rescale h, J to fill the device's programmable ranges (like auto_scale)
  quantise          : finite DAC resolution
  add_ice_noise     : Gaussian control errors on h and J (integrated control errors, ICE)

All of these work on flat numpy arrays (IsingArrays) for speed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import dimod
import numpy as np


@dataclass
class IsingArrays:
    labels: list          # physical qubit labels, order defines column order of samples
    h: np.ndarray         # (n,)
    rows: np.ndarray      # (m,) indices into labels
    cols: np.ndarray      # (m,)
    J: np.ndarray         # (m,)  one entry per coupler (u < v not required)

    @classmethod
    def from_bqm(cls, bqm: dimod.BinaryQuadraticModel) -> "IsingArrays":
        assert bqm.vartype is dimod.SPIN, "convert to SPIN first"
        labels = list(bqm.variables)
        idx = {v: i for i, v in enumerate(labels)}
        h = np.array([bqm.get_linear(v) for v in labels], dtype=float)
        quad = [(idx[u], idx[v], b) for u, v, b in bqm.iter_quadratic()]
        if quad:
            rows, cols, J = (np.array(x) for x in zip(*quad))
        else:
            rows = cols = np.zeros(0, dtype=int)
            J = np.zeros(0)
        return cls(labels, h, rows.astype(int), cols.astype(int), J.astype(float))

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        lin = {v: float(x) for v, x in zip(self.labels, self.h)}
        quad = {(self.labels[i], self.labels[j]): float(b)
                for i, j, b in zip(self.rows, self.cols, self.J)}
        return dimod.BinaryQuadraticModel.from_ising(lin, quad)

    def with_values(self, h=None, J=None) -> "IsingArrays":
        return replace(self, h=self.h if h is None else h, J=self.J if J is None else J)


def _limit(values: np.ndarray, lo: float, hi: float) -> float:
    f = np.inf
    if values.size:
        if values.max() > 0:
            f = min(f, hi / values.max())
        if values.min() < 0:
            f = min(f, lo / values.min())
    return f


def scale_to_hardware(ar: IsingArrays, h_range, j_range, is_chain: np.ndarray | None = None,
                      chain_j_range=None, per_qubit_coupling_range=None):
    """Multiply everything by ONE factor so the tightest of these limits is exactly filled: h vs h_range,
    non-chain J vs j_range, chain J vs chain_j_range (defaults to j_range) and, if given, the signed sum of
    couplings on every qubit vs per_qubit_coupling_range. Take that range from the real solver's properties
    (Hardware.properties) if it reports one; do not guess it. Returns (scaled, factor)."""

    if chain_j_range is None:
        chain_j_range = j_range
    if is_chain is None:
        is_chain = np.zeros(ar.J.shape, dtype=bool)
    f = min(_limit(ar.h, *h_range), _limit(ar.J[~is_chain], *j_range),
            _limit(ar.J[is_chain], *chain_j_range))
    if per_qubit_coupling_range is not None and ar.J.size:
        f = min(f, _limit(_per_qubit_sums(ar), *per_qubit_coupling_range))
    if not np.isfinite(f):
        f = 1.0
    return ar.with_values(h=ar.h * f, J=ar.J * f), float(f)


def _per_qubit_sums(ar: IsingArrays) -> np.ndarray:
    """Signed sum of the couplings incident on each qubit."""
    sums = np.zeros(len(ar.labels))
    np.add.at(sums, ar.rows, ar.J)
    np.add.at(sums, ar.cols, ar.J)
    return sums


def _quantise_values(x: np.ndarray, lo: float, hi: float, bits: int) -> np.ndarray:
    """Uniform grid of step (hi-lo)/2**bits anchored at ZERO (a zero coefficient stays exactly zero), clipped to the programmable range."""
    step = (hi - lo) / (2 ** bits)
    return np.clip(np.round(x / step) * step, lo, hi)


def quantise(ar: IsingArrays, h_range, j_range, bits: int | None, is_chain: np.ndarray | None = None,
             chain_j_range=None, bits_h: int | None = None, bits_j: int | None = None):
    """DAC model. `bits` applies to both h and J; bits_h / bits_j override it for one of them (h and J do not
    have the same resolution on real devices). None for a part -> that part is left unquantised.
    Chain couplers are quantised against chain_j_range, the rest against j_range."""

    bh = bits if bits_h is None else bits_h
    bj = bits if bits_j is None else bits_j
    if bh is None and bj is None:
        return ar
    if chain_j_range is None:
        chain_j_range = j_range
    if is_chain is None:
        is_chain = np.zeros(ar.J.shape, dtype=bool)
    h = ar.h if bh is None else _quantise_values(ar.h, *h_range, bh)
    J = ar.J.copy()
    if bj is not None:
        J[~is_chain] = _quantise_values(ar.J[~is_chain], *j_range, bj)
        J[is_chain] = _quantise_values(ar.J[is_chain], *chain_j_range, bj)
    return ar.with_values(h=h, J=J)


class SystematicICE:
    """The control errors: one offset per physical qubit and per coupler, drawn once and then
    reused for every programming and every problem, because a real device's systematic error belongs to the
    hardware. Offsets are keyed by qubit label (integers, as on dwave_networkx Zephyr graphs), so they do not
    depend on call order. sigma values are calibration parameters, not known constants."""
    def __init__(self, sigma_h: float, sigma_j: float, sigma_j_chain: float | None = None, seed: int = 0):
        self.sigma_h, self.sigma_j, self.sigma_j_chain, self.seed = sigma_h, sigma_j, sigma_j_chain, seed
        self._h, self._j = {}, {}

    def _normal(self, *key) -> float:
        return float(np.random.default_rng([self.seed, *key]).normal())

    def offsets(self, ar: IsingArrays, is_chain: np.ndarray | None = None):
        qs = [int(q) for q in ar.labels]
        for q in qs:
            if q not in self._h:
                self._h[q] = self._normal(0, q)
        dh = np.array([self._h[q] for q in qs]) * self.sigma_h
        z = np.empty(ar.J.shape)
        for k, (i, j) in enumerate(zip(ar.rows, ar.cols)):
            a, b = sorted((qs[i], qs[j]))
            if (a, b) not in self._j:
                self._j[(a, b)] = self._normal(1, a, b)
            z[k] = self._j[(a, b)]
        sj = np.full(ar.J.shape, float(self.sigma_j))
        if is_chain is not None and self.sigma_j_chain is not None:
            sj[is_chain] = self.sigma_j_chain
        return dh, z * sj


def add_ice_noise(ar: IsingArrays, sigma_h: float, sigma_j: float, rng: np.random.Generator,
                  is_chain: np.ndarray | None = None, sigma_j_chain: float | None = None,
                  systematic: SystematicICE | None = None):
    """Control errors in the *scaled* units. The RANDOM part is redrawn on every call (one call = one
    programming). Optional: systematic = a SystematicICE whose fixed offsets are added on top;
    sigma_j_chain = a separate J sigma for chain couplers (needs is_chain).
    With only (ar, sigma_h, sigma_j, rng) it behaves, and consumes the rng, exactly as before."""
    h, J = ar.h.copy(), ar.J.copy()
    if systematic is not None:
        dh, dJ = systematic.offsets(ar, is_chain)
        h += dh
        J += dJ
    if sigma_h > 0:
        h += rng.normal(0.0, sigma_h, h.shape)
    sj = np.full(J.shape, float(sigma_j))
    if is_chain is not None and sigma_j_chain is not None:
        sj[is_chain] = sigma_j_chain
    if (sj > 0).any():
        J += rng.normal(0.0, 1.0, J.shape) * sj
    return ar.with_values(h=h, J=J)


def random_gauge(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=n)


def apply_gauge(ar: IsingArrays, g: np.ndarray) -> IsingArrays:
    """Spin-reversal transform s_i -> g_i s_i: h_i -> g_i h_i, J_ij -> g_i g_j J_ij. Same energies, but the
    (sign-independent) control errors now act with random signs relative to the problem, so they average out
    over gauges. Order: programmed -> apply_gauge -> add_ice_noise -> sample -> undo_gauge."""
    return ar.with_values(h=ar.h * g, J=ar.J * g[ar.rows] * g[ar.cols])


def undo_gauge(samples: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Map samples from the gauged frame back to the original one (g is its own inverse)."""
    return (samples * g[None, :]).astype(np.int8)


def energies(ar: IsingArrays, S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    return S @ ar.h + np.sum(ar.J[None, :] * S[:, ar.rows] * S[:, ar.cols], axis=1)


def _dyn_range(v: np.ndarray) -> float:
    a = np.abs(v[np.abs(v) > 0])
    return float(a.max() / a.min()) if a.size else 1.0


def mapping_diagnostics(ideal: IsingArrays, scaled: IsingArrays, programmed: IsingArrays, h_range, j_range,
                        is_chain: np.ndarray | None = None, chain_j_range=None,
                        bits_h: int | None = None, bits_j: int | None = None,
                        per_qubit_coupling_range=None, sigma_h: float | None = None,
                        sigma_j: float | None = None) -> dict:
    """Describes what scaling and quantisation did. Pass the same ranges/bits you used in scale_to_hardware and
    quantise. `ideal` = embedded model before scaling, `scaled` = after, `programmed` = after quantising."""
    if chain_j_range is None:
        chain_j_range = j_range
    if is_chain is None:
        is_chain = np.zeros(ideal.J.shape, dtype=bool)
    pj = ~is_chain
    limits = {"h_range": _limit(ideal.h, *h_range),
              "problem_J_range": _limit(ideal.J[pj], *j_range),
              "chain_J_range": _limit(ideal.J[is_chain], *chain_j_range)}
    if per_qubit_coupling_range is not None and ideal.J.size:
        limits["per_qubit_coupling_range"] = _limit(_per_qubit_sums(ideal), *per_qubit_coupling_range)
    finite = {k: v for k, v in limits.items() if np.isfinite(v)}
    binding = min(finite, key=finite.get) if finite else None

    nzJ = pj & (np.abs(scaled.J) > 0)
    zeroedJ = int(np.sum(nzJ & (programmed.J == 0)))
    oor = int(np.sum((programmed.h < h_range[0] - 1e-12) | (programmed.h > h_range[1] + 1e-12)))
    oor += int(np.sum((programmed.J[pj] < j_range[0] - 1e-12) | (programmed.J[pj] > j_range[1] + 1e-12)))
    oor += int(np.sum((programmed.J[is_chain] < chain_j_range[0] - 1e-12)
                      | (programmed.J[is_chain] > chain_j_range[1] + 1e-12)))
    d = {
        "n_qubits": len(ideal.labels), "n_couplers": int(ideal.J.size), "n_chain_couplers": int(is_chain.sum()),
        "scale_factor": float(min(finite.values())) if finite else 1.0,
        "binding_constraint": binding,
        "limits_scale_factor": {k: (float(v) if np.isfinite(v) else None) for k, v in limits.items()},
        "dynamic_range_ideal": _dyn_range(np.concatenate([ideal.h, ideal.J])),
        "dynamic_range_programmed": _dyn_range(np.concatenate([programmed.h, programmed.J])),
        "step_h": (h_range[1] - h_range[0]) / 2 ** bits_h if bits_h else None,
        "step_j": (j_range[1] - j_range[0]) / 2 ** bits_j if bits_j else None,
        "problem_J_zeroed": zeroedJ,
        "problem_J_zeroed_pct": 100.0 * zeroedJ / max(1, int(nzJ.sum())),
        "h_zeroed": int(np.sum((np.abs(scaled.h) > 0) & (programmed.h == 0))),
        "max_abs_rounding_err_h": float(np.max(np.abs(programmed.h - scaled.h), initial=0.0)),
        "max_abs_rounding_err_j": float(np.max(np.abs(programmed.J - scaled.J), initial=0.0)),
        "out_of_range_after_quantise": oor,
        "chain_strength_programmed_min": (float(np.min(np.abs(programmed.J[is_chain]))) if is_chain.any() else None),
    }
    if sigma_j is not None and nzJ.any():
        d["problem_J_below_sigma_j_pct"] = float(100.0 * np.mean(np.abs(scaled.J[nzJ]) < sigma_j))
    if sigma_h is not None:
        nzh = np.abs(scaled.h) > 0
        d["h_below_sigma_h_pct"] = float(100.0 * np.mean(np.abs(scaled.h[nzh]) < sigma_h)) if nzh.any() else 0.0
    return d


# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import itertools
    import json
    import sys

    QUICK = False
    NO_OPF = False
    OUT_JSON = "mapping_selftest.json"
    HR, JR, EJR = (-4.0, 4.0), (-1.0, 1.0), (-2.0, 1.0)      # h range, J range, extended (chain) J range
    CHECKS, REPORT = [], {}
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ helpers
    def check(name, ok, detail=""):
        CHECKS.append((name, bool(ok), str(detail)))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def section(title):
        print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else 0.0

    def make_instance(n, m, seed, n_chain=0, spread=1.0):
        """Random Ising model on unique pairs; the first n_chain couplers are strong negative 'chain' couplers."""
        r = np.random.default_rng(seed)
        pairs = list(itertools.combinations(range(n), 2))
        sel = r.choice(len(pairs), size=min(m, len(pairs)), replace=False)
        rows = np.array([pairs[k][0] for k in sel])
        cols = np.array([pairs[k][1] for k in sel])
        J = r.normal(0.0, spread, len(sel))
        is_chain = np.zeros(len(sel), dtype=bool)
        is_chain[:n_chain] = True
        J[is_chain] = -r.uniform(2.0, 6.0, n_chain) * spread
        return IsingArrays(list(range(n)), r.normal(0.0, spread, n), rows, cols, J), is_chain

    def in_range(a, chain):
        ok = a.h.min() >= HR[0] - 1e-9 and a.h.max() <= HR[1] + 1e-9
        o, c = a.J[~chain], a.J[chain]
        ok = ok and (o.size == 0 or (o.min() >= JR[0] - 1e-9 and o.max() <= JR[1] + 1e-9))
        ok = ok and (c.size == 0 or (c.min() >= EJR[0] - 1e-9 and c.max() <= EJR[1] + 1e-9))
        return bool(ok)

    def run_map(a, chain, bits_h, bits_j, pq=None, sig=(None, None)):
        sc, f = scale_to_hardware(a, HR, JR, chain, EJR, per_qubit_coupling_range=pq)
        pr = quantise(sc, HR, JR, None, chain, EJR, bits_h=bits_h, bits_j=bits_j)
        d = mapping_diagnostics(a, sc, pr, HR, JR, chain, EJR, bits_h=bits_h, bits_j=bits_j,
                                per_qubit_coupling_range=pq, sigma_h=sig[0], sigma_j=sig[1])
        return sc, pr, d, f

    print("=== Testing the analog mapping pipeline (scale -> quantise -> noise -> gauge -> diagnostics) ===")
    print(f"mode={'quick' if QUICK else 'full'} | opf={'off' if NO_OPF else 'on'}")

    # ================================================================== [1] scale_to_hardware
    section("[1] scale_to_hardware")
    ar, chain = make_instance(16, 60, seed=1, n_chain=20, spread=5.0)
    scaled, f = scale_to_hardware(ar, HR, JR, chain, EJR)
    f_exp = min(_limit(ar.h, *HR), _limit(ar.J[~chain], *JR), _limit(ar.J[chain], *EJR))
    check("factor equals the tightest of the three limits", abs(f - f_exp) <= 1e-12 * f_exp, f"f={f:.6g}")
    check("scaled model lies inside the h, J and chain-J ranges", in_range(scaled, chain))
    touches = (np.isclose(scaled.h.max(), HR[1]) or np.isclose(scaled.h.min(), HR[0])
               or np.isclose(scaled.J[~chain].max(), JR[1]) or np.isclose(scaled.J[~chain].min(), JR[0])
               or np.isclose(scaled.J[chain].max(), EJR[1]) or np.isclose(scaled.J[chain].min(), EJR[0]))
    check("one range edge is exactly filled (global scale, nothing left on the table)", touches)
    S = rng.choice(np.array([-1, 1], dtype=np.int8), size=(50, 16))
    check("energies scale by exactly the factor (state ordering unchanged)",
          np.allclose(energies(scaled, S), f * energies(ar, S), rtol=1e-9, atol=1e-9))

    _, f_noext = scale_to_hardware(ar, HR, JR, chain, None)           # chain couplers confined to the J range
    check("extended chain range never lowers the factor and gains at most 2x",
          f >= f_noext - 1e-12 and f <= 2.0 * f_noext + 1e-9, f"{f:.4g} vs {f_noext:.4g}")
    sc_plain, _ = scale_to_hardware(ar, HR, JR)
    check("no mask: every coupler confined to the ordinary J range",
          sc_plain.J.min() >= JR[0] - 1e-9 and sc_plain.J.max() <= JR[1] + 1e-9)

    sums = _per_qubit_sums(ar)
    R = 0.5 * float(np.abs(f * sums).max())
    sc_pq, f_pq = scale_to_hardware(ar, HR, JR, chain, EJR, per_qubit_coupling_range=(-R, R))
    s_pq = _per_qubit_sums(sc_pq)
    check("per-qubit coupling limit binds and is respected", abs(f_pq - 0.5 * f) <= 1e-9 * f
          and s_pq.min() >= -R - 1e-9 and s_pq.max() <= R + 1e-9, f"f_pq/f={f_pq / f:.4f}")
    _, f_loose = scale_to_hardware(ar, HR, JR, chain, EJR, per_qubit_coupling_range=(-1e9, 1e9))
    check("a very loose per-qubit limit changes nothing", abs(f_loose - f) <= 1e-12 * f)

    z = IsingArrays([0, 1], np.zeros(2), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    zs, fz = scale_to_hardware(z, HR, JR)
    check("all-zero model: factor 1.0, no crash", fz == 1.0 and zs.h.shape == (2,))

    # ================================================================== [2] quantise
    section("[2] quantise")
    n_q = 300
    rq = np.random.default_rng(2)
    qchain = rq.random(n_q - 1) < 0.4
    qJ = np.where(qchain, rq.uniform(EJR[0], EJR[1], n_q - 1), rq.uniform(JR[0], JR[1], n_q - 1))
    qar = IsingArrays(list(range(n_q)), rq.uniform(HR[0], HR[1], n_q), np.arange(n_q - 1), np.arange(1, n_q), qJ)
    for bits in (3, 5, 8):
        q = quantise(qar, HR, JR, bits, qchain, EJR)
        check(f"{bits} bits: stays inside every programmable range", in_range(q, qchain))
        steps = {"h": (HR[1] - HR[0]) / 2 ** bits, "J": (JR[1] - JR[0]) / 2 ** bits, "c": (EJR[1] - EJR[0]) / 2 ** bits}
        errs = {"h": np.abs(q.h - qar.h).max(), "J": np.abs(q.J[~qchain] - qar.J[~qchain]).max(),
                "c": np.abs(q.J[qchain] - qar.J[qchain]).max()}
        check(f"{bits} bits: error never exceeds half a step (h, J, chain J)",
              all(errs[k] <= steps[k] / 2 + 1e-12 for k in errs), {k: round(float(v), 4) for k, v in errs.items()})
    q5 = quantise(qar, HR, JR, 5, qchain, EJR)
    q5b = quantise(q5, HR, JR, 5, qchain, EJR)
    check("quantising twice equals quantising once", np.allclose(q5.h, q5b.h) and np.allclose(q5.J, q5b.J))
    zq = IsingArrays([0, 1, 2], np.zeros(3), np.array([0, 1]), np.array([1, 2]), np.zeros(2))
    qz = quantise(zq, HR, JR, 4)
    check("zero stays exactly zero", np.all(qz.h == 0) and np.all(qz.J == 0))

    ex = IsingArrays(list(range(5)), np.zeros(5), np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]),
                     np.array([-2.0, 1.0, 1.0, -1.0]))
    exc = np.array([True, True, False, False])
    qe = quantise(ex, HR, JR, 5, exc, EJR)
    check("chain J=-2 lands within half an extended step of -2", abs(qe.J[0] + 2.0) <= 0.09375 / 2 + 1e-12, f"{qe.J[0]}")
    check("chain J=+1 is clipped back to 1.0 (the old grid gave 1.03125)", qe.J[1] <= 1.0 + 1e-12, f"{qe.J[1]}")
    check("ordinary J=+1/-1 are exact on their own grid", qe.J[2] == 1.0 and qe.J[3] == -1.0)

    check("bits=None for h and J returns the input object", quantise(qar, HR, JR, None) is qar)
    qh = quantise(qar, HR, JR, None, bits_h=4)
    check("bits_h alone: J untouched, h on the 4-bit grid", np.array_equal(qh.J, qar.J)
          and np.allclose(qh.h / (8 / 16), np.round(qh.h / (8 / 16))))
    qj = quantise(qar, HR, JR, None, qchain, EJR, bits_j=4)
    check("bits_j alone: h untouched", np.array_equal(qj.h, qar.h) and not np.array_equal(qj.J, qar.J))
    qm = quantise(qar, HR, JR, 3, qchain, EJR, bits_h=6)
    check("bits_h overrides bits for h only", np.abs(qm.h - qar.h).max() <= 8 / 2 ** 6 / 2 + 1e-12
          and np.abs(qm.J[~qchain] - qar.J[~qchain]).max() > 8 / 2 ** 6 / 2)

    # ================================================================== [3] noise
    section("[3] add_ice_noise and SystematicICE")
    N = 20000
    big = IsingArrays(list(range(N)), np.zeros(N), np.arange(N), (np.arange(N) + 1) % N, np.zeros(N))
    bchain = np.zeros(N, dtype=bool)
    bchain[: N // 2] = True
    nb = add_ice_noise(big, 0.02, 0.01, np.random.default_rng(1), is_chain=bchain, sigma_j_chain=0.03)
    dh, dJ = nb.h - big.h, nb.J - big.J
    check("h noise std matches sigma_h", abs(dh.std() / 0.02 - 1) < 0.05, f"{dh.std():.4f}")
    check("problem-coupler noise std matches sigma_j", abs(dJ[~bchain].std() / 0.01 - 1) < 0.05, f"{dJ[~bchain].std():.4f}")
    check("chain-coupler noise std matches sigma_j_chain", abs(dJ[bchain].std() / 0.03 - 1) < 0.05, f"{dJ[bchain].std():.4f}")
    check("noise has zero mean", abs(dh.mean()) < 5 * 0.02 / np.sqrt(N) and abs(dJ.mean()) < 5 * 0.03 / np.sqrt(N))
    nb2 = add_ice_noise(big, 0.0, 0.01, np.random.default_rng(1), sigma_j_chain=0.03)
    check("sigma_j_chain is ignored without an is_chain mask", abs((nb2.J - big.J).std() / 0.01 - 1) < 0.05)

    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    a_new = add_ice_noise(scaled, 0.02, 0.01, r1)
    old_h = scaled.h + r2.normal(0.0, 0.02, scaled.h.shape)
    old_J = scaled.J + r2.normal(0.0, 0.01, scaled.J.shape)
    check("old-style call reproduces the old random stream", np.allclose(a_new.h, old_h) and np.allclose(a_new.J, old_J))
    r3 = np.random.default_rng(5)
    a0 = add_ice_noise(scaled, 0.0, 0.0, r3)
    check("zero sigma: values unchanged and the rng is not consumed",
          np.array_equal(a0.h, scaled.h) and np.array_equal(a0.J, scaled.J) and r3.normal() == np.random.default_rng(5).normal())
    check("add_ice_noise does not modify its input", np.array_equal(scaled.h, ar.h * f))

    M = 4000
    sa_ = IsingArrays(list(range(M)), np.zeros(M), np.arange(M), (np.arange(M) + 1) % M, np.zeros(M))
    sch = np.zeros(M, dtype=bool)
    sch[: M // 2] = True
    sysice = SystematicICE(0.01, 0.005, sigma_j_chain=0.02, seed=3)
    sdh, sdJ = sysice.offsets(sa_, sch)
    check("systematic h offsets have the requested std", abs(sdh.std() / 0.01 - 1) < 0.06, f"{sdh.std():.4f}")
    check("systematic J offsets: problem vs chain std", abs(sdJ[~sch].std() / 0.005 - 1) < 0.06
          and abs(sdJ[sch].std() / 0.02 - 1) < 0.06, f"{sdJ[~sch].std():.4f} / {sdJ[sch].std():.4f}")
    sdh2, sdJ2 = sysice.offsets(sa_, sch)
    check("systematic offsets identical on every call", np.array_equal(sdh, sdh2) and np.array_equal(sdJ, sdJ2))
    n1 = add_ice_noise(sa_, 0.0, 0.0, np.random.default_rng(1), is_chain=sch, systematic=sysice)
    n2 = add_ice_noise(sa_, 0.0, 0.0, np.random.default_rng(2), is_chain=sch, systematic=sysice)
    check("systematic part is independent of the rng", np.array_equal(n1.h, n2.h) and np.array_equal(n1.J, n2.J))

    base, bch = make_instance(10, 20, seed=4, n_chain=5)
    s2 = SystematicICE(0.01, 0.005, 0.02, seed=7)
    dh0, dJ0 = s2.offsets(base, bch)
    perm = np.random.default_rng(9).permutation(len(base.J))
    permuted = IsingArrays(base.labels, base.h, base.rows[perm], base.cols[perm], base.J[perm])
    dh1, dJ1 = s2.offsets(permuted, bch[perm])
    check("offsets follow the coupler, not the array order", np.allclose(dJ1, dJ0[perm]) and np.allclose(dh1, dh0))
    flipped = IsingArrays(base.labels, base.h, base.cols, base.rows, base.J)
    check("coupler (a,b) and (b,a) share one offset", np.allclose(s2.offsets(flipped, bch)[1], dJ0))
    relab = IsingArrays(base.labels[::-1], base.h[::-1], 9 - base.rows, 9 - base.cols, base.J)
    check("offsets follow the qubit label, not its position", np.allclose(s2.offsets(relab, bch)[0], dh0[::-1]))
    other = IsingArrays([5, 6, 7, 8, 9, 10], np.zeros(6), np.array([0]), np.array([1]), np.zeros(1))
    check("a qubit shared by two different problems keeps its offset (hardware property)",
          np.allclose(s2.offsets(other)[0][:5], dh0[5:10]))
    pa = IsingArrays([5, 6], np.zeros(2), np.array([0]), np.array([1]), np.zeros(1))
    pb = IsingArrays([6, 5], np.zeros(2), np.array([0]), np.array([1]), np.zeros(1))
    check("same physical coupler, different problem order: same offset",
          np.allclose(s2.offsets(pa)[1], s2.offsets(pb)[1]))
    ra, rb = s2.offsets(pa, np.array([False]))[1], s2.offsets(pa, np.array([True]))[1]
    check("chain flag rescales the same draw by sigma_j_chain/sigma_j (= 4)", abs(rb[0] / ra[0] - 4.0) < 1e-9)
    check("a different seed gives different offsets", not np.allclose(SystematicICE(0.01, 0.005, seed=8).offsets(base)[0], dh0))

    # ================================================================== [4] gauge + helpers
    section("[4] gauge transforms, energies() and IsingArrays round trip")
    arg, cg = make_instance(14, 50, seed=11, n_chain=10, spread=2.0)
    g = random_gauge(14, np.random.default_rng(3))
    Sg = np.random.default_rng(4).choice(np.array([-1, 1], dtype=np.int8), size=(40, 14))
    check("random_gauge returns only +-1", set(np.unique(g).tolist()) <= {-1, 1})
    gar = apply_gauge(arg, g)
    check("gauged model has the same energies on correspondingly mapped samples",
          np.allclose(energies(arg, Sg), energies(gar, undo_gauge(Sg, g))))
    back = apply_gauge(gar, g)
    check("the same gauge applied twice restores the model", np.allclose(back.h, arg.h) and np.allclose(back.J, arg.J))
    check("undo_gauge is its own inverse", np.array_equal(undo_gauge(undo_gauge(Sg, g), g), Sg))

    bq = arg.to_bqm()
    check("energies() agrees with dimod", np.allclose(energies(arg, Sg), bq.energies((Sg, arg.labels))))
    rt = IsingArrays.from_bqm(bq)
    S_rt = Sg[:, [arg.labels.index(l) for l in rt.labels]]
    check("from_bqm(to_bqm(x)) preserves energies", np.allclose(energies(rt, S_rt), energies(arg, Sg)))

    prog_g, _ = make_instance(40, 300, seed=12)
    sysg = SystematicICE(0.02, 0.01, seed=21)
    G = 200
    rg = np.random.default_rng(6)
    acc_h, acc_J = np.zeros(40), np.zeros(len(prog_g.J))
    for _ in range(G):
        gg = random_gauge(40, rg)
        noisy_g = add_ice_noise(apply_gauge(prog_g, gg), 0.0, 0.0, rg, systematic=sysg)
        eff = apply_gauge(noisy_g, gg)                      # back in the original frame
        acc_h += eff.h - prog_g.h
        acc_J += eff.J - prog_g.J
    off_h, off_J = sysg.offsets(prog_g)
    ratio_h, ratio_J = rms(acc_h / G) / rms(off_h), rms(acc_J / G) / rms(off_J)
    check("averaging over gauges cancels the systematic offsets (residual < 25% of the fixed offset)",
          ratio_h < 0.25 and ratio_J < 0.25, f"h {ratio_h:.3f}, J {ratio_J:.3f}, expected ~{1 / np.sqrt(G):.3f}")
    plain = add_ice_noise(prog_g, 0.0, 0.0, rg, systematic=sysg)
    check("without gauges the systematic offset is the same on every programming",
          np.allclose(plain.h - prog_g.h, off_h) and np.allclose(plain.J - prog_g.J, off_J))

    # ================================================================== [5] diagnostics
    section("[5] mapping_diagnostics")
    cases = {
        "chain_J_range": (IsingArrays([0, 1, 2], np.full(3, 0.1), np.array([0, 1]), np.array([1, 2]), np.array([1.0, -100.0])),
                          np.array([False, True]), None),
        "h_range": (IsingArrays([0, 1, 2], np.array([500.0, -300.0, 100.0]), np.array([0, 1]), np.array([1, 2]),
                                np.array([0.5, -0.2])), None, None),
        "problem_J_range": (IsingArrays([0, 1, 2], np.full(3, 0.1), np.array([0, 1]), np.array([1, 2]),
                                        np.array([100.0, -50.0])), None, None),
        "per_qubit_coupling_range": (IsingArrays(list(range(4)), np.zeros(4), np.array([0, 1, 2]), np.array([1, 2, 3]),
                                                 np.ones(3)), None, (-1.5, 1.5)),
    }
    for expected, (a, ch, pq) in cases.items():
        sc_, pr_, d_, f_ = run_map(a, ch, 5, 5, pq=pq)
        check(f"binding constraint identified: {expected}", d_["binding_constraint"] == expected
              and abs(d_["scale_factor"] - f_) <= 1e-12 * max(f_, 1e-300), f"got {d_['binding_constraint']}, f={f_:.4g}")
    tiny = IsingArrays(list(range(4)), np.zeros(4), np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([1.0, 0.01, -0.02]))
    t_sc, t_f = scale_to_hardware(tiny, HR, JR)
    t_pr = quantise(t_sc, HR, JR, None, bits_j=5)
    d_t = mapping_diagnostics(tiny, t_sc, t_pr, HR, JR, bits_h=None, bits_j=5)
    check("couplers erased by the DAC are counted (2 of 3)", d_t["problem_J_zeroed"] == 2
          and abs(d_t["problem_J_zeroed_pct"] - 200.0 / 3.0) < 1e-9)
    check("step sizes reported (J 5-bit = 2/32, h unquantised = None)", abs(d_t["step_j"] - 2 / 32) < 1e-12 and d_t["step_h"] is None)

    arD, chD = make_instance(30, 150, seed=31, n_chain=40, spread=3.0)
    scD, prD, dD, fD = run_map(arD, chD, 5, 5, sig=(0.02, 0.01))
    indep = int(np.sum((~chD) & (np.abs(scD.J) > 0) & (prD.J == 0)))
    check("zeroed count matches an independent recount", dD["problem_J_zeroed"] == indep, f"{indep}")
    check("no value leaves the programmable range after quantising", dD["out_of_range_after_quantise"] == 0)
    check("sigma-based fields present (J and h below sigma)", "problem_J_below_sigma_j_pct" in dD and "h_below_sigma_h_pct" in dD)
    try:
        json.dumps(dD)
        check("diagnostics are JSON-safe", True)
    except Exception as exc:
        check("diagnostics are JSON-safe", False, exc)
    REPORT["diagnostics_example"] = dD

    # ================================================================== [6] ground-state preservation
    section("[6] ground-state preservation vs DAC bits and noise (exhaustive search, small random models)")
    n_gs = 12
    states = np.array(list(itertools.product((-1, 1), repeat=n_gs)), dtype=np.int8)
    gs_pairs = list(itertools.combinations(range(n_gs), 2))
    n_inst = 10 if QUICK else 40
    bits_list, sig_list = [3, 4, 5, 6, 8, None], [0.0, 0.01, 0.03]
    stat = {(b, s): [0, 0.0] for b in bits_list for s in sig_list}
    rg6 = np.random.default_rng(100)

    def wide(size):                      # log-uniform magnitudes over 2.5 decades, random signs (penalty-like spread)
        return 10 ** rg6.uniform(0.0, 2.5, size) * rg6.choice([-1.0, 1.0], size)

    for _ in range(n_inst):
        sel = rg6.choice(len(gs_pairs), size=40, replace=False)
        inst = IsingArrays(list(range(n_gs)), wide(n_gs), np.array([gs_pairs[i][0] for i in sel]),
                           np.array([gs_pairs[i][1] for i in sel]), wide(40))
        e_ideal = energies(inst, states)
        gs, span = int(np.argmin(e_ideal)), float(e_ideal.max() - e_ideal.min())
        sc6, _ = scale_to_hardware(inst, HR, JR)
        for b in bits_list:
            pr6 = quantise(sc6, HR, JR, b)
            for s in sig_list:
                noisy6 = add_ice_noise(pr6, s, s, rg6) if s > 0 else pr6
                pick = int(np.argmin(energies(noisy6, states)))
                stat[(b, s)][0] += int(pick == gs)
                stat[(b, s)][1] += (e_ideal[pick] - e_ideal[gs]) / span
    print(f"    {n_inst} random {n_gs}-spin models, couplings spanning 2.5 decades. P(ground state unchanged) / mean excess energy:")
    print(f"    {'bits':>6}" + "".join(f"{'sigma=' + str(s):>22}" for s in sig_list))
    for b in bits_list:
        print(f"    {str(b):>6}" + "".join(f"{100.0 * stat[(b, s)][0] / n_inst:>12.0f}% /{100.0 * stat[(b, s)][1] / n_inst:>6.2f}%" for s in sig_list))
    check("no DAC, no noise: ground state always preserved", stat[(None, 0.0)][0] == n_inst)
    check("8-bit preserves the ground state at least as often as 3-bit", stat[(8, 0.0)][0] >= stat[(3, 0.0)][0],
          f"{stat[(8, 0.0)][0]} vs {stat[(3, 0.0)][0]} of {n_inst}")
    REPORT["ground_state_preservation"] = {f"bits={b},sigma={s}": {"preserved_pct": 100.0 * stat[(b, s)][0] / n_inst,
                                                                    "mean_excess_pct": 100.0 * stat[(b, s)][1] / n_inst}
                                           for b in bits_list for s in sig_list}

    # ================================================================== [7] OPF QUBO through the pipeline
    OPF_BQMS = {}
    opf_ok = False
    if not NO_OPF:
        try:
            import copy
            import pandapower as pp
            import pandapower.networks as nw
            from solvers.qubo_formulator import QuboFormulator
            opf_ok = True
        except ImportError as exc:
            print(f"\n[!] Sections 7-8 skipped: cannot import pandapower / formulator ({exc})")

    def sa_spins(a, reads, seed):
        try:
            from dwave.samplers import SimulatedAnnealingSampler as _SA
        except ImportError:
            try:
                from neal import SimulatedAnnealingSampler as _SA
            except ImportError:
                return None
        res = _SA().sample(a.to_bqm(), num_reads=reads, num_sweeps=1000, seed=int(seed))
        pos = {v: i for i, v in enumerate(res.variables)}
        return np.asarray(res.record.sample)[:, [pos[v] for v in a.labels]].astype(np.int8)

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

    if opf_ok:
        section("[7] DC-OPF QUBO -> Ising -> scale -> quantise -> noise -> annealer (SA stand-in) -> decode")
        configs = [("ideal (no DAC, no noise)", None, 0.0), ("8-bit", 8, 0.0), ("6-bit", 6, 0.0), ("5-bit", 5, 0.0),
                   ("4-bit", 4, 0.0), ("3-bit", 3, 0.0), ("5-bit + ICE 0.01", 5, 0.01), ("no DAC + ICE 0.03", None, 0.03)]
        for cname in (["case5"] if QUICK else ["case5", "case9"]):
            print(f"\n  -- {cname}")
            try:
                net = getattr(nw, cname)()
                if not net.ext_grid.empty:
                    net.ext_grid["min_p_mw"] = 0.0
                ref = None
                try:
                    tmp = copy.deepcopy(net)
                    pp.rundcopp(tmp)
                    ref = float(tmp.res_cost)
                except Exception as exc:
                    print(f"     (no classical reference cost: {type(exc).__name__})")
                fm = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0, encoding="radix")
                bqm, cx = fm._formulate_qubo(net)
                if not bqm.variables:
                    print("     0 variables, skipped")
                    continue
            except Exception as exc:
                check(f"{cname}: formulation", False, f"{type(exc).__name__}: {exc}")
                continue
            OPF_BQMS[cname] = bqm
            spin = bqm.change_vartype(dimod.SPIN, inplace=False)
            a7 = IsingArrays.from_bqm(spin)
            print(f"     {len(a7.labels)} spins, {a7.J.size} couplers, reference DC-OPF cost: {ref}")

            X = np.random.default_rng(1).integers(0, 2, size=(30, len(a7.labels))).astype(np.int8)
            check(f"{cname}: IsingArrays(SPIN) + offset reproduces the QUBO energy",
                  np.allclose(energies(a7, 2 * X.astype(int) - 1) + spin.offset, bqm.energies((X, a7.labels)), rtol=1e-9, atol=1e-6))
            dr_formulator = cx.get("hardware_limits", {}).get("ising_dynamic_range")
            dr_here = _dyn_range(np.concatenate([a7.h, a7.J]))
            if dr_formulator is not None:
                check(f"{cname}: dynamic range agrees with the formulator's report",
                      abs(dr_here / dr_formulator - 1) < 1e-3, f"{dr_here:.1f} vs {dr_formulator}")
            sc7, f7 = scale_to_hardware(a7, HR, JR)
            check(f"{cname}: scaled model inside ranges", in_range(sc7, np.zeros(a7.J.shape, dtype=bool)), f"factor {f7:.3e}")

            print(f"     {'configuration':<26}{'zeroed J%':>10}{'dynR prog':>11}{'feasible%':>11}{'gap%':>8}")
            rows7, zeroed_by_bits = [], {}
            for label, b, sg in configs:
                pr7 = quantise(sc7, HR, JR, None, bits_h=b, bits_j=b)
                d7 = mapping_diagnostics(a7, sc7, pr7, HR, JR, bits_h=b, bits_j=b, sigma_h=sg or None, sigma_j=sg or None)
                noisy7 = add_ice_noise(pr7, sg, sg, np.random.default_rng(7)) if sg > 0 else pr7
                spins = sa_spins(noisy7, 200, 42)
                feas = gap = None
                if spins is not None:
                    feas, gap = opf_eval(fm, net, a7.labels, spins, ref)
                if sg == 0.0:
                    zeroed_by_bits[b] = d7["problem_J_zeroed_pct"]
                print(f"     {label:<26}{d7['problem_J_zeroed_pct']:>10.1f}{d7['dynamic_range_programmed']:>11.1f}"
                      f"{('-' if feas is None else format(feas, '.1f')):>11}{('-' if gap is None else format(gap, '.2f')):>8}")
                rows7.append({"config": label, "feasible_pct": feas, "gap_pct": gap, "diagnostics": d7})
                if label.startswith("ideal") and feas is not None:
                    check(f"{cname}: the unquantised, noise-free scaled model still yields feasible samples", feas > 0, f"{feas:.1f}%")
            z = [zeroed_by_bits[b] for b in (8, 6, 5, 4, 3)]
            check(f"{cname}: fewer bits never erase fewer couplers", all(z[i] <= z[i + 1] + 1e-9 for i in range(len(z) - 1)),
                  str([round(v, 1) for v in z]))
            REPORT[f"opf_{cname}"] = rows7

    # ================================================================== [8] embedded problem
    if opf_ok and OPF_BQMS:
        section("[8] embedded OPF problem: chain mask, extended J range, binding constraint, erased couplers")
        try:
            from dwave.embedding import embed_bqm
            from dwave.embedding.chain_strength import uniform_torque_compensation
            from solvers.simulated_quantum_annealing.s1_hardware import Hardware
            from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings
            emb_ok = True
        except ImportError as exc:
            emb_ok = False
            print(f"    skipped: {exc}")
        if emb_ok:
            hw = Hardware.ideal_zephyr(m=12, t=4)
            for cname, bqm8 in OPF_BQMS.items():
                if len(bqm8.variables) > 60:
                    continue
                embs = find_embeddings(bqm8, hw, methods=("clique", "minorminer"), seeds=(0,), timeout=30.0, verbose=False)
                if not embs:
                    print(f"  -- {cname}: no embedding found, skipped")
                    continue
                print(f"\n  -- {cname} ({len(bqm8.variables)} logical variables)")
                print(f"     {'embedding':<18}{'phys':>6}{'chain_s':>9}{'f_ext':>10}{'f_plain':>10}{'binding':>20}{'zeroed5%':>10}{'dynR':>10}")
                logical = bqm8.change_vartype(dimod.SPIN, inplace=False)
                for lab, entry in embs.items():
                    emb = entry["embedding"]
                    cs = float(uniform_torque_compensation(logical, embedding=emb, prefactor=1.414))
                    ideal = IsingArrays.from_bqm(embed_bqm(logical, emb, hw.graph, chain_strength=cs))
                    q2v = {q: v for v, ch in emb.items() for q in ch}
                    owner = np.array([q2v[l] for l in ideal.labels], dtype=object)
                    ischain = owner[ideal.rows] == owner[ideal.cols]
                    expected_chain = sum(hw.graph.subgraph(ch).number_of_edges() for ch in emb.values())
                    check(f"{cname}/{lab}: chain mask matches the hardware edges inside chains",
                          int(ischain.sum()) == expected_chain, f"{int(ischain.sum())} vs {expected_chain}")
                    sc_e, f_e = scale_to_hardware(ideal, HR, JR, ischain, EJR)
                    _, f_p = scale_to_hardware(ideal, HR, JR, ischain, None)
                    check(f"{cname}/{lab}: extended chain range helps by a factor in [1, 2]", 1 - 1e-12 <= f_e / f_p <= 2 + 1e-9,
                          f"{f_e / f_p:.3f}")
                    zs, d5 = [], None
                    for b in (8, 6, 5, 4, 3):
                        _, _, db, _ = run_map(ideal, ischain, b, b)
                        zs.append(db["problem_J_zeroed_pct"])
                        if b == 5:
                            d5 = db
                    check(f"{cname}/{lab}: nothing leaves the programmable range", d5["out_of_range_after_quantise"] == 0)
                    check(f"{cname}/{lab}: fewer bits never erase fewer couplers", all(zs[i] <= zs[i + 1] + 1e-9 for i in range(len(zs) - 1)),
                          str([round(v, 1) for v in zs]))
                    print(f"     {lab:<18}{entry['stats']['physical']:>6}{cs:>9.1f}{f_e:>10.3e}{f_p:>10.3e}"
                          f"{str(d5['binding_constraint']):>20}{d5['problem_J_zeroed_pct']:>10.1f}{d5['dynamic_range_programmed']:>10.1f}")
                    REPORT[f"embedded_{cname}_{lab}"] = {"chain_strength": cs, "scale_ext": f_e, "scale_plain": f_p,
                                                         "zeroed_pct_by_bits(8,6,5,4,3)": zs, "diagnostics_5bit": d5}

    # ================================================================== summary
    section("SUMMARY")
    failed = [c for c in CHECKS if not c[1]]
    print(f"checks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}")
    with open(OUT_JSON, "w") as fh:
        json.dump({"settings": {"quick": QUICK, "no_opf": NO_OPF, "h_range": HR, "j_range": JR, "extended_j_range": EJR},
                   "checks": CHECKS, "report": REPORT}, fh, default=str, indent=1)
    print(f"details written to {OUT_JSON}")
    sys.exit(1 if failed else 0)