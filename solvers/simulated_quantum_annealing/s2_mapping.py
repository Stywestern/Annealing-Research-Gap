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
                      chain_j_range=None):
    """Multiply everything by ONE factor (a real device applies a single global scale, not a
    per-coefficient one) so that the tightest of: h vs h_range, non-chain J vs j_range, and chain J
    vs chain_j_range (defaults to j_range) is exactly filled. Ordering of energies is unchanged.

    is_chain: boolean mask over ar.J, True where that coupler is a chain (intra-logical-variable)
    coupling. Without it every J is treated as a regular problem coupling. Real annealers give chain
    couplers access to an EXTENDED J range (e.g. [-2, 1] vs [-1, 1]) that ordinary problem couplings
    may NOT use; applying the extended range to everything (as an earlier version of this function
    did) lets ordinary couplings claim headroom they would not actually have on hardware, which
    understates precision loss. Returns (scaled, factor)."""
    if chain_j_range is None:
        chain_j_range = j_range
    if is_chain is None:
        is_chain = np.zeros(ar.J.shape, dtype=bool)
    f = min(_limit(ar.h, *h_range), _limit(ar.J[~is_chain], *j_range),
           _limit(ar.J[is_chain], *chain_j_range))
    if not np.isfinite(f):
        f = 1.0
    return ar.with_values(h=ar.h * f, J=ar.J * f), float(f)


def quantise(ar: IsingArrays, h_range, j_range, bits: int | None, is_chain: np.ndarray | None = None,
            chain_j_range=None):
    """Round to a uniform grid with 2**bits levels across each range. bits=None -> no quantisation.
    Crude model of finite DAC precision (real devices are not exactly uniform). is_chain/chain_j_range:
    see scale_to_hardware -- chain couplers are quantised against their own (wider) range."""
    if bits is None:
        return ar
    def q(x, lo, hi):
        step = (hi - lo) / (2 ** bits)
        return np.round(x / step) * step
    if chain_j_range is None:
        chain_j_range = j_range
    if is_chain is None:
        is_chain = np.zeros(ar.J.shape, dtype=bool)
    J = np.empty_like(ar.J)
    J[~is_chain] = q(ar.J[~is_chain], *j_range)
    J[is_chain] = q(ar.J[is_chain], *chain_j_range)
    return ar.with_values(h=q(ar.h, *h_range), J=J)


def add_ice_noise(ar: IsingArrays, sigma_h: float, sigma_j: float, rng: np.random.Generator):
    """Independent Gaussian offsets, in the *scaled* units (h, J of order 1). Applies to chain
    couplers too. sigma values are calibration parameters (step 5), not known constants."""
    h = ar.h + rng.normal(0.0, sigma_h, ar.h.shape) if sigma_h > 0 else ar.h
    J = ar.J + rng.normal(0.0, sigma_j, ar.J.shape) if sigma_j > 0 else ar.J
    return ar.with_values(h=h, J=J)


# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import pandapower.networks as nw
    import dimod
    import numpy as np

    try:
        from solvers.qubo_formulator import QuboFormulator
        # Adjust import path for IsingArrays, scale_to_hardware, quantise, add_ice_noise
    except ImportError:
        print("[-] Could not import QuboFormulator. Ensure script is run from project root.")
        exit(1)

    print("=== Testing Analog Hardware Mapping Pipeline ===")
    
    # 1. Generate Logical BQM
    print("[1] Formulating case5 DC-PTDF QUBO...")
    net = nw.case5()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
    
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0, encoding="radix")
    bqm, _ = formulator._formulate_qubo(net)
    
    if len(bqm.variables) == 0:
        print("[-] ERROR: 0 variables. Domain Truncation deleted the BQM.")
        exit(1)
        
    print(f"    -> Logical BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} edges.")

    # 2. Variable-Level Coefficient Analysis
    print("\n[2] Analyzing Logical Coefficient Extremes...")
    
    # Filter out near-zero coefficients to find the true minimums
    h_items = [(v, abs(bias)) for v, bias in bqm.linear.items() if abs(bias) > 1e-10]
    j_items = [(u, v, abs(bias)) for (u, v), bias in bqm.quadratic.items() if abs(bias) > 1e-10]
    
    if h_items:
        h_max_var, h_max_val = max(h_items, key=lambda item: item[1])
        h_min_var, h_min_val = min(h_items, key=lambda item: item[1])
        print(f"    -> Max |h|: {h_max_val:,.2f} (Variable: '{h_max_var}')")
        print(f"    -> Min |h|: {h_min_val:,.2f} (Variable: '{h_min_var}')")
    
    if j_items:
        j_max_u, j_max_v, j_max_val = max(j_items, key=lambda item: item[2])
        j_min_u, j_min_v, j_min_val = min(j_items, key=lambda item: item[2])
        print(f"    -> Max |J|: {j_max_val:,.2f} (Coupler: '{j_max_u}' <-> '{j_max_v}')")
        print(f"    -> Min |J|: {j_min_val:,.2f} (Coupler: '{j_min_u}' <-> '{j_min_v}')")

    # 3. Abstract into IsingArrays
    print("\n[3] Converting to IsingArrays for bulk numpy operations...")
    ar_logical = IsingArrays.from_bqm(bqm.change_vartype(dimod.SPIN, inplace=False))
    
    # Note: dimod.SPIN conversion shifts QUBO variables (0,1) to Ising variables (-1,1).
    # This mathematically alters the linear and constant terms, but the quadratic J terms 
    # typically divide by 4. The dynamic range relationships remain fundamentally intact.
    
    # 4. Simulate Hardware Scaling (auto_scale)
    print("\n[4] Scaling to hardware limits (h: [-4, 4], J: [-1, 1])...")
    h_range = (-4.0, 4.0)
    j_range = (-1.0, 1.0)
    
    # Calculate headroom based on the Ising converted values
    max_h_ising = np.max(np.abs(ar_logical.h)) if len(ar_logical.h) > 0 else 0
    max_j_ising = np.max(np.abs(ar_logical.J)) if len(ar_logical.J) > 0 else 0
    
    f_h_limit = h_range[1] / max_h_ising if max_h_ising > 0 else np.inf
    f_j_limit = j_range[1] / max_j_ising if max_j_ising > 0 else np.inf
    
    print(f"    -> Available headroom for h: {f_h_limit:e}")
    print(f"    -> Available headroom for J: {f_j_limit:e}")
    
    ar_scaled, scale_factor = scale_to_hardware(ar_logical, h_range, j_range)
    print(f"    -> Applied bottleneck Scale Factor: {scale_factor:e}")
    
    max_h_scaled = np.max(np.abs(ar_scaled.h)) if len(ar_scaled.h) > 0 else 0
    max_j_scaled = np.max(np.abs(ar_scaled.J)) if len(ar_scaled.J) > 0 else 0
    print(f"    -> Post-Scale Max |h|: {max_h_scaled:.4f} (Limit: 4.0)")
    print(f"    -> Post-Scale Max |J|: {max_j_scaled:.4f} (Limit: 1.0)")

    # 5. Simulate Finite DAC Precision
    print("\n[5] Simulating 5-bit DAC Quantization...")
    ar_quantized = quantise(ar_scaled, h_range, j_range, bits=5)
    
    nonzero_scaled_j = np.sum(np.abs(ar_scaled.J) > 1e-10)
    nonzero_quant_j = np.sum(np.abs(ar_quantized.J) > 1e-10)
    erased = nonzero_scaled_j - nonzero_quant_j
    print(f"    -> {erased} J-couplers (out of {nonzero_scaled_j}) were erased to exactly 0.0.")

    # 6. Simulate Analog ICE Noise
    print("\n[6] Injecting Analog ICE Noise (sigma = 0.01)...")
    rng = np.random.default_rng(seed=42)
    ar_noisy = add_ice_noise(ar_quantized, sigma_h=0.01, sigma_j=0.01, rng=rng)
    
    err_h = np.sqrt(np.mean((ar_noisy.h - ar_scaled.h)**2)) if len(ar_noisy.h) > 0 else 0
    err_j = np.sqrt(np.mean((ar_noisy.J - ar_scaled.J)**2)) if len(ar_noisy.J) > 0 else 0
    print(f"    -> RMS |h| deviation from ideal: {err_h:.4f}")
    print(f"    -> RMS |J| deviation from ideal: {err_j:.4f}")
    
    print("\n[+] Analog mapping pipeline executed successfully.")