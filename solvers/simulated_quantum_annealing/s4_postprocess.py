"""Optional greedy steepest-descent postprocessing (numpy, vectorised over reads).

For each sample, repeatedly flip the single spin whose flip lowers the Ising energy the most, until no
single flip helps (a local minimum w.r.t. one-spin moves). Same idea as D-Wave's SteepestDescentSolver
(dwave-greedy), reimplemented so that sample order and array layout stay under our control.

Ising energy convention: E = sum_i h_i s_i + sum_(ij) J_ij s_i s_j (each coupler once).
Flipping spin i changes E by -2 * s_i * (h_i + sum_j J_ij s_j); gain = s_i * field_i > 0 means E drops.
"""
from __future__ import annotations

import dimod
import numpy as np
import scipy.sparse as sp

from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays


def steepest_descent(ar: IsingArrays, samples: np.ndarray, max_steps: int | None = None, tol: float = 1e-12):
    """samples: (num_reads, n) of +/-1 in ar.labels order. Returns (new_samples int8, total_flips)."""
    n, R = len(ar.labels), samples.shape[0]
    if n == 0 or R == 0:
        return samples, 0
    Jm = sp.coo_matrix((np.r_[ar.J, ar.J], (np.r_[ar.rows, ar.cols], np.r_[ar.cols, ar.rows])),
                       shape=(n, n)).tocsr()
    s = samples.T.astype(np.float64)              # (n, R), copy
    h = ar.h[:, None]
    cols = np.arange(R)
    flips = 0
    for _ in range(max_steps or 10 * n):
        gain = s * (h + Jm @ s)                   # > 0 where flipping lowers the energy
        i = gain.argmax(axis=0)                   # steepest move per read
        active = gain[i, cols] > tol
        if not active.any():
            break
        s[i[active], cols[active]] *= -1.0
        flips += int(active.sum())
    return s.T.astype(np.int8), flips


def descend_sampleset(sampleset: dimod.SampleSet, spin_bqm: dimod.BinaryQuadraticModel):
    """Steepest descent on an already-unembedded LOGICAL SPIN sampleset. Keeps num_occurrences and
    extra vectors (e.g. chain_break_fraction); recomputes energies. Returns (sampleset, total_flips)."""
    ar = IsingArrays.from_bqm(spin_bqm)
    pos = {v: i for i, v in enumerate(sampleset.variables)}
    arr = np.asarray(sampleset.record.sample)[:, [pos[v] for v in ar.labels]]
    new, flips = steepest_descent(ar, arr)
    energies = spin_bqm.energies((new, ar.labels))
    extra = {k: sampleset.record[k] for k in sampleset.record.dtype.names
             if k not in ("sample", "energy", "num_occurrences")}
    out = dimod.SampleSet.from_samples((new, ar.labels), energy=energies, vartype=dimod.SPIN,
                                       num_occurrences=sampleset.record.num_occurrences, **extra)
    return out, flips


# =========================================================================
# Execution Block: Post-Processing Cross-Validation & Pure SD Baseline
# =========================================================================
if __name__ == "__main__":
    import time
    import dimod
    import neal
    import numpy as np
    import pandapower.networks as nw

    try:
        from solvers.qubo_formulator import QuboFormulator
        from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
        from solvers.simulated_quantum_annealing.s5_emulators import EmulatedQPU
        from solvers.simulated_quantum_annealing.s4_postprocess import descend_sampleset
    except ImportError as e:
        print(f"[-] Import error: {e}. Ensure script is run from project root.")
        exit(1)

    print("=== Cross-Validating Steepest Descent ===")
    
    # 1. Formulation
    print("\n[1] Formulating case5 DC-PTDF (10 MW precision)...")
    net = nw.case5()
    if not net.ext_grid.empty: 
        net.ext_grid['min_p_mw'] = 0.0
        
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0)
    logical_bqm, _ = formulator._formulate_qubo(net)
    logical_spin = logical_bqm.change_vartype(dimod.SPIN, inplace=False)

    # 2. Embedding
    print("[2] Minor-Embedding to Zephyr Z4...")
    hw = Hardware.ideal_zephyr(m=4, t=4)
    schedule = Schedule.from_csv()
    emb_results = find_embeddings(logical_spin, hw, methods=["minorminer"], seeds=[0], verbose=False)
    _, embedding = select_best(emb_results)

    results = []
    num_reads = 1000

    # =====================================================================
    # TEST A: Pure Steepest Descent (The "Garbage" Baseline)
    # =====================================================================
    print("\n[3] Running Pure Steepest Descent on Random State...")
    rng = np.random.default_rng(42)
    # Generate 1000 completely random initial states
    random_samples = rng.choice([-1, 1], size=(num_reads, len(logical_spin.variables)))
    
    # Create the SampleSet and evaluate initial random energies
    random_sampleset = dimod.SampleSet.from_samples(
        (random_samples, list(logical_spin.variables)), 
        vartype=dimod.SPIN, 
        energy=logical_spin.energies((random_samples, list(logical_spin.variables)))
    )

    t0 = time.time()
    sd_pure_sampleset, sd_pure_flips = descend_sampleset(random_sampleset, logical_spin)
    t_sd_pure = time.time() - t0

    results.append({
        "Engine": "Pure SD",
        "Mode": "Random Init",
        "Flips (Phys/Log)": f"N/A / {sd_pure_flips}",
        "CBF": "N/A",
        "Best Energy": sd_pure_sampleset.first.energy,
        "Time (s)": t_sd_pure
    })

    # =====================================================================
    # TEST B: Pure Logical SA (The Mathematical Quench)
    # =====================================================================
    print("[4] Running Pure Logical SA (Math Baseline)...")
    t0 = time.time()
    sampler_logical = neal.SimulatedAnnealingSampler()
    sa_sampleset = sampler_logical.sample(logical_spin, num_reads=num_reads, num_sweeps=1000, seed=42)
    t_sa_raw = time.time() - t0

    results.append({
        "Engine": "Logical SA",
        "Mode": "None (Raw)",
        "Flips (Phys/Log)": "N/A / 0",
        "CBF": "N/A",
        "Best Energy": sa_sampleset.first.energy,
        "Time (s)": t_sa_raw
    })

    t0 = time.time()
    sa_descended, flips_logical = descend_sampleset(sa_sampleset, logical_spin)
    t_sa_desc = time.time() - t0

    results.append({
        "Engine": "Logical SA",
        "Mode": "Logical Only",
        "Flips (Phys/Log)": f"N/A / {flips_logical}",
        "CBF": "N/A",
        "Best Energy": sa_descended.first.energy,
        "Time (s)": t_sa_raw + t_sa_desc
    })

    # =====================================================================
    # TEST C: Emulated QPU / SAFreezeout (The Thermal & Analog Fuzz)
    # =====================================================================
    print("[5] Running Emulated QPU (SAFreezeout with Capped Chains)...")
    eqpu = EmulatedQPU(hardware=hw, schedule=schedule, sampler="sa", seed=42)

    modes = [None, "physical", "logical", "both"]
    
    for mode in modes:
        t0 = time.time()
        res = eqpu.sample(
            logical_bqm, embedding, 
            num_reads=num_reads, 
            relative_chain_multiplier=1.0, 
            postprocess=mode,
            chain_break_method="majority"
        )
        t_exec = time.time() - t0
        
        cbf = np.mean(res.sampleset.record.chain_break_fraction) * 100
        flips_phys, flips_log = res.info.get("postprocess_flips", (0, 0))
        
        results.append({
            "Engine": "SAFreezeout",
            "Mode": str(mode).capitalize(),
            "Flips (Phys/Log)": f"{flips_phys} / {flips_log}",
            "CBF": f"{cbf:.2f}%",
            "Best Energy": res.sampleset.first.energy,
            "Time (s)": t_exec
        })

    # =====================================================================
    # 6. Output Cross-Validation Table
    # =====================================================================
    print("\n" + "="*95)
    print(f"{'Engine':<15} | {'Post-Process Mode':<17} | {'Flips (Phys/Log)':<18} | {'CBF (%)':<8} | {'Best Energy':<12} | {'Time (s)'}")
    print("="*95)
    
    for r in results:
        print(f"{r['Engine']:<15} | {r['Mode']:<17} | {r['Flips (Phys/Log)']:<18} | {r['CBF']:<8} | {r['Best Energy']:<12.2f} | {r['Time (s)']:.4f}")
        if r['Mode'] == "Random Init" or r['Mode'] == "Logical Only":
            print("-" * 95)
            
    print("="*95 + "\n")