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
from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays

class SAFreezeout:
    def __init__(self, schedule: Schedule, temperature_mk: float = 12.0, freeze_s: float = 0.6,
                 num_sweeps: int = 1000, beta_hot: float = 0.1, use_auto_beta: bool = False):
        """freeze_s is the point on the schedule where dynamics stop (a calibration parameter).
        use_auto_beta: If True, ignores the hardware freeze-out and uses neal's classical heuristic."""
        
        self.schedule, self.temperature_mk = schedule, temperature_mk
        self.freeze_s, self.num_sweeps, self.beta_hot = freeze_s, num_sweeps, beta_hot
        self.use_auto_beta = use_auto_beta

    def beta_final(self) -> float:
        _, B = self.schedule.at(self.freeze_s)
        return beta_eff(B, self.temperature_mk)

    def sample(self, ar: IsingArrays, num_reads: int, seed: int) -> np.ndarray:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError:
            from neal import SimulatedAnnealingSampler
        
        bqm = ar.to_bqm()
        
        if self.use_auto_beta:
            beta_range = None
        else:
            bf = self.beta_final()
            beta_range = (min(self.beta_hot, bf / 10.0), bf)
            
        res = SimulatedAnnealingSampler().sample(
            bqm, num_reads=num_reads, num_sweeps=self.num_sweeps,
            beta_range=beta_range, seed=int(seed))
            
        lab = list(res.variables)
        pos = {v: i for i, v in enumerate(lab)}
        order = [pos[v] for v in ar.labels]
        return np.asarray(res.record.sample)[:, order].astype(np.int8)


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
# Execution Block: Physics Engine Comparison on a Real Grid
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