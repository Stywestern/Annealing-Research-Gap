"""Steps 2-4 glued together: EmulatedQPU.sample(logical_bqm, embedding, ...) -> logical SampleSet."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import dimod
import numpy as np
from dwave.embedding import (MinimizeEnergy, discard, embed_bqm, majority_vote,
                             unembed_sampleset, weighted_random)
from dwave.embedding.chain_strength import uniform_torque_compensation

from solvers.simulated_quantum_annealing.s1b_embedding import embedding_stats
from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
from solvers.simulated_quantum_annealing.s2_mapping import IsingArrays, add_ice_noise, quantise, scale_to_hardware
from solvers.simulated_quantum_annealing.s3_samplers import make_sampler
from solvers.simulated_quantum_annealing.s4_postprocess import descend_sampleset, steepest_descent

@dataclass
class EmulationResult:
    sampleset: dimod.SampleSet            # logical, BINARY, has 'chain_break_fraction'
    chain_strength: float                 # in problem units, before hardware scaling
    scale_factor: float                   # problem units -> device units
    embedding_stats: dict
    target_sampleset: dimod.SampleSet | None = None
    info: dict = field(default_factory=dict)


class EmulatedQPU:
    def __init__(self, hardware: Hardware, sampler: str = "sa",
                 sampler_kwargs: dict | None = None, dac_bits: int | None = 5,
                 ice_sigma_h: float = 0.02, ice_sigma_j: float = 0.01,
                 use_extended_j: bool = True, seed: int = 0):
        """
        dac_bits, ice_sigma_h, ice_sigma_j : PLACEHOLDER defaults. They are the knobs step 5 calibrates
        against real QPU runs; do not read physical meaning into them yet.
        use_extended_j : allow chain couplers (strongly negative J) to use the extended J range.
        """
        self.hw = hardware
        self.schedule = Schedule.from_csv("/home/stywestern/Quantum_Digital_Twin/solvers/simulated_quantum_annealing/standart_annealing_schedule_Ad2Sys1.csv")
        self.sampler_name = sampler
        self.sampler = make_sampler(sampler, self.schedule, hardware.temperature_mk, **(sampler_kwargs or {}))
        self.dac_bits, self.sigma_h, self.sigma_j = dac_bits, ice_sigma_h, ice_sigma_j
        self.use_extended_j = use_extended_j
        self.rng = np.random.default_rng(seed)

    def _chain_break_method(self, name, logical_spin, embedding):
        return {"majority": majority_vote, "discard": discard, "weighted": weighted_random,
                "minimize_energy": MinimizeEnergy(logical_spin, embedding)}[name]

    @staticmethod
    def _local_chain_cap(logical_spin: dimod.BinaryQuadraticModel, embedding: dict,
                         relative_chain_multiplier: float) -> dict:
        """Per-variable cap = relative_chain_multiplier * (sum of |coupling| on edges actually
        incident to that logical variable). A chain only needs to outlast the pull it is actually
        under, so this is anchored locally rather than to the single worst coefficient anywhere in
        the whole problem (which earlier under- or over-constrained every OTHER chain at once)."""
        incident = {v: 0.0 for v in logical_spin.variables}
        for u, v, bias in logical_spin.iter_quadratic():
            incident[u] += abs(bias)
            incident[v] += abs(bias)
        return {v: relative_chain_multiplier * incident[v] for v in embedding}

    def sample(self, logical_bqm: dimod.BinaryQuadraticModel, embedding: dict,
               chain_strength: float | None = None, prefactor: float = 1.414,
               relative_chain_multiplier: float | None = None,
               num_reads: int = 1000, reads_per_programming: int = 100,
               chain_break_method: str = "majority", keep_target: bool = False,
               postprocess: str | None = "both") -> EmulationResult:
        """
        relative_chain_multiplier : if given, caps each logical variable's chain strength at this
                                    multiplier times the TOTAL |coupling| actually incident to that
                                    variable (not the global max coefficient -- a chain only has to
                                    outlast the pull it is actually subject to, and a global cap
                                    either does nothing for most chains or starves the busiest ones).
                                    1.0 means a chain is exactly as strong as everything pulling on
                                    it combined. Per-variable: embed_bqm accepts a dict chain_strength.
        reads_per_programming : the ICE noise realisation is redrawn every this many reads, mimicking
                                re-programming of the device between batches.
        postprocess : None (default) | "physical" | "logical" | "both". Greedy steepest descent, run
                      against the SAME (possibly capped) Hamiltonian that was sampled and scored --
                      not a separate, uncapped one -- so a post-processing step can never reverse the
                      cap or optimize against an objective different from the one being reported.
                      "physical": on the embedded problem, before chain resolution.
                      "logical" : on the logical problem after chain resolution.
        """
        if postprocess not in (None, "physical", "logical", "both"):
            raise ValueError("postprocess must be None, 'physical', 'logical' or 'both'")
        logical_spin = logical_bqm.change_vartype(dimod.SPIN, inplace=False)
        raw_cs = None                                  # uncapped uniform-torque-compensation value, if computed
        if chain_strength is None:
            raw_cs = float(uniform_torque_compensation(logical_spin, embedding=embedding, prefactor=prefactor))
            if relative_chain_multiplier is not None:
                cap = self._local_chain_cap(logical_spin, embedding, relative_chain_multiplier)
                chain_strength = {v: min(raw_cs, c) for v, c in cap.items()}
            else:
                chain_strength = raw_cs

        # 1) embed: chain couplers = -chain_strength, biases spread over chains. ONE Hamiltonian from
        # here on -- the same embedded model is sampled, post-processed, and scored, so nothing can
        # silently optimize against a different objective than the one being reported.
        embedded = embed_bqm(logical_spin, embedding, self.hw.graph, chain_strength=chain_strength)
        ideal = IsingArrays.from_bqm(embedded)

        # which entries of ideal.J are chain couplers (both endpoints in the SAME logical variable's
        # chain) vs. ordinary problem couplings (endpoints in different chains). embed_bqm adds a
        # -chain_strength coupling for every hardware edge inside a chain, so "same chain" <=> "chain
        # edge" exactly -- no need to inspect embed_bqm's internals to get this right.
        qubit_to_var = {q: v for v, chain in embedding.items() for q in chain}
        owner = np.array([qubit_to_var[lbl] for lbl in ideal.labels], dtype=object)
        is_chain = owner[ideal.rows] == owner[ideal.cols]

        # 2) map to device: scale, quantise. Only CHAIN couplers get the extended J range on real
        # hardware; ordinary problem couplings are confined to the narrower j_range even when the
        # device supports extended_j_range. Treating all J alike (as an earlier version did) lets
        # ordinary couplings claim headroom hardware would never give them.
        chain_j_range = self.hw.extended_j_range if self.use_extended_j else self.hw.j_range
        scaled, factor = scale_to_hardware(ideal, self.hw.h_range, self.hw.j_range,
                                           is_chain=is_chain, chain_j_range=chain_j_range)
        programmed = quantise(scaled, self.hw.h_range, self.hw.j_range, self.dac_bits,
                              is_chain=is_chain, chain_j_range=chain_j_range)

        # 3) sample, redrawing control noise for each programming cycle
        batches, done = [], 0
        while done < num_reads:
            r = min(reads_per_programming, num_reads - done)
            noisy = add_ice_noise(programmed, self.sigma_h, self.sigma_j, self.rng)
            batches.append(self.sampler.sample(noisy, r, seed=int(self.rng.integers(2 ** 31 - 1))))
            done += r
        samples = np.vstack(batches)

        flips_phys = flips_log = 0
        if postprocess in ("physical", "both"):
            samples, flips_phys = steepest_descent(ideal, samples)

        # energies are reported w.r.t. the SAME embedded model that was sampled and postprocessed
        # (the capped one, if a cap was applied) -- never a different, uncapped stand-in
        energies = embedded.energies((samples, ideal.labels))
        target = dimod.SampleSet.from_samples((samples, ideal.labels), energy=energies, vartype=dimod.SPIN)

        # 4) resolve chains -> logical samples
        method = self._chain_break_method(chain_break_method, logical_spin, embedding)
        logical = unembed_sampleset(target, embedding, logical_spin,
                                    chain_break_method=method, chain_break_fraction=True)
        if postprocess in ("logical", "both"):
            logical, flips_log = descend_sampleset(logical, logical_spin)
        logical = logical.change_vartype(dimod.BINARY, inplace=False)

        capped = isinstance(chain_strength, dict)
        info = {"sampler": self.sampler_name, "dac_bits": self.dac_bits, "sigma_h": self.sigma_h,
                "sigma_j": self.sigma_j, "num_reads": num_reads,
                "reads_per_programming": reads_per_programming,
                "chain_break_method": chain_break_method,
                "postprocess": postprocess, "postprocess_flips": (flips_phys, flips_log),
                "placeholder_schedule": self.schedule.is_placeholder,
                "chain_strength_capped": capped,
                # when capped, chain_strength varies per logical variable -- report the range so a
                # single-number summary doesn't hide that some chains were constrained more than others
                "chain_strength_range": ((min(chain_strength.values()), max(chain_strength.values()))
                                         if capped else (chain_strength, chain_strength))}
        # EmulationResult.chain_strength stays a single float for backward-compat summaries (e.g.
        # scan_chain_strength): the raw uncapped uniform-torque-compensation value if we computed one
        # (auto chain strength, capped or not), otherwise whatever the caller passed in directly.
        reported_cs = raw_cs if raw_cs is not None else chain_strength
        return EmulationResult(logical, reported_cs, factor, embedding_stats(embedding),
                               target if keep_target else None, info)


# =========================================================================
# Execution Block: Exhaustive Combinatorial Sweep & Architecture Benchmark
# =========================================================================
if __name__ == "__main__":
    import time
    import itertools
    import dimod
    import numpy as np
    import pandapower.networks as nw

    try:
        from solvers.qubo_formulator import QuboFormulator
        from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
        from solvers.simulated_quantum_annealing.s5_emulators import EmulatedQPU
    except ImportError as e:
        print(f"[-] Import error: {e}. Ensure script is run from project root.")
        exit(1)

    print("=== Initializing Exhaustive Combinatorial Emulation Sweep on case5 ===")
    
    # 1. Grid & Formulation
    net = nw.case5()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
    
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0)
    logical_bqm, _ = formulator._formulate_qubo(net)

    # 2. Hardware & Minor-Embedding (Fixed to keep graph routing constant)
    hw = Hardware.ideal_zephyr(m=4, t=4)
    emb_results = find_embeddings(logical_bqm, hw, methods=["minorminer"], seeds=[0], verbose=False)
    _, embedding = select_best(emb_results)

    # 3. Define Combinatorial Parameter Space
    samplers = ["sqa", "svmc", "sa"]
    dac_bits_list = [3, 5, 8]
    multipliers = [None, 0.5, 1.0, 2.0]
    cb_methods = ["majority", "discard"]
    post_modes = [None, "physical", "logical", "both"]

    schedule = Schedule.placeholder()
    num_reads = 100
    seed_val = 42

    combinations = list(itertools.product(samplers, dac_bits_list, multipliers, cb_methods, post_modes))
    total_runs = len(combinations)
    
    print(f"[-] Total configurations in combinatorial space: {total_runs}")
    print("\n" + "="*125)
    print(f"{'Sampler':<6} | {'DAC':<4} | {'Mult':<6} | {'CB Method':<9} | {'Post-Proc':<9} | {'Energy':<10} | {'CBF (%)':<8} | {'Phys':<5} | {'Log':<5} | {'Time (s)'}")
    print("="*125)

    results = []

    for idx, (sampler_name, dac, mult, cb, pp) in enumerate(combinations, 1):
        # Configure sampler-specific arguments
        sampler_kwargs = {}
        if sampler_name == "sqa":
            sampler_kwargs["trotter_slices"] = 16
        elif sampler_name == "svmc":
            sampler_kwargs["num_sweeps"] = 1000

        qpu = EmulatedQPU(
            hardware=hw,
            schedule=schedule,
            sampler=sampler_name,
            sampler_kwargs=sampler_kwargs,
            dac_bits=dac,
            ice_sigma_h=0.01,
            ice_sigma_j=0.01,
            seed=seed_val
        )

        t0 = time.time()
        try:
            res = qpu.sample(
                logical_bqm=logical_bqm,
                embedding=embedding,
                num_reads=num_reads,
                reads_per_programming=50,
                relative_chain_multiplier=mult,
                chain_break_method=cb,
                postprocess=pp
            )
            t_exec = time.time() - t0

            if len(res.sampleset) > 0:
                best_energy = float(res.sampleset.first.energy)
                cbf = float(np.mean(res.sampleset.record.chain_break_fraction) * 100)
            else:
                best_energy = float('nan')
                cbf = 100.0

            phys_flips, log_flips = res.info.get("postprocess_flips", (0, 0))
            status = "Success"

        except Exception as e:
            t_exec = time.time() - t0
            best_energy = float('nan')
            cbf = 100.0
            phys_flips, log_flips = 0, 0
            status = f"Failed: {str(e)}"

        results.append({
            "sampler": sampler_name,
            "dac": dac,
            "mult": mult,
            "cb": cb,
            "pp": str(pp),
            "energy": best_energy,
            "cbf": cbf,
            "phys": phys_flips,
            "log": log_flips,
            "time": t_exec,
            "status": status
        })

        mult_str = str(mult) if mult is not None else "RMS"
        pp_str = str(pp) if pp is not None else "None"
        
        print(f"{sampler_name.upper():<6} | {dac:<4} | {mult_str:<6} | {cb:<9} | {pp_str:<9} | {best_energy:<10.2f} | {cbf:<8.2f} | {phys_flips:<5} | {log_flips:<5} | {t_exec:<8.4f}")

    print("="*125)
    print(f"[+] Completed all {total_runs} emulation runs.")