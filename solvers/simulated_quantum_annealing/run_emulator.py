import time
import numpy as np
import dimod

from solvers.qubo_formulator import QuboFormulator

from solvers.simulated_quantum_annealing.s1_hardware import Hardware
from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
from solvers.simulated_quantum_annealing.s5_emulators import EmulatedQPU

class EmulatedQPUSolver(QuboFormulator):
    """
    End-to-end digital twin of a D-Wave QPU.
    Executes Logical Formulation -> Minor-Embedding -> Hardware Scaling -> DAC Quantization -> 
    ICE Noise Injection -> SVMC Physics Simulation -> Chain Resolution -> Grid Decoding.
    """
    def __init__(self, formulation="dc_ptdf", num_reads=500, num_sweeps=5000, 
                 mw_precision=10.0, hardware_profile=None, sampler="svmc",
                 dac_bits=5, ice_sigma_h=0.01, ice_sigma_j=0.01, use_extended_j=True,
                 reads_per_programming=100, chain_break_method="majority",
                 embedding_methods=("clique", "minorminer"), embedding_seeds=(0, 1, 2),
                 trotter_slices=32, seed=42, **kwargs):
        
        super().__init__(formulation=formulation, mw_precision=mw_precision, **kwargs)
        
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.trotter_slices = trotter_slices  # Added
        
        # Hardware & Analog parameters
        self.hardware = hardware_profile or Hardware.ideal_zephyr(m=15, t=4)
        self.sampler_type = sampler
        self.dac_bits = dac_bits
        self.ice_sigma_h = ice_sigma_h
        self.ice_sigma_j = ice_sigma_j
        self.use_extended_j = use_extended_j
        
        # Execution parameters
        self.reads_per_programming = reads_per_programming
        self.chain_break_method = chain_break_method
        self.embedding_methods = embedding_methods
        self.embedding_seeds = embedding_seeds

    def solve_opf(self, net):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch = "Success", None, {}, {}, {}
        form_time, embed_time, samp_time = 0.0, 0.0, 0.0
        complexity, sampler_stats, hw_metrics = {}, {}, {}
        
        try:
            # 1. Formulation Phase
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0
            logical_vars = len(bqm.variables)

            # --- Calculate Hardware Feasibility (At-Risk Fraction) ---
            at_risk_fraction = 0.0
            if logical_vars > 0:
                max_coeff = max(
                    abs(max(bqm.linear.values(), key=abs, default=0)),
                    abs(max(bqm.quadratic.values(), key=abs, default=0))
                )
                dac_step = max_coeff / (2 ** self.dac_bits)
                at_risk_count = sum(1 for v in list(bqm.linear.values()) + list(bqm.quadratic.values()) 
                                    if 1e-10 < abs(v) < dac_step)
                total_non_zero = sum(1 for v in list(bqm.linear.values()) + list(bqm.quadratic.values()) 
                                     if abs(v) > 1e-10)
                at_risk_fraction = (at_risk_count / total_non_zero) if total_non_zero > 0 else 0.0

            feas_report = {"at_risk_fraction": round(at_risk_fraction, 4)}

            if logical_vars == 0:
                best_sample = {}
                cost = float(bqm.offset)
                feasibility = {"is_feasible": True, "line_flows_mw": {}}
                status = "Success (0 variables, trivial)"
            else:
                # 2. Embedding Phase
                t1 = time.time()
                emb_results = find_embeddings(
                    bqm, self.hardware, methods=self.embedding_methods, 
                    seeds=self.embedding_seeds, verbose=False
                )
                if not emb_results:
                    raise RuntimeError("All embedding heuristics failed.")
                best_label, embedding = select_best(emb_results)
                emb_stats = emb_results[best_label]["stats"]
                embed_time = time.time() - t1

                # 3. Hardware Mapping & Sampling Phase (Single Run with postprocess="both")
                t2 = time.time()
                
                sampler_kwargs = {"num_sweeps": self.num_sweeps}
                if self.sampler_type == "sqa":
                    sampler_kwargs["trotter_slices"] = self.trotter_slices

                qpu = EmulatedQPU(
                    hardware=self.hardware, sampler=self.sampler_type,
                    sampler_kwargs=sampler_kwargs, dac_bits=self.dac_bits,
                    ice_sigma_h=self.ice_sigma_h, ice_sigma_j=self.ice_sigma_j,
                    use_extended_j=self.use_extended_j, seed=self.seed
                )
                
                # Single execution call with "both" (returns final logical postprocessed set, 
                # and we can retain target_sampleset for raw analog metrics if needed)
                qpu.rng = np.random.default_rng(self.seed)
                result = qpu.sample(
                    logical_bqm=bqm, embedding=embedding, num_reads=self.num_reads,
                    reads_per_programming=self.reads_per_programming,
                    relative_chain_multiplier=1.0,  # Protects OPF economics from DAC erasure
                    chain_break_method=self.chain_break_method, 
                    keep_target=True,               # Keeps raw pre-processed physical sample set
                    postprocess="both"
                )
                samp_time = time.time() - t2

                # 4. Decoding Phase: scan the sampleset for the first (lowest-energy)
                decoded = list(result.sampleset.data(
                    ["sample", "energy", "num_occurrences", "chain_break_fraction"], sorted_by="energy"))

                best_feasible, best_any, n_infeasible = None, None, 0
                for d in decoded:
                    disp, sgen_disp, slack_disp, c, feas = self._decode_solution(dict(d.sample), net)
                    row = {"dispatch": disp, "sgen_dispatch": sgen_disp, "slack_dispatch": slack_disp,
                        "cost": c, "feasibility": feas, "qubo_energy": float(d.energy)}
                    if best_any is None:
                        best_any = row
                    if feas["is_feasible"]:
                        best_feasible = row
                        break
                    n_infeasible += 1

                checked = n_infeasible + (1 if best_feasible else 0)
                if best_feasible is not None:
                    chosen, status = best_feasible, "Success"
                else:
                    chosen = best_any
                    status = (f"Infeasible: no feasible sample among {checked} decoded reads "
                            f"(best violation: balance={not chosen['feasibility']['balance_ok']}, "
                            f"lines={not chosen['feasibility']['lines_ok']}, "
                            f"bounds={not chosen['feasibility']['bounds_ok']})")

                dispatch, sgen_dispatch = chosen["dispatch"], chosen["sgen_dispatch"]
                slack_dispatch, cost, feasibility = chosen["slack_dispatch"], chosen["cost"], chosen["feasibility"]
                sampler_stats["reads_checked_for_feasibility"] = checked
                sampler_stats["feasible_read_found"] = best_feasible is not None
                
                # Extract chain diagnostics from embedding
                chain_lens = [len(c) for c in embedding.values()]
                num_chains = sum(1 for l in chain_lens if l > 1)
                single_qubit_vars = sum(1 for l in chain_lens if l == 1)
                mean_chain_len = sum(chain_lens) / len(chain_lens) if chain_lens else 0.0
                chain_variance = float(np.var(chain_lens)) if chain_lens else 0.0

                # Extract raw vs finalized energy metrics
                raw_energies = result.target_sampleset.record.energy if result.target_sampleset else result.sampleset.record.energy
                sds_energies = result.sampleset.record.energy
                cbf_array = getattr(result.sampleset.record, 'chain_break_fraction', np.zeros(len(sds_energies)))
                
                best_sample = result.sampleset.first.sample
                phys_flips, log_flips = result.info.get("postprocess_flips", (0, 0))
                
                raw_best = float(np.min(raw_energies))
                sds_best = float(np.min(sds_energies))

                sampler_stats = {
                    "num_reads_requested": self.num_reads,
                    "num_sweeps_per_read": self.num_sweeps,
                    "unique_states_found": len(result.sampleset),
                    "raw_energy_best_logical": round(raw_best, 2),
                    "raw_energy_mean_logical": round(float(np.mean(raw_energies)), 2),
                    "sds_energy_best_logical": round(sds_best, 2),
                    "sds_energy_mean_logical": round(float(np.mean(sds_energies)), 2),
                    "energy_improvement_delta": round(raw_best - sds_best, 2),
                    "mean_chain_break_fraction": round(float(np.mean(cbf_array)), 4),
                    "postprocess_physical_flips": int(phys_flips),
                    "postprocess_logical_flips": int(log_flips),
                    "total_flips_executed": int(phys_flips + log_flips)
                }
                
                hw_metrics = {
                    "qpu_target": self.hardware.name,
                    "topology": self.hardware.topology,
                    "logical_qubits": logical_vars,
                    "physical_qubits": emb_stats['physical'],
                    "qubit_overhead_factor": round(emb_stats['physical'] / logical_vars, 2) if logical_vars else 0,
                    "max_chain_length": emb_stats['max_chain'],
                    "mean_chain_length": round(mean_chain_len, 2),
                    "chain_length_variance": round(chain_variance, 4),
                    "multi_qubit_chains": num_chains,
                    "single_qubit_variables": single_qubit_vars,
                    "total_chain_qubits": sum(chain_lens),
                    "embedding_heuristic_winner": best_label,
                    "hardware_scale_factor": result.scale_factor,
                    "chain_strength_applied": result.chain_strength,
                    "dac_resolution_bits": self.dac_bits,
                    "dac_step_granularity": round(1.0 / (2 ** self.dac_bits), 5),
                    "ice_sigma_h": self.ice_sigma_h,
                    "ice_sigma_j": self.ice_sigma_j
                }

                # 4. Decoding Phase
                dispatch, sgen_dispatch, slack_dispatch, cost, feasibility = self._decode_solution(best_sample, net)
            
        except Exception as e:
            status = f"Failed: {str(e)}"
            feasibility = {"is_feasible": False, "line_flows_mw": {}}
            print(f"\nCRITICAL ERROR IN EMULATED QPU PIPELINE: {e}\n")

        exec_time = round(time.time() - start_time, 4)
        
        # Safely calculate max line loading percentage
        max_load_pct = None
        if hasattr(self, '_lines') and self._lines:
            loadings = [
                (abs(feasibility.get("line_flows_mw", {}).get(ln['idx'], 0.0)) / ln['p_max']) * 100.0
                for ln in self._lines if ln['p_max'] > 0.0
            ]
            if loadings:
                max_load_pct = round(max(loadings), 3)

        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": max_load_pct,
                "max_voltage_pu": 1.0 if self.formulation == "dc_ptdf" else None,
                "min_voltage_pu": 1.0 if self.formulation == "dc_ptdf" else None,
                "feasibility": feasibility
            }
        }
        
        metadata = {
            "status": status,
            "solver_name": f"emulated_qpu_{self.sampler_type}_{self.formulation}",
            "execution_time_seconds": {
                "total": exec_time,
                "formulation_cpu": round(form_time, 4),
                "embedding_cpu": round(embed_time, 4),
                "sampling_analog_sim": round(samp_time, 4)
            },
            "problem_complexity": complexity,
            "hardware_feasibility_report": feas_report,  # Now correctly populated
            "qubo_parameters": {
                "mw_precision": self.mw_precision,
                "penalty_balance": getattr(self, 'penalty_balance', None),
                "penalty_line": getattr(self, 'penalty_line', None),
            },
            "hardware_metrics": hw_metrics,
            "algorithmic_metrics": {
                "framework": "hardware_aware_analog_simulation",
                "sampler_stats": sampler_stats
            }
        }
        
        return solution, metadata


# =========================================================================
# Execution Block: Submodule Self-Test (EmulatedQPUSolver SAFreezeout)
# =========================================================================
if __name__ == "__main__":
    import pandapower.networks as nw
    
    print("=== Testing EmulatedQPUSolver (SAFreezeout) on case5 ===")
    
    # 1. Load Grid
    net = nw.case5()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
        
    # 2. Configure Solver 
    # (SAFreezeout defaults: sampler="sa", dac_bits=5)
    solver = EmulatedQPUSolver(
        formulation="dc_ptdf",
        mw_precision=10.0,
        sampler="sa",
        num_reads=500,
        num_sweeps=1000
    )
    
    # 3. Solve
    solution, metadata = solver.solve_opf(net)
    
    # 4. Report
    print(f"\n[+] Status: {metadata['status']}")
    if metadata['status'] == 'Success':
        print(f"    Cost: {solution['cost_eur_per_hr']} EUR/hr")
        print(f"    Feasible: {solution['grid_state']['feasibility']['is_feasible']}")
        print(f"    Logical Qubits: {metadata['hardware_metrics']['logical_qubits']}")
        print(f"    Physical Qubits: {metadata['hardware_metrics']['physical_qubits']}")
        print(f"    Chain Breaks (Mean): {metadata['algorithmic_metrics']['sampler_stats']['mean_chain_break_fraction']*100:.2f}%")
        print(f"    Total Flips (SDS): {metadata['algorithmic_metrics']['sampler_stats']['total_flips_executed']}")