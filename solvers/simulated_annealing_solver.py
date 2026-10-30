import time

import neal
import numpy as np

from solvers.qubo_formulator import QuboFormulator

class SimulatedAnnealingSolver(QuboFormulator):
    def __init__(self, formulation="dc", num_reads=500, num_sweeps=3000,
                 mw_precision=10.0, seed=None, max_decode=None, **kwargs):
        """max_decode: cap on how many distinct low-energy reads get decoded/feasibility-checked
        (decoding calls your formulator's PTDF/angle math per sample, so it isn't free). None decodes
        every distinct sample dimod returns (already far fewer than num_reads after aggregation)."""
        super().__init__(formulation=formulation, mw_precision=mw_precision, **kwargs)
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.max_decode = max_decode
        self.sampler = neal.SimulatedAnnealingSampler()

    def solve_opf(self, net):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch = "Success", None, {}, {}, {}
        form_time, samp_time = 0.0, 0.0
        bqm, complexity, sampler_stats = None, {}, {}

        try:
            # 1. Formulation Phase (CPU) via BaseQuboFormulator
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0

            # 2. Sampling Phase (Classical SA mimicking QPU API)
            t1 = time.time()
            response = self.sampler.sample(bqm, num_reads=self.num_reads, num_sweeps=self.num_sweeps,
                                           seed=self.seed)
            samp_time = time.time() - t1

            energies = response.record.energy
            sampler_stats = {
                "num_reads_requested": self.num_reads,
                "num_sweeps_per_read": self.num_sweeps,
                "unique_states_found": len(response.record),
                "energy_best": round(float(np.min(energies)), 2),
                "energy_mean": round(float(np.mean(energies)), 2),
                "energy_worst": round(float(np.max(energies)), 2),
                "energy_std_dev": round(float(np.std(energies)), 2),
            }

            # 3. Decoding Phase: check every distinct read (sorted by energy, so the first feasible
            # one found is also the lowest-energy feasible one), not just response.first.
            decoded = list(response.data(["sample", "energy", "num_occurrences"], sorted_by="energy"))
            if self.max_decode:
                decoded = decoded[: self.max_decode]

            best_feasible, best_any, n_infeasible = None, None, 0
            for d in decoded:
                disp, sgen_disp, slack_disp, c, feas = self._decode_solution(dict(d.sample), net)
                row = {"dispatch": disp, "sgen_dispatch": sgen_disp, "slack_dispatch": slack_disp,
                      "cost": c, "feasibility": feas, "qubo_energy": float(d.energy)}
                if best_any is None:
                    best_any = row                                 # lowest-energy read, feasible or not
                if feas["is_feasible"]:
                    best_feasible = row                             # decoded() is energy-sorted -> first hit wins
                    break
                n_infeasible += 1

            checked = n_infeasible + (1 if best_feasible else 0)
            if best_feasible is not None:
                chosen, status = best_feasible, "Success"
            else:
                # No feasible read among those checked: still report the lowest-energy one (for
                # inspection/debugging) but mark it plainly so it is never mistaken for a real result.
                chosen = best_any
                status = (f"Infeasible: no feasible sample among {checked} decoded reads "
                         f"(best violation: balance={not chosen['feasibility']['balance_ok']}, "
                         f"lines={not chosen['feasibility']['lines_ok']}, "
                         f"bounds={not chosen['feasibility']['bounds_ok']})")

            dispatch, sgen_dispatch = chosen["dispatch"], chosen["sgen_dispatch"]
            slack_dispatch, cost, feasibility = chosen["slack_dispatch"], chosen["cost"], chosen["feasibility"]
            sampler_stats["reads_checked_for_feasibility"] = checked
            sampler_stats["feasible_read_found"] = best_feasible is not None
            sampler_stats["chosen_sample_energy"] = chosen["qubo_energy"]

        except Exception as e:
            status = f"Failed: {str(e)}"
            print(f"\nCRITICAL FORMULATION ERROR: {e}\n")
            raise

        exec_time = round(time.time() - start_time, 4)

        max_load_pct = None
        if hasattr(self, '_lines') and self._lines and feasibility:
            loadings = []
            for ln in self._lines:
                if ln['p_max'] > 0.0:
                    flow = abs(feasibility["line_flows_mw"].get(ln['idx'], 0.0))
                    loadings.append((flow / ln['p_max']) * 100.0)
            if loadings:
                max_load_pct = round(max(loadings), 3)

        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": max_load_pct,
                "max_voltage_pu": 1.0 if self.formulation == "dc" else None,
                "min_voltage_pu": 1.0 if self.formulation == "dc" else None,
                "feasibility": feasibility,
            },
        }

        metadata = {
            "status": status,
            "solver_name": f"neal_simulated_annealing_{self.formulation}",
            "execution_time_seconds": {"total": exec_time, "formulation_cpu": round(form_time, 4),
                                       "sampling_cpu_sa": round(samp_time, 4)},
            "problem_complexity": complexity,
            "qubo_parameters": {"mw_precision": self.mw_precision,
                                "angle_precision": getattr(self, 'angle_precision', None),
                                "penalty_balance": getattr(self, 'penalty_balance', None),
                                "penalty_line": getattr(self, 'penalty_line', None), "seed": self.seed},
            "algorithmic_metrics": {"framework": "pure_qubo_discretization", "num_iterations": 1,
                                    "optimality_gap_percent": 0.0, "sampler_stats": sampler_stats},
            "hardware_metrics": {"qpu_target": "classical_cpu_simulated",
                                 "logical_qubits": len(bqm.variables) if bqm else 0,
                                 "physical_qubits": len(bqm.variables) if bqm else 0,
                                 "max_chain_length": 1, "circuit_depth": 0},
        }
        return solution, metadata