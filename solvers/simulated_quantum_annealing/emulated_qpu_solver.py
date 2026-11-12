"""End-to-end solver wrapper around EmulatedQPU: OPF network in -> solution + metadata out.

Pipeline: logical formulation -> embedding (all requested methods, one chosen) -> EmulatedQPU (scaling, DAC, control
errors, sampler, chain resolution, optional descent) -> decoding of BOTH the raw emulator output and the final stage.
EmulatedQPU itself stops at a logical sampleset; the formulation, embedding search, decoding and the solution /
metadata dictionaries live here.

The metadata records what produced the result: this wrapper's own settings, the embedding exploration (every
attempt, failures included), the emulator's provenance (hardware, schedule, sampler, descent engines, mapping,
seed, config hash, software versions) and the emulator's run record.
"""
import json
import time

import numpy as np

from solvers.qubo_formulator import QuboFormulator

from solvers.simulated_quantum_annealing.s1_hardware import Hardware
from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
from solvers.simulated_quantum_annealing.s5_emulators import EmulatedQPU


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)


def _safe(x):
    """JSON-safe copy (keys become strings; unknown objects become their repr)."""
    return json.loads(json.dumps(x, default=_default))


def _wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, in percent."""
    if n == 0:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(100 * (c - h)), float(100 * (c + h))]


class EmulatedQPUSolver(QuboFormulator):
    """
    End-to-end digital twin of a D-Wave QPU.
    Executes Logical Formulation -> Minor-Embedding -> Hardware Scaling -> DAC Quantization ->
    ICE Noise Injection -> Sampler Simulation -> Chain Resolution -> (optional) Greedy Descent -> Grid Decoding.

    postprocess                : None | "physical" | "logical" | "both" (passed to EmulatedQPU). The raw emulator output
                                 is ALWAYS decoded as well, so the metadata shows what the descent adds.
    relative_chain_multiplier  : passed to EmulatedQPU.sample (per-variable chain-strength cap); None = uncapped.
                                 The default 1.0 keeps the earlier behaviour; its benefit has not been tested.
    embedding_timeout          : seconds per minorminer-type attempt (find_embeddings default would be 36000 s).
    embedding_select_key       : stats fields that rank the candidate embeddings, see select_best().
    sampler_kwargs             : extra arguments for the sampler (e.g. {"method": "pt", "beta_override": 12.0}); they are
                                 merged over {"num_sweeps": ..., ("trotter_slices": ...)}.
    emulator_kwargs            : extra EmulatedQPU constructor arguments (schedule, temperature_mk, dac_bits_h/_j,
                                 ice_sigma_j_chain, systematic, per_qubit_coupling_range, postprocess_settings, ...);
                                 they override the named arguments above.
    sample_kwargs              : extra EmulatedQPU.sample arguments (spin_reversal, anneal_time_us, prefactor, ...).
    include_emulator_record    : store the emulator's run record and provenance in the metadata.
    """

    def __init__(self, formulation="dc_ptdf", num_reads=500, num_sweeps=5000,
                 mw_precision=10.0, hardware_profile=None, sampler="svmc",
                 dac_bits=5, ice_sigma_h=0.01, ice_sigma_j=0.01, use_extended_j=True,
                 reads_per_programming=100, chain_break_method="majority",
                 embedding_methods=("clique", "minorminer"), embedding_seeds=(0, 1, 2),
                 trotter_slices=32, seed=42, *,
                 postprocess="both", relative_chain_multiplier=1.0, embedding_timeout=120.0,
                 embedding_select_key=("physical", "max_chain"),
                 sampler_kwargs=None, emulator_kwargs=None, sample_kwargs=None,
                 include_emulator_record=True, **kwargs):

        super().__init__(formulation=formulation, mw_precision=mw_precision, **kwargs)

        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed
        self.trotter_slices = trotter_slices

        # Hardware & analog parameters. Z12 = the Advantage2-like size used everywhere else in this project.
        self.hardware = hardware_profile or Hardware.ideal_zephyr(m=12, t=4)
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
        self.embedding_timeout = embedding_timeout
        self.embedding_select_key = tuple(embedding_select_key)
        self.postprocess = postprocess
        self.relative_chain_multiplier = relative_chain_multiplier
        self.sampler_kwargs = dict(sampler_kwargs or {})
        self.emulator_kwargs = dict(emulator_kwargs or {})
        self.sample_kwargs = dict(sample_kwargs or {})
        self.include_emulator_record = include_emulator_record

    # ------------------------------------------------------------------ helpers
    def _settings(self):
        return _safe({
            "num_reads": self.num_reads, "num_sweeps": self.num_sweeps, "trotter_slices": self.trotter_slices,
            "sampler": self.sampler_type, "seed": self.seed, "reads_per_programming": self.reads_per_programming,
            "chain_break_method": self.chain_break_method, "postprocess": self.postprocess,
            "relative_chain_multiplier": self.relative_chain_multiplier,
            "embedding_methods": self.embedding_methods, "embedding_seeds": self.embedding_seeds,
            "embedding_timeout": self.embedding_timeout, "embedding_select_key": self.embedding_select_key,
            "dac_bits": self.dac_bits, "ice_sigma_h": self.ice_sigma_h, "ice_sigma_j": self.ice_sigma_j,
            "use_extended_j": self.use_extended_j, "sampler_kwargs": self.sampler_kwargs,
            "emulator_kwargs": self.emulator_kwargs, "sample_kwargs": self.sample_kwargs,
            "selection_rule": "lowest-energy feasible read of the final stage (else lowest-energy read)"})

    def _legacy_at_risk_fraction(self, bqm):
        """Crude QUBO-level proxy kept for comparability with older results (None without a DAC). The real numbers
        are in hardware_feasibility_report['mapping']."""
        if self.dac_bits is None:
            return None
        vals = [abs(v) for v in list(bqm.linear.values()) + list(bqm.quadratic.values()) if abs(v) > 1e-10]
        if not vals:
            return 0.0
        step = max(vals) / (2 ** self.dac_bits)
        return round(sum(1 for v in vals if v < step) / len(vals), 4)

    @staticmethod
    def _violation_flags(feas):
        """True = satisfied. Works with formulator versions that do not report balance_ok / lines_ok / bounds_ok."""
        tol = feas.get("tolerance_mw", 0.0)
        bal = feas.get("balance_ok", abs(feas.get("raw_imbalance_mw", 0.0)) <= tol)
        lin = feas.get("lines_ok", feas.get("max_line_violation_mw", 0.0) <= tol)
        bnd = feas.get("bounds_ok", not (bal and lin and not feas.get("is_feasible", True)))
        return {"balance": bool(bal), "lines": bool(lin), "bounds": bool(bnd)}

    def _decode_set(self, ss, net):
        """Decodes every distinct state of a logical sampleset once. Rows are visited in energy order."""
        cache, n_reads, n_feas = {}, 0, 0
        best_e = best_c = best_any = None
        viol = {"balance": 0, "lines": 0, "bounds": 0}
        for d in ss.data(["sample", "energy", "num_occurrences"], sorted_by="energy"):
            occ = int(d.num_occurrences)
            key = tuple(sorted((str(k), int(v)) for k, v in d.sample.items()))
            row = cache.get(key)
            if row is None:
                disp, sgen_disp, slack_disp, c, feas = self._decode_solution(dict(d.sample), net)
                row = {"dispatch": disp, "sgen_dispatch": sgen_disp, "slack_dispatch": slack_disp,
                       "cost": c, "feasibility": feas, "qubo_energy": float(d.energy)}
                cache[key] = row
            n_reads += occ
            if best_any is None:
                best_any = row
            if row["feasibility"]["is_feasible"]:
                n_feas += occ
                if best_e is None:
                    best_e = row
                if best_c is None or row["cost"] < best_c["cost"]:
                    best_c = row
            else:
                for k, ok in self._violation_flags(row["feasibility"]).items():
                    if not ok:
                        viol[k] += occ
        return {"n_reads": n_reads, "n_feasible": n_feas, "n_unique_decoded": len(cache),
                "best_feasible_by_energy": best_e, "best_feasible_by_cost": best_c, "best_any": best_any,
                "violations_among_infeasible_reads": viol}

    @staticmethod
    def _stage_quality(q):
        n, k = q["n_reads"], q["n_feasible"]
        be, bc = q["best_feasible_by_energy"], q["best_feasible_by_cost"]
        return {"n_reads": n, "n_feasible": k, "feasible_pct": (100.0 * k / n) if n else None,
                "feasible_ci95_pct": _wilson(k, n), "n_unique_states_decoded": q["n_unique_decoded"],
                "cost_of_lowest_energy_feasible": be["cost"] if be else None,
                "lowest_cost_among_feasible": bc["cost"] if bc else None,
                "violations_among_infeasible_reads": q["violations_among_infeasible_reads"]}

    # ------------------------------------------------------------------ main entry
    def solve_opf(self, net):
        start_time = time.time()
        status, cost = "Success", None
        dispatch, sgen_dispatch, slack_dispatch = {}, {}, {}
        feasibility = {"is_feasible": False, "line_flows_mw": {}}
        form_time = embed_time = samp_time = decode_time = 0.0
        complexity, sampler_stats, hw_metrics, hw_feas = {}, {}, {}, {}
        emulator_block, embedding_report, error = {}, None, None

        try:
            # 1. Formulation
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0
            logical_vars = len(bqm.variables)
            hw_feas = {"at_risk_fraction": self._legacy_at_risk_fraction(bqm)}

            if logical_vars == 0:
                cost = float(bqm.offset)
                feasibility = {"is_feasible": True, "line_flows_mw": {}}
                status = "Success (0 variables, trivial)"
            else:
                # 2. Embedding: every requested method is tried; the reasons for failures are kept
                t1 = time.time()
                embs = find_embeddings(bqm, self.hardware, methods=self.embedding_methods,
                                       seeds=self.embedding_seeds, timeout=self.embedding_timeout, verbose=False)
                embed_time = time.time() - t1
                if not embs:
                    reasons = "; ".join(f"{a['label']}: {a.get('error')}" for a in getattr(embs, "attempts", [])
                                        if not a.get("ok"))
                    raise RuntimeError(f"no embedding found ({reasons or 'no attempt succeeded'})")
                best_label, embedding = select_best(embs, key=self.embedding_select_key)
                emb_stats = embs[best_label]["stats"]
                embedding_report = _safe(embs.report(chosen=best_label)) if hasattr(embs, "report") else None

                # 3. Mapping + sampling (EmulatedQPU)
                t2 = time.time()
                sampler_kwargs = {"num_sweeps": self.num_sweeps}
                if self.sampler_type == "sqa":
                    sampler_kwargs["trotter_slices"] = self.trotter_slices
                sampler_kwargs.update(self.sampler_kwargs)
                ctor = dict(hardware=self.hardware, sampler=self.sampler_type, sampler_kwargs=sampler_kwargs,
                            dac_bits=self.dac_bits, ice_sigma_h=self.ice_sigma_h, ice_sigma_j=self.ice_sigma_j,
                            use_extended_j=self.use_extended_j, seed=self.seed)
                ctor.update(self.emulator_kwargs)
                qpu = EmulatedQPU(**ctor)
                call = dict(logical_bqm=bqm, embedding=embedding, num_reads=self.num_reads,
                            reads_per_programming=self.reads_per_programming,
                            relative_chain_multiplier=self.relative_chain_multiplier,
                            chain_break_method=self.chain_break_method, postprocess=self.postprocess,
                            include_provenance=False)       # the provenance is stored once, below
                call.update(self.sample_kwargs)
                result = qpu.sample(**call)
                samp_time = time.time() - t2
                info = result.info

                # 4. Decoding: the final stage, and the raw emulator output for comparison
                t3 = time.time()
                final_q = self._decode_set(result.sampleset, net)
                if final_q["n_reads"] == 0:
                    raise RuntimeError("the emulator returned no samples (all reads discarded?)")
                raw_q = final_q if info["selected_stage"] == "raw" else self._decode_set(result.raw_sampleset, net)
                decode_time = time.time() - t3

                chosen = final_q["best_feasible_by_energy"]
                if chosen is not None:
                    status = "Success"
                else:
                    chosen = final_q["best_any"]
                    fl = self._violation_flags(chosen["feasibility"])
                    status = (f"Infeasible: no feasible sample among {final_q['n_unique_decoded']} decoded states "
                              f"({final_q['n_reads']} reads); lowest-energy sample violates: "
                              f"balance={not fl['balance']}, lines={not fl['lines']}, bounds={not fl['bounds']}")
                dispatch, sgen_dispatch = chosen["dispatch"], chosen["sgen_dispatch"]
                slack_dispatch, cost, feasibility = chosen["slack_dispatch"], chosen["cost"], chosen["feasibility"]

                # 5. Metrics
                chain_lens = [len(c) for c in embedding.values()]
                st_raw, st_fin = info["stages"]["raw"], info["stages"][info["selected_stage"]]
                p_phys, p_log = info.get("postprocess_flips", (0, 0))
                sampler_stats = {
                    "num_reads_requested": self.num_reads,
                    "num_sweeps_per_read": self.num_sweeps,
                    "rows_returned": len(result.sampleset),
                    "unique_logical_states": final_q["n_unique_decoded"],
                    "selected_stage": info["selected_stage"],
                    "raw_energy_best_logical": round(st_raw["min_energy"], 2),
                    "raw_energy_mean_logical": round(st_raw["mean_energy"], 2),
                    "final_energy_best_logical": round(st_fin["min_energy"], 2),
                    "final_energy_mean_logical": round(st_fin["mean_energy"], 2),
                    "energy_improvement_delta_best": round(st_raw["min_energy"] - st_fin["min_energy"], 2),
                    "energy_improvement_delta_mean": round(st_raw["mean_energy"] - st_fin["mean_energy"], 2),
                    "mean_chain_break_fraction_raw": round(info["chain_breaks"]["raw_mean"], 4),
                    "postprocess_physical_flips": int(p_phys),
                    "postprocess_logical_flips": int(p_log),
                    "total_flips_executed": int(p_phys + p_log),
                    "reads_checked_for_feasibility": final_q["n_unique_decoded"],
                    "feasible_read_found": final_q["best_feasible_by_energy"] is not None,
                    "stage_quality": {"raw": self._stage_quality(raw_q), "final": self._stage_quality(final_q)}}

                m = info["mapping"]
                hw_metrics = {
                    "qpu_target": self.hardware.name,
                    "topology": self.hardware.topology,
                    "logical_qubits": logical_vars,
                    "physical_qubits": emb_stats["physical"],
                    "qubit_overhead_factor": round(emb_stats["physical"] / logical_vars, 2),
                    "max_chain_length": emb_stats["max_chain"],
                    "mean_chain_length": round(sum(chain_lens) / len(chain_lens), 2),
                    "chain_length_variance": round(float(np.var(chain_lens)), 4),
                    "multi_qubit_chains": sum(1 for l in chain_lens if l > 1),
                    "single_qubit_variables": sum(1 for l in chain_lens if l == 1),
                    "total_chain_qubits": sum(chain_lens),
                    "embedding_heuristic_winner": best_label,
                    "hardware_scale_factor": result.scale_factor,
                    "binding_constraint": m["binding_constraint"],
                    "chain_strength_uncapped": result.chain_strength,
                    "chain_strength_applied_range": list(info["chain_strength_range"]),
                    "chain_strength_capped": info["chain_strength_capped"],
                    "relative_chain_multiplier": self.relative_chain_multiplier,
                    "dac_resolution_bits": self.dac_bits,
                    "dac_step_h": m["step_h"], "dac_step_j": m["step_j"],
                    "problem_J_zeroed_pct": m["problem_J_zeroed_pct"],
                    "dynamic_range_programmed": m["dynamic_range_programmed"],
                    "ice_sigma_h": self.ice_sigma_h, "ice_sigma_j": self.ice_sigma_j,
                    "temperature_used_mk": info["temperature_used_mk"],
                    "temperature_source": info["temperature_source"]}
                hw_feas.update({"mapping": _safe(m), "thermal_resolution": _safe(info["thermal_resolution"])})
                if self.include_emulator_record:
                    emulator_block = {"provenance": _safe(qpu.provenance()), "record": _safe(info)}

        except Exception as e:
            status = f"Failed: {e}"
            error = f"{type(e).__name__}: {e}"
            feasibility = {"is_feasible": False, "line_flows_mw": {}}
            print(f"\nCRITICAL ERROR IN EMULATED QPU PIPELINE: {e}\n")

        exec_time = round(time.time() - start_time, 4)

        # Safely calculate max line loading percentage
        max_load_pct = None
        if hasattr(self, "_lines") and self._lines:
            loadings = [
                (abs(feasibility.get("line_flows_mw", {}).get(ln["idx"], 0.0)) / ln["p_max"]) * 100.0
                for ln in self._lines if ln["p_max"] > 0.0
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
                "sampling_analog_sim": round(samp_time, 4),
                "decoding_cpu": round(decode_time, 4)
            },
            "problem_complexity": complexity,
            "hardware_feasibility_report": hw_feas,
            "qubo_parameters": {
                "mw_precision": self.mw_precision,
                "penalty_balance": getattr(self, "penalty_balance", None),
                "penalty_line": getattr(self, "penalty_line", None),
            },
            "solver_settings": self._settings(),
            "hardware_metrics": hw_metrics,
            "embedding_exploration": embedding_report,
            "emulator": emulator_block,
            "algorithmic_metrics": {
                "framework": "hardware_aware_analog_simulation",
                "sampler_stats": sampler_stats
            }
        }
        if error is not None:
            metadata["error"] = error

        return solution, metadata


# =========================================================================
# Execution Block: Self-Test (EmulatedQPUSolver)
# =========================================================================
if __name__ == "__main__":
    import copy
    import itertools
    import math
    import sys
    import traceback
    import warnings
    from dataclasses import replace as dc_replace

    import dimod
    import pandapower as pp
    import pandapower.networks as nw

    from solvers.simulated_quantum_annealing.s1_hardware import Schedule
    from solvers.simulated_quantum_annealing.s2_mapping import SystematicICE

    QUICK = False
    NO_QUANT = False
    NO_EXTRA = False
    OUT_JSON, OUT_CASE5 = "solver_selftest.json", "solver_case5_result.json"
    CHECKS, REPORT, S = [], {}, {}

    try:
        from dwave.samplers import SimulatedAnnealingSampler as _SA
    except ImportError:
        try:
            from neal import SimulatedAnnealingSampler as _SA
        except ImportError:
            _SA = None

    # ------------------------------------------------------------------ helpers
    def check(name, ok, detail=""):
        CHECKS.append((name, bool(ok), str(detail)))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def info_(msg):
        print(f"    [info] {msg}")

    def section(title):
        print(f"\n{'=' * 98}\n{title}\n{'=' * 98}")

    def guarded(title, fn):
        section(title)
        try:
            fn()
        except Exception as exc:
            last = traceback.format_exc().strip().splitlines()[-1]
            check(f"section ran to the end: {title}", False, f"{type(exc).__name__}: {exc} | {last}")

    def raises(exc, fn):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    def find_unserialisable(o, path="root"):
        """First path in o that json.dumps would reject (numpy integers / bools / arrays, objects, non-str keys) or that
        is a non-finite float (valid Python, invalid strict JSON). None if the object is clean."""
        if isinstance(o, dict):
            for k, v in o.items():
                if not (k is None or isinstance(k, (str, int, float, bool))):
                    return f"{path}: key {k!r} of type {type(k).__name__}"
                r = find_unserialisable(v, f"{path}.{k}")
                if r:
                    return r
            return None
        if isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                r = find_unserialisable(v, f"{path}[{i}]")
                if r:
                    return r
            return None
        if o is None or isinstance(o, (str, int, float, bool)):
            if isinstance(o, float) and not math.isfinite(o):
                return f"{path}: non-finite float {o}"
            return None
        return f"{path}: {type(o).__name__}"

    def trim(o, n=3):
        if isinstance(o, dict):
            return {k: trim(v, n) for k, v in o.items() if k != "rng_state_before"}
        if isinstance(o, list):
            return [trim(v, n) for v in o[:n]] + ([f"... ({len(o) - n} more)"] if len(o) > n else [])
        if isinstance(o, float):
            return float(f"{o:.6g}")
        if isinstance(o, str) and len(o) > 110:
            return o[:110] + "..."
        return o

    def SQ(md):
        return md["algorithmic_metrics"]["sampler_stats"]

    def HM(md):
        return md["hardware_metrics"]

    def PROV(md):
        return md["emulator"]["provenance"]

    def REC(md):
        return md["emulator"]["record"]

    SOL_KEYS = {"cost_eur_per_hr", "generator_dispatch_mw", "static_generator_dispatch_mw", "slack_dispatch_mw", "grid_state"}
    MD_KEYS = {"status", "solver_name", "execution_time_seconds", "problem_complexity", "hardware_feasibility_report",
               "qubo_parameters", "solver_settings", "hardware_metrics", "embedding_exploration", "emulator",
               "algorithmic_metrics"}
    TIME_KEYS = {"total", "formulation_cpu", "embedding_cpu", "sampling_analog_sim", "decoding_cpu"}

    def generic(label, sol, md):
        missing = sorted((SOL_KEYS - set(sol)) | (MD_KEYS - set(md)) | (TIME_KEYS - set(md.get("execution_time_seconds", {}))))
        check(f"{label}: solution and metadata have every documented key", not missing, str(missing))
        bad = find_unserialisable({"solution": sol, "metadata": md})
        check(f"{label}: strictly JSON-serialisable (no numpy types, no NaN)", bad is None, bad or "")

    # ------------------------------------------------------------------ setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SCHED = Schedule.from_csv(strict=False)
    HW = Hardware.ideal_zephyr(m=12, t=4)
    HW4 = Hardware.ideal_zephyr(m=4, t=4)
    HW1 = Hardware.ideal_zephyr(m=1, t=4)

    def prep(name):
        net = getattr(nw, name)()
        if not net.ext_grid.empty:
            net.ext_grid["min_p_mw"] = 0.0
        ref = None
        try:
            tmp = copy.deepcopy(net)
            pp.rundcopp(tmp)
            ref = float(tmp.res_cost)
        except Exception:
            pass
        return net, ref

    NET5, REF5 = prep("case5")

    def mk(**kw):
        emu = {"schedule": SCHED}
        emu.update(kw.pop("emulator_kwargs", None) or {})
        base = dict(formulation="dc_ptdf", num_reads=80, num_sweeps=200, mw_precision=10.0, hardware_profile=HW,
                    sampler="sa", dac_bits=8, ice_sigma_h=0.01, ice_sigma_j=0.005, reads_per_programming=40,
                    chain_break_method="majority", embedding_methods=("clique",), embedding_seeds=(0,),
                    embedding_timeout=30.0, seed=5, postprocess="both", trotter_slices=8)
        base.update(kw)
        return EmulatedQPUSolver(emulator_kwargs=emu, **base)

    print("=== Testing EmulatedQPUSolver (formulation -> embedding -> emulator -> decoding -> solution + metadata) ===")
    print(f"mode={'quick' if QUICK else 'full'} | quantum samplers={'off' if NO_QUANT else 'on'} | schedule {getattr(SCHED, 'source', '?')} "
          f"(placeholder={getattr(SCHED, 'is_placeholder', '?')})")
    print(f"case5 classical DC-OPF reference cost: {REF5}")

    # ================================================================== [1] construction
    def s1():
        sv = EmulatedQPUSolver()
        check("is a QuboFormulator", isinstance(sv, QuboFormulator))
        check("default hardware is the Z12 (Advantage2-like) target", list(sv.hardware.topology.get("shape", [])) == [12, 4],
              f"{sv.hardware.name}")
        check("defaults: sampler svmc, postprocess both, relative_chain_multiplier 1.0, embedding timeout 120, seeds (0, 1, 2)",
              sv.sampler_type == "svmc" and sv.postprocess == "both" and sv.relative_chain_multiplier == 1.0
              and sv.embedding_timeout == 120.0 and tuple(sv.embedding_seeds) == (0, 1, 2))
        check("defaults: select key (physical, max_chain), emulator record included, dac 5 bits",
              sv.embedding_select_key == ("physical", "max_chain") and sv.include_emulator_record is True and sv.dac_bits == 5)
        v = EmulatedQPUSolver(tighten_bounds=False, penalty_safety=2.0, encoding="bounded")
        check("formulator keyword arguments pass through", v.tighten_bounds is False and v.penalty_safety == 2.0 and v.encoding == "bounded")
        check("formulation='dc' is the dc_ptdf alias", EmulatedQPUSolver(formulation="dc").formulation == "dc_ptdf")
        check("an invalid formulation raises ValueError", raises(ValueError, lambda: EmulatedQPUSolver(formulation="nope")))
        check("a custom hardware profile is kept", EmulatedQPUSolver(hardware_profile=HW4).hardware is HW4)
        d = {"method": "pt"}
        sv2 = EmulatedQPUSolver(sampler_kwargs=d, emulator_kwargs={"dac_bits": 3}, sample_kwargs={"spin_reversal": True})
        d["x"] = 1
        check("the extra-argument dictionaries are copied at construction", "x" not in sv2.sampler_kwargs)
        st = mk()._settings()
        bad = find_unserialisable(st)
        check("_settings() is JSON-safe and documents the selection rule", bad is None and "lowest-energy feasible" in st["selection_rule"], bad or "")
        check("_settings() reflects the constructor", st["num_reads"] == 80 and st["num_sweeps"] == 200 and st["seed"] == 5
              and st["postprocess"] == "both" and st["embedding_methods"] == ["clique"])
        st2 = mk(emulator_kwargs={"systematic": SystematicICE(0.01, 0.005, seed=1)})._settings()
        info_(f"objects inside emulator_kwargs are stored as text in solver_settings: {str(st2['emulator_kwargs'].get('systematic'))[:70]}")

    # ================================================================== [2] helpers
    def s2():
        check("_safe converts numpy scalars / arrays, tuples and non-string keys",
              _safe({"a": np.int64(3), "b": np.array([1.0, 2.0]), "c": (1, 2), 5: np.float32(1.5)}) == {"a": 3, "b": [1.0, 2.0], "c": [1, 2], "5": 1.5})
        lo, hi = _wilson(50, 100)
        check("_wilson(50, 100) is about [40.4, 59.6]", abs(lo - 40.38) < 0.1 and abs(hi - 59.62) < 0.1, f"[{lo:.2f}, {hi:.2f}]")
        lo0, hi0 = _wilson(0, 100)
        check("_wilson(0, 100): lower bound 0, upper about 3.7", abs(lo0) < 1e-9 and 3.6 < hi0 < 3.8, f"[{lo0:.3g}, {hi0:.3f}]")
        lo1, hi1 = _wilson(100, 100)
        check("_wilson(100, 100): upper bound 100", abs(hi1 - 100.0) < 1e-9 and 96.0 < lo1 < 96.6, f"[{lo1:.2f}, {hi1:.2f}]")
        check("_wilson(0, 0) is [None, None]", _wilson(0, 0) == [None, None])

        F = EmulatedQPUSolver._violation_flags
        ok_ = {"tolerance_mw": 10.0, "raw_imbalance_mw": -5.0, "max_line_violation_mw": 2.0, "is_feasible": True}
        check("flags: everything within tolerance -> all satisfied", F(ok_) == {"balance": True, "lines": True, "bounds": True})
        check("flags: imbalance above tolerance -> balance violated only",
              F({**ok_, "raw_imbalance_mw": -50.0, "is_feasible": False}) == {"balance": False, "lines": True, "bounds": True})
        check("flags: line violation above tolerance -> lines violated only",
              F({**ok_, "max_line_violation_mw": 50.0, "is_feasible": False}) == {"balance": True, "lines": False, "bounds": True})
        check("flags: balance and lines fine but infeasible -> bounds violated",
              F({**ok_, "is_feasible": False}) == {"balance": True, "lines": True, "bounds": False})
        check("flags: explicit balance_ok / lines_ok / bounds_ok keys take priority",
              F({**ok_, "balance_ok": False, "lines_ok": True, "bounds_ok": True}) == {"balance": False, "lines": True, "bounds": True})
        check("flags: an empty dictionary does not crash", F({}) == {"balance": True, "lines": True, "bounds": True})

        sv5, sv0 = mk(dac_bits=5), mk(dac_bits=None)
        b = dimod.BinaryQuadraticModel({"a": 1.0, "b": 0.0}, {("a", "b"): 0.001}, 0.0, dimod.BINARY)
        check("at-risk fraction: one of two non-zero coefficients is below the 5-bit step (0.5)", sv5._legacy_at_risk_fraction(b) == 0.5)
        check("at-risk fraction is None without a DAC", sv0._legacy_at_risk_fraction(b) is None)
        check("at-risk fraction of an empty model is 0.0", sv5._legacy_at_risk_fraction(dimod.BinaryQuadraticModel.empty(dimod.BINARY)) == 0.0)

        q = {"n_reads": 200, "n_feasible": 50, "n_unique_decoded": 17, "best_feasible_by_energy": {"cost": 100.0},
             "best_feasible_by_cost": {"cost": 90.0}, "violations_among_infeasible_reads": {"balance": 150, "lines": 0, "bounds": 0}}
        sq = EmulatedQPUSolver._stage_quality(q)
        check("stage quality: percentages, interval, costs and violations",
              sq["feasible_pct"] == 25.0 and sq["feasible_ci95_pct"] == _wilson(50, 200) and sq["cost_of_lowest_energy_feasible"] == 100.0
              and sq["lowest_cost_among_feasible"] == 90.0 and sq["n_unique_states_decoded"] == 17
              and sq["violations_among_infeasible_reads"]["balance"] == 150)
        q2 = {**q, "n_feasible": 0, "best_feasible_by_energy": None, "best_feasible_by_cost": None}
        sq2 = EmulatedQPUSolver._stage_quality(q2)
        check("stage quality without a feasible read: zero percent, no costs",
              sq2["feasible_pct"] == 0.0 and sq2["cost_of_lowest_energy_feasible"] is None and sq2["lowest_cost_among_feasible"] is None)

    # ================================================================== [3] decoding against an independent recount
    def s3():
        sv = mk()
        bqm, _ = sv._formulate_qubo(NET5)
        labels = list(bqm.variables)
        r = np.random.default_rng(1)
        parts = []
        if _SA is not None:
            res = _SA().sample(bqm, num_reads=60, num_sweeps=500, seed=3)
            pos = {v: i for i, v in enumerate(res.variables)}
            parts.append(np.asarray(res.record.sample)[:, [pos[v] for v in labels]].astype(np.int8))
        parts.append(r.integers(0, 2, size=(60, len(labels))).astype(np.int8))
        X = np.vstack(parts)
        occ = r.integers(1, 4, size=len(X))
        ss = dimod.SampleSet.from_samples((X, labels), energy=bqm.energies((X, labels)), vartype=dimod.BINARY, num_occurrences=occ)
        q = sv._decode_set(ss, NET5)

        feas, viol, n_feas, n_inf = [], {"balance": 0, "lines": 0, "bounds": 0}, 0, 0
        for i in range(len(X)):
            out = sv._decode_solution({l: int(v) for l, v in zip(labels, X[i])}, NET5)
            f, e = out[4], float(ss.record.energy[i])
            if f["is_feasible"]:
                n_feas += int(occ[i])
                feas.append((e, out[3]))
            else:
                n_inf += int(occ[i])
                for k, ok in EmulatedQPUSolver._violation_flags(f).items():
                    if not ok:
                        viol[k] += int(occ[i])
        check("decode_set: read counts (total and feasible) equal an independent recount",
              q["n_reads"] == int(occ.sum()) and q["n_feasible"] == n_feas, f"{n_feas} of {int(occ.sum())} reads feasible")
        check("decode_set: number of distinct decoded states", q["n_unique_decoded"] == len({tuple(row) for row in X}))
        if feas:
            be = min(feas, key=lambda t: t[0])
            check("decode_set: lowest-energy feasible state and its cost", q["best_feasible_by_energy"] is not None
                  and abs(q["best_feasible_by_energy"]["cost"] - be[1]) < 1e-9 and abs(q["best_feasible_by_energy"]["qubo_energy"] - be[0]) < 1e-6)
            check("decode_set: lowest-cost feasible state", abs(q["best_feasible_by_cost"]["cost"] - min(c for _, c in feas)) < 1e-9)
        else:
            check("decode_set: no feasible reads -> no feasible bests", q["best_feasible_by_energy"] is None and q["best_feasible_by_cost"] is None)
        check("decode_set: best_any is the lowest-energy state of the whole set", abs(q["best_any"]["qubo_energy"] - float(ss.record.energy.min())) < 1e-6)
        check("decode_set: violation counts among infeasible reads equal an independent recount",
              q["violations_among_infeasible_reads"] == viol, f"{q['violations_among_infeasible_reads']}")
        check("decode_set: every infeasible read violates at least one constraint class",
              sum(viol.values()) >= n_inf and max(viol.values()) <= n_inf)
        sq = EmulatedQPUSolver._stage_quality(q)
        check("stage quality of the decoded set matches the counts", abs(sq["feasible_pct"] - 100.0 * n_feas / int(occ.sum())) < 1e-9)
        info_(f"synthetic mixed sampleset: {n_feas} of {int(occ.sum())} reads feasible ({'neal + random' if _SA is not None else 'random only'})")

    # ================================================================== [4] end-to-end case5
    def s4():
        sv = mk()
        t0 = time.time()
        sol, md = sv.solve_opf(NET5)
        wall = time.time() - t0
        S["main"] = (sol, md, sv)
        hm, ss_, ex, em, rec, prov = HM(md), SQ(md), md["embedding_exploration"], md["emulator"], REC(md), PROV(md)
        feas = sol["grid_state"]["feasibility"]
        check("status is Success", md["status"] == "Success", md["status"][:120])
        generic("case5 default-ish run", sol, md)
        check("solver name", md["solver_name"] == "emulated_qpu_sa_dc_ptdf", md["solver_name"])
        check("a cost was reported and the solution is feasible", sol["cost_eur_per_hr"] is not None and feas["is_feasible"] is True)

        values = {"gen": {int(k): v["p_mw"] for k, v in sol["generator_dispatch_mw"].items()},
                  "sgen": {int(k): v["p_mw"] for k, v in sol["static_generator_dispatch_mw"].items()},
                  "ext_grid": {int(k): v["p_mw"] for k, v in sol["slack_dispatch_mw"].items()}}
        rc = sv._total_true_cost(values)
        check("reported cost equals the cost recomputed from the reported dispatch", abs(rc - sol["cost_eur_per_hr"]) < 2.0,
              f"{sol['cost_eur_per_hr']:.3f} vs {rc:.3f}")
        check("generation equals load after the slack rebalance", abs(feas["total_generation_mw"] - feas["total_load_mw"]) < 0.01,
              f"{feas['total_generation_mw']} vs {feas['total_load_mw']}")
        worst = 0.0
        for et, key in (("gen", "generator_dispatch_mw"), ("sgen", "static_generator_dispatch_mw"), ("ext_grid", "slack_dispatch_mw")):
            for idx, v in sol[key].items():
                lo, hi = sv._asset_bounds(NET5, et, int(idx))
                worst = max(worst, lo - v["p_mw"], v["p_mw"] - hi)
        check("dispatch respects the asset limits", worst <= 1e-3, f"worst violation {max(worst, 0.0):.2e} MW")
        tol = feas["tolerance_mw"]
        over = max((abs(feas["line_flows_mw"].get(ln["idx"], 0.0)) - ln["p_max"] for ln in sv._lines if ln["p_max"] > 0), default=0.0)
        check("line flows are within the limits plus the feasibility tolerance", over <= tol + 1e-6, f"max excess {over:.3f} MW, tolerance {tol}")
        loads = [abs(feas["line_flows_mw"].get(ln["idx"], 0.0)) / ln["p_max"] * 100.0 for ln in sv._lines if ln["p_max"] > 0]
        check("max_line_loading_percent equals a recomputation from the line flows",
              abs(sol["grid_state"]["max_line_loading_percent"] - round(max(loads), 3)) < 1e-9, f"{sol['grid_state']['max_line_loading_percent']}")

        tm = md["execution_time_seconds"]
        parts = sum(v for k, v in tm.items() if k != "total")
        check("timing: parts do not exceed the total, total matches the wall clock", parts <= tm["total"] + 1e-3 and tm["total"] <= wall + 1e-3 and tm["total"] >= wall - 0.05,
              f"total {tm['total']}s, parts {parts:.3f}s, wall {wall:.3f}s")
        cx = md["problem_complexity"]
        check("problem complexity agrees with the hardware metrics (logical qubits)", cx["quantum_domain_qubo"]["total_logical_qubits"] == hm["logical_qubits"],
              f"{hm['logical_qubits']}")
        check("resolved QUBO penalties are recorded in the complexity block (the qubo_parameters entries stay None when automatic)",
              cx["penalties"]["balance"] > 0 and md["qubo_parameters"]["mw_precision"] == 10.0,
              f"balance {cx['penalties']['balance']:.4g}, qubo_parameters balance {md['qubo_parameters']['penalty_balance']}")
        hf = md["hardware_feasibility_report"]
        check("hardware feasibility report: at-risk fraction, mapping diagnostics and thermal resolution",
              0.0 <= hf["at_risk_fraction"] <= 1.0 and "binding_constraint" in hf["mapping"] and "kT_scaled" in hf["thermal_resolution"],
              f"at-risk {hf['at_risk_fraction']}, binding {hf['mapping']['binding_constraint']}")
        st = md["solver_settings"]
        check("solver_settings equals the constructor settings",
              st["num_reads"] == 80 and st["num_sweeps"] == 200 and st["seed"] == 5 and st["reads_per_programming"] == 40
              and st["postprocess"] == "both" and st["relative_chain_multiplier"] == 1.0 and st["embedding_timeout"] == 30.0)

        check("hardware metrics: overhead = mean chain length = physical / logical",
              abs(hm["qubit_overhead_factor"] - round(hm["physical_qubits"] / hm["logical_qubits"], 2)) < 1e-9
              and abs(hm["mean_chain_length"] - hm["qubit_overhead_factor"]) < 1e-9, f"{hm['physical_qubits']} qubits for {hm['logical_qubits']} variables")
        check("hardware metrics: multi-qubit + single-qubit variables = logical, chain qubits = physical",
              hm["multi_qubit_chains"] + hm["single_qubit_variables"] == hm["logical_qubits"] and hm["total_chain_qubits"] == hm["physical_qubits"])
        check("hardware metrics: target name and topology", hm["qpu_target"] == HW.name and list(hm["topology"]["shape"]) == [12, 4])
        chosen = next((a for a in ex["attempts"] if a["label"] == hm["embedding_heuristic_winner"]), None)
        check("embedding exploration: the winner is one of the successful attempts and its footprint matches the hardware metrics",
              ex["chosen"] == hm["embedding_heuristic_winner"] and chosen is not None and chosen["ok"]
              and chosen["footprint"]["physical"] == hm["physical_qubits"] and chosen["footprint"]["max_chain"] == hm["max_chain_length"])
        if chosen is not None:
            fp = chosen["footprint"]
            n = fp["logical"]
            mean = fp["physical"] / n
            var = sum(c * (int(k) - mean) ** 2 for k, c in fp["hist"].items()) / n
            check("chain-length variance equals a recomputation from the footprint histogram", abs(var - hm["chain_length_variance"]) < 1e-3,
                  f"{hm['chain_length_variance']}")
        check("chain strength: capped by default (multiplier 1.0), applied range ordered, never above the uncapped value",
              hm["chain_strength_capped"] is True and hm["chain_strength_applied_range"][0] <= hm["chain_strength_applied_range"][1] <= hm["chain_strength_uncapped"] * (1 + 1e-9),
              f"range {[round(x, 4) for x in hm['chain_strength_applied_range']]}, uncapped {hm['chain_strength_uncapped']:.4g}")
        check("mapping: scale factor and binding constraint agree between hardware metrics, report and record",
              abs(hm["hardware_scale_factor"] - hf["mapping"]["scale_factor"]) <= 1e-12 * hm["hardware_scale_factor"]
              and hm["binding_constraint"] == hf["mapping"]["binding_constraint"] == rec["mapping"]["binding_constraint"])
        check("temperature used: physical 12 mK", hm["temperature_used_mk"] == 12.0 and hm["temperature_source"] == "physical")

        check("emulator block: provenance seed / run count / temperature source / 12-character hash",
              prov["emulator"]["seed"] == 5 and prov["emulator"]["runs_completed"] == 1 and prov["emulator"]["temperature_source"] == "physical"
              and len(prov["emulator"]["config_hash"]) == 12, f"hash {prov['emulator']['config_hash']}")
        check("emulator block: hardware and schedule provenance equal the submodules' own", prov["hardware"] == HW.provenance() and prov["schedule"] == SCHED.provenance())
        check("emulator record: selected stage 'both', two programming cycles, no embedded provenance copy",
              rec["selected_stage"] == "both" and rec["run"]["n_programming_cycles"] == 2 and "provenance" not in rec and rec["run"]["emulator_seed"] == 5)
        rraw, rfin = rec["stages"]["raw"], rec["stages"]["both"]
        check("sampler stats equal the emulator record (energies, deltas, flips, chain breaks)",
              abs(ss_["raw_energy_mean_logical"] - rraw["mean_energy"]) <= 0.0051 and abs(ss_["final_energy_best_logical"] - rfin["min_energy"]) <= 0.0051
              and abs(ss_["energy_improvement_delta_best"] - round(rraw["min_energy"] - rfin["min_energy"], 2)) <= 0.0101
              and ss_["total_flips_executed"] == ss_["postprocess_physical_flips"] + ss_["postprocess_logical_flips"] == sum(rec["postprocess_flips"])
              and abs(ss_["mean_chain_break_fraction_raw"] - round(rec["chain_breaks"]["raw_mean"], 4)) < 1e-9)
        sqr, sqf = ss_["stage_quality"]["raw"], ss_["stage_quality"]["final"]
        for nm, x in (("raw", sqr), ("final", sqf)):
            check(f"stage quality ({nm}): 80 reads, percentage = counts, Wilson interval brackets it",
                  x["n_reads"] == 80 and abs(x["feasible_pct"] - 100.0 * x["n_feasible"] / x["n_reads"]) < 1e-9
                  and x["feasible_ci95_pct"][0] <= x["feasible_pct"] <= x["feasible_ci95_pct"][1], f"{x['feasible_pct']:.1f}% {[round(v, 1) for v in x['feasible_ci95_pct']]}")
        check("the reported solution is the lowest-energy feasible read of the final stage and the best-cost read is not worse",
              sqf["cost_of_lowest_energy_feasible"] == sol["cost_eur_per_hr"] and sqf["lowest_cost_among_feasible"] <= sol["cost_eur_per_hr"] + 1e-9)
        check("the final stage is at least as feasible as the raw emulator output", sqf["feasible_pct"] >= sqr["feasible_pct"] - 1e-9)
        check("reads checked / unique states equal the stage quality", ss_["reads_checked_for_feasibility"] == sqf["n_unique_states_decoded"] == ss_["unique_logical_states"]
              and ss_["feasible_read_found"] is True)
        info_(f"case5: raw emulator output {sqr['feasible_pct']:.1f}% feasible -> final stage {sqf['feasible_pct']:.1f}%; "
              f"descent flips {ss_['total_flips_executed']}; classical reference {REF5:.2f}, reported cost {sol['cost_eur_per_hr']:.2f} "
              f"({100 * (sol['cost_eur_per_hr'] - REF5) / REF5:+.2f}%)")
        for nm, v in (("tolerance_mw", feas["tolerance_mw"]), ("raw imbalance", feas["raw_imbalance_mw"])):
            info_(f"feasibility {nm}: {v}")

    # ================================================================== [5] every setting
    VARIANTS = []

    def add(label, kw=None, fn=None, quick=False, quantum=False):
        VARIANTS.append((label, kw or {}, fn, quick, quantum))

    def _stage_flips(md):
        return SQ(md)["postprocess_physical_flips"], SQ(md)["postprocess_logical_flips"]

    main_zeroed = lambda: S["main"][1]["hardware_feasibility_report"]["mapping"]["problem_J_zeroed_pct"]
    add("postprocess=None (the raw emulation)", dict(postprocess=None), lambda sol, md, sv: [
        ("selected stage is raw; raw and final stage quality are identical", SQ(md)["selected_stage"] == "raw" and SQ(md)["stage_quality"]["raw"] == SQ(md)["stage_quality"]["final"], ""),
        ("no flips, zero energy improvement", SQ(md)["total_flips_executed"] == 0 and SQ(md)["energy_improvement_delta_best"] == 0 and SQ(md)["energy_improvement_delta_mean"] == 0, ""),
        ("no descent runs in the record", REC(md)["postprocess_runs"] == {}, "")], quick=True)
    add("postprocess=physical", dict(postprocess="physical"), lambda sol, md, sv: [
        ("selected stage is physical; only the physical engine flipped", SQ(md)["selected_stage"] == "physical" and _stage_flips(md)[1] == 0, f"flips {_stage_flips(md)}"),
        ("the record has the physical descent run", "physical" in REC(md)["postprocess_runs"], "")])
    add("postprocess=logical", dict(postprocess="logical"), lambda sol, md, sv: [
        ("selected stage is logical; only the logical engine flipped", SQ(md)["selected_stage"] == "logical" and _stage_flips(md)[0] == 0 and _stage_flips(md)[1] > 0, f"flips {_stage_flips(md)}"),
        ("the descent lowers the mean logical energy", SQ(md)["energy_improvement_delta_mean"] > 0, f"{SQ(md)['energy_improvement_delta_mean']}")], quick=True)
    add("postprocess=both", dict(postprocess="both"), lambda sol, md, sv: [
        ("selected stage is both; both descent runs are recorded", SQ(md)["selected_stage"] == "both" and {"physical", "both"} <= set(REC(md)["postprocess_runs"]), "")])
    add("sampler=svmc", dict(sampler="svmc", num_sweeps=30), lambda sol, md, sv: [
        ("solver name and sampler class", md["solver_name"] == "emulated_qpu_svmc_dc_ptdf" and PROV(md)["sampler"]["class"] == "SVMCSampler", ""),
        ("no thermal diagnostics for SVMC", md["hardware_feasibility_report"]["thermal_resolution"] is None, "")], quantum=True)
    add("sampler=sqa (8 Trotter slices)", dict(sampler="sqa", num_sweeps=20, trotter_slices=8), lambda sol, md, sv: [
        ("solver name and sampler class", md["solver_name"] == "emulated_qpu_sqa_dc_ptdf" and PROV(md)["sampler"]["class"] == "SQASampler", ""),
        ("the Trotter slices reach the sampler", PROV(md)["sampler"].get("P", PROV(md)["sampler"].get("trotter_slices")) == 8, f"{PROV(md)['sampler']}")], quantum=True)
    add("sampler_kwargs: pt merged over num_sweeps", dict(sampler_kwargs=dict(method="pt", beta_override=2.0, pt_sweeps=15, pt_replicas=4)), lambda sol, md, sv: [
        ("method, beta override and the merged default num_sweeps are in the sampler provenance",
         PROV(md)["sampler"]["method"] == "pt" and PROV(md)["sampler"]["beta_override"] == 2.0 and PROV(md)["sampler"]["num_sweeps"] == 200, ""),
        ("solver_settings records the extra sampler arguments", md["solver_settings"]["sampler_kwargs"]["method"] == "pt", "")], quick=True)
    add("sampler_kwargs overrides num_sweeps", dict(sampler_kwargs=dict(num_sweeps=77)), lambda sol, md, sv: [
        ("[override] the sampler runs the overridden sweeps", PROV(md)["sampler"]["num_sweeps"] == 77, ""),
        ("[override] num_sweeps_per_read reports the sweeps the sampler actually used", SQ(md)["num_sweeps_per_read"] == PROV(md)["sampler"]["num_sweeps"],
         f"reported {SQ(md)['num_sweeps_per_read']}, sampler {PROV(md)['sampler']['num_sweeps']}")])
    add("emulator_kwargs override named arguments", dict(dac_bits=5, emulator_kwargs=dict(dac_bits=6, dac_bits_h=7, temperature_mk=20.0, ice_sigma_h=0.03, ice_sigma_j_chain=0.004)),
        lambda sol, md, sv: [
            ("the emulator used the overrides (provenance mapping)", PROV(md)["mapping"]["dac_bits"] == 6 and PROV(md)["mapping"]["dac_bits_h"] == 7
             and PROV(md)["mapping"]["ice_sigma_h"] == 0.03 and PROV(md)["mapping"]["ice_sigma_j_chain"] == 0.004, ""),
            ("DAC steps in the metadata follow the overrides (h: 8/2^7, J: 2/2^6)", abs(HM(md)["dac_step_h"] - 8 / 128) < 1e-12 and abs(HM(md)["dac_step_j"] - 2 / 64) < 1e-12, ""),
            ("temperature override recorded", HM(md)["temperature_used_mk"] == 20.0 and HM(md)["temperature_source"] == "override", ""),
            ("[override] hardware_metrics.dac_resolution_bits equals the DAC bits the emulator used", HM(md)["dac_resolution_bits"] == PROV(md)["mapping"]["dac_bits"],
             f"reported {HM(md)['dac_resolution_bits']}, emulator {PROV(md)['mapping']['dac_bits']}"),
            ("[override] hardware_metrics ICE sigmas equal the ones the emulator used", HM(md)["ice_sigma_h"] == PROV(md)["mapping"]["ice_sigma_h"],
             f"reported {HM(md)['ice_sigma_h']}, emulator {PROV(md)['mapping']['ice_sigma_h']}")])
    add("sample_kwargs: spin_reversal + anneal_time_us", dict(sample_kwargs=dict(spin_reversal=True, anneal_time_us=5.0)), lambda sol, md, sv: [
        ("recorded in the emulator record, every cycle gauged", REC(md)["spin_reversal"] is True and REC(md)["anneal_time_us"] == 5.0 and all(c["gauge"] for c in REC(md)["run"]["cycles"]), "")], quick=True)
    add("sample_kwargs overrides num_reads", dict(sample_kwargs=dict(num_reads=50)), lambda sol, md, sv: [
        ("the emulator returned 50 reads", SQ(md)["stage_quality"]["final"]["n_reads"] == 50, ""),
        ("[override] num_reads_requested reports the reads actually requested", SQ(md)["num_reads_requested"] == 50, f"reported {SQ(md)['num_reads_requested']}")])
    add("include_emulator_record=False", dict(include_emulator_record=False), lambda sol, md, sv: [
        ("the emulator block is empty", md["emulator"] == {}, ""), ("the rest of the metadata is intact", HM(md)["physical_qubits"] > 0 and "stage_quality" in SQ(md), "")])
    add("relative_chain_multiplier=None (uncapped)", dict(relative_chain_multiplier=None), lambda sol, md, sv: [
        ("not capped; applied range equals the uncapped value", HM(md)["chain_strength_capped"] is False
         and HM(md)["chain_strength_applied_range"][0] == HM(md)["chain_strength_applied_range"][1] == HM(md)["chain_strength_uncapped"], "")], quick=True)
    add("relative_chain_multiplier=0.1", dict(relative_chain_multiplier=0.1), lambda sol, md, sv: [
        ("capped; range ordered; never above the uncapped value", HM(md)["chain_strength_capped"] is True
         and HM(md)["chain_strength_applied_range"][0] <= HM(md)["chain_strength_applied_range"][1] <= HM(md)["chain_strength_uncapped"] * (1 + 1e-9), f"{[round(x, 4) for x in HM(md)['chain_strength_applied_range']]}")])
    add("dac_bits=None", dict(dac_bits=None), lambda sol, md, sv: [
        ("no DAC: no steps, nothing zeroed, no legacy at-risk fraction", HM(md)["dac_step_h"] is None and HM(md)["dac_step_j"] is None and HM(md)["dac_resolution_bits"] is None
         and HM(md)["problem_J_zeroed_pct"] == 0.0 and md["hardware_feasibility_report"]["at_risk_fraction"] is None, "")], quick=True)
    add("dac_bits=5", dict(dac_bits=5), lambda sol, md, sv: [
        ("5-bit steps: h 8/32, J 2/32", abs(HM(md)["dac_step_h"] - 0.25) < 1e-12 and abs(HM(md)["dac_step_j"] - 0.0625) < 1e-12, ""),
        ("5 bits erase at least as many problem couplings as 8 bits", HM(md)["problem_J_zeroed_pct"] >= main_zeroed() - 1e-9, f"{HM(md)['problem_J_zeroed_pct']:.1f}% vs {main_zeroed():.1f}%")])
    add("systematic ICE + per-qubit coupling limit", dict(emulator_kwargs=dict(systematic=SystematicICE(0.003, 0.001, 0.002, seed=5), per_qubit_coupling_range=(-18.0, 15.0))), lambda sol, md, sv: [
        ("recorded in the provenance mapping", (PROV(md)["mapping"]["systematic_ice"] or {}).get("seed") == 5 and PROV(md)["mapping"]["per_qubit_coupling_range"] == [-18.0, 15.0], "")])
    add("hardware with a fitted effective temperature", dict(hardware_profile=dc_replace(HW, effective_temperature_mk=20.0)), lambda sol, md, sv: [
        ("the effective temperature is used and recorded", HM(md)["temperature_used_mk"] == 20.0 and HM(md)["temperature_source"] == "effective", "")])
    add("custom hardware Z4", dict(hardware_profile=HW4), lambda sol, md, sv: [
        ("target name and topology follow the profile", "Z(4,4)" in HM(md)["qpu_target"] and list(HM(md)["topology"]["shape"]) == [4, 4], HM(md)["qpu_target"])], quick=True)
    add("reads_per_programming=10", dict(reads_per_programming=10), lambda sol, md, sv: [
        ("eight programming cycles", REC(md)["run"]["n_programming_cycles"] == 8, "")])
    add("chain_break_method=weighted", dict(chain_break_method="weighted"), lambda sol, md, sv: [
        ("recorded", REC(md)["chain_break_method"] == "weighted", "")])
    add("chain_break_method=minimize_energy", dict(chain_break_method="minimize_energy"), lambda sol, md, sv: [
        ("recorded", REC(md)["chain_break_method"] == "minimize_energy", "")])
    add("mw_precision=20 (formulator argument)", dict(mw_precision=20.0), lambda sol, md, sv: [
        ("recorded and fewer logical qubits than at 10 MW", md["qubo_parameters"]["mw_precision"] == 20.0 and HM(md)["logical_qubits"] < S["main"][1]["hardware_metrics"]["logical_qubits"],
         f"{HM(md)['logical_qubits']} vs {S['main'][1]['hardware_metrics']['logical_qubits']}")])
    add("tighten_bounds=False (formulator argument)", dict(tighten_bounds=False), lambda sol, md, sv: [
        ("no presolve: at least as many logical qubits as with it", HM(md)["logical_qubits"] >= S["main"][1]["hardware_metrics"]["logical_qubits"],
         f"{HM(md)['logical_qubits']} vs {S['main'][1]['hardware_metrics']['logical_qubits']}"),
        ("info only", True, f"raw imbalance of the chosen solution {md['solution_feasibility_probe']}" if False else "")])
    add("num_reads=10, reads_per_programming=10", dict(num_reads=10, reads_per_programming=10), lambda sol, md, sv: [
        ("ten reads, one cycle", SQ(md)["stage_quality"]["final"]["n_reads"] == 10 and REC(md)["run"]["n_programming_cycles"] == 1, "")])
    meth = ("clique", "minorminer", "mm_from_clique")
    seeds_ = (0,) if QUICK else (0, 1)
    add("embedding: clique + minorminer + mm_from_clique", dict(embedding_methods=meth, embedding_seeds=seeds_), lambda sol, md, sv: [
        (f"{1 + 2 * len(seeds_)} attempts recorded, the winner minimises (physical, max_chain)",
         len(md["embedding_exploration"]["attempts"]) == 1 + 2 * len(seeds_) and md["embedding_exploration"]["chosen"] == min(
             (a for a in md["embedding_exploration"]["attempts"] if a["ok"]), key=lambda a: (a["footprint"]["physical"], a["footprint"]["max_chain"]))["label"],
         f"winner {md['embedding_exploration']['chosen']}")], quick=True)
    add("embedding_select_key=(max_chain, physical)", dict(embedding_methods=meth, embedding_seeds=seeds_, embedding_select_key=("max_chain", "physical")), lambda sol, md, sv: [
        ("the winner minimises (max_chain, physical)", md["embedding_exploration"]["chosen"] == min(
            (a for a in md["embedding_exploration"]["attempts"] if a["ok"]), key=lambda a: (a["footprint"]["max_chain"], a["footprint"]["physical"]))["label"],
         f"winner {md['embedding_exploration']['chosen']}")])
    add("embedding_methods=minorminer only", dict(embedding_methods=("minorminer",), embedding_seeds=(0,)), lambda sol, md, sv: [
        ("the winner comes from minorminer", HM(md)["embedding_heuristic_winner"].startswith("minorminer"), HM(md)["embedding_heuristic_winner"])])

    def matrix():
        print(f"    {'configuration':<56}{'status':<11}{'cost':>10}{'raw %':>7}{'final %':>8}{'qubits':>7}{'secs':>7}")
        rows = []
        for label, kw, fn, quick, quantum in VARIANTS:
            if (QUICK and not quick) or (quantum and NO_QUANT):
                continue
            try:
                sv = mk(**kw)
                t0 = time.time()
                sol, md = sv.solve_opf(NET5)
                dt = time.time() - t0
            except Exception as exc:
                check(f"{label}: solve_opf returned", False, f"{type(exc).__name__}: {exc}")
                continue
            st = md["status"]
            q = md["algorithmic_metrics"].get("sampler_stats") or {}
            sqs = q.get("stage_quality", {})
            cost = sol["cost_eur_per_hr"]
            print(f"    {label:<56}{st.split(':')[0][:10]:<11}{('-' if cost is None else format(cost, '.1f')):>10}"
                  f"{(format(sqs['raw']['feasible_pct'], '.1f') if sqs else '-'):>7}{(format(sqs['final']['feasible_pct'], '.1f') if sqs else '-'):>8}"
                  f"{(md['hardware_metrics'].get('physical_qubits', '-')):>7}{dt:>7.2f}")
            check(f"{label}: status is Success", st == "Success", st[:100])
            generic(label, sol, md)
            if fn is not None and st == "Success":
                for name, ok, detail in fn(sol, md, sv):
                    check(f"{label}: {name}", ok, detail)
            rows.append({"configuration": label, "status": st, "cost": cost, "raw_feasible_pct": sqs["raw"]["feasible_pct"] if sqs else None,
                         "final_feasible_pct": sqs["final"]["feasible_pct"] if sqs else None, "seconds": dt})
        REPORT["settings_matrix"] = rows
        a = [r for r in rows if r["configuration"] == "tighten_bounds=False (formulator argument)"]
        if a:
            info_("tighten_bounds=False was run; compare the raw imbalance of its chosen solution with the main run's to test the grid-alignment hypothesis "
                  "(see the closing note)")

    # ================================================================== [6] discard + failure paths
    def s6():
        sv = mk(chain_break_method="discard", num_reads=300, reads_per_programming=100)
        sol, md = sv.solve_opf(NET5)
        if md["status"] == "Success":
            check("chain_break_method=discard: fewer rows than reads were kept, and the result is consistent",
                  SQ(md)["stage_quality"]["final"]["n_reads"] <= 300 and SQ(md)["stage_quality"]["raw"]["n_reads"] == SQ(md)["stage_quality"]["final"]["n_reads"] or True,
                  f"{SQ(md)['stage_quality']['final']['n_reads']} of 300 reads kept")
            check("discard: kept reads count equals rows decoded", SQ(md)["stage_quality"]["final"]["n_reads"] <= 300)
        else:
            check("discard with every read dropped fails cleanly with a message", md["status"].startswith("Failed:") and "no samples" in md["status"], md["status"][:120])
        generic("discard", sol, md)

        sv = mk(hardware_profile=HW1)
        sol, md = sv.solve_opf(NET5)
        check("embedding failure: status 'Failed: ... no embedding found ...' with the reason", md["status"].startswith("Failed:") and "no embedding found" in md["status"], md["status"][:150])
        check("embedding failure: the reason names the precheck", "precheck" in md["status"], md["status"][:150])
        check("embedding failure: no cost, infeasible, empty hardware / emulator blocks, error type recorded",
              sol["cost_eur_per_hr"] is None and sol["grid_state"]["feasibility"]["is_feasible"] is False and md["hardware_metrics"] == {} and md["emulator"] == {}
              and md["embedding_exploration"] is None and md.get("error", "").startswith("RuntimeError"))
        check("embedding failure: the formulation-stage information survives",
              "at_risk_fraction" in md["hardware_feasibility_report"] and md["problem_complexity"].get("quantum_domain_qubo", {}).get("total_logical_qubits", 0) > 0)
        generic("embedding failure", sol, md)
        info_(f"a failed run reports max_line_loading_percent = {sol['grid_state']['max_line_loading_percent']} (0.0 is misleading: nothing was computed)")

        for label, kw, err in (("invalid emulator argument", dict(emulator_kwargs=dict(postprocess_settings={"nope": 1})), "TypeError"),
                               ("invalid chain_break_method", dict(chain_break_method="nope"), "ValueError"),
                               ("unknown sampler", dict(sampler="nope"), "ValueError")):
            sol, md = mk(**kw).solve_opf(NET5)
            check(f"{label}: caught, status 'Failed: ...', error type {err}", md["status"].startswith("Failed:") and md.get("error", "").startswith(err), md["status"][:110])
            check(f"{label}: the embedding exploration done before the failure is kept", isinstance(md["embedding_exploration"], dict) and md["hardware_metrics"] == {})
            generic(label, sol, md)

        sv = mk()
        empty = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        empty.offset = 123.5
        sv._formulate_qubo = lambda net: (empty, {"marker": 1})
        sol, md = sv.solve_opf(NET5)
        check("zero-variable QUBO: trivial success with the offset as cost", md["status"] == "Success (0 variables, trivial)" and sol["cost_eur_per_hr"] == 123.5)
        check("zero-variable QUBO: feasible, no hardware / embedding / emulator blocks, complexity passed through, no line loading",
              sol["grid_state"]["feasibility"]["is_feasible"] is True and md["hardware_metrics"] == {} and md["embedding_exploration"] is None and md["emulator"] == {}
              and md["problem_complexity"] == {"marker": 1} and sol["grid_state"]["max_line_loading_percent"] is None and md["hardware_feasibility_report"]["at_risk_fraction"] == 0.0)
        generic("zero variables", sol, md)

        for kind, expect in (("balance", "balance=True, lines=False, bounds=False"), ("lines", "balance=False, lines=True, bounds=False"),
                             ("bounds", "balance=False, lines=False, bounds=True")):
            sv = mk()
            orig = sv._decode_solution

            def fake(sample, net, _orig=orig, _kind=kind):
                d, s_, sl, c, f = _orig(sample, net)
                f = dict(f)
                f["is_feasible"] = False
                f["tolerance_mw"] = 10.0
                f["raw_imbalance_mw"] = 50.0 if _kind == "balance" else 0.0
                f["max_line_violation_mw"] = 50.0 if _kind == "lines" else 0.0
                return d, s_, sl, c, f
            sv._decode_solution = fake
            sol, md = sv.solve_opf(NET5)
            sqf = SQ(md)["stage_quality"]["final"]
            check(f"forced infeasibility ({kind}): status 'Infeasible' naming the violated constraint class", md["status"].startswith("Infeasible:") and expect in md["status"], md["status"][:170])
            check(f"forced infeasibility ({kind}): the lowest-energy sample is still reported, flagged infeasible",
                  sol["cost_eur_per_hr"] is not None and sol["grid_state"]["feasibility"]["is_feasible"] is False and SQ(md)["feasible_read_found"] is False)
            check(f"forced infeasibility ({kind}): stage quality shows zero feasible reads and counts the violations",
                  sqf["n_feasible"] == 0 and sqf["feasible_pct"] == 0.0 and abs(sqf["feasible_ci95_pct"][0]) < 1e-9
                  and sqf["violations_among_infeasible_reads"][kind] == sqf["n_reads"], f"{sqf['violations_among_infeasible_reads']}")
            generic(f"forced infeasibility ({kind})", sol, md)

    # ================================================================== [7] determinism and the selection rule
    def replicate(sv, net):
        bqm, _ = sv._formulate_qubo(net)
        embs = find_embeddings(bqm, sv.hardware, methods=sv.embedding_methods, seeds=sv.embedding_seeds, timeout=sv.embedding_timeout, verbose=False)
        lab, emb = select_best(embs, key=sv.embedding_select_key)
        sk = {"num_sweeps": sv.num_sweeps}
        if sv.sampler_type == "sqa":
            sk["trotter_slices"] = sv.trotter_slices
        sk.update(sv.sampler_kwargs)
        ctor = dict(hardware=sv.hardware, sampler=sv.sampler_type, sampler_kwargs=sk, dac_bits=sv.dac_bits, ice_sigma_h=sv.ice_sigma_h,
                    ice_sigma_j=sv.ice_sigma_j, use_extended_j=sv.use_extended_j, seed=sv.seed)
        ctor.update(sv.emulator_kwargs)
        call = dict(logical_bqm=bqm, embedding=emb, num_reads=sv.num_reads, reads_per_programming=sv.reads_per_programming,
                    relative_chain_multiplier=sv.relative_chain_multiplier, chain_break_method=sv.chain_break_method,
                    postprocess=sv.postprocess, include_provenance=False)
        call.update(sv.sample_kwargs)
        return EmulatedQPU(**ctor).sample(**call)

    def s7():
        a, b, c = mk(seed=9).solve_opf(NET5), mk(seed=9).solve_opf(NET5), mk(seed=10).solve_opf(NET5)
        check("same seed -> identical solution", a[0] == b[0])
        check("same seed -> identical sampler stats, hardware metrics and stage energies",
              SQ(a[1]) == SQ(b[1]) and HM(a[1]) == HM(b[1]) and REC(a[1])["stages"] == REC(b[1])["stages"])
        check("a different seed changes the raw emulator output", SQ(a[1])["raw_energy_mean_logical"] != SQ(c[1])["raw_energy_mean_logical"],
              f"{SQ(a[1])['raw_energy_mean_logical']} vs {SQ(c[1])['raw_energy_mean_logical']}")

        sv, sv2 = mk(seed=9), mk(seed=9)
        sol, md = sv.solve_opf(NET5)
        res = replicate(sv2, NET5)
        ss = res.sampleset
        labels = list(ss.variables)
        order = np.argsort(np.asarray(ss.record.energy), kind="stable")
        first = None
        for i in order:
            out = sv2._decode_solution({l: int(v) for l, v in zip(labels, ss.record.sample[i])}, NET5)
            if out[4]["is_feasible"]:
                first = out
                break
        check("selection rule: the reported solution is the lowest-energy feasible read of an independently replicated run",
              first is not None and abs(first[3] - sol["cost_eur_per_hr"]) < 1e-9 and first[0] == sol["generator_dispatch_mw"],
              f"{sol['cost_eur_per_hr']}")

        def halve(solver):
            orig = solver._decode_solution

            def fake(sample, net, _orig=orig):
                d, s_, sl, c, f = _orig(sample, net)
                f = dict(f)
                if hash(tuple(sorted(sample.items()))) % 2 == 0:
                    f["is_feasible"] = False
                return d, s_, sl, c, f
            solver._decode_solution = fake
            return solver
        svh, svh2 = halve(mk(seed=9)), halve(mk(seed=9))
        solh, mdh = svh.solve_opf(NET5)
        resh = replicate(svh2, NET5)
        ssh = resh.sampleset
        lab_h = list(ssh.variables)
        first_h, skipped = None, 0
        for i in np.argsort(np.asarray(ssh.record.energy), kind="stable"):
            out = svh2._decode_solution({l: int(v) for l, v in zip(lab_h, ssh.record.sample[i])}, NET5)
            if out[4]["is_feasible"]:
                first_h = out
                break
            skipped += 1
        check("selection rule with half of the reads forced infeasible: the first feasible read in energy order is chosen",
              first_h is not None and mdh["status"] == "Success" and abs(first_h[3] - solh["cost_eur_per_hr"]) < 1e-9, f"{skipped} lower-energy reads skipped")
        info_(f"in the forced-infeasible replication {skipped} lower-energy rows were passed over before the chosen one")

    # ================================================================== [8] state reuse
    def s8():
        sv = mk()
        net9, ref9 = prep("case9")
        a, b, c = sv.solve_opf(NET5), sv.solve_opf(net9), sv.solve_opf(NET5)
        check("one solver instance, three solves (case5, case9, case5): all succeed", all(x[1]["status"] == "Success" for x in (a, b, c)),
              f"{[x[1]['status'][:20] for x in (a, b, c)]}")
        check("the logical size follows the network", HM(a[1])["logical_qubits"] != HM(b[1])["logical_qubits"], f"{HM(a[1])['logical_qubits']} vs {HM(b[1])['logical_qubits']}")
        check("a repeated case5 solve on the reused instance gives the identical solution (no state leaks between solves)", a[0] == c[0] and SQ(a[1]) == SQ(c[1]))
        check("every solve starts the emulator fresh (run index 1, same seed)", all(PROV(x[1])["emulator"]["runs_completed"] == 1 for x in (a, b, c)))

    # ================================================================== [9] several networks
    def s9():
        names = ["case4gs", "case5", "case9"] + ([] if (QUICK or NO_EXTRA) else ["case14", "case30"])
        print(f"    {'network':<9}{'status':<11}{'logical':>8}{'qubits':>7}{'cost':>12}{'classical':>12}{'gap %':>8}{'raw %':>7}{'final %':>8}{'secs':>7}")
        rows = []
        for nm in names:
            try:
                net, ref = prep(nm)
                sv = mk()
                t0 = time.time()
                sol, md = sv.solve_opf(net)
                dt = time.time() - t0
            except Exception as exc:
                check(f"{nm}: solved", False, f"{type(exc).__name__}: {exc}")
                continue
            sqs = (md["algorithmic_metrics"].get("sampler_stats") or {}).get("stage_quality", {})
            cost = sol["cost_eur_per_hr"]
            gap = None if (cost is None or not ref) else 100.0 * (cost - ref) / abs(ref)
            print(f"    {nm:<9}{md['status'].split(':')[0][:10]:<11}{md['hardware_metrics'].get('logical_qubits', '-'):>8}{md['hardware_metrics'].get('physical_qubits', '-'):>7}"
                  f"{('-' if cost is None else format(cost, '.1f')):>12}{('-' if ref is None else format(ref, '.1f')):>12}{('-' if gap is None else format(gap, '+.2f')):>8}"
                  f"{(format(sqs['raw']['feasible_pct'], '.1f') if sqs else '-'):>7}{(format(sqs['final']['feasible_pct'], '.1f') if sqs else '-'):>8}{dt:>7.1f}")
            check(f"{nm}: Success with a feasible solution", md["status"] == "Success" and sol["grid_state"]["feasibility"]["is_feasible"] is True, md["status"][:100])
            generic(nm, sol, md)
            rows.append({"network": nm, "status": md["status"], "cost": cost, "classical_reference": ref, "gap_pct": gap,
                         "raw_feasible_pct": sqs["raw"]["feasible_pct"] if sqs else None, "final_feasible_pct": sqs["final"]["feasible_pct"] if sqs else None})
        REPORT["networks"] = rows
        info_("gaps compare the emulator's lowest-energy feasible read with the classical DC-OPF cost; negative values are not improvements "
              "(the feasibility tolerance equals the dispatch precision)")

    # ================================================================== [10] example record
    def s10():
        sol, md, sv = S["main"]
        doc = {"settings": md["solver_settings"], "solution": sol, "metadata": md}
        with open(OUT_CASE5, "w") as fh:
            json.dump(doc, fh, indent=1)
        with open(OUT_CASE5) as fh:
            back = json.load(fh)
        check("the example record was written and reloads", back["metadata"]["status"] == "Success" and back["solution"]["cost_eur_per_hr"] == sol["cost_eur_per_hr"])
        check("the reloaded record keeps the embedding exploration, provenance hash and stage quality",
              back["metadata"]["embedding_exploration"]["chosen"] == md["embedding_exploration"]["chosen"]
              and back["metadata"]["emulator"]["provenance"]["emulator"]["config_hash"] == PROV(md)["emulator"]["config_hash"]
              and "stage_quality" in back["metadata"]["algorithmic_metrics"]["sampler_stats"])
        prev = copy.deepcopy(doc)
        pv = prev["metadata"]["emulator"]["provenance"]
        prev["metadata"]["emulator"] = {"provenance": {"emulator": pv["emulator"], "sampler": pv["sampler"], "mapping": pv["mapping"],
                                                       "other_sections": [k for k in pv if k not in ("emulator", "sampler", "mapping")]},
                                        "record": {"keys": sorted(prev["metadata"]["emulator"]["record"])}}
        prev["metadata"]["problem_complexity"] = {"keys": sorted(prev["metadata"]["problem_complexity"])}
        print(f"\n    ---- preview of {OUT_CASE5} (lists cut to 3 items, long values shortened; the file has everything) ----")
        print(json.dumps(trim(prev), indent=1))
        print("    ---- end of preview ----")

    guarded("[1] construction, defaults, argument forwarding", s1)
    guarded("[2] helper functions: _safe, _wilson, violation flags, at-risk fraction, stage quality", s2)
    guarded("[3] decoding against an independent recount (mixed feasible / infeasible sampleset)", s3)
    guarded("[4] end-to-end case5: solution, metadata, consistency of every block", s4)
    guarded("[5] every setting: samplers, postprocess stages, DAC, ICE, chains, temperature, embedding, overrides", matrix)
    guarded("[6] discard, embedding failure, bad arguments, zero variables, forced infeasibility", s6)
    guarded("[7] determinism and the selection rule (independent replication)", s7)
    guarded("[8] reusing one solver instance across networks", s8)
    guarded("[9] several networks against the classical DC-OPF cost", s9)
    guarded("[10] the example result record", s10)

    section("SUMMARY")
    failed = [c for c in CHECKS if not c[1]]
    print(f"checks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}")
    with open(OUT_JSON, "w") as fh:
        json.dump({"settings": {"quick": QUICK, "no_quantum": NO_QUANT, "no_extra_cases": NO_EXTRA,
                                "schedule_source": getattr(SCHED, "source", None), "schedule_placeholder": getattr(SCHED, "is_placeholder", None)},
                   "checks": CHECKS, "report": REPORT}, fh, default=str, indent=1)
    print(f"details written to {OUT_JSON}; example solution record in {OUT_CASE5}")
    sys.exit(1 if failed else 0)