###############################################################################################################
#                                           SETUP
###############################################################################################################
# Library imports
import copy
import datetime
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

# Module imports
from homemade_grids.small_grids import case3_low_gen, case3_high_gen, case4_low_gen, case4_high_gen, extract_network_info

from solvers.qubo_formulator import QuboFormulator

from solvers.classical_ip_solver import ClassicalIPSolver
from solvers.simulated_quantum_annealing.emulated_qpu_solver import EmulatedQPUSolver

# Pipeline layout (one pass of run_pipeline):
#   1. Classical IP            continuous optimum (ground truth for the real-valued problem)
#   2. Discrete exact          exact optimum of the discretised problem (the best the encoding allows) [PLACEHOLDER]
#   3. Logical SA              simulated annealing on the discretised QUBO, no hardware effects     [PLACEHOLDER] 
#   4. Emulated QPU            hardware-aware emulation: embedding, scaling, DAC, noise, sampler, chains, decoding
# Comparing 1 -> 3 isolates the discretisation error, 3 -> 2 the sampler's search error, 2 -> 4 the hardware effects.


###############################################################################################################
#                                           Helpers: files
###############################################################################################################

class NumpyEncoder(json.JSONEncoder):
    """Safely converts numpy types to native Python types for JSON serialization. Unknown objects are stored as
    their repr instead of crashing the save (a crash at save time would lose a whole run)."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        if isinstance(obj, (set, frozenset)): return sorted(obj, key=str)
        return repr(obj)


def save_payload(data: dict, filepath: str):
    """Writes through a temporary file so that an interrupted save never leaves a half-written payload behind."""
    save_dir = os.path.dirname(filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    os.replace(tmp, filepath)


def load_payload(filepath: str) -> dict:
    """Loads an existing JSON payload from disk."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


###############################################################################################################
#                                           Helpers: configuration signatures and cache
###############################################################################################################
# A payload is only reused when it was produced by the SAME configuration (network, formulation, solver settings).
# Each saved payload carries a "run_config" block: {"signature_hash", "signature", "created_utc"}.

def _canon(o, _depth=0):
    """Canonical JSON-able form of a configuration value (used for hashing, never for results)."""
    if _depth > 6:
        return repr(o)[:120]
    if isinstance(o, dict):
        return {str(k): _canon(v, _depth + 1) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_canon(v, _depth + 1) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted((_canon(v, _depth + 1) for v in o), key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(o, np.ndarray):
        return {"ndarray": list(o.shape), "sha1": hashlib.sha1(np.ascontiguousarray(o).tobytes()).hexdigest()[:12]}
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if o is None or isinstance(o, (str, int, float)):
        return o
    prov = getattr(o, "provenance", None)             # Hardware, Schedule, ... describe themselves
    if callable(prov):
        try:
            return {"__provenance__": type(o).__name__, "data": _canon(prov(), _depth + 1)}
        except Exception:
            pass
    if callable(o):
        return {"__callable__": getattr(o, "__qualname__", repr(o))}
    if hasattr(o, "__dict__"):
        return {"__type__": type(o).__name__,
                **{k: _canon(v, _depth + 1) for k, v in vars(o).items() if not k.startswith("_")}}
    return repr(o)[:120]


def config_hash(signature: dict) -> str:
    return hashlib.sha1(json.dumps(_canon(signature), sort_keys=True, default=str).encode()).hexdigest()[:12]


def network_fingerprint(net) -> str:
    """Short hash of everything in a pandapower network that defines the OPF problem."""
    try:
        parts = {}
        for name in ("bus", "line", "trafo", "load", "gen", "sgen", "ext_grid", "shunt", "switch", "poly_cost", "pwl_cost"):
            tbl = getattr(net, name, None)
            if tbl is not None and len(tbl):
                parts[name] = tbl.to_json(orient="split", double_precision=9)
        parts["sn_mva"] = float(getattr(net, "sn_mva", 0.0))
        parts["f_hz"] = float(getattr(net, "f_hz", 0.0))
        return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]
    except Exception:
        return "fingerprint-unavailable"


def _try_cache(filename: str, signature_hash: str, policy: str, label: str):
    """Returns a reusable payload or None.
    policy "match": reuse only if the stored signature equals the current one (legacy payloads without a signature are re-run)
    policy "any"  : reuse whatever file exists (the behaviour of the earlier version of this script)
    policy "off"  : never reuse
    Payloads whose status is a failure or a placeholder are never reused."""
    if policy == "off" or not os.path.exists(filename):
        return None
    try:
        payload = load_payload(filename)
    except Exception as exc:
        print(f"[Cache Miss] {label}: {filename} could not be read ({type(exc).__name__}); re-running")
        return None
    status = str(((payload.get("opf_results") or {}).get("metadata") or {}).get("status", ""))
    if status.startswith("Failed") or status.startswith("Not implemented"):
        print(f"[Cache Miss] {label}: stored status is '{status[:50]}'; re-running")
        return None
    if policy == "any":
        print(f"[Cache Hit] {label}: loading existing payload (cache='any'): {filename}")
        return payload
    stored = (payload.get("run_config") or {}).get("signature_hash")
    if stored == signature_hash:
        print(f"[Cache Hit] {label}: configuration unchanged ({signature_hash}); loading {filename}")
        return payload
    reason = "legacy payload without a run_config block" if stored is None else f"configuration changed ({stored} -> {signature_hash})"
    print(f"[Cache Miss] {label}: {reason}; re-running and overwriting {filename}")
    return None


def _placeholder_payload(problem_parameters: dict, label: str, note: str) -> dict:
    """In-memory stand-in for a solver that does not exist yet. Same layout as a real payload, never saved to disk
    (so it can never be mistaken for a cached result)."""
    return {
        "problem_parameters": problem_parameters,
        "opf_results": {
            "solution": {
                "cost_eur_per_hr": None, "generator_dispatch_mw": {}, "static_generator_dispatch_mw": {},
                "slack_dispatch_mw": {},
                "grid_state": {"max_line_loading_percent": None, "max_voltage_pu": None, "min_voltage_pu": None,
                               "feasibility": {"is_feasible": None, "line_flows_mw": {}}}},
            "metadata": {"status": "Not implemented (placeholder)", "solver_name": label,
                         "execution_time_seconds": {"total": 0.0}, "note": note}}}


def _failed_payload(problem_parameters: dict, label: str, exc: Exception) -> dict:
    """In-memory record of a step that raised. Not saved, so a later run retries it."""
    return {
        "problem_parameters": problem_parameters,
        "opf_results": {
            "solution": {
                "cost_eur_per_hr": None, "generator_dispatch_mw": {}, "static_generator_dispatch_mw": {},
                "slack_dispatch_mw": {},
                "grid_state": {"max_line_loading_percent": None, "max_voltage_pu": None, "min_voltage_pu": None,
                               "feasibility": {"is_feasible": False, "line_flows_mw": {}}}},
            "metadata": {"status": f"Failed: {exc}", "solver_name": label, "error": f"{type(exc).__name__}: {exc}",
                         "execution_time_seconds": {"total": 0.0}}}}


def _run_step(label, filename, signature, problem_parameters, solve_fn, cache):
    """One pipeline step: cache check -> solve -> save. solve_fn() returns (solution, metadata)."""
    sig_hash = config_hash(signature)
    cached = _try_cache(filename, sig_hash, cache, label)
    if cached is not None:
        return cached
    t0 = time.time()
    try:
        solution, metadata = solve_fn()
    except NotImplementedError as exc:
        print(f"[Placeholder] {label}: not implemented yet ({exc})")
        return _placeholder_payload(problem_parameters, label, str(exc))
    except Exception as exc:
        print(f"[Error] {label} raised {type(exc).__name__}: {exc}  (nothing saved; the next run will retry this step)")
        return _failed_payload(problem_parameters, label, exc)
    payload = {
        "problem_parameters": problem_parameters,
        "opf_results": {"solution": solution, "metadata": metadata},
        "run_config": {"signature_hash": sig_hash, "signature": _canon(signature),
                       "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}}
    save_payload(payload, filename)
    print(f"{label} time: {round(time.time() - t0, 4)} seconds  -> {filename}")
    return payload


###############################################################################################################
#                                           Solver plug-ins
###############################################################################################################

def solve_discrete_exact(net, formulator_config):
    """PLACEHOLDER: exact optimum of the DISCRETISED problem (the best solution the chosen encoding and precision allow).
    It is the reference against which the logical SA and the emulator should be judged: the gap between it and the
    continuous classical optimum is pure discretisation error."""
    raise NotImplementedError("discrete exact solver not written yet")


def solve_logical_sa(net, formulator_config, num_reads, num_sweeps):
    """PLACEHOLDER: simulated annealing on the discretised QUBO (no embedding, no hardware effects).
    Replace the body with the call into the new module once it exists. The earlier version of this script ran
        SimulatedAnnealingSolver(**formulator_config, num_reads=num_reads, num_sweeps=num_sweeps).solve_opf(net)
    and saved the result as sa_<formulation>_enc_<encoding>_prec_<precision>.json."""
    raise NotImplementedError("logical SA solver module not written yet")

###############################################################################################################
#                                           Summary tables
###############################################################################################################

def _opf(payload):
    return (payload or {}).get("opf_results") or {}


def _solution(payload):
    return _opf(payload).get("solution") or {}


def _metadata(payload):
    return _opf(payload).get("metadata") or {}


def _cost(payload):
    return _solution(payload).get("cost_eur_per_hr")


def _status(payload):
    return str(_metadata(payload).get("status", "Failed"))


def _usable(payload):
    """True when the payload holds a cost that is worth comparing (not a failure, not infeasible, not a placeholder)."""
    s = _status(payload)
    return _cost(payload) is not None and not s.startswith(("Failed", "Infeasible", "Not implemented"))


def _gap(cost, ref):
    if cost is None or ref is None or ref == 0:
        return None
    return 100.0 * (cost - ref) / abs(ref)


def _fmt(v, spec, suffix=""):
    return f"{format(v, spec)}{suffix}" if v is not None else "N/A"


def print_execution_summary(payloads: dict, config: dict = None):
    """Parses saved payloads and prints pivot tables of key OPF metrics and hardware complexity."""
    print("\n" + "="*110)
    print(" PIPELINE EXECUTION SUMMARY ".center(110, "="))
    print("="*110)

    ref_ip = next((_cost(p) for k, p in payloads.items() if k.startswith("Classical IP") and _usable(p)), None)
    ref_ex = next((_cost(p) for k, p in payloads.items() if k.startswith("Discrete Exact") and _usable(p)), None)

    rows_perf = []
    rows_res = []

    for solver_name, payload in payloads.items():
        opf_sol = _solution(payload)
        opf_meta = _metadata(payload)

        # --- 1. Performance & Economics Table ---
        opf_cost = opf_sol.get("cost_eur_per_hr")

        slack_mw = None
        if opf_sol.get("slack_dispatch_mw"):
            slack_mw = sum(d.get("p_mw", 0.0) for d in opf_sol["slack_dispatch_mw"].values())

        pf_loading = None
        if opf_sol.get("grid_state"):
            pf_loading = opf_sol["grid_state"].get("max_line_loading_percent")

        feasible = ((opf_sol.get("grid_state") or {}).get("feasibility") or {}).get("is_feasible")
        exec_time = round((opf_meta.get("execution_time_seconds") or {}).get("total", 0.0), 4)
        status = _status(payload)
        usable = _usable(payload)
        is_ip = solver_name.startswith("Classical IP")

        rows_perf.append({
            "Solver": solver_name,
            "OPF Status": status if len(status) <= 38 else status[:35] + "...",
            "Feasible": "N/A" if feasible is None else ("yes" if feasible else "no"),
            "Cost (EUR/hr)": f"{opf_cost:.2f}" if opf_cost is not None else "N/A",
            "Gap vs IP (%)": "-" if is_ip else (_fmt(_gap(opf_cost, ref_ip), "+.2f") if usable else "N/A"),
            "Gap vs Exact (%)": "-" if solver_name.startswith("Discrete Exact") else (_fmt(_gap(opf_cost, ref_ex), "+.2f") if usable else "N/A"),
            "Slack (MW)": f"{slack_mw:.2f}" if slack_mw is not None else "N/A",
            "Max Load (%)": f"{pf_loading:.2f}" if pf_loading is not None else "N/A",
            "Time (s)": exec_time
        })

        # --- 2. Hardware & Complexity Table ---
        if not is_ip:
            comp = opf_meta.get("problem_complexity") or {}
            hw = opf_meta.get("hardware_metrics") or {}
            alg = opf_meta.get("algorithmic_metrics") or {}

            feas = opf_meta.get("hardware_feasibility_report") or {}

            form = comp.get("formulation", "N/A")
            enc = comp.get("encoding", "N/A")
            prec = config.get("mw_precision", "N/A") if config else "N/A"

            q_domain = comp.get("quantum_domain_qubo") or {}
            l_qub = q_domain.get("total_logical_qubits", "N/A")
            dens = q_domain.get("density_percent", "N/A")

            hw_limits = comp.get("hardware_limits") or {}
            dr = hw_limits.get("ising_dynamic_range", "N/A")

            p_qub = hw.get("physical_qubits", "N/A")
            max_chain = hw.get("max_chain_length", "N/A")

            sampler_stats = alg.get("sampler_stats") or {}
            cb_frac = sampler_stats.get("mean_chain_break_fraction_raw", sampler_stats.get("mean_chain_break_fraction"))
            cb = f"{cb_frac * 100:.3f}%" if cb_frac is not None else "N/A"

            risk_frac = feas.get("at_risk_fraction")
            risk = f"{risk_frac * 100:.2f}%" if risk_frac is not None else "N/A"

            dens_str = f"{dens}%" if isinstance(dens, (int, float)) else "N/A"
            dr_str = f"{dr:,.1f}" if isinstance(dr, (int, float)) else "N/A"

            stq = sampler_stats.get("stage_quality") or {}
            raw_f = (stq.get("raw") or {}).get("feasible_pct")
            fin_f = (stq.get("final") or {}).get("feasible_pct")

            rows_res.append({
                "Solver": solver_name,
                "Form": form,
                "Enc": enc,
                "Prec": prec,
                "L-Qubits": l_qub,
                "P-Qubits": p_qub,
                "Max-Chain": max_chain,
                "Density": dens_str,
                "Dyn-Range": dr_str,
                "At-Risk": risk,
                "Chain-Break": cb,
                "Raw Feas%": _fmt(raw_f, ".1f"),
                "Final Feas%": _fmt(fin_f, ".1f"),
                "Stage": sampler_stats.get("selected_stage", "N/A"),
            })

    # Print First Table
    df_perf = pd.DataFrame(rows_perf)
    print(df_perf.to_string(index=False))
    print("  (gaps compare cost only: the feasibility tolerance equals the dispatch precision, so small negative gaps are not improvements)")

    # Print Second Table
    print("\n" + "-"*110)
    print(" QUBO COMPLEXITY & HARDWARE METRICS ".center(110, "-"))
    print("-" * 110)
    if rows_res:
        df_res = pd.DataFrame(rows_res)
        print(df_res.to_string(index=False))
    else:
        print("No QUBO solvers executed to report hardware metrics.")
    print("="*110 + "\n")


###############################################################################################################
#                                           Runner
###############################################################################################################

def run_pipeline(formulator_config, pristine_net, num_reads=500, num_sweeps=2000, trotter_slices=16, *,
                 emulator_config=None, cache="match", output_root="output", emulator_tag=None):
    """
    formulator_config : QuboFormulator options (formulation, encoding, mw_precision, ...), shared by every QUBO-based step.
    emulator_config   : extra EmulatedQPUSolver arguments (sampler, postprocess, hardware_profile, dac_bits, sampler_kwargs,
                        emulator_kwargs, sample_kwargs, seed, ...). They override the defaults built from the arguments above
                        (sampler defaults to "sa").
    cache             : "match" (default) reuse a saved payload only if it was produced by the same configuration,
                        "any" reuse whatever file exists (old behaviour), "off" always re-run.
                        The signature covers the network and the settings, NOT the solver code: use cache="off" after
                        changing a solver module.
    output_root       : payloads go to <output_root>/<grid name>/.
    emulator_tag      : appended to the emulator file name so that several emulator configurations can sit side by side.
    Returns the dictionary of payloads (the same ones the summary prints).
    """
    if cache not in ("match", "any", "off"):
        raise ValueError("cache must be 'match', 'any' or 'off'")
    t_pipeline = time.time()

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 1: Initialize the problem and enforce rules
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    original_problem_parameters = extract_network_info(pristine_net)
    grid_name = original_problem_parameters.get("network_name", "unknown_grid").replace(" ", "_").lower()

    output_dir = os.path.join(output_root, grid_name)
    os.makedirs(output_dir, exist_ok=True)

    if not pristine_net.ext_grid.empty:
        pristine_net.ext_grid['min_p_mw'] = 0.0

    net_id = network_fingerprint(pristine_net)       # computed AFTER the ext_grid rule, it is part of the problem

    # Setup naming configurations
    form_type = formulator_config.get("formulation", "dc_ptdf")
    prec = formulator_config.get("mw_precision", 10.0)
    encoding_type = formulator_config.get("encoding", "radix")
    tail = f"{form_type}_enc_{encoding_type}_prec_{prec}"

    print(f"Network '{grid_name}' (fingerprint {net_id}) | formulation {form_type}, encoding {encoding_type}, precision {prec} MW | cache='{cache}'")

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 2: Continuous Ground Truth (Classical IP OPF)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    def _solve_ip():
        return ClassicalIPSolver(formulation="dc").solve_opf(copy.deepcopy(pristine_net))

    ground_truth_payload = _run_step(
        "Classical IP (dc)", os.path.join(output_dir, "classical_ip_dc.json"),
        {"step": "classical_ip", "formulation": "dc", "network": net_id},
        original_problem_parameters, _solve_ip, cache)

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 3: Discrete solvers on the QUBO, without hardware effects (both PLACEHOLDERS for now)
    # ------------------------------------------------------------------------------------------------------------------------------------ #

    # 3a. Discrete exact: the true optimum of the discretised problem
    exact_payload = _run_step(
        f"Discrete Exact ({form_type})", os.path.join(output_dir, f"discrete_exact_{tail}.json"),
        {"step": "discrete_exact", "network": net_id, "formulator_config": formulator_config},
        original_problem_parameters,
        lambda: solve_discrete_exact(copy.deepcopy(pristine_net), formulator_config), cache)
    
    # 3b. Logical SA: search error of a classical annealer on the discretised problem
    sa_payload = _run_step(
        f"Logical SA ({form_type})", os.path.join(output_dir, f"sa_{tail}.json"),
        {"step": "logical_sa", "network": net_id, "formulator_config": formulator_config,
         "num_reads": num_reads, "num_sweeps": num_sweeps},
        original_problem_parameters,
        lambda: solve_logical_sa(copy.deepcopy(pristine_net), formulator_config, num_reads, num_sweeps), cache)

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 4: Hardware-Aware Analog Simulation (Emulated QPU)
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    emu_args = {**formulator_config, "num_reads": num_reads, "num_sweeps": num_sweeps,
                "trotter_slices": trotter_slices, "sampler": "sa"}
    emu_args.update(emulator_config or {})
    sampler_name = emu_args["sampler"]
    suffix = f"_{emulator_tag}" if emulator_tag else ""
    filename_emu = os.path.join(output_dir, f"sqa_{sampler_name}_{tail}{suffix}.json")   # the legacy name is sqa_sa_<...>.json

    def _solve_emulator():
        return EmulatedQPUSolver(**emu_args).solve_opf(copy.deepcopy(pristine_net))

    emu_payload = _run_step(
        f"Emulated QPU ({sampler_name.upper()})", filename_emu,
        {"step": "emulator", "network": net_id, "emulator_args": emu_args},
        original_problem_parameters, _solve_emulator, cache)

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 5: Execution Summary
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    all_payloads = {
        "Classical IP (dc)": ground_truth_payload,
        f"Discrete Exact ({form_type})": exact_payload,
        f"Logical SA ({form_type})": sa_payload,
        f"Emulated QPU ({sampler_name.upper()})": emu_payload
    }

    print_execution_summary(all_payloads, formulator_config)
    print(f"Pipeline time: {round(time.time() - t_pipeline, 4)} seconds")
    return all_payloads

###############################################################################################################
#                               Execution Block
###############################################################################################################

if __name__ == "__main__":
    default_config = {
        "formulation": "dc_ptdf",
        "encoding": "radix",
        "mw_precision": 10.0,
    }

    num_reads = 1000
    num_sweeps = 10000

    # Optional emulator settings (anything EmulatedQPUSolver accepts), for example:
    # emulator_config = dict(sampler="sa", postprocess="logical", dac_bits=8, seed=1)

    run_pipeline(default_config, pn.case5(), num_reads=num_reads, num_sweeps=num_sweeps)