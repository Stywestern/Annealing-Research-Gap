"""Logical simulated annealing solver: the QUBO is sampled directly with SA, with no embedding and no hardware effects.

It is the perfect-logical reference of the pipeline. Its search error (against the discrete exact optimum) is what a
good classical annealer achieves on the discretised problem; whatever the emulated QPU loses relative to it comes from
embedding, chains, limited precision, control errors, temperature and the sampler's dynamics.

The output has the same layout as the emulated-QPU solver so the two can be compared directly: stage quality with a
feasibility rate and its 95% interval, violation counts, and the best costs.

postprocess : None (default) | "logical". "logical" runs greedy steepest descent (Ocean's SteepestDescentSolver) on every
              read, like the emulator's logical stage. The raw SA output is ALWAYS decoded and reported as well, so the
              effect of the descent is visible. For a like-for-like comparison with an emulator run, give both the same
              postprocess setting.
Penalty diagnostics: whether the lowest-QUBO-energy state is feasible, and how many distinct infeasible states lie below
              the best feasible one. These depend only on the formulation (penalty weights), not on any solver.
"""
import datetime
import json
import math
import time
from importlib import metadata as _md

import numpy as np

from solvers.qubo_formulator import QuboFormulator

try:
    from dwave.samplers import SimulatedAnnealingSampler as _SASampler
except ImportError:                                  # older installations
    from neal import SimulatedAnnealingSampler as _SASampler
try:
    from dwave.samplers import SteepestDescentSolver as _SteepestDescent
except ImportError:
    _SteepestDescent = None

_RESERVED = ("num_reads", "num_sweeps", "seed")


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return repr(o)


def _safe(x):
    """JSON-safe copy (numpy scalars converted, unknown objects stored as text)."""
    return json.loads(json.dumps(x, default=_default))


def _wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, in percent."""
    if n == 0:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100.0 * (c - h), 100.0 * (c + h)]


def _version(*names):
    for n in names:
        try:
            return {n: _md.version(n)}
        except _md.PackageNotFoundError:
            continue
    return {}


class SimulatedAnnealingSolver(QuboFormulator):
    def __init__(self, formulation="dc_ptdf", num_reads=500, num_sweeps=3000,
                 mw_precision=10.0, seed=42, max_decode=None, *,
                 postprocess=None, sampler_kwargs=None, **kwargs):
        """
        seed        : integer, or None for a random seed (the seed actually used is always recorded in the metadata).
                      The default was None in earlier versions.
        max_decode  : cap on how many DISTINCT states (lowest energy first) are decoded and feasibility-checked per stage.
                      None decodes all of them. With a cap, the feasibility percentages describe the decoded
                      (lowest-energy) part only; the metadata says so (decode_capped).
        postprocess : None | "logical", see the module docstring.
        sampler_kwargs : extra arguments of the SA sampler, e.g. {"beta_range": (0.1, 10.0), "beta_schedule_type": "linear",
                      "num_sweeps_per_beta": 2}. num_reads, num_sweeps and seed are set by this class.
        """
        super().__init__(formulation=formulation, mw_precision=mw_precision, **kwargs)
        if int(num_reads) < 1 or int(num_sweeps) < 1:
            raise ValueError("num_reads and num_sweeps must be >= 1")
        if postprocess not in (None, "logical"):
            raise ValueError("postprocess must be None or 'logical'")
        if postprocess == "logical" and _SteepestDescent is None:
            raise ValueError("postprocess='logical' needs dwave.samplers.SteepestDescentSolver")
        bad = [k for k in (sampler_kwargs or {}) if k in _RESERVED]
        if bad:
            raise ValueError(f"sampler_kwargs must not contain {bad}; set them through the constructor")
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)
        self.seed = seed
        self.max_decode = max_decode
        self.postprocess = postprocess
        self.sampler_kwargs = dict(sampler_kwargs or {})
        self.sampler = _SASampler()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _violation_flags(feas):
        """True = satisfied. Works with formulator versions that do not report balance_ok / lines_ok / bounds_ok."""
        tol = feas.get("tolerance_mw", 0.0)
        bal = feas.get("balance_ok", abs(feas.get("raw_imbalance_mw", 0.0)) <= tol)
        lin = feas.get("lines_ok", feas.get("max_line_violation_mw", 0.0) <= tol)
        bnd = feas.get("bounds_ok", not (bal and lin and not feas.get("is_feasible", True)))
        return {"balance": bool(bal), "lines": bool(lin), "bounds": bool(bnd)}

    @staticmethod
    def _energy_stats(ss):
        """Energy statistics over the READS of a sampleset (weighted by num_occurrences)."""
        e = np.asarray(ss.record.energy, dtype=float)
        w = np.asarray(ss.record.num_occurrences, dtype=float)
        mean = float(np.average(e, weights=w))
        emin = float(e.min())
        return {"best": emin, "mean": mean, "worst": float(e.max()),
                "std": float(math.sqrt(np.average((e - mean) ** 2, weights=w))),
                "fraction_of_reads_at_lowest_energy": float(w[e <= emin + 1e-9 * max(1.0, abs(emin))].sum() / w.sum())}

    def _decode_set(self, ss, net):
        """Decodes the distinct states of a sampleset, lowest energy first. Counts are weighted by num_occurrences."""
        agg = ss.aggregate()
        rows = list(agg.data(["sample", "energy", "num_occurrences"], sorted_by="energy"))
        n_total = int(sum(int(r.num_occurrences) for r in rows))
        capped = bool(self.max_decode) and len(rows) > int(self.max_decode)
        n_unique_total = len(rows)
        if capped:
            rows = rows[: int(self.max_decode)]
        n_reads = n_feas = n_infeasible_states_before = 0
        best_e = best_c = best_any = None
        viol = {"balance": 0, "lines": 0, "bounds": 0}
        for d in rows:
            occ = int(d.num_occurrences)
            disp, sgen_disp, slack_disp, c, feas = self._decode_solution(dict(d.sample), net)
            row = {"dispatch": disp, "sgen_dispatch": sgen_disp, "slack_dispatch": slack_disp,
                   "cost": c, "feasibility": feas, "qubo_energy": float(d.energy)}
            n_reads += occ
            if best_any is None:
                best_any = row                      # lowest-energy state, feasible or not
            if feas["is_feasible"]:
                n_feas += occ
                if best_e is None:
                    best_e = row                    # first feasible in energy order = lowest-energy feasible
                if best_c is None or c < best_c["cost"]:
                    best_c = row
            else:
                if best_e is None:
                    n_infeasible_states_before += 1
                for k, ok in self._violation_flags(feas).items():
                    if not ok:
                        viol[k] += occ
        return {"n_reads": n_reads, "n_reads_total": n_total, "capped": capped, "n_feasible": n_feas,
                "n_unique_decoded": len(rows), "n_unique_total": n_unique_total,
                "best_feasible_by_energy": best_e, "best_feasible_by_cost": best_c, "best_any": best_any,
                "violations_among_infeasible_reads": viol,
                "n_infeasible_states_below_first_feasible": n_infeasible_states_before}

    @staticmethod
    def _stage_quality(q, es):
        n, k = q["n_reads"], q["n_feasible"]
        be, bc, ba = q["best_feasible_by_energy"], q["best_feasible_by_cost"], q["best_any"]
        return {"n_reads": n, "n_reads_total": q["n_reads_total"], "decode_capped": q["capped"],
                "n_feasible": k, "feasible_pct": (100.0 * k / n) if n else None, "feasible_ci95_pct": _wilson(k, n),
                "n_unique_states_decoded": q["n_unique_decoded"], "n_unique_states_total": q["n_unique_total"],
                "cost_of_lowest_energy_feasible": be["cost"] if be else None,
                "lowest_cost_among_feasible": bc["cost"] if bc else None,
                "violations_among_infeasible_reads": q["violations_among_infeasible_reads"],
                "lowest_energy_state_is_feasible": bool(ba["feasibility"]["is_feasible"]) if ba else None,
                "infeasible_states_below_best_feasible": q["n_infeasible_states_below_first_feasible"],
                "energy": es}

    # ------------------------------------------------------------------ main entry
    def solve_opf(self, net):
        t_start = time.time()
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        status, cost = "Success", None
        dispatch, sgen_dispatch, slack_dispatch = {}, {}, {}
        feasibility = {"is_feasible": False, "line_flows_mw": {}}
        form_time = samp_time = post_time = decode_time = 0.0
        bqm, complexity, sampler_stats, run, error = None, {}, {}, {}, None
        seed_used = None

        try:
            # 1. Formulation (CPU)
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0
            n_vars = len(bqm.variables)

            if n_vars == 0:
                cost = float(bqm.offset)
                feasibility = {"is_feasible": True, "line_flows_mw": {}}
                status = "Success (0 variables, trivial)"
            else:
                # 2. Sampling: plain classical SA on the QUBO
                seed_used = (int(self.seed) if self.seed is not None
                             else int(np.random.SeedSequence().entropy % (2 ** 31 - 1)))
                t1 = time.time()
                response = self.sampler.sample(bqm, num_reads=self.num_reads, num_sweeps=self.num_sweeps,
                                               seed=seed_used, **self.sampler_kwargs)
                samp_time = time.time() - t1
                run["sampler_info"] = _safe(dict(response.info))     # includes the beta range that was actually used

                # 3. Optional greedy descent on every read (the raw reads are kept)
                raw_ss, final_ss = response, response
                if self.postprocess == "logical":
                    t2 = time.time()
                    final_ss = _SteepestDescent().sample(bqm, initial_states=raw_ss, seed=seed_used)
                    post_time = time.time() - t2

                # 4. Decoding: every distinct state of each stage (lowest energy first)
                t3 = time.time()
                es_raw = self._energy_stats(raw_ss)
                raw_q = self._decode_set(raw_ss, net)
                if final_ss is raw_ss:
                    es_fin, final_q = es_raw, raw_q
                else:
                    es_fin = self._energy_stats(final_ss)
                    final_q = self._decode_set(final_ss, net)
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

                stage_raw = self._stage_quality(raw_q, es_raw)
                stage_fin = stage_raw if final_ss is raw_ss else self._stage_quality(final_q, es_fin)
                sampler_stats = {
                    # keys of the earlier version (unique_states_found now counts distinct states, not rows)
                    "num_reads_requested": self.num_reads,
                    "num_sweeps_per_read": self.num_sweeps,
                    "unique_states_found": raw_q["n_unique_total"],
                    "energy_best": round(es_raw["best"], 2),
                    "energy_mean": round(es_raw["mean"], 2),
                    "energy_worst": round(es_raw["worst"], 2),
                    "energy_std_dev": round(es_raw["std"], 2),
                    "reads_checked_for_feasibility": final_q["n_unique_decoded"],
                    "feasible_read_found": final_q["best_feasible_by_energy"] is not None,
                    "chosen_sample_energy": chosen["qubo_energy"],
                    # new
                    "selected_stage": "logical" if final_ss is not raw_ss else "raw",
                    "postprocess": self.postprocess,
                    "fraction_of_raw_reads_at_lowest_energy": es_raw["fraction_of_reads_at_lowest_energy"],
                    "stage_quality": {"raw": stage_raw, "final": stage_fin}}

        except Exception as e:
            status = f"Failed: {e}"
            error = f"{type(e).__name__}: {e}"
            feasibility = {"is_feasible": False, "line_flows_mw": {}}
            cost, dispatch, sgen_dispatch, slack_dispatch = None, {}, {}, {}
            print(f"\nCRITICAL ERROR IN LOGICAL SA SOLVER: {e}\n")

        exec_time = round(time.time() - t_start, 4)

        max_load_pct = None
        if hasattr(self, "_lines") and self._lines and feasibility.get("line_flows_mw"):
            loadings = [abs(feasibility["line_flows_mw"].get(ln["idx"], 0.0)) / ln["p_max"] * 100.0
                        for ln in self._lines if ln["p_max"] > 0.0]
            if loadings:
                max_load_pct = round(max(loadings), 3)

        flat = self.formulation in ("dc_ptdf", "dc_theta")          # DC assumes flat voltages
        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": max_load_pct,
                "max_voltage_pu": 1.0 if flat else None,
                "min_voltage_pu": 1.0 if flat else None,
                "feasibility": feasibility,
            },
        }

        nv = len(bqm.variables) if bqm is not None else 0
        run.update({"started_utc": started, "seed_given": self.seed, "seed_used": seed_used,
                    "software": {**_version("dwave-samplers", "dwave-neal"), **_version("dimod"), **_version("numpy")}})
        metadata = {
            "status": status,
            "solver_name": f"neal_simulated_annealing_{self.formulation}",
            "execution_time_seconds": {"total": exec_time, "formulation_cpu": round(form_time, 4),
                                       "sampling_cpu_sa": round(samp_time, 4),
                                       "postprocess_cpu": round(post_time, 4), "decoding_cpu": round(decode_time, 4)},
            "problem_complexity": complexity,
            "hardware_feasibility_report": {"applicable": False,
                                            "note": "logical solver: no embedding, scaling, quantisation or noise"},
            "qubo_parameters": {"mw_precision": self.mw_precision,
                                "angle_precision": getattr(self, "angle_precision", None),
                                "penalty_balance": getattr(self, "penalty_balance", None),
                                "penalty_line": getattr(self, "penalty_line", None), "seed": self.seed},
            "solver_settings": {"num_reads": self.num_reads, "num_sweeps": self.num_sweeps, "seed": self.seed,
                                "seed_used": seed_used, "max_decode": self.max_decode, "postprocess": self.postprocess,
                                "sampler_kwargs": self.sampler_kwargs,
                                "selection_rule": "lowest-energy feasible read of the final stage (else lowest-energy read)"},
            "algorithmic_metrics": {"framework": "pure_qubo_discretization", "num_iterations": 1,
                                    "optimality_gap_percent": None,      # earlier versions wrote 0.0, which claimed optimality
                                    "sampler_stats": sampler_stats},
            "hardware_metrics": {"qpu_target": "classical_cpu_simulated", "perfect_logical_solver": True,
                                 "logical_qubits": nv, "physical_qubits": nv, "qubit_overhead_factor": 1.0,
                                 "max_chain_length": 1, "mean_chain_length": 1.0, "circuit_depth": 0},
            "run": run,
        }
        if error is not None:
            metadata["error"] = error
        return solution, _safe(metadata)

# =========================================================================
# Execution Block: Self-Test (SimulatedAnnealingSolver)
# =========================================================================
if __name__ == "__main__":
    import copy
    import sys
    import traceback

    import dimod
    import pandapower as pp
    import pandapower.networks as nw

    QUICK = False
    NO_BRUTE = False
    NO_EXTRA = False
    OUT_JSON, OUT_CASE5 = "sa_solver_selftest.json", "sa_solver_case5_result.json"
    CHECKS, REPORT, S = [], {}, {}

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
        """First path in o that json.dumps would reject (numpy types, objects, odd keys) or that is a non-finite float.
        None if the object is clean."""
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
            return {k: trim(v, n) for k, v in o.items()}
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

    SOL_KEYS = {"cost_eur_per_hr", "generator_dispatch_mw", "static_generator_dispatch_mw", "slack_dispatch_mw", "grid_state"}
    MD_KEYS = {"status", "solver_name", "execution_time_seconds", "problem_complexity", "hardware_feasibility_report",
               "qubo_parameters", "solver_settings", "algorithmic_metrics", "hardware_metrics", "run"}
    TIME_KEYS = {"total", "formulation_cpu", "sampling_cpu_sa", "postprocess_cpu", "decoding_cpu"}
    STAGE_KEYS = {"n_reads", "n_feasible", "feasible_pct", "feasible_ci95_pct", "n_unique_states_decoded",
                  "cost_of_lowest_energy_feasible", "lowest_cost_among_feasible", "violations_among_infeasible_reads"}

    def generic(label, sol, md):
        missing = sorted((SOL_KEYS - set(sol)) | (MD_KEYS - set(md)) | (TIME_KEYS - set(md.get("execution_time_seconds", {}))))
        check(f"{label}: solution and metadata have every documented key", not missing, str(missing))
        bad = find_unserialisable({"solution": sol, "metadata": md})
        check(f"{label}: strictly JSON-serialisable (no numpy types, no NaN)", bad is None, bad or "")

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

    def mk(**kw):
        base = dict(formulation="dc_ptdf", num_reads=80, num_sweeps=200, mw_precision=10.0, seed=7)
        base.update(kw)
        return SimulatedAnnealingSolver(**base)

    def replicate(sv, net):
        """Independent re-run of exactly what solve_opf samples: (bqm, raw sampleset, final sampleset)."""
        bqm, _ = sv._formulate_qubo(net)
        raw = _SASampler().sample(bqm, num_reads=sv.num_reads, num_sweeps=sv.num_sweeps, seed=sv.seed, **sv.sampler_kwargs)
        fin = raw
        if sv.postprocess == "logical":
            fin = _SteepestDescent().sample(bqm, initial_states=raw, seed=sv.seed)
        return bqm, raw, fin

    def one_flip_minima(bqm, ss, tol=1e-6):
        """Per row: True when no single-variable flip lowers the QUBO energy (independent of the solver's code)."""
        labels = list(ss.variables)
        X = np.asarray(ss.record.sample)
        e0 = np.asarray(bqm.energies((X, labels)), dtype=float)
        ok = np.ones(len(X), dtype=bool)
        for j in range(X.shape[1]):
            Y = X.copy()
            Y[:, j] = 1 - Y[:, j]
            ok &= np.asarray(bqm.energies((Y, labels)), dtype=float) >= e0 - tol
        return ok

    def decode_rows(sv, ss, net, bqm):
        """Independent per-row decoding: returns a list of (energy, feasible, cost, occurrences, flags)."""
        labels = list(ss.variables)
        out = []
        for row, occ, e in zip(np.asarray(ss.record.sample), ss.record.num_occurrences, ss.record.energy):
            d = sv._decode_solution({l: int(v) for l, v in zip(labels, row)}, net)
            f = d[4]
            out.append((float(e), bool(f["is_feasible"]), float(d[3]), int(occ), SimulatedAnnealingSolver._violation_flags(f)))
        return out

    # ------------------------------------------------------------------ setup
    NET5, REF5 = prep("case5")
    print("=== Testing SimulatedAnnealingSolver (QUBO -> classical SA -> decoding -> solution + metadata) ===")
    print(f"mode={'quick' if QUICK else 'full'} | brute force={'off' if NO_BRUTE else 'on'} | "
          f"steepest descent available: {_SteepestDescent is not None} | case5 classical DC-OPF reference cost: {REF5}")

    # ================================================================== [1] construction
    def s1():
        sv = SimulatedAnnealingSolver()
        check("is a QuboFormulator", isinstance(sv, QuboFormulator))
        check("defaults: dc_ptdf, 500 reads, 3000 sweeps, seed 42, no postprocess, no decode cap",
              sv.formulation == "dc_ptdf" and sv.num_reads == 500 and sv.num_sweeps == 3000 and sv.seed == 42
              and sv.postprocess is None and sv.max_decode is None and sv.sampler_kwargs == {}, f"{sv.formulation}")
        check("formulation='dc' is the dc_ptdf alias", SimulatedAnnealingSolver(formulation="dc").formulation == "dc_ptdf")
        check("an invalid formulation raises ValueError", raises(ValueError, lambda: SimulatedAnnealingSolver(formulation="nope")))
        check("num_reads < 1 raises ValueError", raises(ValueError, lambda: SimulatedAnnealingSolver(num_reads=0)))
        check("num_sweeps < 1 raises ValueError", raises(ValueError, lambda: SimulatedAnnealingSolver(num_sweeps=0)))
        check("postprocess='nope' raises ValueError", raises(ValueError, lambda: SimulatedAnnealingSolver(postprocess="nope")))
        for k in ("num_reads", "num_sweeps", "seed"):
            check(f"sampler_kwargs containing '{k}' raises ValueError", raises(ValueError, lambda k=k: SimulatedAnnealingSolver(sampler_kwargs={k: 1})))
        d = {"beta_range": (0.1, 5.0)}
        sv2 = SimulatedAnnealingSolver(sampler_kwargs=d)
        d["x"] = 1
        check("sampler_kwargs are copied at construction", "x" not in sv2.sampler_kwargs)
        v = SimulatedAnnealingSolver(tighten_bounds=False, penalty_safety=2.0, encoding="bounded")
        check("formulator keyword arguments pass through", v.tighten_bounds is False and v.penalty_safety == 2.0 and v.encoding == "bounded")
        check("the sampler is Ocean's simulated-annealing sampler", type(sv.sampler).__name__ == "SimulatedAnnealingSampler")
        check("seed=None is accepted", SimulatedAnnealingSolver(seed=None).seed is None)

    # ================================================================== [2] helpers
    def s2():
        W = SimulatedAnnealingSolver._wilson if hasattr(SimulatedAnnealingSolver, "_wilson") else _wilson
        lo, hi = W(50, 100)
        check("_wilson(50, 100) is about [40.4, 59.6]", abs(lo - 40.38) < 0.1 and abs(hi - 59.62) < 0.1, f"[{lo:.2f}, {hi:.2f}]")
        lo0, hi0 = W(0, 100)
        check("_wilson(0, 100): lower bound 0, upper about 3.7", abs(lo0) < 1e-9 and 3.6 < hi0 < 3.8)
        lo1, hi1 = W(100, 100)
        check("_wilson(100, 100): upper bound 100", abs(hi1 - 100.0) < 1e-9 and 96.0 < lo1 < 96.6)
        check("_wilson(0, 0) is [None, None]", W(0, 0) == [None, None])
        check("_safe converts numpy scalars / arrays, tuples and non-string keys",
              _safe({"a": np.int64(3), "b": np.array([1.0, 2.0]), "c": (1, 2), 5: np.float32(1.5)}) == {"a": 3, "b": [1.0, 2.0], "c": [1, 2], "5": 1.5})

        F = SimulatedAnnealingSolver._violation_flags
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

        ss = dimod.SampleSet.from_samples(([[0], [1], [0]], ["x"]), energy=[1.0, 2.0, 3.0], vartype=dimod.BINARY, num_occurrences=[1, 1, 2])
        es = SimulatedAnnealingSolver._energy_stats(ss)
        mean = 2.25
        std = math.sqrt(((1 - mean) ** 2 + (2 - mean) ** 2 + 2 * (3 - mean) ** 2) / 4.0)
        check("energy stats are weighted by num_occurrences",
              abs(es["best"] - 1.0) < 1e-12 and abs(es["worst"] - 3.0) < 1e-12 and abs(es["mean"] - mean) < 1e-12 and abs(es["std"] - std) < 1e-12, f"mean {es['mean']}, std {es['std']:.4f}")
        check("fraction of reads at the lowest energy = 1/4", abs(es["fraction_of_reads_at_lowest_energy"] - 0.25) < 1e-12)

        q = {"n_reads": 200, "n_reads_total": 220, "capped": True, "n_feasible": 50, "n_unique_decoded": 17, "n_unique_total": 40,
             "best_feasible_by_energy": {"cost": 100.0}, "best_feasible_by_cost": {"cost": 90.0},
             "best_any": {"feasibility": {"is_feasible": False}},
             "violations_among_infeasible_reads": {"balance": 150, "lines": 0, "bounds": 0}, "n_infeasible_states_below_first_feasible": 3}
        sq = SimulatedAnnealingSolver._stage_quality(q, {"best": 1.0})
        check("stage quality: percentages, interval, costs, caps and violations",
              sq["feasible_pct"] == 25.0 and sq["feasible_ci95_pct"] == W(50, 200) and sq["cost_of_lowest_energy_feasible"] == 100.0
              and sq["lowest_cost_among_feasible"] == 90.0 and sq["decode_capped"] is True and sq["n_reads_total"] == 220
              and sq["n_unique_states_total"] == 40 and sq["lowest_energy_state_is_feasible"] is False
              and sq["infeasible_states_below_best_feasible"] == 3 and sq["energy"] == {"best": 1.0})
        q2 = {**q, "n_feasible": 0, "best_feasible_by_energy": None, "best_feasible_by_cost": None}
        sq2 = SimulatedAnnealingSolver._stage_quality(q2, {})
        check("stage quality without a feasible read: zero percent, no costs",
              sq2["feasible_pct"] == 0.0 and sq2["cost_of_lowest_energy_feasible"] is None and sq2["lowest_cost_among_feasible"] is None)

    # ================================================================== [3] decoding against an independent recount
    def s3():
        sv = mk()
        bqm, _ = sv._formulate_qubo(NET5)
        labels = list(bqm.variables)
        r = np.random.default_rng(1)
        X = np.vstack([np.asarray(_SASampler().sample(bqm, num_reads=60, num_sweeps=500, seed=3).record.sample)[:, :]
                       if False else r.integers(0, 2, size=(5, len(labels))).astype(np.int8)])
        res = _SASampler().sample(bqm, num_reads=60, num_sweeps=500, seed=3)
        pos = {v: i for i, v in enumerate(res.variables)}
        sa_rows = np.asarray(res.record.sample)[:, [pos[v] for v in labels]].astype(np.int8)
        X = np.vstack([sa_rows, r.integers(0, 2, size=(60, len(labels))).astype(np.int8), sa_rows[:10]])   # duplicates on purpose
        occ = r.integers(1, 4, size=len(X))
        ss = dimod.SampleSet.from_samples((X, labels), energy=bqm.energies((X, labels)), vartype=dimod.BINARY, num_occurrences=occ)
        q = sv._decode_set(ss, NET5)

        distinct, n_feas, viol = {}, 0, {"balance": 0, "lines": 0, "bounds": 0}
        for i in range(len(X)):
            key = tuple(int(v) for v in X[i])
            out = sv._decode_solution({l: int(v) for l, v in zip(labels, X[i])}, NET5)
            f, e = out[4], float(ss.record.energy[i])
            ent = distinct.setdefault(key, {"energy": e, "feasible": bool(f["is_feasible"]), "cost": out[3], "occ": 0})
            ent["occ"] += int(occ[i])
            if f["is_feasible"]:
                n_feas += int(occ[i])
            else:
                for k, ok in SimulatedAnnealingSolver._violation_flags(f).items():
                    if not ok:
                        viol[k] += int(occ[i])
        ordered = sorted(distinct.values(), key=lambda d: d["energy"])
        before = 0
        for d in ordered:
            if d["feasible"]:
                break
            before += 1
        feas = [d for d in ordered if d["feasible"]]
        check("decode_set: total and feasible read counts equal an independent recount",
              q["n_reads"] == int(occ.sum()) == q["n_reads_total"] and q["n_feasible"] == n_feas, f"{n_feas} of {int(occ.sum())} reads feasible")
        check("decode_set: distinct states are aggregated (duplicates merged)", q["n_unique_total"] == len(distinct) == q["n_unique_decoded"] and not q["capped"],
              f"{q['n_unique_total']} distinct states from {len(X)} rows")
        if feas:
            check("decode_set: lowest-energy feasible state (energy and cost)", abs(q["best_feasible_by_energy"]["qubo_energy"] - feas[0]["energy"]) < 1e-6
                  and abs(q["best_feasible_by_energy"]["cost"] - feas[0]["cost"]) < 1e-9)
            check("decode_set: lowest-cost feasible state", abs(q["best_feasible_by_cost"]["cost"] - min(d["cost"] for d in feas)) < 1e-9)
        else:
            check("decode_set: no feasible state -> no feasible bests", q["best_feasible_by_energy"] is None and q["best_feasible_by_cost"] is None)
        check("decode_set: best_any is the lowest-energy state of the set", abs(q["best_any"]["qubo_energy"] - ordered[0]["energy"]) < 1e-6)
        check("decode_set: violation counts among infeasible reads equal an independent recount", q["violations_among_infeasible_reads"] == viol, f"{viol}")
        check("decode_set: infeasible states below the first feasible one", q["n_infeasible_states_below_first_feasible"] == before, f"{before}")

        qc = mk(max_decode=5)._decode_set(ss, NET5)
        top5 = ordered[:5]
        check("decode cap: only the lowest-energy states are decoded and the cap is flagged",
              qc["capped"] is True and qc["n_unique_decoded"] == 5 and qc["n_unique_total"] == len(distinct) and qc["n_reads_total"] == int(occ.sum())
              and qc["n_reads"] == sum(d["occ"] for d in top5), f"{qc['n_reads']} of {qc['n_reads_total']} reads decoded")
        qn = mk(max_decode=10 ** 6)._decode_set(ss, NET5)
        check("a cap larger than the number of states changes nothing", qn["capped"] is False and qn["n_reads"] == q["n_reads"])

    # ================================================================== [4] end-to-end case5
    def s4():
        sv = mk()
        t0 = time.time()
        sol, md = sv.solve_opf(NET5)
        wall = time.time() - t0
        S["main"] = (sol, md, sv)
        ss_, run = SQ(md), md["run"]
        feas = sol["grid_state"]["feasibility"]
        check("status is Success", md["status"] == "Success", md["status"][:120])
        generic("case5", sol, md)
        check("solver name", md["solver_name"] == "neal_simulated_annealing_dc_ptdf", md["solver_name"])
        check("a cost was reported and the solution is feasible", sol["cost_eur_per_hr"] is not None and feas["is_feasible"] is True)

        values = {"gen": {int(k): v["p_mw"] for k, v in sol["generator_dispatch_mw"].items()},
                  "sgen": {int(k): v["p_mw"] for k, v in sol["static_generator_dispatch_mw"].items()},
                  "ext_grid": {int(k): v["p_mw"] for k, v in sol["slack_dispatch_mw"].items()}}
        rc = sv._total_true_cost(values)
        check("reported cost equals the cost recomputed from the reported dispatch", abs(rc - sol["cost_eur_per_hr"]) < 2.0, f"{sol['cost_eur_per_hr']:.3f} vs {rc:.3f}")
        check("generation equals load after the slack rebalance", abs(feas["total_generation_mw"] - feas["total_load_mw"]) < 0.01)
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
        check("DC formulation: flat voltages are reported (1.0 pu)", sol["grid_state"]["max_voltage_pu"] == 1.0 and sol["grid_state"]["min_voltage_pu"] == 1.0)

        tm = md["execution_time_seconds"]
        parts = sum(v for k, v in tm.items() if k != "total")
        check("timing: parts do not exceed the total, total matches the wall clock", parts <= tm["total"] + 1e-3 and tm["total"] <= wall + 1e-3 and tm["total"] >= wall - 0.05,
              f"total {tm['total']}s, parts {parts:.3f}s, wall {wall:.3f}s")
        check("no postprocessing: postprocess_cpu is zero", tm["postprocess_cpu"] == 0.0)
        cx = md["problem_complexity"]
        hm = HM(md)
        check("problem complexity agrees with the hardware metrics (logical qubits)", cx["quantum_domain_qubo"]["total_logical_qubits"] == hm["logical_qubits"], f"{hm['logical_qubits']}")
        check("a perfect logical solver: logical = physical qubits, chains of length 1, overhead 1.0, flagged as such",
              hm["logical_qubits"] == hm["physical_qubits"] and hm["max_chain_length"] == 1 and hm["mean_chain_length"] == 1.0 and hm["qubit_overhead_factor"] == 1.0
              and hm["perfect_logical_solver"] is True and hm["qpu_target"] == "classical_cpu_simulated")
        check("hardware feasibility report says it does not apply", md["hardware_feasibility_report"].get("applicable") is False)
        check("the optimality gap is not claimed (None, not 0.0)", md["algorithmic_metrics"]["optimality_gap_percent"] is None
              and md["algorithmic_metrics"]["framework"] == "pure_qubo_discretization")
        check("qubo parameters are recorded", md["qubo_parameters"]["mw_precision"] == 10.0 and md["qubo_parameters"]["seed"] == 7)
        st = md["solver_settings"]
        check("solver_settings equals the constructor settings",
              st["num_reads"] == 80 and st["num_sweeps"] == 200 and st["seed"] == 7 and st["seed_used"] == 7 and st["postprocess"] is None
              and st["max_decode"] is None and "lowest-energy feasible" in st["selection_rule"])
        check("run record: start time, seeds, software versions", bool(run["started_utc"]) and run["seed_given"] == 7 and run["seed_used"] == 7 and "dimod" in run["software"],
              f"{run['software']}")
        check("legacy sampler_stats keys are still there",
              {"num_reads_requested", "num_sweeps_per_read", "unique_states_found", "energy_best", "energy_mean", "energy_worst", "energy_std_dev",
               "reads_checked_for_feasibility", "feasible_read_found", "chosen_sample_energy"} <= set(ss_))
        check("stage_quality has the keys shared with the emulator wrapper", STAGE_KEYS <= set(ss_["stage_quality"]["raw"]) and STAGE_KEYS <= set(ss_["stage_quality"]["final"]))
        raw = ss_["stage_quality"]["raw"]
        check("no postprocessing: raw and final stage are the same object content", ss_["stage_quality"]["raw"] == ss_["stage_quality"]["final"] and ss_["selected_stage"] == "raw")
        check("80 reads; the percentage equals the counts; the Wilson interval brackets it",
              raw["n_reads"] == 80 and abs(raw["feasible_pct"] - 100.0 * raw["n_feasible"] / raw["n_reads"]) < 1e-9 and raw["feasible_ci95_pct"][0] <= raw["feasible_pct"] <= raw["feasible_ci95_pct"][1],
              f"{raw['feasible_pct']:.1f}% {[round(v, 1) for v in raw['feasible_ci95_pct']]}")
        check("the reported solution is the lowest-energy feasible read (stage cost and chosen energy agree)",
              raw["cost_of_lowest_energy_feasible"] == sol["cost_eur_per_hr"] and raw["lowest_cost_among_feasible"] <= sol["cost_eur_per_hr"] + 1e-9)
        check("checked reads and distinct states are consistent", ss_["reads_checked_for_feasibility"] == raw["n_unique_states_decoded"] == ss_["unique_states_found"]
              and ss_["feasible_read_found"] is True and not raw["decode_capped"])

        # independent replication of the whole sampling + selection
        bqm, raw_ss, fin_ss = replicate(sv, NET5)
        e = np.asarray(raw_ss.record.energy, dtype=float)
        check("energy statistics equal an independent replication (best / mean / worst)",
              abs(ss_["energy_best"] - round(float(e.min()), 2)) < 1e-6 and abs(ss_["energy_mean"] - round(float(e.mean()), 2)) < 1e-6
              and abs(ss_["energy_worst"] - round(float(e.max()), 2)) < 1e-6, f"best {ss_['energy_best']}, mean {ss_['energy_mean']}")
        check("distinct-state count equals an independent aggregation", ss_["unique_states_found"] == len(raw_ss.aggregate()), f"{ss_['unique_states_found']}")
        rows = decode_rows(sv, raw_ss, NET5, bqm)
        k_ind = sum(r[3] for r in rows if r[1])
        check("feasible read count equals an independent per-row decoding", raw["n_feasible"] == k_ind, f"{k_ind} of {len(rows)}")
        ef = min((r[0] for r in rows if r[1]), default=None)
        check("selection rule: the chosen read has the lowest QUBO energy among all feasible reads",
              ef is not None and abs(ss_["chosen_sample_energy"] - ef) < 1e-6, f"{ss_['chosen_sample_energy']}")
        viol_ind = {"balance": 0, "lines": 0, "bounds": 0}
        for r in rows:
            if not r[1]:
                for k, ok in r[4].items():
                    if not ok:
                        viol_ind[k] += r[3]
        check("violation counts equal an independent per-row recount", raw["violations_among_infeasible_reads"] == viol_ind, f"{viol_ind}")
        info_(f"case5: raw SA {raw['feasible_pct']:.1f}% feasible, {ss_['fraction_of_raw_reads_at_lowest_energy'] * 100:.1f}% of reads at the lowest energy; "
              f"lowest-energy state feasible: {raw['lowest_energy_state_is_feasible']}; infeasible states below the best feasible: {raw['infeasible_states_below_best_feasible']}")
        info_(f"cost {sol['cost_eur_per_hr']:.2f} vs classical continuous optimum {REF5:.2f} ({100 * (sol['cost_eur_per_hr'] - REF5) / REF5:+.2f}%); "
              f"feasibility tolerance {tol} MW, raw imbalance {feas['raw_imbalance_mw']}")

    # ================================================================== [5] every setting
    VARIANTS = []

    def add(label, kw=None, fn=None, quick=False, needs_descent=False, expect_success=True):
        VARIANTS.append((label, kw or {}, fn, quick, needs_descent, expect_success))

    add("postprocess=logical", dict(postprocess="logical"), lambda sol, md, sv: [
        ("selected stage logical; raw and final stage differ", SQ(md)["selected_stage"] == "logical" and SQ(md)["stage_quality"]["raw"] != SQ(md)["stage_quality"]["final"], ""),
        ("descent does not raise the mean energy", SQ(md)["stage_quality"]["final"]["energy"]["mean"] <= SQ(md)["stage_quality"]["raw"]["energy"]["mean"] + 1e-6,
         f"{SQ(md)['stage_quality']['raw']['energy']['mean']:.2f} -> {SQ(md)['stage_quality']['final']['energy']['mean']:.2f}"),
        ("descent does not lower the feasible share", SQ(md)["stage_quality"]["final"]["feasible_pct"] >= SQ(md)["stage_quality"]["raw"]["feasible_pct"] - 1e-9, ""),
        ("postprocess_cpu was measured", md["execution_time_seconds"]["postprocess_cpu"] >= 0.0 and md["solver_settings"]["postprocess"] == "logical", "")],
        quick=True, needs_descent=True)
    add("sampler_kwargs: beta_range (0.1, 5.0)", dict(sampler_kwargs=dict(beta_range=(0.1, 5.0))), lambda sol, md, sv: [
        ("recorded in solver_settings", md["solver_settings"]["sampler_kwargs"].get("beta_range") in ([0.1, 5.0], (0.1, 5.0)), ""),
        ("the sampler used it (its own info)", "beta_range" not in md["run"]["sampler_info"] or list(md["run"]["sampler_info"]["beta_range"]) == [0.1, 5.0],
         f"{md['run']['sampler_info'].get('beta_range')}")], quick=True)
    add("sampler_kwargs: linear schedule, 2 sweeps per beta", dict(sampler_kwargs=dict(beta_schedule_type="linear", num_sweeps_per_beta=2)), lambda sol, md, sv: [
        ("the schedule type reached the sampler", "beta_schedule_type" not in md["run"]["sampler_info"] or md["run"]["sampler_info"]["beta_schedule_type"] == "linear",
         f"{md['run']['sampler_info'].get('beta_schedule_type')}")])
    add("max_decode=3", dict(max_decode=3), lambda sol, md, sv: [
        ("only 3 distinct states decoded, cap flagged, all reads accounted for",
         SQ(md)["stage_quality"]["raw"]["n_unique_states_decoded"] <= 3 and SQ(md)["stage_quality"]["raw"]["decode_capped"] is True
         and SQ(md)["stage_quality"]["raw"]["n_reads_total"] == 80 and SQ(md)["stage_quality"]["raw"]["n_reads"] <= 80, f"{SQ(md)['stage_quality']['raw']['n_reads']} reads decoded")], quick=True)
    add("seed=None", dict(seed=None), lambda sol, md, sv: [
        ("a seed was generated and recorded", md["run"]["seed_given"] is None and isinstance(md["run"]["seed_used"], int) and md["solver_settings"]["seed_used"] == md["run"]["seed_used"], f"{md['run']['seed_used']}"),
        ("the qubo parameters keep the seed as given (None)", md["qubo_parameters"]["seed"] is None, "")])
    add("num_reads=1", dict(num_reads=1), lambda sol, md, sv: [("a single read", SQ(md)["stage_quality"]["raw"]["n_reads"] == 1 and SQ(md)["unique_states_found"] == 1, "")])
    add("num_sweeps=1 (barely annealed)", dict(num_sweeps=1), lambda sol, md, sv: [("recorded", SQ(md)["num_sweeps_per_read"] == 1, "")], expect_success=False)
    add("num_reads=400, num_sweeps=1000", dict(num_reads=400, num_sweeps=1000), lambda sol, md, sv: [("400 reads", SQ(md)["stage_quality"]["raw"]["n_reads"] == 400, "")])
    add("mw_precision=20", dict(mw_precision=20.0), lambda sol, md, sv: [
        ("recorded and fewer logical qubits than at 10 MW", md["qubo_parameters"]["mw_precision"] == 20.0 and HM(md)["logical_qubits"] < S["main"][1]["hardware_metrics"]["logical_qubits"],
         f"{HM(md)['logical_qubits']} vs {S['main'][1]['hardware_metrics']['logical_qubits']}")], quick=True)
    add("tighten_bounds=False", dict(tighten_bounds=False), lambda sol, md, sv: [
        ("no presolve: at least as many logical qubits", HM(md)["logical_qubits"] >= S["main"][1]["hardware_metrics"]["logical_qubits"], f"{HM(md)['logical_qubits']} vs {S['main'][1]['hardware_metrics']['logical_qubits']}"),
        ("info", True, f"raw imbalance of the chosen solution {sol['grid_state']['feasibility'].get('raw_imbalance_mw')} (the presolve run had {S['main'][0]['grid_state']['feasibility']['raw_imbalance_mw']})")])
    add("penalty_safety=2.0", dict(penalty_safety=2.0), lambda sol, md, sv: [
        ("the penalty weights double", abs(md["problem_complexity"]["penalties"]["balance"] / S["main"][1]["problem_complexity"]["penalties"]["balance"] - 2.0) < 1e-9, "")])
    add("encoding=unary", dict(encoding="unary"), lambda sol, md, sv: [("encoding recorded", md["problem_complexity"]["encoding"] == "unary", f"{HM(md)['logical_qubits']} logical qubits")])
    add("encoding=bounded", dict(encoding="bounded"), lambda sol, md, sv: [("encoding recorded", md["problem_complexity"]["encoding"] == "bounded", f"{HM(md)['logical_qubits']} logical qubits")])
    add("formulation=dc_theta", dict(formulation="dc_theta", num_sweeps=400), lambda sol, md, sv: [
        ("flat voltages are reported for dc_theta too", sol["grid_state"]["max_voltage_pu"] == 1.0 and sol["grid_state"]["min_voltage_pu"] == 1.0, ""),
        ("the formulation is recorded", md["problem_complexity"]["formulation"] == "dc_theta" and md["solver_name"].endswith("dc_theta"), f"{HM(md)['logical_qubits']} logical qubits")],
        expect_success=False)

    def matrix():
        print(f"    {'configuration':<46}{'status':<11}{'cost':>10}{'raw %':>7}{'final %':>8}{'states':>7}{'bestE':>14}{'secs':>7}")
        rows = []
        for label, kw, fn, quick, needs_descent, expect_success in VARIANTS:
            if (QUICK and not quick) or (needs_descent and _SteepestDescent is None):
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
            print(f"    {label:<46}{st.split(':')[0][:10]:<11}{('-' if cost is None else format(cost, '.1f')):>10}"
                  f"{(format(sqs['raw']['feasible_pct'], '.1f') if sqs else '-'):>7}{(format(sqs['final']['feasible_pct'], '.1f') if sqs else '-'):>8}"
                  f"{q.get('unique_states_found', '-'):>7}{(format(q['energy_best'], '.1f') if 'energy_best' in q else '-'):>14}{dt:>7.2f}")
            if expect_success:
                check(f"{label}: status is Success", st == "Success", st[:100])
            else:
                check(f"{label}: returned a result (Success or Infeasible, never Failed)", not st.startswith("Failed"), st[:100])
            generic(label, sol, md)
            if fn is not None and not st.startswith("Failed"):
                for name, ok, detail in fn(sol, md, sv):
                    check(f"{label}: {name}", ok, detail)
            rows.append({"configuration": label, "status": st, "cost": cost, "raw_feasible_pct": sqs["raw"]["feasible_pct"] if sqs else None,
                         "final_feasible_pct": sqs["final"]["feasible_pct"] if sqs else None, "seconds": dt})
        REPORT["settings_matrix"] = rows

    # ================================================================== [6] failure paths, trivial and forced cases
    def s6():
        sv = mk()

        def boom(net):
            raise RuntimeError("boom")
        sv._formulate_qubo = boom
        sol, md = sv.solve_opf(NET5)
        check("a formulation exception is caught: status 'Failed: boom', error type recorded, no raise", md["status"] == "Failed: boom" and md.get("error") == "RuntimeError: boom")
        check("failure: no cost, infeasible, empty dispatch", sol["cost_eur_per_hr"] is None and sol["grid_state"]["feasibility"]["is_feasible"] is False
              and sol["generator_dispatch_mw"] == {} and sol["grid_state"]["max_line_loading_percent"] is None)
        check("failure: the hardware block is consistent (0 logical qubits)", HM(md)["logical_qubits"] == 0 and md["algorithmic_metrics"]["sampler_stats"] == {})
        generic("formulation failure", sol, md)

        sv = mk()
        sv._decode_solution = lambda sample, net: (_ for _ in ()).throw(ValueError("decode broke"))
        sol, md = sv.solve_opf(NET5)
        check("a decoding exception is caught", md["status"].startswith("Failed:") and "decode broke" in md["status"] and md.get("error", "").startswith("ValueError"), md["status"][:90])
        check("failure after formulation: the formulation information survives", HM(md)["logical_qubits"] > 0 and md["problem_complexity"].get("quantum_domain_qubo", {}).get("total_logical_qubits", 0) > 0)
        generic("decoding failure", sol, md)

        sol, md = mk(sampler_kwargs=dict(beta_range="nope")).solve_opf(NET5)
        check("an invalid sampler argument is caught and reported as a failed solve", md["status"].startswith("Failed:") and "error" in md, md["status"][:100])
        generic("invalid sampler argument", sol, md)

        sv = mk()
        empty = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        empty.offset = 123.5
        sv._formulate_qubo = lambda net: (empty, {"marker": 1})
        sol, md = sv.solve_opf(NET5)
        check("zero-variable QUBO: trivial success with the offset as cost", md["status"] == "Success (0 variables, trivial)" and sol["cost_eur_per_hr"] == 123.5)
        check("zero-variable QUBO: feasible, no sampler stats, complexity passed through, no line loading",
              sol["grid_state"]["feasibility"]["is_feasible"] is True and md["algorithmic_metrics"]["sampler_stats"] == {} and md["problem_complexity"] == {"marker": 1}
              and sol["grid_state"]["max_line_loading_percent"] is None and HM(md)["logical_qubits"] == 0)
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
            check(f"forced infeasibility ({kind}): zero feasible reads, violations counted, lowest-energy state flagged infeasible",
                  sqf["n_feasible"] == 0 and sqf["feasible_pct"] == 0.0 and abs(sqf["feasible_ci95_pct"][0]) < 1e-9
                  and sqf["violations_among_infeasible_reads"][kind] == sqf["n_reads"] and sqf["lowest_energy_state_is_feasible"] is False, f"{sqf['violations_among_infeasible_reads']}")
            generic(f"forced infeasibility ({kind})", sol, md)

    # ================================================================== [7] determinism, seeds, state reuse
    def s7():
        a, b, c = mk(seed=9).solve_opf(NET5), mk(seed=9).solve_opf(NET5), mk(seed=10).solve_opf(NET5)
        check("same seed -> identical solution and sampler statistics", a[0] == b[0] and SQ(a[1]) == SQ(b[1]))
        check("a different seed changes the raw energies", SQ(a[1])["energy_mean"] != SQ(c[1])["energy_mean"], f"{SQ(a[1])['energy_mean']} vs {SQ(c[1])['energy_mean']}")
        r1, r2 = mk(seed=None).solve_opf(NET5), mk(seed=None).solve_opf(NET5)
        s1, s2 = r1[1]["run"]["seed_used"], r2[1]["run"]["seed_used"]
        check("seed=None draws a new seed for every solver", s1 != s2, f"{s1} vs {s2}")
        again = mk(seed=s1).solve_opf(NET5)
        check("feeding the recorded seed_used back in reproduces the run exactly", again[0] == r1[0] and SQ(again[1]) == SQ(r1[1]))
        if _SteepestDescent is not None:
            p1, p2 = mk(seed=9, postprocess="logical").solve_opf(NET5), mk(seed=9, postprocess="logical").solve_opf(NET5)
            check("postprocess='logical': same seed -> identical result", p1[0] == p2[0] and SQ(p1[1]) == SQ(p2[1]))

        sv = mk()
        net9, ref9 = prep("case9")
        x, y, z = sv.solve_opf(NET5), sv.solve_opf(net9), sv.solve_opf(NET5)
        check("one solver instance, three solves (case5, case9, case5): all succeed", all(t[1]["status"] == "Success" for t in (x, y, z)), f"{[t[1]['status'][:20] for t in (x, y, z)]}")
        check("the logical size follows the network", HM(x[1])["logical_qubits"] != HM(y[1])["logical_qubits"], f"{HM(x[1])['logical_qubits']} vs {HM(y[1])['logical_qubits']}")
        check("a repeated case5 solve on the reused instance is identical (no state leaks)", x[0] == z[0] and SQ(x[1]) == SQ(z[1]))

    # ================================================================== [8] postprocess stage against an independent replication
    def s8():
        if _SteepestDescent is None:
            info_("dwave.samplers.SteepestDescentSolver not available; skipped")
            return
        sv = mk(postprocess="logical", num_reads=120, num_sweeps=100, seed=4)
        sol, md = sv.solve_opf(NET5)
        bqm, raw_ss, fin_ss = replicate(sv, NET5)
        ss_ = SQ(md)
        er, ef = np.asarray(raw_ss.record.energy, dtype=float), np.asarray(fin_ss.record.energy, dtype=float)
        check("replication: the final stage has the same number of reads as the raw stage",
              int(np.sum(fin_ss.record.num_occurrences)) == int(np.sum(raw_ss.record.num_occurrences)) == ss_["stage_quality"]["raw"]["n_reads"] == ss_["stage_quality"]["final"]["n_reads_total"],
              f"{int(np.sum(fin_ss.record.num_occurrences))} reads")
        check("replication: every read of the final stage is a one-flip local minimum of the QUBO (independent check)", bool(one_flip_minima(bqm, fin_ss).all()))
        check("replication: the descent lowers or keeps the mean and the best energy", ef.mean() <= er.mean() + 1e-9 and ef.min() <= er.min() + 1e-9,
              f"mean {er.mean():.2f} -> {ef.mean():.2f}")
        check("the metadata's final-stage energy statistics equal the replication",
              abs(ss_["stage_quality"]["final"]["energy"]["mean"] - float(np.average(ef, weights=fin_ss.record.num_occurrences))) < 1e-6
              and abs(ss_["stage_quality"]["final"]["energy"]["best"] - float(ef.min())) < 1e-6)
        rows = decode_rows(sv, fin_ss, NET5, bqm)
        k_ind = sum(r[3] for r in rows if r[1])
        check("final-stage feasible read count equals an independent per-row decoding", ss_["stage_quality"]["final"]["n_feasible"] == k_ind, f"{k_ind} of {sum(r[3] for r in rows)}")
        ef_min = min((r[0] for r in rows if r[1]), default=None)
        check("selection rule on the final stage: the chosen read has the lowest QUBO energy among feasible reads",
              ef_min is not None and abs(ss_["chosen_sample_energy"] - ef_min) < 1e-6)
        raw_rows = decode_rows(sv, raw_ss, NET5, bqm)
        check("the raw stage reported in the metadata is the undescended output", ss_["stage_quality"]["raw"]["n_feasible"] == sum(r[3] for r in raw_rows if r[1]))
        info_(f"descent effect on case5: feasible reads {ss_['stage_quality']['raw']['feasible_pct']:.1f}% -> {ss_['stage_quality']['final']['feasible_pct']:.1f}%; "
              f"mean QUBO energy {er.mean():.1f} -> {ef.mean():.1f}")

    # ================================================================== [9] exhaustive enumeration
    def s9():
        if NO_BRUTE:
            info_("skipped (--no-brute-force)")
            return
        sv = None
        for prec in (100.0, 150.0, 200.0, 250.0, 300.0, 400.0):
            cand = mk(mw_precision=prec, num_reads=300, num_sweeps=2000, seed=3)
            bqm_c, _ = cand._formulate_qubo(NET5)
            if 1 <= len(bqm_c.variables) <= 18:
                sv, bqm, prec_used = cand, bqm_c, prec
                break
        if sv is None:
            info_("no coarse precision gives a QUBO with at most 18 variables; skipped")
            return
        n = len(bqm.variables)
        ex = dimod.ExactSolver().sample(bqm)
        rows = list(ex.data(["sample", "energy"], sorted_by="energy"))
        E0 = float(rows[0].energy)
        unique_ground = len(rows) < 2 or float(rows[1].energy) > E0 + 1e-9
        limit = len(rows) if n <= 14 else 4000
        first_feas_e, infeas_before, best_cost, best_cost_e, found = None, 0, None, None, False
        for i, d in enumerate(rows[:limit]):
            disp, sg, sl, c, f = sv._decode_solution(dict(d.sample), NET5)
            if f["is_feasible"]:
                if not found:
                    first_feas_e, found = float(d.energy), True
                if best_cost is None or c < best_cost:
                    best_cost, best_cost_e = c, float(d.energy)
            elif not found:
                infeas_before += 1
        complete = limit == len(rows)
        print(f"    precision {prec_used} MW -> {n} logical variables, {len(rows)} states enumerated ({'all decoded' if complete else f'first {limit} decoded'}); "
              f"ground energy {E0:.4f}{' (unique)' if unique_ground else ' (degenerate)'}")
        sol, md = sv.solve_opf(NET5)
        raw = SQ(md)["stage_quality"]["raw"]
        check("the solver found the exact ground state of the QUBO", abs(SQ(md)["energy_best"] - round(E0, 2)) < 1e-6, f"{SQ(md)['energy_best']} vs {round(E0, 2)}")
        check("a good share of the reads reached the ground energy", SQ(md)["fraction_of_raw_reads_at_lowest_energy"] > 0.05, f"{100 * SQ(md)['fraction_of_raw_reads_at_lowest_energy']:.1f}%")
        if unique_ground:
            ground_feas = bool(sv._decode_solution(dict(rows[0].sample), NET5)[4]["is_feasible"])
            check("penalty diagnostic: 'lowest-energy state is feasible' equals the exact answer", raw["lowest_energy_state_is_feasible"] == ground_feas, f"exact {ground_feas}")
        else:
            info_("the ground state is degenerate; the lowest-energy-state-feasible comparison is skipped")
        if found:
            check("the reported solution has the exact lowest QUBO energy among FEASIBLE states", md["status"] == "Success" and abs(SQ(md)["chosen_sample_energy"] - first_feas_e) < 1e-6,
                  f"{SQ(md)['chosen_sample_energy']} vs exact {first_feas_e}")
            check("infeasible states seen below the best feasible one never exceed the exact count", raw["infeasible_states_below_best_feasible"] <= infeas_before,
                  f"solver {raw['infeasible_states_below_best_feasible']}, exact {infeas_before}")
            check("no feasible cost reported below the exact discrete optimum" + ("" if complete else " (of the decoded states)"),
                  best_cost is None or raw["lowest_cost_among_feasible"] >= best_cost - 1e-9, f"solver {raw['lowest_cost_among_feasible']} vs exact {best_cost}")
            if complete and REF5:
                info_(f"discretisation at {prec_used} MW: exact discrete optimum cost {best_cost:.2f} vs continuous optimum {REF5:.2f} ({100 * (best_cost - REF5) / REF5:+.2f}%); "
                      f"the solver's own best feasible cost {raw['lowest_cost_among_feasible']:.2f}")
                REPORT["brute_force"] = {"precision": prec_used, "n_variables": n, "ground_energy": E0, "unique_ground": unique_ground,
                                         "exact_best_feasible_cost": best_cost, "continuous_reference": REF5, "solver_best_feasible_cost": raw["lowest_cost_among_feasible"]}
        else:
            check("a feasible state exists among the decoded states", False, "none found; the exact reference is unavailable")

    # ================================================================== [10] several networks
    def s10():
        names = ["case4gs", "case5", "case9"] + ([] if (QUICK or NO_EXTRA) else ["case14", "case30"])
        print(f"    {'network':<9}{'status':<11}{'logical':>8}{'cost':>12}{'classical':>12}{'gap %':>8}{'raw %':>7}{'bestE feas?':>12}{'secs':>7}")
        rows = []
        for nm in names:
            try:
                net, ref = prep(nm)
                sv = mk(num_reads=200, num_sweeps=500)
                t0 = time.time()
                sol, md = sv.solve_opf(net)
                dt = time.time() - t0
            except Exception as exc:
                check(f"{nm}: solved", False, f"{type(exc).__name__}: {exc}")
                continue
            sqs = (md["algorithmic_metrics"].get("sampler_stats") or {}).get("stage_quality", {})
            cost = sol["cost_eur_per_hr"]
            gap = None if (cost is None or not ref) else 100.0 * (cost - ref) / abs(ref)
            lf = sqs["raw"]["lowest_energy_state_is_feasible"] if sqs else None
            print(f"    {nm:<9}{md['status'].split(':')[0][:10]:<11}{HM(md).get('logical_qubits', '-'):>8}{('-' if cost is None else format(cost, '.1f')):>12}"
                  f"{('-' if ref is None else format(ref, '.1f')):>12}{('-' if gap is None else format(gap, '+.2f')):>8}{(format(sqs['raw']['feasible_pct'], '.1f') if sqs else '-'):>7}{str(lf):>12}{dt:>7.1f}")
            if ref is not None:
                check(f"{nm}: Success with a feasible solution (a classical reference exists)", md["status"] == "Success" and sol["grid_state"]["feasibility"]["is_feasible"] is True, md["status"][:100])
            else:
                check(f"{nm}: no classical reference (the DC-OPF itself failed); the solver returned a well-formed result", not md["status"].startswith("Failed"), md["status"][:100])
                info_(f"{nm}: pandapower's own DC-OPF failed, so this network may be infeasible after the ext_grid modification; solver status: {md['status'][:80]}")
            generic(nm, sol, md)
            rows.append({"network": nm, "status": md["status"], "cost": cost, "classical_reference": ref, "gap_pct": gap,
                         "raw_feasible_pct": sqs["raw"]["feasible_pct"] if sqs else None, "lowest_energy_state_feasible": lf})
        REPORT["networks"] = rows
        info_("gaps compare the solver's lowest-energy feasible read with the classical DC-OPF cost; small negative values are not improvements "
              "(the feasibility tolerance equals the dispatch precision)")

    # ================================================================== [11] example record
    def s11():
        sol, md, sv = S["main"]
        doc = {"settings": md["solver_settings"], "solution": sol, "metadata": md}
        with open(OUT_CASE5, "w") as fh:
            json.dump(doc, fh, indent=1)
        with open(OUT_CASE5) as fh:
            back = json.load(fh)
        check("the example record was written and reloads", back["metadata"]["status"] == "Success" and back["solution"]["cost_eur_per_hr"] == sol["cost_eur_per_hr"])
        check("the reloaded record keeps the stage quality and the seed", "stage_quality" in back["metadata"]["algorithmic_metrics"]["sampler_stats"] and back["metadata"]["run"]["seed_used"] == 7)
        prev = copy.deepcopy(doc)
        prev["metadata"]["problem_complexity"] = {"keys": sorted(prev["metadata"]["problem_complexity"])}
        print(f"\n    ---- preview of {OUT_CASE5} (lists cut to 3 items, long values shortened; the file has everything) ----")
        print(json.dumps(trim(prev), indent=1))
        print("    ---- end of preview ----")

    guarded("[1] construction, defaults, validation", s1)
    guarded("[2] helper functions: _wilson, _safe, violation flags, energy statistics, stage quality", s2)
    guarded("[3] decoding against an independent recount (mixed feasible / infeasible sampleset, duplicates, decode cap)", s3)
    guarded("[4] end-to-end case5: solution, metadata, independent replication of sampling and selection", s4)
    guarded("[5] every setting: postprocess, sampler arguments, decode cap, seeds, reads / sweeps, formulator arguments, formulations", matrix)
    guarded("[6] failure paths, trivial problem and forced infeasibility", s6)
    guarded("[7] determinism, seeds and state reuse", s7)
    guarded("[8] the logical postprocessing stage against an independent replication", s8)
    guarded("[9] exhaustive enumeration of a coarse-precision QUBO (the strongest check)", s9)
    guarded("[10] several networks", s10)
    guarded("[11] the example result record", s11)

    section("SUMMARY")
    failed = [c for c in CHECKS if not c[1]]
    print(f"checks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}")
    with open(OUT_JSON, "w") as fh:
        json.dump({"settings": {"quick": QUICK, "no_brute_force": NO_BRUTE, "no_extra_cases": NO_EXTRA}, "checks": CHECKS, "report": REPORT}, fh, default=str, indent=1)
    print(f"details written to {OUT_JSON}; example solution record in {OUT_CASE5}")
    sys.exit(1 if failed else 0)