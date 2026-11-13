"""Exact optimum of the DISCRETISED OPF: the baseline the samplers are judged against.

Model (solved to proven optimality by HiGHS, a deterministic branch-and-bound MILP):
  * Every dispatch asset takes one value from the set of values its QUBO encoding can represent (all subset sums of the
    encoding weights above the lower bound), so the search space is exactly the one the QUBO samplers explore. One binary
    per (asset, value), exactly one chosen per asset. Costs are the TRUE costs (exact PWL / polynomial), not the quadratic fit.
  * Constraints: power balance and line limits (full PTDF, the decoder's own physics).
Two solves, both reported:
  primary          feasibility exactly as the decoder / samplers define it: |imbalance| <= tolerance (default: the formulator's
                   feasibility tolerance, i.e. mw_precision), line flows <= limit + tolerance, the reference ext_grid absorbs the
                   imbalance. Compare samplers against THIS cost. (With a non-linear reference cost the rebalance cost is not
                   modelled; metadata says so.)
  strict reference balance and line limits to 1e-6 MW, no tolerance: the true discretised optimum. The gap between this and the
                   continuous optimum is pure discretisation error; if the grid cannot hit the balance exactly it is Infeasible.
The primary solution is passed through the formulator's own decoder, so the output layout and feasibility fields match the
other solvers, and the MILP objective is cross-checked against the decoder's cost.
Limits: dc_theta is solved as the dispatch-grid problem (angle discretisation not modelled); the value set per asset is capped
(max_values_per_asset). Optimality is only claimed when HiGHS proves it (status code 0); otherwise the status says so.
"""
import datetime
import json
import math
import time
from importlib import metadata as _md

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from solvers.qubo_formulator import QuboFormulator

_STRICT_TOL = 1e-6


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


def _version(*names):
    for n in names:
        try:
            return {n: _md.version(n)}
        except _md.PackageNotFoundError:
            continue
    return {}


class ExactDiscreteSolver(QuboFormulator):
    def __init__(self, formulation="dc_ptdf", mw_precision=10.0, time_limit_s=60.0, mip_rel_gap=0.0,
                 max_values_per_asset=20000, compute_strict=True, balance_tol_mw=None, line_tol_mw=None, **kwargs):
        """time_limit_s / mip_rel_gap : HiGHS limits (gap 0 = prove optimality). max_values_per_asset: cap on the number of
        representable values per asset (raise mw_precision or use another encoding if exceeded). compute_strict: also solve the
        strict reference. balance_tol_mw / line_tol_mw: tolerances of the primary solve (None = the decoder's tolerance)."""
        super().__init__(formulation=formulation, mw_precision=mw_precision, **kwargs)
        if time_limit_s <= 0 or mip_rel_gap < 0 or max_values_per_asset < 1:
            raise ValueError("time_limit_s > 0, mip_rel_gap >= 0 and max_values_per_asset >= 1 are required")
        if (balance_tol_mw is not None and balance_tol_mw < 0) or (line_tol_mw is not None and line_tol_mw < 0):
            raise ValueError("tolerances must be >= 0")
        self.time_limit_s, self.mip_rel_gap = float(time_limit_s), float(mip_rel_gap)
        self.max_values_per_asset, self.compute_strict = int(max_values_per_asset), bool(compute_strict)
        self.balance_tol_mw, self.line_tol_mw = balance_tol_mw, line_tol_mw

    # ------------------------------------------------------------------ model
    def _values(self, reg):
        """All values an encoded asset can take, with one bit mask per value (bit k <-> reg['bits'][k])."""
        vals = {round(reg["min"], 9): 0}
        for k, w in enumerate(reg["weights"]):
            new = dict(vals)
            for v, m in vals.items():
                new.setdefault(round(v + w, 9), m | (1 << k))
            vals = new
            if len(vals) > self.max_values_per_asset:
                raise RuntimeError(f"an asset has more than {self.max_values_per_asset} representable values; "
                                   f"use a coarser mw_precision or another encoding")
        items = sorted(vals.items())
        return np.array([v for v, _ in items], dtype=float), [m for _, m in items]

    def _milp(self, assets, tol_b, tol_l, adjust_ref):
        pos, L = self._bus_pos, float(sum(self._load_mw.values()))
        H = self._H_full if self.decode_with_full_ptdf else self._H
        offs = np.cumsum([0] + [len(a["v"]) for a in assets])
        n = int(offs[-1])
        rows, lb, ub = [], [], []

        def block_row(fn):
            r = np.zeros(n)
            for a, o in zip(assets, offs):
                r[o:o + len(a["v"])] = fn(a)
            return r

        for a, o in zip(assets, offs):                                          # exactly one value per asset
            r = np.zeros(n)
            r[o:o + len(a["v"])] = 1.0
            rows.append(r); lb.append(1.0); ub.append(1.0)
        rows.append(block_row(lambda a: a["v"])); lb.append(L - tol_b); ub.append(L + tol_b)      # power balance
        for ln in self._lines:                                                  # line limits
            if ln["p_max"] <= 0.0:
                continue
            h = H[ln["idx"]]
            base = -sum(h[pos[b]] * ld for b, ld in self._load_mw.items())
            rows.append(block_row(lambda a: h[pos[a["bus"]]] * a["v"]))
            lb.append(-(ln["p_max"] + tol_l) - base); ub.append(ln["p_max"] + tol_l - base)
        ref = next(a for a in assets if a["ref"])
        if adjust_ref:                                                          # rebalanced reference stays inside its bounds
            rows.append(block_row(lambda a: 0.0 * a["v"] if a["ref"] else a["v"]))
            lb.append(L - ref["reg"]["max"]); ub.append(L - ref["reg"]["min"])

        c0r, c1r, c2r = self._cost_coeffs("ext_grid", self._ref_ext)
        entry = self._costs["ext_grid"].get(self._ref_ext)
        ref_linear = c2r == 0.0 and not (entry is not None and entry["pwl"] is not None)
        adjust = bool(adjust_ref and ref_linear)        # reference value = L - (others), so its cost is linear in the others
        c, const = np.zeros(n), (c1r * L + c0r if adjust else 0.0)
        for a, o in zip(assets, offs):
            for j, v in enumerate(a["v"]):
                if a["ref"] and adjust:
                    continue
                c[o + j] = self._true_cost(a["et"], a["idx"], float(v)) - (c1r * float(v) if adjust else 0.0)

        t0 = time.time()
        res = milp(c, constraints=LinearConstraint(sp.csr_matrix(np.array(rows)), np.array(lb), np.array(ub)),
                   integrality=np.ones(n), bounds=Bounds(0.0, 1.0),
                   options={"time_limit": self.time_limit_s, "mip_rel_gap": self.mip_rel_gap, "disp": False})
        secs = time.time() - t0
        x = res.x if (res.x is not None and res.status in (0, 1)) else None
        gap = getattr(res, "mip_gap", None)
        bound = getattr(res, "mip_dual_bound", None)
        info = {"status_code": int(res.status), "message": str(res.message), "proven_optimal": bool(res.status == 0 and x is not None),
                "objective": (float(res.fun) + const) if x is not None else None,
                "dual_bound": (float(bound) + const) if bound is not None and math.isfinite(bound) else None,
                "gap_pct": (100.0 * float(gap)) if gap is not None and math.isfinite(gap) else None,
                "nodes": getattr(res, "mip_node_count", None), "seconds": secs,
                "balance_tol_mw": tol_b, "line_tol_mw": tol_l, "rebalance_cost_modelled": adjust,
                "rows": len(rows), "columns": n}
        return info, x, offs

    @staticmethod
    def _sample(assets, offs, x):
        sample = {}
        for a, o in zip(assets, offs):
            m = a["m"][int(np.argmax(x[o:o + len(a["v"])]))]
            for k, bit in enumerate(a["reg"]["bits"]):
                sample[bit] = (m >> k) & 1
        return sample

    # ------------------------------------------------------------------ main entry
    def solve_opf(self, net):
        t_start = time.time()
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        status, cost = "Success", None
        dispatch, sgen_dispatch, slack_dispatch = {}, {}, {}
        feasibility = {"is_feasible": False, "line_flows_mw": {}}
        form_t = build_t = solve_t = dec_t = 0.0
        complexity, stats, error, nvars = {}, {}, None, 0

        try:
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)                  # same discretisation (grid, presolve bounds) as the samplers
            form_t, nvars = time.time() - t0, len(bqm.variables)

            t0 = time.time()
            assets = []
            for et, idx, reg, bus in self._dispatch_assets(net):
                v, m = self._values(reg)
                assets.append({"et": et, "idx": int(idx), "bus": bus, "reg": reg, "v": v, "m": m,
                               "ref": et == "ext_grid" and int(idx) == self._ref_ext})
            if not any(a["ref"] for a in assets):
                raise RuntimeError("the reference ext_grid has no dispatch variable")
            build_t = time.time() - t0

            tol_dec = self.feasibility_tol_mw if self.feasibility_tol_mw is not None else self.mw_precision
            tol_b = tol_dec if self.balance_tol_mw is None else float(self.balance_tol_mw)
            tol_l = tol_dec if self.line_tol_mw is None else float(self.line_tol_mw)

            t0 = time.time()
            p, xp, offs = self._milp(assets, tol_b, tol_l, adjust_ref=tol_b > _STRICT_TOL)
            strict = self._milp(assets, _STRICT_TOL, _STRICT_TOL, adjust_ref=False) if self.compute_strict else None
            solve_t = time.time() - t0

            t0 = time.time()
            stats = {"model": {"n_assets": len(assets), "n_binary_columns": p["columns"], "n_constraints": p["rows"],
                               "values_per_asset": {f"{a['et']}_{a['idx']}": len(a["v"]) for a in assets},
                               "log10_search_space": round(sum(math.log10(len(a["v"])) for a in assets), 2),
                               "angle_discretisation_modelled": False if self.formulation == "dc_theta" else None},
                     "decoder_tolerance_mw": tol_dec}
            stats["primary"] = {k: v for k, v in p.items()}
            if p["status_code"] == 0 or xp is not None:
                sample = self._sample(assets, offs, xp)
                dispatch, sgen_dispatch, slack_dispatch, cost, feasibility = self._decode_solution(sample, net)
                stats["primary"]["decoded_cost"] = cost
                stats["primary"]["objective_minus_decoded_cost"] = p["objective"] - cost
                if not feasibility["is_feasible"] or abs(p["objective"] - cost) > 0.01 + 1e-6 * abs(cost):
                    status = ("Failed: the MILP optimum disagrees with the decoder "
                              f"(feasible={feasibility['is_feasible']}, objective {p['objective']:.3f} vs decoded cost {cost:.3f})")
                elif p["proven_optimal"]:
                    status = "Success"
                else:
                    status = (f"Success (not proven optimal: HiGHS stopped with gap "
                              f"{'n/a' if p['gap_pct'] is None else format(p['gap_pct'], '.3f') + '%'}; {p['message'][:60]})")
            elif p["status_code"] == 2:
                status = (f"Infeasible: no discrete dispatch satisfies balance (+-{tol_b} MW) and the line limits (+{tol_l} MW)")
            else:
                status = f"Failed: HiGHS ended without a solution ({p['message'][:100]})"

            if strict is not None:
                sinfo, sx, soffs = strict
                ref_doc = {k: v for k, v in sinfo.items()}
                if sx is not None:
                    sd = self._decode_solution(self._sample(assets, soffs, sx), net)
                    ref_doc.update({"decoded_cost": sd[3], "decoded_feasible": sd[4]["is_feasible"],
                                    "raw_imbalance_mw": sd[4]["raw_imbalance_mw"]})
                    if cost is not None:
                        ref_doc["primary_minus_strict_cost"] = cost - sd[3]
                ref_doc["verdict"] = ("optimal" if sinfo["proven_optimal"] else "infeasible: the grid cannot satisfy balance and lines "
                                      "exactly (check grid alignment / tighten_bounds)" if sinfo["status_code"] == 2 else sinfo["message"][:80])
                stats["strict_reference"] = ref_doc
            dec_t = time.time() - t0

        except Exception as e:
            status, error = f"Failed: {e}", f"{type(e).__name__}: {e}"
            cost, dispatch, sgen_dispatch, slack_dispatch = None, {}, {}, {}
            feasibility = {"is_feasible": False, "line_flows_mw": {}}
            print(f"\nCRITICAL ERROR IN EXACT DISCRETE SOLVER: {e}\n")

        exec_time = round(time.time() - t_start, 4)
        max_load_pct = None
        if hasattr(self, "_lines") and self._lines and feasibility.get("line_flows_mw"):
            loads = [abs(feasibility["line_flows_mw"].get(ln["idx"], 0.0)) / ln["p_max"] * 100.0 for ln in self._lines if ln["p_max"] > 0.0]
            if loads:
                max_load_pct = round(max(loads), 3)

        flat = self.formulation in ("dc_ptdf", "dc_theta")
        solution = {
            "cost_eur_per_hr": cost, "generator_dispatch_mw": dispatch, "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {"max_line_loading_percent": max_load_pct, "max_voltage_pu": 1.0 if flat else None,
                           "min_voltage_pu": 1.0 if flat else None, "feasibility": feasibility}}
        prim = stats.get("primary", {})
        metadata = {
            "status": status,
            "solver_name": f"highs_milp_discrete_exact_{self.formulation}",
            "execution_time_seconds": {"total": exec_time, "formulation_cpu": round(form_t, 4), "model_build_cpu": round(build_t, 4),
                                       "solve_cpu_milp": round(solve_t, 4), "decoding_cpu": round(dec_t, 4)},
            "problem_complexity": complexity,
            "hardware_feasibility_report": {"applicable": False, "note": "exact classical solver: no embedding, scaling, quantisation or noise"},
            "qubo_parameters": {"mw_precision": self.mw_precision, "angle_precision": getattr(self, "angle_precision", None),
                                "penalty_balance": getattr(self, "penalty_balance", None), "penalty_line": getattr(self, "penalty_line", None)},
            "solver_settings": {"time_limit_s": self.time_limit_s, "mip_rel_gap": self.mip_rel_gap, "max_values_per_asset": self.max_values_per_asset,
                                "compute_strict": self.compute_strict, "balance_tol_mw": self.balance_tol_mw, "line_tol_mw": self.line_tol_mw,
                                "feasibility": "decoder definition (tolerance + slack rebalance) for the primary solve; 1e-6 MW for the strict reference"},
            "algorithmic_metrics": {"framework": "exact_discrete_opf_milp", "num_iterations": 1, "proven_optimal": prim.get("proven_optimal", False),
                                    "optimality_gap_percent": prim.get("gap_pct"), "dual_bound": prim.get("dual_bound"), "solver_stats": stats},
            "hardware_metrics": {"qpu_target": "classical_cpu_exact_milp", "perfect_logical_solver": True, "logical_qubits": nvars,
                                 "physical_qubits": nvars, "qubit_overhead_factor": 1.0, "max_chain_length": 1, "mean_chain_length": 1.0, "circuit_depth": 0},
            "run": {"started_utc": started, "software": {**_version("scipy"), **_version("numpy")}}}
        if error is not None:
            metadata["error"] = error
        return solution, _safe(metadata)