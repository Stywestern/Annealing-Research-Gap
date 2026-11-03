"""Classical (Lossless) DC Optimal Power Flow (DC-OPF) QUBO Formulation."""

import heapq
import math
import time
import copy

import dimod
import numpy as np
import pandapower as pp
from scipy.optimize import linprog

# ===============================================================================================================
#                                         QUBO FORMULATOR
# ===============================================================================================================

class QuboFormulator:
    # Asset constraints
    UNCONSTRAINED_I_KA = 100.0
    UNCONSTRAINED_MW = 1000.0
    DEFAULT_MAX_MW = 500.0
    ANGLE_HEADROOM = 1.1

    FORMULATIONS = ("dc_ptdf", "dc_theta")
    ENCODINGS = ("radix", "unary", "bounded")
    _DISPATCH_TYPES = ("gen", "sgen", "ext_grid")
    _BIT_PREFIX = {"gen": "gen", "sgen": "sgen", "ext_grid": "ext"}

    def __init__(self, formulation="dc_ptdf", mw_precision=10.0, angle_precision=None, 
                 penalty_balance=None, penalty_line=None, encoding="radix", penalty_safety=1.0, 
                 ptdf_threshold=0.05, feasibility_tol_mw=None, rebalance_slack=True,
                 scale_line_constraints=True, smart_slack_side=True, slack_precision_factor=1.0, 
                 decode_with_full_ptdf=True, ptdf_rel_threshold=None, ptdf_round_to=None, 
                 snap_weights=True, ext_fallback_mult=2.0, risk_noise_frac=0.03125,
                 tighten_bounds=True, theta_smart_slack=True, single_reference_angle=True, 
                 bounded_mu=4, penalty_mode="heuristic", **kwargs):

        self.formulation = "dc_ptdf" if formulation in ("dc", "dc_ptdf") else formulation
        if self.formulation not in self.FORMULATIONS:
            raise ValueError(f"formulation must be {self.FORMULATIONS}, got {formulation}")
        if encoding not in self.ENCODINGS:
            raise ValueError(f"encoding must be {self.ENCODINGS}, got {encoding}")

        # Core Parameters
        self.mw_precision = mw_precision
        self.angle_precision = angle_precision
        self.encoding = encoding
        self.feasibility_tol_mw = feasibility_tol_mw
        
        # Penalties
        self.penalty_balance = penalty_balance
        self.penalty_line = penalty_line
        self.penalty_safety = penalty_safety
        self.penalty_mode = penalty_mode
        self.lambda_balance = penalty_balance
        self.lambda_line = penalty_line

        # Scaling & Precision
        self.scale_line_constraints = scale_line_constraints
        self.smart_slack_side = smart_slack_side
        self.slack_precision_factor = slack_precision_factor
        self.snap_weights = snap_weights
        self.rebalance_slack = rebalance_slack
        self.ext_fallback_mult = ext_fallback_mult
        self.risk_noise_frac = risk_noise_frac

        # PTDF Specifics
        self.ptdf_threshold = ptdf_threshold
        self.ptdf_rel_threshold = ptdf_rel_threshold
        self.ptdf_round_to = ptdf_round_to
        self.decode_with_full_ptdf = decode_with_full_ptdf

        # Theta Specifics
        self.theta_smart_slack = theta_smart_slack
        self.single_reference_angle = single_reference_angle
        self.bounded_mu = bounded_mu
        self.tighten_bounds = tighten_bounds
        
        # Internal State
        self._tight_bounds = {}
        self._line_flow_range = {}
        self.presolve_info = {}
        self.var_registry = {}
        self.cost_model_warnings = []
        self.angle_bound_info = {}
        self._line_slack_sign = {}

    # ==================================================================
    # Utility Helpers
    # ==================================================================
    
    @staticmethod
    def _val(x, default):
        try:
            return default if x is None or math.isnan(float(x)) else float(x)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _active(net, et):
        tbl = getattr(net, et, None)
        return list(tbl.index[tbl.in_service.astype(bool)]) if tbl is not None and not tbl.empty else []

    def _asset_bounds(self, net, et, idx):
        row = getattr(net, et).loc[idx]
        default_max = self._val(row.get('p_mw'), 0.0) if et == 'sgen' else self.DEFAULT_MAX_MW
        p_min = self._val(row.get('min_p_mw'), 0.0)
        p_max = self._val(row.get('max_p_mw'), default_max)
        if p_max < p_min - 1e-9:
            raise ValueError(f"{et} {idx}: max_p_mw ({p_max}) < min_p_mw ({p_min})")
        return p_min, p_max

    def _dispatch_assets(self, net):
        for et in self._DISPATCH_TYPES:
            tbl = getattr(net, et)
            for idx, reg in self.var_registry.get(et, {}).items():
                yield et, idx, reg, int(tbl.at[idx, 'bus'])

    # ==================================================================
    # Variable Encoders
    # ==================================================================

    def _get_radix_weights(self, total_range, precision):
        if total_range <= 1e-12: return []
        n = max(1, int(math.floor(total_range / precision + 1e-9)))
        K = int(math.floor(math.log2(n)))
        weights = [precision * (2 ** k) for k in range(K)]
        weights.append(total_range - (2 ** K - 1) * precision)
        return weights

    def _get_unary_weights(self, total_range, precision):
        if total_range <= 1e-12: return []
        n = max(1, int(math.floor(total_range / precision + 1e-9)))
        weights = [precision] * n
        remainder = total_range - n * precision
        if remainder > 1e-12: weights.append(remainder)
        return weights

    @staticmethod
    def _get_bounded_coefficient_weights(kappa, mu, precision):
        kappa, mu = int(kappa), int(mu)
        if kappa <= 0: return []
        mu = max(1, min(mu, kappa))
        
        if kappa < 2 ** (int(math.floor(math.log2(mu))) + 1):
            n = int(math.floor(math.log2(kappa)))
            c = [2 ** i for i in range(n)]
            c.append(kappa - sum(c))
        else:
            rho = int(math.floor(math.log2(mu))) + 1
            c = [2 ** i for i in range(rho)]
            nu = kappa - sum(c)
            eta = nu // mu
            c += [mu] * eta
            if nu - eta * mu > 0: c.append(nu - eta * mu)
            
        assert sum(c) == kappa
        return [x * precision for x in c]

    def _get_bounded_weights(self, total_range, precision):
        if total_range <= 1e-12 or not precision or precision <= 0: return []
        kappa = int(math.floor(total_range / precision + 1e-9))
        if kappa <= 0: return [total_range]
        
        weights = self._get_bounded_coefficient_weights(kappa, self.bounded_mu, precision)
        remainder = total_range - kappa * precision
        if remainder > 1e-12: weights.append(remainder)
        return weights

    def _get_weights(self, total_range, precision):
        if total_range <= 0: return []
        
        enc_map = {
            "radix": self._get_radix_weights,
            "unary": self._get_unary_weights,
            "bounded": self._get_bounded_weights
        }
        weights = enc_map[self.encoding](total_range, precision)

        if not self.snap_weights or not precision or precision <= 0:
            return weights

        snapped = [k * precision for x in weights if (k := math.floor(x / precision + 1e-9)) > 0]
        return snapped if snapped else weights

    def _register(self, group, key, prefix, abs_lo, abs_hi, precision):
        weights = self._get_weights(abs_hi - abs_lo, precision) if abs_hi > abs_lo else []
        bits = [f"{prefix}_bit_{k}" for k in range(len(weights))]
        self.var_registry.setdefault(group, {})[key] = {
            'min': abs_lo, 'max': abs_hi, 'precision': precision, 'bits': bits, 'weights': weights
        }
        return len(bits)

    @staticmethod
    def _decode_value(reg, sample):
        return reg['min'] + sum(w for bit, w in zip(reg['bits'], reg['weights']) if sample.get(bit, 0) == 1)

    @staticmethod
    def _reg_span(reg):
        return reg['min'], reg['min'] + sum(reg['weights'])

    # ==================================================================
    # Cost Model
    # ==================================================================
    
    @staticmethod
    def _parse_pwl(segments, label):
        segs = [(float(s[0]), float(s[1]), float(s[2])) for s in segments if len(s) == 3]
        if len(segs) != len(segments):
            raise ValueError(f"pwl_cost of {label}: expected (p_from, p_to, slope)")
        return sorted(segs)

    @staticmethod
    def _pwl_eval(segs, p):
        p = np.asarray(p, dtype=float)
        total = np.zeros_like(p)
        for k, (a, b, slope) in enumerate(segs):
            if k == len(segs) - 1:
                total += slope * np.maximum(p - a, 0.0)
            else:
                total += slope * np.clip(p - a, 0.0, b - a)
        return total

    def _fit_quadratic(self, segs, p_lo, p_hi, label):
        if p_hi <= p_lo:
            return float(self._pwl_eval(segs, [p_lo])[0]), 0.0, 0.0
        
        xs = np.linspace(p_lo, p_hi, 201)
        ys = self._pwl_eval(segs, xs)
        c2, c1, c0 = np.polyfit(xs, ys, 2)
        
        if c2 < 1e-12:
            c1, c0 = np.polyfit(xs, ys, 1)
            c2 = 0.0
            
        err = float(np.max(np.abs(ys - (c2 * xs ** 2 + c1 * xs + c0))))
        if err > 1e-6 * max(1.0, float(np.max(np.abs(ys)))):
            self.cost_model_warnings.append(f"pwl_cost of {label} approximated by a quadratic in QUBO")
            
        return float(c0), float(c1), float(c2)

    def _build_cost_table(self, net):
        self.cost_model_warnings = []
        table = {et: {} for et in self._DISPATCH_TYPES}

        if hasattr(net, 'poly_cost') and not net.poly_cost.empty:
            for _, row in net.poly_cost.iterrows():
                et = row['et']
                if et in table:
                    table[et][int(row['element'])] = {
                        'pwl': None, 
                        'coeffs': (self._val(row.get('cp0_eur'), 0.0), 
                                   self._val(row.get('cp1_eur_per_mw'), 1.0), 
                                   self._val(row.get('cp2_eur_per_mw2'), 0.0))
                    }

        if hasattr(net, 'pwl_cost') and not net.pwl_cost.empty:
            for _, row in net.pwl_cost.iterrows():
                et, el = row['et'], int(row['element'])
                if et not in table or el in table[et] or row.get('power_type', 'p') != 'p' or el not in getattr(net, et).index:
                    continue
                p_lo, p_hi = self._asset_bounds(net, et, el)
                segs = self._parse_pwl(row['points'], f"{et} {el}")
                table[et][el] = {'pwl': segs, 'coeffs': self._fit_quadratic(segs, p_lo, p_hi, f"{et} {el}")}

        self._fallback_c1 = max([e['coeffs'][1] for e in table['gen'].values()] + [100.0])
        self._costs = table

    def _cost_coeffs(self, et, idx):
        if entry := self._costs[et].get(int(idx)):
            return entry['coeffs']
        mult = self.ext_fallback_mult if et == 'ext_grid' else 1.0
        return 0.0, self._fallback_c1 * mult, 0.0

    def _true_cost(self, et, idx, p):
        if (entry := self._costs[et].get(int(idx))) and entry['pwl'] is not None:
            return float(self._pwl_eval(entry['pwl'], [p])[0])
        c0, c1, c2 = self._cost_coeffs(et, idx)
        return c2 * p * p + c1 * p + c0

    def _total_true_cost(self, values):
        return sum(self._true_cost(et, idx, v) for et in self._DISPATCH_TYPES for idx, v in values[et].items())

    # ==================================================================
    # Network Preprocessing
    # ==================================================================

    def _get_line_physics(self, net, line_idx):
        line = net.line.loc[line_idx]
        vn_kv = net.bus.at[int(line['from_bus']), 'vn_kv']
        sn_mva = getattr(net, 'sn_mva', 100.0)

        z_base = (vn_kv ** 2) / sn_mva
        x_pu = (line['x_ohm_per_km'] * line['length_km'] / self._val(line.get('parallel'), 1.0)) / z_base
        b_mw_rad = sn_mva / x_pu if x_pu != 0 else 0.0

        max_i_ka = self._val(line.get('max_i_ka'), None)
        max_loading = self._val(line.get('max_loading_percent'), 100.0) / 100.0

        if max_i_ka is None or max_i_ka > self.UNCONSTRAINED_I_KA:
            p_max_mw = 0.0
        else:
            p_max_mw = math.sqrt(3) * vn_kv * max_i_ka * max_loading
            if p_max_mw > self.UNCONSTRAINED_MW: p_max_mw = 0.0

        return int(line['from_bus']), int(line['to_bus']), b_mw_rad, p_max_mw

    def _get_trafo_physics(self, net, trafo_idx):
        trafo = net.trafo.loc[trafo_idx]
        sn_mva_trafo, vk_percent = trafo['sn_mva'], trafo['vk_percent']
        parallel = self._val(trafo.get('parallel'), 1.0)
        
        b_mw_rad = (sn_mva_trafo * parallel * 100.0) / vk_percent if vk_percent != 0 else 0.0
        max_loading = self._val(trafo.get('max_loading_percent'), 100.0) / 100.0
        p_max_mw = sn_mva_trafo * parallel * max_loading

        return int(trafo['hv_bus']), int(trafo['lv_bus']), b_mw_rad, p_max_mw

    def _prepare_network(self, net):
        self._active_buses = {int(b) for b in net.bus.index[net.bus.in_service.astype(bool)]}
        self._load_mw = {}
        
        if not net.load.empty:
            for _, ld in net.load[net.load.in_service.astype(bool)].iterrows():
                if (bus := int(ld['bus'])) in self._active_buses:
                    self._load_mw[bus] = self._load_mw.get(bus, 0.0) + ld['p_mw'] * self._val(ld.get('scaling'), 1.0)

        self._lines, self._max_b = [], {}
        
        def _add_edge(f, t, b, p_max, prefix, idx):
            if b != 0.0 and f in self._active_buses and t in self._active_buses:
                self._lines.append({'idx': f"{prefix}_{idx}", 'f': f, 't': t, 'b': b, 'p_max': p_max})
                self._max_b[f] = max(self._max_b.get(f, 0.0), abs(b))
                self._max_b[t] = max(self._max_b.get(t, 0.0), abs(b))

        for l_idx in net.line.index[net.line.in_service.astype(bool)]:
            _add_edge(*self._get_line_physics(net, l_idx), "line", l_idx)

        if hasattr(net, 'trafo') and not net.trafo.empty:
            for t_idx in net.trafo.index[net.trafo.in_service.astype(bool)]:
                _add_edge(*self._get_trafo_physics(net, t_idx), "trafo", t_idx)

        self._build_cost_table(net)
        self._build_ptdf(net)

        self._tight_bounds, self._line_flow_range = {}, {}
        if self.tighten_bounds:
            self._tighten_dispatch_bounds(net)

        if self.formulation == "dc_theta":
            self._angle_bounds = self._compute_angle_bounds(net)

    def _build_ptdf(self, net):
        ext = self._active(net, 'ext_grid')
        if not ext: raise ValueError("DC-OPF needs >= 1 ext_grid as angle reference.")
        self._ref_ext = int(ext[0])
        self._ref_bus = int(net.ext_grid.at[self._ref_ext, 'bus'])
        if self._ref_bus not in self._active_buses:
            raise ValueError(f"ext_grid {self._ref_ext} sits on an out-of-service bus.")

        buses = sorted(self._active_buses)
        self._bus_pos = pos = {b: i for i, b in enumerate(buses)}
        n = len(buses)

        adj = {b: [] for b in buses}
        for ln in self._lines:
            adj[ln['f']].append(ln['t'])
            adj[ln['t']].append(ln['f'])
            
        seen, stack = {self._ref_bus}, [self._ref_bus]
        while stack:
            for v in adj[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
                    
        if missing := sorted(set(buses) - seen):
            raise ValueError(f"Buses {missing} disconnected from reference bus {self._ref_bus}.")

        Bm = np.zeros((n, n))
        for ln in self._lines:
            i, j, b = pos[ln['f']], pos[ln['t']], ln['b']
            Bm[i, i] += b; Bm[j, j] += b
            Bm[i, j] -= b; Bm[j, i] -= b
            
        keep = [i for i in range(n) if i != pos[self._ref_bus]]
        X = np.zeros((n, n))
        if keep:
            try:
                X[np.ix_(keep, keep)] = np.linalg.inv(Bm[np.ix_(keep, keep)])
            except np.linalg.LinAlgError:
                X[np.ix_(keep, keep)] = np.linalg.pinv(Bm[np.ix_(keep, keep)])
        self._X = X

        self._H_full = {ln['idx']: ln['b'] * (X[pos[ln['f']]] - X[pos[ln['t']]]) for ln in self._lines}
        self._H = {}

        for ln in self._lines:
            raw = self._H_full[ln['idx']]
            thr = max(self.ptdf_threshold, (self.ptdf_rel_threshold or 0.0) * float(np.max(np.abs(raw))))
            h = np.where(np.abs(raw) < thr, 0.0, raw)
            if self.ptdf_round_to:
                h = np.round(h / self.ptdf_round_to) * self.ptdf_round_to
            self._H[ln['idx']] = h

            if getattr(self, 'decode_with_full_ptdf', False):
                full_mag, kept_mag = float(np.sum(np.abs(raw))), float(np.sum(np.abs(h)))
                if full_mag > 1e-9 and (drop_pct := 1.0 - kept_mag / full_mag) > 0.3:
                    self.cost_model_warnings.append(
                        f"{ln['idx']}: ptdf_threshold dropped {drop_pct*100:.0f}% of full PTDF magnitude."
                    )

    def _tighten_dispatch_bounds(self, net):
        assets = [(et, int(idx), int(getattr(net, et).at[idx, 'bus']), *self._asset_bounds(net, et, idx))
                  for et in self._DISPATCH_TYPES for idx in self._active(net, et)
                  if int(getattr(net, et).at[idx, 'bus']) in self._active_buses]
        
        n = len(assets)
        if not n: return
        
        bounds = [(a[3], a[4]) for a in assets]
        A_eq, b_eq = np.ones((1, n)), [sum(self._load_mw.values())]
        
        def flow_row(h):
            return -sum(h[self._bus_pos[b]] * ld for b, ld in self._load_mw.items()), np.array([h[self._bus_pos[a[2]]] for a in assets])
            
        def lp(c, A=None, b=None, bnds=bounds):
            return linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")

        H_phys = self._H_full if self.formulation == "dc_theta" or self.decode_with_full_ptdf else self._H
        rows, rhs = [], []
        
        for ln in self._lines:
            if ln['p_max'] > 0.0:
                base, coef = flow_row(H_phys[ln['idx']])
                rows.extend([coef, -coef])
                rhs.extend([ln['p_max'] - base, ln['p_max'] + base])
                
        A_ub, b_ub = (np.array(rows), np.array(rhs)) if rows else (None, None)

        if lp(np.zeros(n), A_ub, b_ub).status != 0:
            self.cost_model_warnings.append("bound tightening skipped: constraint-only LP infeasible")
            return

        n_tight, n_free = 0, 0
        for k, (et, idx, bus, lo, hi) in enumerate(assets):
            c = np.zeros(n); c[k] = 1.0
            r_min, r_max = lp(c, A_ub, b_ub), lp(-c, A_ub, b_ub)
            new_lo = max(lo, r_min.fun - 1e-7) if r_min.status == 0 else lo
            new_hi = min(hi, -r_max.fun + 1e-7) if r_max.status == 0 else hi
            new_hi = max(new_hi, new_lo)
            if new_lo > lo + 1e-9 or new_hi < hi - 1e-9: n_tight += 1
            self._tight_bounds[(et, idx)] = (new_lo, new_hi)

        tb = [self._tight_bounds[(a[0], a[1])] for a in assets]
        H_qubo = self._H if self.formulation == "dc_ptdf" else self._H_full
        
        for ln in self._lines:
            if ln['p_max'] > 0.0:
                base, coef = flow_row(H_qubo[ln['idx']])
                r_min, r_max = lp(coef, bnds=tb), lp(-coef, bnds=tb)
                if r_min.status == 0 and r_max.status == 0:
                    rng = (base + r_min.fun, base - r_max.fun)
                    self._line_flow_range[ln['idx']] = rng
                    if rng[0] >= -ln['p_max'] - 1e-9 and rng[1] <= ln['p_max'] + 1e-9: n_free += 1
                    
        self.presolve_info = {"dispatch_bounds_tightened": n_tight, "lines_never_binding": n_free}

    def _compute_angle_bounds(self, net):
        total_load = max(sum(self._load_mw.values()), self.mw_precision)
        slack = {self._ref_bus} if self.single_reference_angle else {int(net.ext_grid.at[i, 'bus']) for i in self._active(net, 'ext_grid')}
        adj = {b: [] for b in self._active_buses} if self.single_reference_angle else {int(b): [] for b in net.bus.index}

        for ln in self._lines:
            w = (ln['p_max'] if ln['p_max'] > 0.0 else total_load) / abs(ln['b'])
            adj[ln['f']].append((ln['t'], w))
            adj[ln['t']].append((ln['f'], w))

        dist = {b: 0.0 for b in slack}
        heap = [(0.0, b) for b in slack]
        
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf): continue
            for v, w in adj.get(u, []):
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(heap, (d + w, v))

        return {int(b): dist[int(b)] * self.ANGLE_HEADROOM if int(b) in dist and int(b) not in slack else 0.0 for b in net.bus.index}

    def _interval_angle_ranges(self, net):
        pos, X, n = self._bus_pos, self._X, len(self._bus_pos)
        inj_lo, inj_hi = np.zeros(n), np.zeros(n)
        
        for et, idx, reg, bus in self._dispatch_assets(net):
            lo, hi = self._reg_span(reg)
            inj_lo[pos[bus]] += lo; inj_hi[pos[bus]] += hi
            
        for bus, load in self._load_mw.items():
            inj_lo[pos[bus]] -= load; inj_hi[pos[bus]] -= load
            
        a, b = X * inj_lo[None, :], X * inj_hi[None, :]
        lo, hi = np.minimum(a, b).sum(axis=1), np.maximum(a, b).sum(axis=1)
        return {bus: (float(lo[i]), float(hi[i])) for bus, i in pos.items()}

    def _encode_angles(self, net):
        ranges = self._interval_angle_ranges(net) if getattr(self, 'tight_angle_bounds', True) else {}
        self.angle_bound_info = {}
        total = 0
        
        for bus in (int(b) for b in net.bus.index):
            d, max_b = self._angle_bounds.get(bus, 0.0), self._max_b.get(bus, 0.0)
            lo = hi = prec = bits_path = n = 0.0
            
            if d > 0.0 and max_b > 0.0:
                prec = self.angle_precision if self.angle_precision is not None else self.mw_precision / max_b
                if bus in ranges:
                    r_lo, r_hi = ranges[bus]
                    c, half = 0.5 * (r_lo + r_hi), 0.5 * (r_hi - r_lo) * self.ANGLE_HEADROOM
                    lo, hi = max(c - half, -d), min(c + half, d)
                    if hi < lo: lo, hi = c - half, c + half
                else:
                    lo, hi = -d, d
                    
            if hi > lo and prec > 0.0:
                n = self._register('bus', bus, f"bus_{bus}", lo, hi, prec)
                bits_path = len(self._get_weights(2.0 * d, prec))
            else:
                n = self._register('bus', bus, f"bus_{bus}", lo, lo, 0.0)
                
            total += n
            self.angle_bound_info[bus] = {"path_half_width": d, "range": (lo, hi), "bits_path_only": bits_path, "bits_tight": n}
        return total

    def _encode_variables(self, net):
        self.var_registry = {'gen': {}, 'sgen': {}, 'ext_grid': {}, 'bus': {}, 'slack_lines': {}}
        total_qubits = 0

        for et in self._DISPATCH_TYPES:
            for idx in self._active(net, et):
                if int(getattr(net, et).at[idx, 'bus']) in self._active_buses:
                    p_min, p_max = self._tight_bounds.get((et, int(idx))) or self._asset_bounds(net, et, idx)
                    total_qubits += self._register(et, int(idx), f"{self._BIT_PREFIX[et]}_{idx}", p_min, p_max, self.mw_precision)

        if self.formulation == "dc_theta":
            total_qubits += self._encode_angles(net)
            
        return total_qubits

    # ==================================================================
    # Penalty Resolution
    # ==================================================================

    def _resolve_penalties(self, net):
        bus_costs, marginals = {}, []
        for et, idx, reg, bus in self._dispatch_assets(net):
            _, c1, c2 = self._cost_coeffs(et, idx)
            c_val = abs(c1) + 2.0 * abs(c2) * reg['max']
            marginals.append(c_val)
            bus_costs[bus] = max(bus_costs.get(bus, 0.0), c_val)

        c_max_global = max(marginals + [1.0])
        if self.penalty_mode == "dual":
            if (price := self._reference_lmp_max(net)) is not None:
                c_max_global = min(c_max_global, max(price, 1.0))
                
        self.lambda_balance = self.penalty_balance if self.penalty_balance is not None else self.penalty_safety * c_max_global / self.mw_precision
        self.lambda_line = {}
        
        for ln in self._lines:
            if self.penalty_line is not None:
                self.lambda_line[ln['idx']] = self.penalty_line
            else:
                if self.formulation == "dc_ptdf":
                    h = self._H.get(ln['idx'], [])
                    local_marginals = [c for bus, c in bus_costs.items() if self._bus_pos[bus] < len(h) and abs(h[self._bus_pos[bus]]) > 1e-9]
                else:
                    local_marginals = [bus_costs.get(ln['f'], 0.0), bus_costs.get(ln['t'], 0.0)]
                    
                local_c = max([m for m in local_marginals if m >= 1.0], default=1.0)
                self.lambda_line[ln['idx']] = self.penalty_safety * local_c / self.mw_precision

    def _reference_lmp_max(self, net):
        try:
            tmp = copy.deepcopy(net)
            pp.rundcopp(tmp)
            lam = float(tmp.res_bus["lam_p"].abs().max())
            return lam if np.isfinite(lam) else None
        except Exception:
            return None

    # ==================================================================
    # QUBO Construction
    # ==================================================================
    
    @staticmethod
    def _add_polynomial_cost(bqm, reg, c0, c1, c2):
        p_min, bits, weights = reg['min'], reg['bits'], reg['weights']
        bqm.offset += (c2 * p_min ** 2) + (c1 * p_min) + c0
        for k, bit in enumerate(bits):
            bqm.add_linear(bit, (2 * c2 * p_min * weights[k]) + (c2 * weights[k] ** 2) + (c1 * weights[k]))
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                bqm.add_quadratic(bits[i], bits[j], 2 * c2 * weights[i] * weights[j])

    def _add_squared_penalty(self, bqm, terms_dict, constant, penalty_weight):
        terms = [(bit, w) for bit, w in terms_dict.items() if w != 0.0]
        bqm.offset += penalty_weight * (constant ** 2)
        for bit, w in terms:
            bqm.add_linear(bit, penalty_weight * ((2 * constant * w) + (w ** 2)))
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                bqm.add_quadratic(terms[i][0], terms[j][0], penalty_weight * 2 * terms[i][1] * terms[j][1])

    def _build_objective(self, bqm, net):
        for et in self._DISPATCH_TYPES:
            for idx, reg in self.var_registry.get(et, {}).items():
                self._add_polynomial_cost(bqm, reg, *self._cost_coeffs(et, idx))
        return bqm

    def _build_power_balance_ptdf(self, bqm, net):
        terms, const = {}, -sum(self._load_mw.values())
        for _, _, reg, _ in self._dispatch_assets(net):
            const += reg['min']
            for bit, w in zip(reg['bits'], reg['weights']):
                terms[bit] = terms.get(bit, 0.0) + w
        self._add_squared_penalty(bqm, terms, const, self.lambda_balance)
        return bqm

    def _build_line_limits_ptdf(self, bqm, net):
        pos, assets = self._bus_pos, list(self._dispatch_assets(net))
        slack_prec = self.mw_precision * self.slack_precision_factor

        for ln in self._lines:
            if ln['p_max'] <= 0.0: continue
            h, p_max = self._H[ln['idx']], ln['p_max']
            base_flow = -sum(h[pos[b]] * ld for b, ld in self._load_mw.items())
            
            if not self.scale_line_constraints:
                max_f = min_f = base_flow
                for _, _, reg, bus in assets:
                    val1, val2 = h[pos[bus]] * reg['min'], h[pos[bus]] * reg['max']
                    max_f += max(val1, val2); min_f += min(val1, val2)
                    
                if max_f <= p_max and min_f >= -p_max: continue
                
                terms, const = {}, p_max + base_flow
                for _, _, reg, bus in assets:
                    const += h[pos[bus]] * reg['min']
                    for bit, w in zip(reg['bits'], reg['weights']):
                        terms[bit] = terms.get(bit, 0.0) + h[pos[bus]] * w
                        
                ub = min(p_max + max_f, 2.0 * p_max)
                self._line_slack_sign[ln['idx']] = 1.0
                self._register('slack_lines', ln['idx'], f"slack_line_{ln['idx']}", 0.0, ub, self.mw_precision)
                slack = self.var_registry['slack_lines'][ln['idx']]
                for bit, w in zip(slack['bits'], slack['weights']):
                    terms[bit] = terms.get(bit, 0.0) - w
                self._add_squared_penalty(bqm, terms, const, self.lambda_line[ln['idx']])
                continue

            max_f = min_f = base_flow
            coefs, marg = [], []
            for et, idx, reg, bus in assets:
                c = h[pos[bus]]
                v1, v2 = c * reg['min'], c * reg['max']
                max_f += max(v1, v2); min_f += min(v1, v2)
                if c != 0.0 and reg['bits']:
                    coefs.append(abs(c))
                    c1, c2 = self._cost_coeffs(et, idx)[1:]
                    marg.append(abs(c1) + 2.0 * abs(c2) * reg['max'])

            if (lp_rng := self._line_flow_range.get(ln['idx'])):
                min_f, max_f = max(min_f, lp_rng[0]), min(max_f, lp_rng[1])

            upper, lower = max_f > p_max + 1e-9, min_f < -p_max - 1e-9
            if not (upper or lower): continue
            if not coefs: continue

            sgn, ub = (1.0, (2.0 * p_max if (upper and lower) else p_max + max_f)) if (upper and lower or (lower and not upper)) else (-1.0, p_max - min_f)
            if not self.smart_slack_side: sgn, ub = 1.0, min(p_max + max_f, 2.0 * p_max)
            if ub <= 1e-9: continue

            sigma = max(coefs)
            self.lambda_line[ln['idx']] = self.penalty_line if self.penalty_line is not None else self.penalty_safety * max(marg) / self.mw_precision
            
            terms, const = {}, p_max / sigma + sgn * base_flow / sigma
            for _, _, reg, bus in assets:
                if (c := sgn * h[pos[bus]] / sigma) == 0.0: continue
                const += c * reg['min']
                for bit, w in zip(reg['bits'], reg['weights']): terms[bit] = terms.get(bit, 0.0) + c * w

            self._line_slack_sign[ln['idx']] = sgn
            self._register('slack_lines', ln['idx'], f"slack_line_{ln['idx']}", 0.0, ub / sigma, slack_prec)
            slack = self.var_registry['slack_lines'][ln['idx']]
            for bit, w in zip(slack['bits'], slack['weights']): terms[bit] = terms.get(bit, 0.0) - w
            self._add_squared_penalty(bqm, terms, const, self.lambda_line[ln['idx']])
            
        return bqm

    def _build_power_balance_theta(self, bqm, net):
        terms = {b: {} for b in self._active_buses}
        const = {b: -self._load_mw.get(b, 0.0) for b in self._active_buses}

        def add(bus, reg, coeff):
            const[bus] += coeff * reg['min']
            for bit, w in zip(reg['bits'], reg['weights']):
                terms[bus][bit] = terms[bus].get(bit, 0.0) + coeff * w

        for _, _, reg, bus in self._dispatch_assets(net): add(bus, reg, 1.0)
        
        for ln in self._lines:
            f, t, b = ln['f'], ln['t'], ln['b']
            reg_f, reg_t = self.var_registry['bus'][f], self.var_registry['bus'][t]
            add(f, reg_f, -b); add(f, reg_t, +b)
            add(t, reg_f, +b); add(t, reg_t, -b)

        for bus in sorted(self._active_buses):
            self._add_squared_penalty(bqm, terms[bus], const[bus], self.lambda_balance)
        return bqm

    def _build_line_limits_theta(self, bqm, net):
        slack_prec = self.mw_precision * self.slack_precision_factor
        
        for ln in self._lines:
            if ln['p_max'] <= 0.0: continue
            f, t, b, p_max = ln['f'], ln['t'], ln['b'], ln['p_max']
            reg_f, reg_t = self.var_registry['bus'][f], self.var_registry['bus'][t]
            
            if not self.theta_smart_slack:
                max_flow = abs(b) * (reg_f['max'] - reg_t['min'])
                min_flow = -abs(b) * (reg_f['max'] - reg_t['min'])
                if max_flow <= p_max and min_flow >= -p_max: continue
                
                ub = min(p_max + max_flow, 2.0 * p_max)
                terms, const = {}, p_max + b * reg_f['min'] - b * reg_t['min']
                for k, bit in enumerate(reg_f['bits']): terms[bit] = terms.get(bit, 0.0) + b * reg_f['weights'][k]
                for k, bit in enumerate(reg_t['bits']): terms[bit] = terms.get(bit, 0.0) - b * reg_t['weights'][k]
                
                self._line_slack_sign[ln['idx']] = 1.0
                self._register('slack_lines', ln['idx'], f"slack_line_{ln['idx']}", 0.0, ub, self.mw_precision)
                slack = self.var_registry['slack_lines'][ln['idx']]
                for bit, w in zip(slack['bits'], slack['weights']): terms[bit] = terms.get(bit, 0.0) - w
                self._add_squared_penalty(bqm, terms, const, self.lambda_line[ln['idx']])
                continue

            f_lo, f_hi = self._reg_span(reg_f)
            t_lo, t_hi = self._reg_span(reg_t)
            fl_lo, fl_hi = (b * (f_lo - t_hi), b * (f_hi - t_lo)) if b >= 0 else (b * (f_hi - t_lo), b * (f_lo - t_hi))

            if (lp_rng := self._line_flow_range.get(ln['idx'])):
                fl_lo, fl_hi = max(fl_lo, lp_rng[0]), min(fl_hi, lp_rng[1])

            upper, lower = fl_hi > p_max + 1e-9, fl_lo < -p_max - 1e-9
            if not (upper or lower): continue

            sgn, ub = (1.0, 2.0 * p_max) if upper and lower else (1.0, p_max + fl_hi) if lower else (-1.0, p_max - fl_lo)
            if ub <= 1e-9: continue

            const = p_max + sgn * b * (reg_f['min'] - reg_t['min'])
            terms = {}
            for bit, w in zip(reg_f['bits'], reg_f['weights']): terms[bit] = terms.get(bit, 0.0) + sgn * b * w
            for bit, w in zip(reg_t['bits'], reg_t['weights']): terms[bit] = terms.get(bit, 0.0) - sgn * b * w

            self._line_slack_sign[ln['idx']] = sgn
            self._register('slack_lines', ln['idx'], f"slack_line_{ln['idx']}", 0.0, ub, slack_prec)
            slack = self.var_registry['slack_lines'][ln['idx']]
            for bit, w in zip(slack['bits'], slack['weights']): terms[bit] = terms.get(bit, 0.0) - w
            self._add_squared_penalty(bqm, terms, const, self.lambda_line[ln['idx']])
            
        return bqm

    # ==================================================================
    # Public & Solution Interfaces
    # ==================================================================

    def _formulate_qubo(self, net):
        self._line_slack_sign = {}
        self._prepare_network(net)
        self._encode_variables(net)
        self._resolve_penalties(net)

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        bqm = self._build_objective(bqm, net)
        
        if self.formulation == "dc_ptdf":
            bqm = self._build_power_balance_ptdf(bqm, net)
            bqm = self._build_line_limits_ptdf(bqm, net)
        else:
            bqm = self._build_power_balance_theta(bqm, net)
            bqm = self._build_line_limits_theta(bqm, net)

        R = self.var_registry
        n_vars = len(bqm.variables)
        max_int = (n_vars * (n_vars - 1)) / 2
        
        mags = [abs(v) for v in bqm.linear.values() if abs(v) > 1e-12] + [abs(v) for v in bqm.quadratic.values() if abs(v) > 1e-12]
        max_mag, min_mag = max(mags, default=0.0), min(mags, default=0.0)
        
        complexity = {
            "formulation": self.formulation,
            "encoding": self.encoding,
            "classical_domain_input": {
                "continuous_variables": sum(len(R[et]) for et in self._DISPATCH_TYPES) + sum(1 for r in R.get('bus', {}).values() if r['bits']),
                "equality_constraints": 1 if self.formulation == "dc_ptdf" else len(self._active_buses),
                "inequality_constraints": 2 * sum(1 for ln in self._lines if ln['p_max'] > 0.0),
            },
            "quantum_domain_qubo": {
                "total_logical_qubits": n_vars,
                "qubits_used_for_variables": n_vars - sum(len(r['bits']) for r in R.get('slack_lines', {}).values()),
                "qubits_used_for_dispatch": sum(len(r['bits']) for et in self._DISPATCH_TYPES for r in R.get(et, {}).values()),
                "qubits_used_for_angles": sum(len(r['bits']) for r in R.get('bus', {}).values()),
                "qubits_wasted_on_slack": sum(len(r['bits']) for r in R.get('slack_lines', {}).values()),
                "num_interactions": bqm.num_interactions,
                "density_percent": round((bqm.num_interactions / max_int * 100) if max_int > 0 else 0.0, 2),
                "offset": float(bqm.offset),
            },
            "hardware_limits": {
                "max_coefficient_magnitude": round(float(max_mag), 4),
                "min_coefficient_magnitude": round(float(min_mag), 4),
                "dac_precision_warning": bool((max_mag / min_mag if min_mag > 0 else 1.0) > 256.0)
            },
            "penalties": {
                "balance": float(self.lambda_balance),
                "line": self.lambda_line if isinstance(self.lambda_line, dict) else float(self.lambda_line)
            },
        }
        
        ising = bqm.change_vartype(dimod.SPIN, inplace=False)
        i_mags = [abs(b) for b in ising.linear.values() if abs(b) > 1e-12] + [abs(b) for _, _, b in ising.iter_quadratic() if abs(b) > 1e-12]
        if i_mags:
            mx, mn = max(i_mags), min(i_mags)
            complexity["hardware_limits"].update({
                "ising_max": round(mx, 4), "ising_min": round(mn, 6),
                "ising_dynamic_range": round(mx / mn, 2),
                "ising_at_risk_percent": round(100 * sum(1 for m in i_mags if m < self.risk_noise_frac * mx) / len(i_mags), 2)
            })

        return bqm, complexity

    def _decode_solution(self, sample, net):
        truncated = getattr(self, '_H', {})
        if getattr(self, 'decode_with_full_ptdf', False) and hasattr(self, '_H_full'):
            self._H = self._H_full

        try:
            raw = {et: {idx: self._decode_value(reg, sample) for idx, reg in self.var_registry.get(et, {}).items()} for et in self._DISPATCH_TYPES}
            pos, inj, bounds_ok = self._bus_pos, np.zeros(len(self._bus_pos)), True
            
            for et, idx, reg, bus in self._dispatch_assets(net):
                val = raw[et][idx]
                if val < reg['min'] - 1e-6 or val > reg['max'] + 1e-6: bounds_ok = False
                inj[pos[bus]] += val
                
            for bus, load in self._load_mw.items(): inj[pos[bus]] -= load
            
            imbalance = float(inj.sum())
            reported, rebalance_ok = {et: dict(v) for et, v in raw.items()}, True
            raw_ref = raw['ext_grid'][self._ref_ext]
            
            if self.rebalance_slack:
                new_ref = raw_ref - imbalance
                reported['ext_grid'][self._ref_ext] = new_ref
                ref_reg = self.var_registry['ext_grid'][self._ref_ext]
                rebalance_ok = ref_reg['min'] - 1e-6 <= new_ref <= ref_reg['max'] + 1e-6
                inj[pos[int(net.ext_grid.at[self._ref_ext, 'bus'])]] += (new_ref - raw_ref)

            flows = {idx: float(h @ inj) for idx, h in self._H.items()}
            angles = self._X @ inj
            max_line_violation = max([abs(flows[ln['idx']]) - ln['p_max'] for ln in self._lines if ln['p_max'] > 0.0] + [0.0])

            tol = self.feasibility_tol_mw if self.feasibility_tol_mw is not None else self.mw_precision
            balance_ok, lines_ok = abs(imbalance) <= tol, max_line_violation <= tol

            feasibility = {
                "total_generation_mw": round(float(sum(sum(v.values()) for v in reported.values())), 3),
                "total_load_mw": round(float(sum(self._load_mw.values())), 3),
                "raw_imbalance_mw": round(imbalance, 3),
                "cost_raw_dispatch_eur": round(self._total_true_cost(raw), 3),
                "max_line_violation_mw": round(float(max_line_violation), 3),
                "tolerance_mw": tol,
                "is_feasible": bool(balance_ok and lines_ok and bounds_ok and rebalance_ok),
                "line_flows_mw": {k: round(v, 3) for k, v in flows.items()},
                "bus_angles_rad": {b: round(float(angles[i]), 6) for b, i in pos.items()},
            }

            dispatch = {i: {'p_mw': round(v, 3)} for i, v in reported['gen'].items()}
            sgen_dispatch = {i: {'p_mw': round(v, 3)} for i, v in reported['sgen'].items()}
            slack_dispatch = {i: {'p_mw': round(v, 3)} for i, v in reported['ext_grid'].items()}

            return dispatch, sgen_dispatch, slack_dispatch, round(self._total_true_cost(reported), 3), feasibility
        finally:
            self._H = truncated

    # ==================================================================
    # Reporting
    # ==================================================================
    
    @staticmethod
    def _fmt_weights(weights, max_show=10):
        w = [f"{x:.4g}" for x in weights]
        return ", ".join(w[:4] + ["..."] + w[-2:]) if len(w) > max_show else ", ".join(w)

    @staticmethod
    def _variable_symbol(group, idx):
        return {"gen": f"P_gen{idx}", "sgen": f"P_sgen{idx}", "ext_grid": f"P_ext{idx}", "bus": f"theta_{idx}"}[group]

    def _print_problem_parameters(self, net, complexity):
        line = "=" * 80
        ptdf = self.formulation == "dc_ptdf"
        print("\n" + line)
        print(f"   QUBO {self.formulation.upper()}-OPF: COMPLETE PROBLEM DEFINITION")
        print(line)
        print(f"Total Grid Demand (Load): {round(sum(self._load_mw.values()), 3)} MW")
        print(f"MW precision: {self.mw_precision} MW | Encoding: {self.encoding} | Reference bus: {self._ref_bus}\n")

        labels = {'gen': 'Gen', 'sgen': 'SGen', 'ext_grid': 'Ext_Grid (Slack)'}

        # 1) objective ------------------------------------------------------
        print("--- OBJECTIVE: minimise total generation cost ---")
        for et in self._DISPATCH_TYPES:
            tbl = getattr(net, et)
            for idx in self.var_registry.get(et, {}):
                c0, c1, c2 = self._cost_coeffs(et, idx)
                sym = self._variable_symbol(et, idx)
                entry = self._costs[et].get(int(idx))
                note = "   [fallback cost: none given in net]" if entry is None else (
                    "   [pwl_cost -> quadratic fit in QUBO]" if entry['pwl'] is not None else "")
                print(f" {labels[et]} {idx} (Bus {tbl.at[idx, 'bus']}): "
                      f"{c2:.6g}*{sym}^2 + {c1:.6g}*{sym} + {c0:.6g}{note}")
        for w in self.cost_model_warnings:
            print(f" [cost warning] {w}")
        print(" QUBO energy = sum(costs) + lambda_balance * (balance residual)^2"
              + (" + lambda_line * sum_lines(line residual)^2"))
        print(" Reported cost = exact cost function of the physically balanced operating point.")

        # 2) variables ------------------------------------------------------
        print(f"\n--- DECISION VARIABLES ({self.encoding} encoding: value = min + sum_k w_k * bit_k) ---")
        for et in self._DISPATCH_TYPES:
            for idx, reg in self.var_registry.get(et, {}).items():
                sym = self._variable_symbol(et, idx)
                print(f" {sym:<10} in [{reg['min']}, {reg['max']}] MW | {len(reg['bits']):>4} bits | "
                      f"weights: {self._fmt_weights(reg['weights'])}")
        for bus, reg in self.var_registry.get('bus', {}).items():
            sym = self._variable_symbol('bus', bus)
            if reg['bits']:
                print(f" {sym:<10} in [{reg['min']:.4f}, {reg['max']:.4f}] rad | {len(reg['bits']):>4} bits | "
                      f"step {reg['precision']:.3g} rad")
            elif abs(reg['min']) > 1e-12:
                print(f" {sym:<10} fixed at {reg['min']:.4f} rad (no freedom left)")
            else:
                print(f" {sym:<10} fixed at 0 (slack reference or no lines)")
        for l_idx, reg in self.var_registry.get('slack_lines', {}).items():
            print(f" s_line{l_idx:<4} in [0, {reg['max']:.4g}] MW | {len(reg['bits']):>4} bits | "
                  f"(inequality slack, not a physical variable)")
        if ptdf:
            print(" (no angle variables: angles are recovered afterwards as theta = X @ injections)")

        # 3) network and constraints ---------------------------------------
        if ptdf:
            print(f"\n--- TRANSMISSION LINES (PTDF: flow_l = sum_i PTDF[l,i] * (P_i - load_i), bus {self._ref_bus} = reference) ---")
        else:
            print("\n--- TRANSMISSION LINES (flow_ij = B_ij * (theta_i - theta_j)) ---")
        inv_pos = {i: b for b, i in self._bus_pos.items()}
        for ln in self._lines:
            limit_str = f"|flow| <= {round(ln['p_max'], 2)} MW" if ln['p_max'] > 0 else "unconstrained"
            print(f" Line {ln['idx']} ({ln['f']} -> {ln['t']}): B = {round(ln['b'], 2)} MW/rad, {limit_str}")
            if ptdf and ln['p_max'] > 0:
                row = ", ".join(f"bus{inv_pos[i]} {v:+.4f}" for i, v in enumerate(self._H[ln['idx']]) if abs(v) > 1e-9)
                print(f"     PTDF: {row}")

        print("\n--- CONSTRAINTS (soft, as quadratic penalties) ---")
        if ptdf:
            syms = " + ".join(self._variable_symbol(et, idx) for et in self._DISPATCH_TYPES for idx in self.var_registry.get(et, {}))
            print(f" Balance: {syms} - {round(sum(self._load_mw.values()), 3)} MW load = 0")
        else:
            for bus in sorted(self._active_buses):
                inj = [self._variable_symbol(et, idx) for et, idx, reg, b in self._dispatch_assets(net) if b == bus]
                flows = []
                for ln in self._lines:
                    if ln['f'] == bus:
                        flows.append(f"flow({ln['f']}->{ln['t']})")
                    elif ln['t'] == bus:
                        flows.append(f"flow({ln['t']}->{ln['f']})")
                load = round(self._load_mw.get(bus, 0.0), 3)
                print(f" KCL bus {bus}: {' + '.join(inj) if inj else '0'} - {load} MW load "
                      f"- outgoing[{', '.join(flows)}] = 0")
        for l_idx in self.var_registry.get('slack_lines', {}):
            ln = next(x for x in self._lines if x['idx'] == l_idx)
            op = "+" if self._line_slack_sign.get(l_idx, 1.0) > 0 else "-"
            print(f" Line {l_idx}: {round(ln['p_max'], 2)} {op} flow({ln['f']}->{ln['t']}) - s_line{l_idx} = 0")

        auto_b = "auto" if self.penalty_balance is None else "manual"
        auto_l = "auto" if self.penalty_line is None else "manual"

        if isinstance(self.lambda_line, dict):
            l_min = min(self.lambda_line.values()) if self.lambda_line else 0.0
            l_max = max(self.lambda_line.values()) if self.lambda_line else 0.0
            line_str = f"[{l_min:.2f} to {l_max:.2f}]"
        else:
            line_str = f"{self.lambda_line:.6g}"

        print(f" lambda_balance = {self.lambda_balance:.6g} ({auto_b}) | lambda_line = {line_str} ({auto_l})")

        # 4) size summary ---------------------------------------------------
        c, q = complexity['classical_domain_input'], complexity['quantum_domain_qubo']
        print("\n--- PROBLEM SIZE ---")
        print(f" Classical: {c['continuous_variables']} continuous variables, "
              f"{c['equality_constraints']} equality and {c['inequality_constraints']} inequality constraints")
        print(f" QUBO ({self.encoding}): {q['total_logical_qubits']} logical qubits "
              f"(dispatch {q['qubits_used_for_dispatch']}, angles {q['qubits_used_for_angles']}, "
              f"line-limit slack {q['qubits_wasted_on_slack']}), "
              f"{q['num_interactions']} quadratic terms, constant offset {q['offset']:.6g}")
        print(line + "\n")

    def view_problem_definition(self, net):
        bqm, complexity = self._formulate_qubo(net)
        self._print_problem_parameters(net, complexity)
        return bqm, complexity

# ===============================================================================================================
#                                            Execution Block
# ===============================================================================================================
if __name__ == "__main__":
    import time

    import pandapower as pp

    # Adjust import paths if needed based on your directory structure
    from homemade_grids.small_grids import case4_low_gen
    from solvers.simulated_quantum_annealing.s1_hardware import Hardware
    from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
    from solvers.simulated_quantum_annealing.run_emulator import EmulatedQPU

    # ----------------------------------------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------------------------------------
    GRID_FACTORY = case4_low_gen
    BASE = dict(formulation="dc_ptdf", encoding="radix", mw_precision=10.0, ptdf_threshold=0.05)
    SHOW_EXPANSION = True                       # full algebraic problem definition for baseline ptdf and theta
    RUN_SECTIONS = ("3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8")   # drop entries to run fewer sections
    SEED = 42
    GAP_OK = 1.0                                # percent; "acceptable" threshold for the final ranking
    PRECISIONS = (50.0, 25.0, 20.0, 10.0, 5.0)
    PENALTY_SAFETY = (0.25, 0.5, 1.0, 2.0, 5.0, 20.0)
    MU_VALUES = (1, 2, 4, 8, 16, 64)
    FORMS = ("dc_ptdf", "dc_theta")

    # annealing emulator
    HW_FILE = None                              # path of a saved Hardware (Hardware.save); None -> ideal Zephyr Z(12,4)
    EMBED_TIMEOUT = 120.0                       # seconds per minorminer seed (busclique clique embedding has no timeout)
    EMU_QPU_KW = dict(sampler="sa", sampler_kwargs=dict(num_sweeps=1000),
                      dac_bits=5, ice_sigma_h=0.02, ice_sigma_j=0.01, use_extended_j=True)
    EMU_SAMPLE_KW = dict(num_reads=200, reads_per_programming=50, chain_break_method="majority",
                         postprocess=None)      # None = raw annealer output; "logical" / "both" add steepest descent

    # logical (unembedded, noise-free) SA = the "mathematical ceiling" of each formulation
    LOGICAL_CEILING = True
    LOGICAL_READS, LOGICAL_SWEEPS = 200, 2000
    MAX_DECODE = 400                            # decode at most this many distinct samples (lowest energy first)

    SA = None
    if LOGICAL_CEILING:
        try:
            from dwave.samplers import SimulatedAnnealingSampler as SA
        except ImportError:
            try:
                from neal import SimulatedAnnealingSampler as SA
            except ImportError:
                print("[note] no SA sampler found (dwave-samplers / neal): logical-ceiling columns are skipped")

    # ----------------------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------------------
    def short(form):
        return form.replace("dc_", "")

    def describe_grid(net):
        print("\n--- LOADS (Demand) ---")
        load_cols = [c for c in ['bus', 'p_mw', 'scaling', 'in_service'] if c in net.load.columns]
        print(net.load[load_cols].to_string())

        print("\n--- GENERATORS (Supply) ---")
        if not net.ext_grid.empty:
            print("\nExt Grid (Slack Reference):")
            ext_cols = [c for c in ['bus', 'min_p_mw', 'max_p_mw'] if c in net.ext_grid.columns]
            print(net.ext_grid[ext_cols].to_string())
        if not net.gen.empty:
            print("\nStandard Generators:")
            gen_cols = [c for c in ['bus', 'min_p_mw', 'max_p_mw'] if c in net.gen.columns]
            print(net.gen[gen_cols].to_string())

        print("\n--- TRANSMISSION LINES (Routing) ---")
        line_cols = [c for c in ['from_bus', 'to_bus', 'length_km', 'x_ohm_per_km', 'max_i_ka',
                                 'max_loading_percent'] if c in net.line.columns]
        print(net.line[line_cols].to_string())

        print("\n--- COSTS (Economics) ---")
        if hasattr(net, 'poly_cost') and not net.poly_cost.empty:
            print("\nPolynomial Costs:")
            poly_cols = [c for c in ['et', 'element', 'cp0_eur', 'cp1_eur_per_mw', 'cp2_eur_per_mw2']
                         if c in net.poly_cost.columns]
            print(net.poly_cost[poly_cols].to_string())
        if hasattr(net, 'pwl_cost') and not net.pwl_cost.empty:
            print("\nPiecewise Linear Costs:")
            pwl_cols = [c for c in ['et', 'element', 'points'] if c in net.pwl_cost.columns]
            print(net.pwl_cost[pwl_cols].to_string())

    def reference_solution(net):
        """Continuous classical optimum (pandapower DC-OPF): the yardstick for every gap below."""
        try:
            pp.rundcopp(net)
            cost = float(net.res_cost)
            print(f"\nClassical DC-OPF (pandapower rundcopp) optimum: {cost:.3f} EUR/h")
            if not net.res_gen.empty:
                print(net.res_gen[['p_mw']].to_string())
            if not net.res_ext_grid.empty:
                print(net.res_ext_grid[['p_mw']].to_string())
            return cost
        except Exception as exc:
            print(f"[warn] classical reference failed ({type(exc).__name__}: {exc}); gap columns will be empty")
            return None

    def evaluate(f, bqm, net, sampleset, ref_cost):
        """Decode distinct samples in increasing QUBO energy (energies recomputed on the formulated BQM, so they are
        comparable across solvers). Same selection rule as the solvers: best = lowest-energy FEASIBLE sample.
        Returns feasible share (% of reads), best feasible cost, gap (%) to the classical optimum, and whether the
        lowest-energy sample is feasible."""
        labels = list(sampleset.variables)
        arr = np.asarray(sampleset.record.sample)
        occ = np.asarray(sampleset.record.num_occurrences)
        agg = dimod.SampleSet.from_samples_bqm((arr, labels), bqm, num_occurrences=occ).aggregate()
        total = feas = decoded = 0
        best_cost, lowest_ok = None, None
        for datum in agg.data(['sample', 'num_occurrences'], sorted_by='energy'):
            if decoded >= MAX_DECODE:
                break
            _, _, _, cost, fe = f._decode_solution(dict(datum.sample), net)
            ok = bool(fe['is_feasible'])
            if lowest_ok is None:
                lowest_ok = ok
            decoded += 1
            total += int(datum.num_occurrences)
            if ok:
                feas += int(datum.num_occurrences)
                if best_cost is None:
                    best_cost = cost
        gap = None
        if best_cost is not None and ref_cost:
            gap = 100.0 * (best_cost - ref_cost) / abs(ref_cost)
        return {"feas": 100.0 * feas / total if total else None, "gap": gap, "lowest_ok": lowest_ok}

    # ---- hardware, embedding cache -------------------------------------------------------------
    HW = Hardware.load(HW_FILE) if HW_FILE else Hardware.ideal_zephyr(m=12, t=4)
    print(HW.summary())
    EMB_CACHE = {}

    def embed(bqm):
        """Best embedding of the BQM's interaction graph. Cached: configurations that share the same graph
        (e.g. every emulator-knob row) reuse one embedding instead of paying the search again."""
        key = (tuple(sorted((str(v) for v in bqm.variables))),
               tuple(sorted(tuple(sorted((str(u), str(v)))) for u, v in bqm.quadratic)))
        if key not in EMB_CACHE:
            t0 = time.time()
            embs = find_embeddings(bqm, HW, methods=("clique", "minorminer"), seeds=(0,), timeout=EMBED_TIMEOUT,
                                   verbose=False, skip_minorminer_if_clique_found=True)
            if embs:
                label, emb = select_best(embs)
                EMB_CACHE[key] = (emb, embs[label]["stats"], time.time() - t0)
            else:
                EMB_CACHE[key] = (None, None, time.time() - t0)
        return EMB_CACHE[key]

    def run_config(net, ref_cost, label, **opts):
        """opts: QuboFormulator arguments, plus optional reserved keys
        _qpu    : overrides for EmulatedQPU(...)      (sampler, dac_bits, ice_sigma_h, ...)
        _sample : overrides for EmulatedQPU.sample()  (prefactor, postprocess, chain_break_method, ...)"""
        opts = dict(opts)
        qpu_over, sample_over = opts.pop("_qpu", {}), opts.pop("_sample", {})
        params = {**BASE, **opts}
        row = {"label": label}
        try:
            f = QuboFormulator(**params)
            bqm, cx = f._formulate_qubo(net)
            q, hw = cx["quantum_domain_qubo"], cx["hardware_limits"]
            degs = [bqm.degree(v) for v in bqm.variables]
            row.update(nq=q["total_logical_qubits"], disp=q["qubits_used_for_dispatch"],
                       ang=q["qubits_used_for_angles"], slk=q["qubits_wasted_on_slack"],
                       coup=q["num_interactions"], dens=q["density_percent"],
                       dmax=max(degs) if degs else 0, dmean=(sum(degs) / len(degs)) if degs else 0.0,
                       dr=hw.get("ising_dynamic_range"), risk=hw.get("ising_at_risk_percent"))
            if not bqm.variables:
                return row

            # logical ceiling: unembedded, noise-free SA on the formulated QUBO
            if SA is not None:
                ss = SA().sample(bqm, num_reads=LOGICAL_READS, num_sweeps=LOGICAL_SWEEPS, seed=SEED)
                r = evaluate(f, bqm, net, ss, ref_cost)
                row.update(lsa_feas=r["feas"], lsa_gap=r["gap"])

            # annealing emulator: embed, scale, quantise, ICE noise, sample, unembed
            emb, stats, emb_s = embed(bqm)
            row["emb_s"] = emb_s
            if emb is None:
                row["emb_fail"] = True
                return row
            row.update(phys=stats["physical"], maxch=stats["max_chain"])
            t0 = time.time()
            qpu = EmulatedQPU(HW, seed=SEED, **{**EMU_QPU_KW, **qpu_over})
            res = qpu.sample(bqm, emb, **{**EMU_SAMPLE_KW, **sample_over})
            row["emu_s"] = time.time() - t0
            r = evaluate(f, bqm, net, res.sampleset, ref_cost)
            row.update(emu_feas=r["feas"], emu_gap=r["gap"], e0=r["lowest_ok"])
            try:
                row["cb"] = 100.0 * float(np.mean(res.sampleset.record["chain_break_fraction"]))
            except Exception:
                pass
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    def num(v, spec, w):
        return (format(v, spec) if v is not None else "-").rjust(w)

    HEADER = (f"{'configuration':<38}{'qub':>5}{'disp':>6}{'ang':>5}{'slk':>5}{'coupl':>7}{'dens%':>7}"
              f"{'dmax':>6}{'dmean':>7}{'DynRng':>10}{'risk%':>7}"
              f"{'phys':>6}{'maxch':>6}{'embs':>7}{'CB%':>6}{'emus':>6}"
              f"{'LSAfeas%':>9}{'LSAgap%':>8}{'EMUfeas%':>9}{'EMUgap%':>8}{'E0':>5}")

    def print_row(r):
        if "error" in r:
            print(f"{r['label'][:37]:<38}ERROR {r['error'][:110]}")
            return
        e0 = {True: "ok", False: "INF", None: "-"}[r.get("e0")]
        tail = f"{num(r.get('lsa_feas'), '.1f', 9)}{num(r.get('lsa_gap'), '.2f', 8)}"
        if r.get("emb_fail"):
            tail += f"{'NO EMBEDDING':>22}"
        else:
            tail += f"{num(r.get('emu_feas'), '.1f', 9)}{num(r.get('emu_gap'), '.2f', 8)}{e0.rjust(5)}"
        print(f"{r['label'][:37]:<38}{num(r['nq'], 'd', 5)}{num(r['disp'], 'd', 6)}{num(r['ang'], 'd', 5)}"
              f"{num(r['slk'], 'd', 5)}{num(r['coup'], 'd', 7)}{num(r['dens'], '.1f', 7)}{num(r['dmax'], 'd', 6)}"
              f"{num(r['dmean'], '.1f', 7)}{num(r['dr'], '.1f', 10)}{num(r['risk'], '.1f', 7)}"
              f"{num(r.get('phys'), 'd', 6)}{num(r.get('maxch'), 'd', 6)}{num(r.get('emb_s'), '.1f', 7)}"
              f"{num(r.get('cb'), '.1f', 6)}{num(r.get('emu_s'), '.1f', 6)}" + tail)

    ALL_ROWS = []

    def section(net, ref_cost, key, title, note, configs):
        if key not in RUN_SECTIONS:
            return
        print("\n" + "=" * len(HEADER))
        print(f" {key} {title}")
        print("=" * len(HEADER))
        for line in note.strip().splitlines():
            print("  " + line.strip())
        print(HEADER)
        print("-" * len(HEADER))
        for label, opts in configs:
            row = run_config(net, ref_cost, label, **opts)
            row["section"] = f"{key} {title}"
            ALL_ROWS.append(row)
            print_row(row)

    # ----------------------------------------------------------------------------------------------
    # STEP 1: raw physical grid and classical reference
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" STEP 1: RAW PHYSICAL GRID DEFINITION (Pandapower)".center(90))
    print("=" * 90)

    net = GRID_FACTORY()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0          # standard physical rule: slack cannot absorb power
    describe_grid(net)
    REF = reference_solution(net)

    # ----------------------------------------------------------------------------------------------
    # STEP 2: algebraic expansion of the baseline problem, both formulations
    # ----------------------------------------------------------------------------------------------
    if SHOW_EXPANSION:
        print("\n\n" + "=" * 90)
        print(" STEP 2: MATHEMATICAL QUBO TRANSFORMATION (baseline options)".center(90))
        print("=" * 90)
        for form in FORMS:
            QuboFormulator(**{**BASE, "formulation": form}).view_problem_definition(net)

    # ----------------------------------------------------------------------------------------------
    # STEP 3: what does every option change?
    # ----------------------------------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print(" STEP 3: OPTION SWEEPS (one knob at a time, everything else at BASE)".center(90))
    print("=" * 90)
    print(f"BASE         = {BASE}")
    print(f"emulator     = {EMU_QPU_KW}")
    print(f"emu sampling = {EMU_SAMPLE_KW}")
    print("Columns: qub = logical qubits | disp/ang/slk = dispatch / angle / line-slack qubits | coupl = quadratic terms")
    print("         dens% = density | dmax, dmean = max / mean logical degree | DynRng = Ising max/min coefficient")
    print("         risk% = coefficients below risk_noise_frac * max (analog noise-floor exposure)")
    print("         phys, maxch = physical qubits and longest chain of the embedding | embs = embedding search seconds")
    print("         CB% = mean chain-break fraction | emus = emulator seconds")
    print("         LSA = logical SA (no embedding, no analog noise): the formulation's own ceiling")
    print("         EMU = annealing emulator; feas% = share of reads decoding to a feasible dispatch, gap% = best feasible")
    print("               vs the classical optimum; E0 = is the lowest-energy sample feasible (INF = the QUBO rewards")
    print("               infeasible states -> penalty/formulation problem, not a sampler problem)")
    print("Reading rule: LSA good but EMU bad -> hardware effects (chains, DAC, noise). Both bad -> the formulation.")

    # 3.1 formulation x encoding
    section(net, REF, "3.1", "formulation x encoding",
            """dc_ptdf eliminates flows -> fewest qubits, densest graph, longest chains. dc_theta keeps angles + slacks -> more
            qubits but local KCL cliques (lower degree). unary: most qubits, flattest coefficients. bounded: weights capped
            by bounded_mu.""",
            [(f"{short(form)} / {enc}", dict(formulation=form, encoding=enc))
             for form in FORMS for enc in ("radix", "unary", "bounded")])

    # 3.2 precision
    section(net, REF, "3.2", "mw_precision (discretisation step)",
            """Finer step -> more bits per variable and a smaller discretisation gap, but larger cliques, dynamic range and
            chains. Look for the precision where the gap stops improving: below it you only pay qubits and chain breaks.""",
            [(f"{short(form)} / prec {p:g} MW", dict(formulation=form, mw_precision=p))
             for form in FORMS for p in PRECISIONS])

    # 3.3 PTDF-specific options
    P = dict(formulation="dc_ptdf")
    section(net, REF, "3.3", "dc_ptdf options",
            """scale_line_constraints: sigma-normalisation of line rows (dynamic range). smart_slack_side: one-sided slack (fewer
            bits). slack_precision_factor: coarser slack (fewer bits, rounding residual). ptdf_threshold / rel / round:
            sparsify the PTDF (fewer couplers and shorter chains, but the QUBO then differs from the physics the decoder
            checks). decode_with_full_ptdf: judges feasibility on the untruncated PTDF (changes feasibility columns only).
            snap_weights: weights rounded to the grid.""",
            [("baseline", P),
             ("scale_line_constraints OFF", {**P, "scale_line_constraints": False}),
             ("smart_slack_side OFF", {**P, "smart_slack_side": False}),
             ("slack_precision_factor 2", {**P, "slack_precision_factor": 2.0}),
             ("slack_precision_factor 4", {**P, "slack_precision_factor": 4.0}),
             ("ptdf_threshold 0.0 (dense)", {**P, "ptdf_threshold": 0.0}),
             ("ptdf_threshold 0.15", {**P, "ptdf_threshold": 0.15}),
             ("ptdf_threshold 0.30", {**P, "ptdf_threshold": 0.30}),
             ("ptdf_rel_threshold 0.2", {**P, "ptdf_rel_threshold": 0.2}),
             ("ptdf_round_to 0.05", {**P, "ptdf_round_to": 0.05}),
             ("decode_with_full_ptdf OFF", {**P, "decode_with_full_ptdf": False}),
             ("snap_weights OFF", {**P, "snap_weights": False})])

    # 3.4 theta-specific options (angle step derived from the grid so the multipliers are meaningful)
    T = dict(formulation="dc_theta")
    theta_configs = [
        ("theta: all improvements ON", T),
        ("tight_angle_bounds OFF", {**T, "tight_angle_bounds": False}),
        ("theta_smart_slack OFF", {**T, "theta_smart_slack": False}),
        ("single_reference_angle OFF", {**T, "single_reference_angle": False}),
        ("ALL THREE OFF (previous behaviour)", {**T, "tight_angle_bounds": False, "theta_smart_slack": False,
                                                "single_reference_angle": False}),
        ("slack_precision_factor 2", {**T, "slack_precision_factor": 2.0}),
    ]
    if "3.4" in RUN_SECTIONS:
        try:
            _probe = QuboFormulator(**{**BASE, **T})
            _probe._formulate_qubo(net)
            default_step = BASE["mw_precision"] / max(_probe._max_b.values())
            for m in (0.5, 2.0, 4.0):
                theta_configs.append((f"angle_precision x{m:g} ({default_step * m:.4g} rad)",
                                      {**T, "angle_precision": default_step * m}))
        except Exception as exc:
            print(f"[note] angle_precision sweep skipped ({type(exc).__name__}: {exc})")
    section(net, REF, "3.4", "dc_theta options",
            """tight_angle_bounds: angle range from X @ injections (fewer angle bits). theta_smart_slack: skip lines that cannot
            bind, one-sided slack elsewhere. single_reference_angle: pin only the reference bus. angle_precision: finer angle
            grid = more bits but KCL closer to exact; coarser grid = fewer bits and a larger KCL residual.""",
            theta_configs)

    # 3.5 penalties and feasibility rules
    pen_configs = []
    for form in FORMS:
        for s in PENALTY_SAFETY:
            pen_configs.append((f"{short(form)} / penalty_safety {s:g}", dict(formulation=form, penalty_safety=s)))
    tol2 = 2.0 * BASE["mw_precision"]
    for form in FORMS:
        pen_configs.append((f"{short(form)} / rebalance_slack OFF", dict(formulation=form, rebalance_slack=False)))
        pen_configs.append((f"{short(form)} / feasibility_tol 0 MW", dict(formulation=form, feasibility_tol_mw=0.0)))
        pen_configs.append((f"{short(form)} / feasibility_tol {tol2:g} MW",
                            dict(formulation=form, feasibility_tol_mw=tol2)))
    section(net, REF, "3.5", "penalties and feasibility rules",
            """penalty_safety scales lambda: too small -> the QUBO minimum cheats (E0 = INF); too large -> dynamic range grows and
            the cost signal is compressed into the analog noise floor. rebalance_slack / feasibility_tol only change how
            states are judged, not the QUBO: watch the feasibility columns, not the qubit columns.""",
            pen_configs)

    # 3.6 bounded-coefficient cap
    section(net, REF, "3.6", "bounded encoding: coefficient cap mu",
            """mu = largest single weight in units of the precision. mu=1 is unary; large mu is radix. Small mu: more bits, flatter
            coefficients (lower DynRng, less analog exposure); large mu: fewer bits, wider range.""",
            [(f"{short(form)} / bounded mu={mu}", dict(formulation=form, encoding="bounded", bounded_mu=mu))
             for form in FORMS for mu in MU_VALUES])

    # 3.7 knobs that only change reporting / fallback costs
    section(net, REF, "3.7", "reporting and fallback-cost knobs",
            """risk_noise_frac only moves the risk% column (a metric, not the model). ext_fallback_mult prices an ext_grid that
            has no cost entry: it changes the objective and therefore the penalties derived from marginal costs.""",
            [(f"risk_noise_frac {r:g}", {**P, "risk_noise_frac": r}) for r in (0.01, 0.03125, 0.1)]
            + [(f"ext_fallback_mult {m:g}", {**P, "ext_fallback_mult": m}) for m in (1.0, 2.0, 10.0)])

    # 3.8 emulator knobs (same QUBO and embedding for every row -> only the hardware model changes)
    emu_configs = [("emulator baseline", P)]
    emu_configs += [(f"sampler {k} (slow for sqa)", {**P, "_qpu": dict(sampler=k, sampler_kwargs=kw)})
                    for k, kw in (("svmc", dict(num_sweeps=300)), ("sqa", dict(num_sweeps=100, trotter_slices=16)))]
    emu_configs += [(f"dac_bits {b}", {**P, "_qpu": dict(dac_bits=b)}) for b in (None, 8, 6, 4, 3)]
    emu_configs += [(f"ICE sigma_h={sh:g} sigma_j={sj:g}", {**P, "_qpu": dict(ice_sigma_h=sh, ice_sigma_j=sj)})
                    for sh, sj in ((0.0, 0.0), (0.01, 0.005), (0.05, 0.025), (0.1, 0.05))]
    emu_configs += [("use_extended_j OFF", {**P, "_qpu": dict(use_extended_j=False)})]
    emu_configs += [(f"chain prefactor {pf:g}", {**P, "_sample": dict(prefactor=pf)}) for pf in (0.7, 1.414, 2.5)]
    emu_configs += [(f"relative_chain_multiplier {m:g}", {**P, "_sample": dict(relative_chain_multiplier=m)})
                    for m in (1.0, 3.0)]
    emu_configs += [(f"chain_break {m}", {**P, "_sample": dict(chain_break_method=m)})
                    for m in ("minimize_energy", "discard", "weighted")]
    emu_configs += [(f"postprocess {m}", {**P, "_sample": dict(postprocess=m)})
                    for m in ("physical", "logical", "both")]
    section(net, REF, "3.8", "annealing-emulator knobs (baseline QUBO)",
            """Same QUBO and embedding in every row. dac_bits / ICE sigma: analog precision and control noise. chain prefactor:
            chain strength (low -> chain breaks, high -> chains crush the problem coefficients after scaling to the device
            range). chain_break: how broken chains are resolved. postprocess: greedy steepest descent (physical = on the
            embedded problem, logical = after unembedding): it repairs bit flips but is classical post-processing.""",
            emu_configs)

    # ----------------------------------------------------------------------------------------------
    # STEP 4: ranking across everything that ran
    # ----------------------------------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print(f" STEP 4: RANKING (emulator |gap| <= {GAP_OK:g} %, lowest-energy sample feasible)".center(90))
    print("=" * 90)
    good = [r for r in ALL_ROWS
            if "error" not in r and not r.get("emb_fail") and r.get("emu_gap") is not None
            and abs(r["emu_gap"]) <= GAP_OK and r.get("e0") is not False]
    if not good:
        print("No configuration met the threshold; relax GAP_OK, or raise reads / sweeps in EMU_SAMPLE_KW.")
    else:
        for title, key in (("fewest physical qubits", "phys"), ("shortest longest-chain", "maxch"),
                           ("lowest dynamic range", "dr"), ("lowest chain-break fraction", "cb")):
            print(f"\n--- {title} ---")
            print(HEADER)
            for r in sorted((x for x in good if x.get(key) is not None), key=lambda x: x[key])[:5]:
                print_row(r)

    # ----------------------------------------------------------------------------------------------
    # STEP 5: the baseline matrix itself
    # ----------------------------------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print(" STEP 5: BASELINE QUBO MATRIX INSPECTION".center(90))
    print("=" * 90)
    base_formulator = QuboFormulator(**BASE)
    bqm, complexity = base_formulator._formulate_qubo(net)
    print(f"Total Logical Variables (Qubits): {len(bqm.variables)}")
    print(f"Total Quadratic Interactions (Couplers): {len(bqm.quadratic)}\n")
    print("--- First 5 Linear Terms (Q_ii) ---")
    for var, bias in list(bqm.linear.items())[:5]:
        print(f"  {var:<20}: {bias:.2f}")
    print("\n--- First 5 Quadratic Terms (Q_ij) ---")
    for (u, v), bias in list(bqm.quadratic.items())[:5]:
        print(f"  ({u:<15}, {v:<15}): {bias:.2f}")
    print(f"\n[+] Hardware Dynamic Range: {complexity['hardware_limits'].get('ising_dynamic_range', 'N/A')}")