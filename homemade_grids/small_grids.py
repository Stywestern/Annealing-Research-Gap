import pandapower as pp

def case3_low_gen():
    """3 Buses, 2 Dispatchable Assets (1 Slack, 1 Gen)"""
    net = pp.create_empty_network(name="3-Bus Low Gen")
    
    b0 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 0")
    b1 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 2")

    ext = pp.create_ext_grid(net, bus=b0, vm_pu=1.0, max_p_mw=100.0, name="Slack")
    g1 = pp.create_gen(net, bus=b1, p_mw=0, min_p_mw=0.0, max_p_mw=100.0, controllable=True, name="Gen 1")

    pp.create_load(net, bus=b2, p_mw=120.0, name="Load 1")

    line_params = {'length_km': 1.0, 'r_ohm_per_km': 1.0, 'x_ohm_per_km': 10.0, 'c_nf_per_km': 0.0}
    pp.create_line_from_parameters(net, b0, b1, max_i_ka=0.2, name="L0-1", **line_params)
    pp.create_line_from_parameters(net, b1, b2, max_i_ka=0.2, name="L1-2", **line_params)
    pp.create_line_from_parameters(net, b0, b2, max_i_ka=0.2, name="L0-2", **line_params)

    pp.create_poly_cost(net, ext, 'ext_grid', cp1_eur_per_mw=40.0)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=20.0)
    return net


def case3_high_gen():
    """3 Buses, 3 Dispatchable Assets (1 Slack, 2 Gens)"""
    net = pp.create_empty_network(name="3-Bus High Gen")
    
    b0 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b1 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b2 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)

    ext = pp.create_ext_grid(net, bus=b0, vm_pu=1.0, max_p_mw=100.0)
    g1 = pp.create_gen(net, bus=b1, p_mw=0, min_p_mw=0.0, max_p_mw=80.0, controllable=True)
    g2 = pp.create_gen(net, bus=b2, p_mw=0, min_p_mw=0.0, max_p_mw=80.0, controllable=True)

    pp.create_load(net, bus=b1, p_mw=50.0)
    pp.create_load(net, bus=b2, p_mw=100.0)

    line_params = {'length_km': 1.0, 'r_ohm_per_km': 1.0, 'x_ohm_per_km': 10.0, 'c_nf_per_km': 0.0}
    pp.create_line_from_parameters(net, b0, b1, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b1, b2, max_i_ka=0.15, **line_params)
    pp.create_line_from_parameters(net, b0, b2, max_i_ka=0.2, **line_params)

    pp.create_poly_cost(net, ext, 'ext_grid', cp1_eur_per_mw=40.0)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=25.0)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=15.0)
    return net


def case4_low_gen():
    """4 Buses, 2 Dispatchable Assets (1 Slack, 1 Gen)"""
    net = pp.create_empty_network(name="4-Bus Low Gen")
    
    b0 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b1 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b2 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b3 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)

    ext = pp.create_ext_grid(net, bus=b0, vm_pu=1.0, max_p_mw=120.0)
    g1 = pp.create_gen(net, bus=b2, p_mw=0, min_p_mw=0.0, max_p_mw=120.0, controllable=True)

    pp.create_load(net, bus=b1, p_mw=70.0)
    pp.create_load(net, bus=b3, p_mw=90.0)

    line_params = {'length_km': 1.0, 'r_ohm_per_km': 1.0, 'x_ohm_per_km': 10.0, 'c_nf_per_km': 0.0}
    pp.create_line_from_parameters(net, b0, b1, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b1, b2, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b2, b3, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b3, b0, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b0, b2, max_i_ka=0.2, **line_params)

    pp.create_poly_cost(net, ext, 'ext_grid', cp1_eur_per_mw=40.0)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=20.0)
    return net


def case4_high_gen():
    """4 Buses, 4 Dispatchable Assets (1 Slack, 3 Gens)"""
    net = pp.create_empty_network(name="4-Bus High Gen")
    
    b0 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b1 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b2 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)
    b3 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1)

    ext = pp.create_ext_grid(net, bus=b0, vm_pu=1.0, min_p_mw=0.0, max_p_mw=100.0)
    g1 = pp.create_gen(net, bus=b1, p_mw=0, min_p_mw=0.0, max_p_mw=80.0, controllable=True)
    g2 = pp.create_gen(net, bus=b2, p_mw=0, min_p_mw=0.0, max_p_mw=80.0, controllable=True)
    g3 = pp.create_gen(net, bus=b3, p_mw=0, min_p_mw=0.0, max_p_mw=80.0, controllable=True)

    pp.create_load(net, bus=b1, p_mw=40.0)
    pp.create_load(net, bus=b2, p_mw=60.0)
    pp.create_load(net, bus=b3, p_mw=80.0)

    line_params = {'length_km': 1.0, 'r_ohm_per_km': 1.0, 'x_ohm_per_km': 10.0, 'c_nf_per_km': 0.0}
    pp.create_line_from_parameters(net, b0, b1, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b1, b2, max_i_ka=0.15, **line_params)
    pp.create_line_from_parameters(net, b2, b3, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b3, b0, max_i_ka=0.2, **line_params)
    pp.create_line_from_parameters(net, b0, b2, max_i_ka=0.2, **line_params)

    pp.create_poly_cost(net, ext, 'ext_grid', cp1_eur_per_mw=40.0)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=30.0)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=20.0)
    pp.create_poly_cost(net, g3, 'gen', cp1_eur_per_mw=10.0)
    return net

# ------------------------------------------------------------------ #
# Info extraction from the original problem itself
# ------------------------------------------------------------------ #
def extract_network_info(net: pp.pandapowerNet) -> dict:
    """Extracts structural and operational parameters from a pandapower network

    relevant to DC-OPF formulation and benchmarking.
    """
    # 1. Component counts
    n_buses = len(net.bus)
    n_lines = len(net.line)
    n_trafos = len(net.trafo) if hasattr(net, "trafo") else 0
    n_branches = n_lines + n_trafos

    n_gen = len(net.gen) if hasattr(net, "gen") else 0
    n_ext_grid = len(net.ext_grid) if hasattr(net, "ext_grid") else 0
    n_dispatchable = n_gen + n_ext_grid

    n_loads = len(net.load) if hasattr(net, "load") else 0
    n_sgens = len(net.sgen) if hasattr(net, "sgen") else 0

    # 2. Demand and Capacity (MW)
    total_load_mw = float(net.load["p_mw"].sum()) if n_loads > 0 else 0.0

    # Include fixed generation from static generators if present
    total_sgen_mw = float(net.sgen["p_mw"].sum()) if n_sgens > 0 else 0.0
    net_demand_mw = total_load_mw - total_sgen_mw

    # Dispatchable capacity limits
    gen_max_mw = float(net.gen["max_p_mw"].sum()) if n_gen > 0 else 0.0
    ext_max_mw = (
        float(net.ext_grid["max_p_mw"].sum())
        if n_ext_grid > 0 and "max_p_mw" in net.ext_grid.columns
        else 0.0
    )
    total_gen_capacity_mw = gen_max_mw + ext_max_mw

    gen_min_mw = float(net.gen["min_p_mw"].sum()) if n_gen > 0 else 0.0
    ext_min_mw = (
        float(net.ext_grid["min_p_mw"].sum())
        if n_ext_grid > 0 and "min_p_mw" in net.ext_grid.columns
        else 0.0
    )
    total_gen_min_mw = gen_min_mw + ext_min_mw

    # 3. Cost coefficients (linear term cp1_eur_per_mw for DC-OPF)
    costs = {}
    if hasattr(net, "poly_cost") and len(net.poly_cost) > 0:
        for _, row in net.poly_cost.iterrows():
            element_type = row["et"]
            element_id = int(row["element"])
            cost_val = float(row.get("cp1_eur_per_mw", 0.0))
            costs[f"{element_type}_{element_id}"] = cost_val

    # 4. Network topology connectivity check
    is_connected = True
    try:
        import networkx as nx

        mg = pp.topology.create_nxgraph(net, include_trafos=True)
        is_connected = nx.is_connected(mg.to_undirected())
    except Exception:
        pass

    return {
        "network_name": getattr(net, "name", "unnamed_network"),
        "topology": {
            "buses": n_buses,
            "branches": n_branches,
            "lines": n_lines,
            "trafos": n_trafos,
            "is_connected": is_connected,
        },
        "dispatchable_assets": {
            "total_count": n_dispatchable,
            "generators": n_gen,
            "external_grids": n_ext_grid,
            "capacity_max_mw": total_gen_capacity_mw,
            "capacity_min_mw": total_gen_min_mw,
        },
        "demand": {
            "load_count": n_loads,
            "total_load_mw": total_load_mw,
            "static_gen_mw": total_sgen_mw,
            "net_active_demand_mw": net_demand_mw,
        },
        "feasibility_precheck": {
            "capacity_sufficient": total_gen_capacity_mw >= net_demand_mw
        },
        "linear_costs_eur_per_mw": costs,
    }

def extract_pf_info(net: pp.pandapowerNet) -> dict:
    """
    Extracts structural and operational parameters relevant to a pure Power Flow (PF) problem.
    In PF, generator setpoints are fixed inputs, economic costs are ignored, and 
    line capacities are not enforced during the solve (only monitored afterwards).
    """
    # 1. Bus counts by type
    n_buses = len(net.bus)
    slack_buses = net.ext_grid["bus"].tolist() if not net.ext_grid.empty else []
    pv_buses = net.gen["bus"].tolist() if not net.gen.empty else []
    pq_buses = [b for b in net.bus.index if b not in slack_buses and b not in pv_buses]

    # 2. Fixed Power Injections (The deterministic inputs)
    total_load_mw = float(net.load["p_mw"].sum()) if not net.load.empty else 0.0
    
    # In PF, p_mw is a rigid input, NOT a variable. 
    # If p_mw is 0, the generator acts as if it is turned off.
    fixed_gen_mw = float(net.gen["p_mw"].sum()) if not net.gen.empty else 0.0
    
    # 3. Expected Slack Behavior
    # The slack bus must absorb the exact difference between fixed load and fixed generation.
    expected_slack_mw = total_load_mw - fixed_gen_mw

    # 4. Limit Violations (Pre-solve warning)
    # PF will execute regardless, but we can flag if the fixed inputs force a slack violation.
    slack_max = float(net.ext_grid["max_p_mw"].sum()) if "max_p_mw" in net.ext_grid.columns else float('inf')
    slack_violation = expected_slack_mw > slack_max

    return {
        "problem_type": "Power Flow (Deterministic Root-Finding)",
        "network_name": getattr(net, "name", "unnamed_network"),
        "bus_types": {
            "total": n_buses,
            "slack_buses": len(slack_buses),
            "pv_buses_fixed_gen": len(pv_buses),
            "pq_buses_load": len(pq_buses)
        },
        "deterministic_inputs": {
            "total_load_demand_mw": total_load_mw,
            "fixed_generation_input_mw": fixed_gen_mw,
        },
        "physics_requirements": {
            "required_slack_injection_mw": expected_slack_mw,
            "slack_capacity_exceeded": bool(slack_violation)
        }
    }