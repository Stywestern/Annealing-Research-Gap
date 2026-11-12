"""
End-to-End Demonstration Script for Quantum Annealing DC-OPF Solver
Submodule: solvers/simulated_quantum_annealing/run_dc_opf_demo.py
"""

import numpy as np
import pandapower as pp
import pandas as pd
import json
import pandapower.plotting as pp_plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import dimod

import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from solvers.qubo_formulator import QuboFormulator
from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
from solvers.simulated_quantum_annealing.s2_mapping import (
    IsingArrays, scale_to_hardware, quantise, add_ice_noise, mapping_diagnostics
)
from solvers.simulated_quantum_annealing.s3_samplers import make_sampler
from solvers.simulated_quantum_annealing.s4_postprocess import steepest_descent
from solvers.simulated_quantum_annealing.s5_emulators import EmulatedQPU

def custom_problem():
    """5 Buses, 3 Dispatchable Assets (1 Slack, 2 Gen), 2 Loads, 7 Lines (Meshed Ring + Radial Extension)."""
    net = pp.create_empty_network(name="5-Bus Natural OPF")
    
    b0 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 0")
    b1 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 3")
    b4 = pp.create_bus(net, vn_kv=230, min_vm_pu=0.9, max_vm_pu=1.1, name="Bus 4")

    # Slack and Generation Setup
    ext = pp.create_ext_grid(net, bus=b0, vm_pu=1.0, min_p_mw=0.0, max_p_mw=150.0, name="Slack")
    g1 = pp.create_gen(net, bus=b1, p_mw=0, min_p_mw=0.0, max_p_mw=150.0, controllable=True, name="Gen 1")
    g2 = pp.create_gen(net, bus=b3, p_mw=0, min_p_mw=0.0, max_p_mw=100.0, controllable=True, name="Gen 2")

    # Distributed Loads placed on separate load-serving buses (Bus 2 and Bus 4)
    pp.create_load(net, bus=b2, p_mw=120.0, name="Load 1")
    pp.create_load(net, bus=b4, p_mw=70.0, name="Load 2")

    # Natural Transmission Topology (Meshed core with radial load feeds)
    line_params = {'length_km': 1.0, 'r_ohm_per_km': 1.0, 'x_ohm_per_km': 10.0, 'c_nf_per_km': 0.0}
    pp.create_line_from_parameters(net, b0, b1, max_i_ka=0.3, name="L0-1", **line_params)
    pp.create_line_from_parameters(net, b1, b2, max_i_ka=0.25, name="L1-2", **line_params)
    pp.create_line_from_parameters(net, b0, b3, max_i_ka=0.3, name="L0-3", **line_params)
    pp.create_line_from_parameters(net, b1, b3, max_i_ka=0.2, name="L1-3", **line_params)
    pp.create_line_from_parameters(net, b2, b3, max_i_ka=0.25, name="L2-3", **line_params)
    pp.create_line_from_parameters(net, b2, b4, max_i_ka=0.2, name="L2-4", **line_params)
    pp.create_line_from_parameters(net, b3, b4, max_i_ka=0.2, name="L3-4", **line_params)

    # Cost Coefficients
    pp.create_poly_cost(net, ext, 'ext_grid', cp1_eur_per_mw=45.0)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=25.0)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=15.0)

    return net

def plot_grid(net):
    # 1. Generate artificial coordinates automatically
    pp_plot.create_generic_coordinates(net, library="igraph")

    # 2. Extract coordinates robustly
    if hasattr(net, "bus_geodata") and len(net.bus_geodata) > 0:
        coords = list(zip(net.bus_geodata['x'].values, net.bus_geodata['y'].values))
        get_xy = lambda b_idx: (net.bus_geodata.at[b_idx, 'x'], net.bus_geodata.at[b_idx, 'y'])
    else:
        coords = [json.loads(g)["coordinates"] for g in net.bus["geo"]]
        coord_dict = {i: c for i, c in zip(net.bus.index, coords)}
        get_xy = lambda b_idx: coord_dict[b_idx]

    # 3. Determine bus colors based on your exact network assets
    ext_buses = set(net.ext_grid.bus.values) if len(net.ext_grid) > 0 else set()
    gen_buses = set(net.gen.bus.values) if len(net.gen) > 0 else set()
    load_buses = set(net.load.bus.values) if len(net.load) > 0 else set()

    bus_colors = []
    for idx in net.bus.index:
        if idx in ext_buses:
            bus_colors.append('orange')  # Slack / External Grid (Bus 0)
        elif idx in gen_buses:
            bus_colors.append('green')   # Generators (Bus 1, Bus 3)
        elif idx in load_buses:
            bus_colors.append('red')     # Loads (Bus 2, Bus 4)
        else:
            bus_colors.append('blue')

    # 4. Create base network collections (Lines and Buses only - no annotation collection)
    bc = pp_plot.create_bus_collection(net, net.bus.index, size=0.04, color=bus_colors, zorder=2)
    lc = pp_plot.create_line_collection(net, net.line.index, color='grey', linewidth=1.5, zorder=1)
    
    collections = [lc, bc]
    
    if hasattr(net, "trafo") and len(net.trafo) > 0:
        tc = pp_plot.create_trafo_collection(net, net.trafo.index, size=0.03, color='green', zorder=2)
        collections.insert(1, tc)

    # Draw collections
    main_ax = pp_plot.draw_collections(collections, figsize=(8, 6))

    # --- CLEAN MANUALLY MANAGED BUS LABELS (Avoiding artifacts) ---
    for idx, row in net.bus.iterrows():
        name_val = row['name']
        b_name = name_val if pd.notna(name_val) and str(name_val).strip() != "" else f"Bus {idx}"
        label_text = f"{b_name}\n(B{idx})"
        
        bx, by = get_xy(idx)
        main_ax.text(bx, by + 0.12, label_text, 
                     fontsize=9, fontweight='bold', color='darkblue', 
                     ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # --- DRAW LINE LABELS MANUALLY ---
    for idx, row in net.line.iterrows():
        f_bus = int(row['from_bus'])
        t_bus = int(row['to_bus'])
        line_name = row['name'] if pd.notna(row['name']) and str(row['name']).strip() != "" else f"L{f_bus}-{t_bus}"
        
        fx, fy = get_xy(f_bus)
        tx, ty = get_xy(t_bus)
        
        mx, my = (fx + tx) / 2.0, (fy + ty) / 2.0
        main_ax.text(mx, my, line_name, fontsize=8, color='dimgrey', fontweight='semibold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    # --- LEGEND MATCHING ASSET COLOR CODING ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Slack Bus (Ext Grid)', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Generator Bus', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Load Bus', markerfacecolor='red', markersize=8),
        Line2D([0], [0], color='grey', lw=2, label='Transmission Line')
    ]

    main_ax.legend(handles=legend_elements, loc='lower left', frameon=True, edgecolor='black')

    plt.savefig("custom_grid_topology.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGrid plotted with clean asset-based coloring, accurate labels, and legend successfully.")


def export_embedding_graphs(bqm, hw, embedding):
    # Color palette grouping logical variables by asset (ext_0, gen_0, gen_1)
    sorted_vars = sorted(bqm.variables, key=str)
    cmap = plt.get_cmap("tab20").resampled(len(sorted_vars))
    var_colors = {v: cmap(i) for i, v in enumerate(sorted_vars)}

    # -------------------------------------------------------------------------
    # 0. Unembedded Raw Zephyr Topology
    # -------------------------------------------------------------------------
    G_hw = getattr(hw, "graph", None) or getattr(hw, "G", None) or dnx.zephyr_graph(1, 4)
    fig, ax = plt.subplots(figsize=(7, 7))
    dnx.draw_zephyr(G_hw, ax=ax, node_size=80, node_color="#4a7c59", edge_color="#888888", width=0.8)
    ax.set_title(f"Unembedded Zephyr Topology ({G_hw.number_of_nodes()} Qubits, {G_hw.number_of_edges()} Couplers)", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig("raw_zephyr_topology.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 1. Logical Problem Graph (K_12)
    # -------------------------------------------------------------------------
    G_logical = dimod.to_networkx_graph(bqm)
    fig, ax = plt.subplots(figsize=(7, 7))
    pos_logical = nx.circular_layout(G_logical)

    labels = {
        v: str(v).replace("ext_0_bit_", "ext0_b").replace("gen_0_bit_", "g0_b").replace("gen_1_bit_", "g1_b")
        for v in sorted_vars
    }

    nx.draw_networkx_edges(G_logical, pos_logical, ax=ax, edge_color="gray", alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(
        G_logical, pos_logical, ax=ax,
        nodelist=sorted_vars,
        node_color=[var_colors[v] for v in sorted_vars],
        node_size=900, edgecolors="black", linewidths=1.2
    )
    nx.draw_networkx_labels(G_logical, pos_logical, labels=labels, ax=ax, font_size=7.5, font_weight="bold")

    ax.set_title("Logical QUBO Graph $G_L = K_{12}$ (12 Qubits, 66 Edges)", fontsize=11, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("logical_problem_graph.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Physical Embedded Subgraph on Zephyr Z(4,4)
    # -------------------------------------------------------------------------
    G_hw = getattr(hw, "graph", None) or getattr(hw, "G", None) or dnx.zephyr_graph(4, 4)

    fig, ax = plt.subplots(figsize=(8, 8))
    pos_hw = dnx.zephyr_layout(G_hw)

    # Do not pass `pos` here; dnx.draw_zephyr computes zephyr_layout(G_hw) internally
    dnx.draw_zephyr(G_hw, ax=ax, node_size=15, node_color="#dddddd", edge_color="#eeeeee", width=0.4)

    # Do not pass `pos` here either for the same reason
    dnx.draw_zephyr_embedding(
        G_hw, emb=embedding, ax=ax,
        chain_color=var_colors,
        node_size=140,
        width=2.5,
        show_labels=False
    )

    # Zoom view onto the active 22 physical qubits using pos_hw coordinates
    active_nodes = [q for chain in embedding.values() for q in chain]
    xs = [pos_hw[q][0] for q in active_nodes]
    ys = [pos_hw[q][1] for q in active_nodes]
    margin_x = (max(xs) - min(xs)) * 0.15 + 0.05
    margin_y = (max(ys) - min(ys)) * 0.15 + 0.05

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"{labels[v]} (len={len(embedding[v])})",
                   markerfacecolor=var_colors[v], markeredgecolor="black", markersize=8)
        for v in sorted_vars
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7.5, framealpha=0.9, title="Logical Chains")
    ax.set_title("Minor-Embedded Subgraph on Zephyr Z(4,4) (22 Physical Qubits)", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig("physical_embedded_graph.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved logical_problem_graph.png and physical_embedded_graph.png")


def run_pipeline():
    print("=" * 80)
    print(" STEP 1: INITIALIZE PANDAPOWER NETWORK & QUBO FORMULATOR")
    print("=" * 80)
    net = custom_problem()
    
    formulator = QuboFormulator(
        formulation="dc_ptdf",
        mw_precision=10.0,
        encoding="radix",
        penalty_safety=1.0,
        tighten_bounds=True
    )
    
    # View exact problem definition, variables, and coefficients
    bqm, complexity = formulator.view_problem_definition(net)
    print(f"Generated BQM with {len(bqm.variables)} variables and {bqm.num_interactions} interactions.")

    print("\n" + "=" * 80)
    print(" PENALTY MULTIPLIERS & OPEN-FORM EQUATION INSPECTION")
    print("=" * 80)
    
    # Print individual constraint penalty weights and scaling factors
    print(f"Balance Penalty Multiplier (lambda_balance): {formulator.lambda_balance}")
    print(f"Line Limit Penalty Multipliers (lambda_line): {formulator.lambda_line}")
    
    # Reconstruct and print the open-form polynomial equation from the BQM
    print("\nFull Open-Form Quadratic Polynomial H(b):")
    polynomial_lines = [f"H(b) = {bqm.offset:.4f}"]
    
    # Add linear terms
    for var, coef in sorted(bqm.linear.items(), key=lambda x: str(x[0])):
        if abs(coef) > 1e-6:
            polynomial_lines.append(f"       {coef:+.4f} * {var}")
            
    # Add quadratic coupling terms
    for (u, v), coef in sorted(bqm.quadratic.items(), key=lambda x: (str(x[0]), str(x[1]))):
        if abs(coef) > 1e-6:
            polynomial_lines.append(f"       {coef:+.4f} * {u} * {v}")
            
    print("\n".join(polynomial_lines))

    print("\n" + "=" * 80)
    print(" STEP 2: HARDWARE TOPOLOGY & MINOR EMBEDDING")
    print("=" * 80)
    # Define an ideal Zephyr hardware graph (Z(4,4) or Z(12,4))
    hw = Hardware.ideal_zephyr(m=2, t=3)
    print(hw.summary())
    
    embeddings = find_embeddings(bqm, hw, methods=("clique", "minorminer", "layout", "mm_from_clique"), seeds=(0,1,2))
    chosen_label, embedding = select_best(embeddings, key=("physical", "max_chain", "mean_chain"))

    print(f"Chosen embedding method: {chosen_label}")
    print(f"Embedding stats: {embeddings[chosen_label]['stats']}")
    
    print("\n" + "=" * 80)
    print(" STEP 3: EMULATED QPU PIPELINE EXECUTION")
    print("=" * 80)

    export_embedding_graphs(bqm, hw, embedding)

    # Emulator part
    emulator = EmulatedQPU(
        hardware=hw,
        sampler="sa",
        sampler_kwargs={"num_sweeps": 500},
        dac_bits=5,
        ice_sigma_h=0.02,
        ice_sigma_j=0.01,
        schedule=None,
        seed=42
    )

    print("Running EmulatedQPU sampling and decoding...")
    result = emulator.sample(
        logical_bqm=bqm,
        embedding=embedding,
        num_reads=100,
        reads_per_programming=50,
        chain_break_method="majority",
        postprocess="logical"
    )

    print("\n" + "=" * 80)
    print(" STEP 4: RESULTS AND FEASIBILITY ANALYSIS")
    print("=" * 80)
    print(f"Chain Strength Used: {result.chain_strength}")
    print(f"Hardware Scale Factor: {result.scale_factor}")
    print(f"Embedding Footprint: {result.embedding_stats}")
    
    print("\nDecoding solution back to power flow domain...")
    sample_dict = result.sampleset.first.sample
    dispatch, sgen_dispatch, slack_dispatch, total_cost, feasibility = formulator._decode_solution(sample_dict, net)
    
    print(f"Feasible: {feasibility['is_feasible']}")
    print(f"Total Generation: {feasibility['total_generation_mw']} MW (Load: {feasibility['total_load_mw']} MW)")
    print(f"Raw Imbalance: {feasibility['raw_imbalance_mw']} MW")
    print(f"Total True Cost: {total_cost} EUR")
    print(f"Max Line Violation: {feasibility['max_line_violation_mw']} MW")
    print("Line Flows:")
    for l_id, flow in feasibility['line_flows_mw'].items():
        print(f"  {l_id}: {flow} MW")

if __name__ == "__main__":
    net = custom_problem()
    #plot_grid(net)
    

    run_pipeline()