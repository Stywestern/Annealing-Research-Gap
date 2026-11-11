"""
End-to-End Demonstration Script for Quantum Annealing DC-OPF Solver
Submodule: solvers/simulated_quantum_annealing/run_dc_opf_demo.py
"""

import numpy as np
import pandapower as pp
import pandapower.plotting as pp_plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import dimod

from solvers.qubo_formulator import QuboFormulator
from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best
from solvers.simulated_quantum_annealing.s2_mapping import (
    IsingArrays, scale_to_hardware, quantise, add_ice_noise, mapping_diagnostics
)
from solvers.simulated_quantum_annealing.s3_samplers import make_sampler
from solvers.simulated_quantum_annealing.s4_postprocess import steepest_descent
from solvers.simulated_quantum_annealing.s5_emulators import EmulatedQPU

import pandapower as pp

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
    g2 = pp.create_gen(net, bus=b2, p_mw=0, min_p_mw=0.0, max_p_mw=100.0, controllable=True, name="Gen 2")

    # Distributed Loads placed on separate load-serving buses (Bus 2 and Bus 4)
    pp.create_load(net, bus=b2, p_mw=50.0, name="Load 1")
    pp.create_load(net, bus=b4, p_mw=60.0, name="Load 2")

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
    print(" STEP 2: HARDWARE TOPOLOGY & MINOR EMBEDDING")
    print("=" * 80)
    # Define an ideal Zephyr hardware graph (Z(4,4) or Z(12,4))
    hw = Hardware.ideal_zephyr(m=4, t=4)
    print(hw.summary())
    
    embeddings = find_embeddings(bqm, hw, methods=("clique", "minorminer", "layout", "mm_from_clique"), seeds=(0,1,2))
    chosen_label, embedding = select_best(embeddings, key=("physical", "max_chain", "mean_chain"))

    print(f"Chosen embedding method: {chosen_label}")
    print(f"Embedding stats: {embeddings[chosen_label]['stats']}")
    
    print("\n" + "=" * 80)
    print(" STEP 3: EMULATED QPU PIPELINE EXECUTION")
    print("=" * 80)

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
    # 7. Configure the plot
    pp_plot.simple_plot(net, 
                    plot_loads=True, 
                    plot_gens=True, 
                    ext_grid_size=1.5, 
                    bus_color='b', 
                    line_color='grey', 
                    show_plot=False)

    fig = plt.gcf()
    main_ax = fig.axes[0]

    # Remove the 'Normal' and 'Colormap' widget buttons by dropping extra axes
    for widget_ax in fig.axes[1:]:
        widget_ax.remove()

    # Custom path for a thick circle containing two small dots side-by-side inside
    theta = np.linspace(0, 2 * np.pi, 40)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Outer circle path
    path_data = [(mpath.Path.MOVETO, (circle_x[0], circle_y[0]))]
    for x, y in zip(circle_x[1:], circle_y[1:]):
        path_data.append((mpath.Path.LINETO, (x, y)))
    path_data.append((mpath.Path.CLOSEPOLY, (circle_x[0], circle_y[0])))

    # Left dot path (small circle)
    dot_r = 0.15
    dot1_cx, dot1_cy = -0.35, 0.0
    path_data.append((mpath.Path.MOVETO, (dot1_cx + dot_r, dot1_cy)))
    for t in theta:
        path_data.append((mpath.Path.LINETO, (dot1_cx + dot_r * np.cos(t), dot1_cy + dot_r * np.sin(t))))
    path_data.append((mpath.Path.CLOSEPOLY, (dot1_cx + dot_r, dot1_cy)))

    # Right dot path (small circle)
    dot2_cx, dot2_cy = 0.35, 0.0
    path_data.append((mpath.Path.MOVETO, (dot2_cx + dot_r, dot2_cy)))
    for t in theta:
        path_data.append((mpath.Path.LINETO, (dot2_cx + dot_r * np.cos(t), dot2_cy + dot_r * np.sin(t))))
    path_data.append((mpath.Path.CLOSEPOLY, (dot2_cx + dot_r, dot2_cy)))

    codes, verts = zip(*path_data)
    gen_path = mpath.Path(verts, codes)

    # 8. Create a clean legend matching the detailed system explanations
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Bus (Vertex)', markerfacecolor='b', markersize=8),
        Line2D([0], [0], color='grey', lw=2, label='Transmission Line'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='xx', label='Slack Bus (External Grid)'),
        Line2D([0], [0], marker=gen_path, color='w', label='Generator (Synchronous Machine)', markeredgecolor='black', markerfacecolor='black', markersize=14, markeredgewidth=1.5),
        Line2D([0], [0], marker='v', color='w', label='Load (Power Consumer)', markeredgecolor='black', markerfacecolor='none', markersize=10)
    ]

    # Attach the legend to the main axis
    main_ax.legend(handles=legend_elements, loc='lower left', frameon=True, edgecolor='black')

    # 9. Save the visualization
    plt.savefig("custom_grid_topology.png", dpi=300, bbox_inches='tight')
    print("\nGrid plotted and saved successfully.")

    #run_pipeline()