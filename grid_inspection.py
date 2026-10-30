import pandapower.networks as nw
import pandapower.plotting as plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

# Instantiate the IEEE network
net = nw.case5()

GRID_REGISTRY = {
    "case4": nw.case4gs,
    "case5": nw.case5,
    "case9": nw.case9,
    "case14": nw.case14
}

# 1. Bus Information
print("--- Bus Data (Voltage Limits) ---")
print(net.bus[['name', 'vn_kv', 'min_vm_pu', 'max_vm_pu']])

# 2. Load Information
print("\n--- Load Data ---")
print(net.load[['name', 'bus', 'p_mw', 'q_mvar']])

# 3. View all standard generators
print("\n--- Generator Constraints ---")
print(net.gen[['name', 'bus', 'p_mw', 'min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']])

# 4. View the slack bus
print("\n--- Slack Bus (External Grid) ---")
print(net.ext_grid[['name', 'bus', 'vm_pu', 'va_degree', 'min_p_mw', 'max_p_mw']])

# 5. View the line constraints
print("\n--- Line Constraints & Impedances ---")
print(net.line[['name', 'from_bus', 'to_bus', 'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka']])

# 6. View the generation cost polynomials
print("\n--- Cost Polynomials ---")
print(net.poly_cost[['element', 'et', 'cp0_eur', 'cp1_eur_per_mw', 'cp2_eur_per_mw2']])

# 7. Configure the plot
plot.simple_plot(net, 
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
plt.savefig("ieee5_grid_topology.png", dpi=300, bbox_inches='tight')
print("\nGrid plotted and saved successfully.")