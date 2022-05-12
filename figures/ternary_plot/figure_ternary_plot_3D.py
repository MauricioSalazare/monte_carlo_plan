import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_data(load: float,
              pv: float,
              quant_value: str,
              file_path: Path=None,):

    assert (0 <= pv <= 1.0) and (0 <= load <= 1.0), "Load and PV must be between [0,1]"

    if file_path is None:
        path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
    else:
        path_file_parent = file_path

    file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quant_value}.pkl"
    with open(path_file_parent / file_name_solutions_dictionary, "rb") as pickle_file:
        solutions_dict = pickle.load(pickle_file)

    mixture_count = 0
    solutions_ternary = {}
    for solution_key in solutions_dict:
        mixture_, load_, pv_ = solution_key
        if (load == load_) and (pv == pv_):
            mixture_count += 1
            solutions_ternary[mixture_] = solutions_dict[solution_key]["max_q_" + quant_value].max()

    print(f"Total mixture in load-PV combination: {mixture_count}")
    print(f"Min. voltage: {min(solutions_ternary.values())}")
    print(f"Max. voltage: {max(solutions_ternary.values())}")

    mixture_values = np.array(list(solutions_ternary.keys()))
    voltage_values = np.array(list(solutions_ternary.values()))

    mixture_frame = pd.DataFrame(mixture_values, columns=["cloudy", "sunny", "overcast"])
    max_voltage_frame = pd.DataFrame(voltage_values, columns=["voltage"])

    ternary_data = pd.concat([mixture_frame, max_voltage_frame], axis=1)

    return ternary_data

def tr_b2c2b():
    # returns the transformation matrix from barycentric to cartesian coordinates and conversely
    tri_verts = np.array([[0.5, np.sqrt(3)/2], [0, 0], [1, 0]])# reference triangle
    M = np.array([tri_verts[:,0], tri_verts[:, 1], np.ones(3)])
    return M, np.linalg.inv(M)

CMIN_VMIN = 1.03
CMAX_VMAX = 1.061
EVERY = 0.00105
N_LEVELS = np.floor((CMAX_VMAX-CMIN_VMIN)/EVERY).astype(int)

LOAD = 0.5
PV = 0.4

data_voltage = load_data(load=LOAD, pv=PV, quant_value="90")

A = data_voltage["cloudy"].values
C = data_voltage["sunny"].values
B = data_voltage["overcast"].values
z = data_voltage["voltage"].values

M, invM =  tr_b2c2b()
cartes_coord_points = np.einsum('ik, kj -> ij', M, np.stack((A, B, C)))
xx, yy = cartes_coord_points[:2]
a, b = xx.min(), xx.max()
c, d = yy.min(), yy.max()

N=150
gr_x = np.linspace(a,b, N)
gr_y = np.linspace(c,d, N)
grid_x, grid_y = np.meshgrid(gr_x, gr_y)

grid_z = griddata(cartes_coord_points[:2].T, z, (grid_x, grid_y), method='cubic')

# Go back to barycentric coordinates
bar_coords = np.einsum('ik, kmn -> imn', invM, np.stack((grid_x, grid_y, np.ones(grid_x.shape))))
bar_coords[np.where(bar_coords<0)] = None # invalidate the points outside of the reference triangle
xy1 = np.einsum('ik, kmn -> imn', M, bar_coords) # recompute back the cartesian coordinates of bar_coords with invalid
                                                 # positions and extract indices where x are nan

I = np.where(np.isnan(xy1[0]))
grid_z[I] = None

levels_ = np.linspace(CMIN_VMIN, CMAX_VMAX, N_LEVELS)

#%%
fig, ax = plt.subplots(1 , 1, figsize=(5, 5))
ax.contourf(grid_x, grid_y, grid_z, levels=levels_)
cs = ax.contour(grid_x, grid_y, grid_z, levels=levels_,
                 colors="k", linewidths=0.4)
ax.clabel(cs, inline=True, fontsize=10, colors ='k')

#%%
norm_individual = mpl.colors.Normalize(vmin=CMIN_VMIN, vmax=CMAX_VMAX)
# v_safe = norm_individual(1.045)
# v_caution = norm_individual(1.05)

alpha = 0.9
green_color = (127/255,191/255,127/255, alpha)
orange_color = (255/255,192/255,76/255, alpha)
red_color = (219/255,76/255,76/255, alpha)

kwargs = {'format': '%.2f'}

cmap3 = mpl.colors.ListedColormap([green_color, orange_color, red_color])
bounds = [CMIN_VMIN, 1.045, 1.0499, CMAX_VMAX]
norm3 = mpl.colors.BoundaryNorm(bounds, cmap3.N)

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
plt.subplots_adjust(left=0.1, right=0.9)
# ax1.scatter(x_recovered_high_res, y_recovered_high_res, z_recovered_high_res, marker="o", s=1, facecolors='none', edgecolors='r')
ax1.plot_wireframe(grid_x, grid_y, grid_z, color='grey', linewidth=0.4, alpha=0.8)
# cs = ax1.contourf(grid_x, grid_y, grid_z, levels=levels_ )
ax1.contourf(grid_x, grid_y, grid_z, levels=levels_, zdir='z', offset=1.010,
                  vmin=CMIN_VMIN, vmax=CMAX_VMAX, cmap=cmap3, norm=norm3)
cs = ax1.contour(grid_x, grid_y, grid_z, levels=levels_, zdir='z', offset=1.010,
                 colors="k", linewidths=0.4)
ax1.set_zlim((1.005, ax1.get_zlim()[1]))
ax1.clabel(cs, inline=True, fontsize=10, colors ='k')
ax1.set_xlabel('x', fontsize="small")
ax1.set_ylabel('y', fontsize="small")
ax1.set_zlabel('voltage', fontsize="small")
ax1.tick_params(axis='both', which='major', labelsize="small")
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.grid(False)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm3, cmap=cmap3),
                    ax=ax1, fraction=0.02, pad=0.08, **kwargs)
cbar.ax.set_ylabel("Max. grid voltage", fontsize="medium")

z_min_ = np.nanmin(grid_z)