import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from glob import glob
import os
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn import preprocessing


def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator

def load_solutions(file_name_solutions):
    with open(file_name_solutions, "rb") as pickle_file:
        solutions = pickle.load(pickle_file)

    return solutions

file_name_scenario_generator_model = "../models/scenario_generator_model_new_AWS.pkl"
# file_name_solutions = "solutions.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
# solutions = load_solutions(file_name_solutions)
cases_combinations = scenario_generator.cases_combinations

# assert len(cases_combinations) == len(solutions), "The labeling could be wrong"

# # Assign the case combination to the solutions.
# solutions_dict = {}
# for case_, solution in zip(cases_combinations, solutions):  # The order of the combinations is the same as the solutions
#     solutions_dict[case_] = solution

x = scenario_generator.percentages_pv_growth
y = scenario_generator.percentages_load_growth


#%%
path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
quant_name = "90"
file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quant_name}.pkl"
with open(path_file_parent / file_name_solutions_dictionary, "rb") as pickle_file:
    solutions_dict = pickle.load(pickle_file)

load = 0.6
pv = 0.5
max_technical_limit = 1.045

# Mixture is (cloudy, sunny, dark)


mixture_count = 0
solutions_ternary = {}
for solution_key in solutions_dict:
    mixture_, load_, pv_ = solution_key
    if (load == load_) and (pv == pv_):
        mixture_count += 1
        solutions_ternary[mixture_] = solutions_dict[solution_key]["max_q_" + quant_name].max()

print(f"Total mixture in load-PV combination: {mixture_count}")
print(f"Min. voltage: {min(solutions_ternary.values())}")
print(f"Max. voltage: {max(solutions_ternary.values())}")


import plotly.figure_factory as ff
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

# Al = np.array([0. , 0. , 0., 0., 1./3, 1./3, 1./3, 2./3, 2./3, 1.])
# Cu = np.array([0., 1./3, 2./3, 1., 0., 1./3, 2./3, 0., 1./3, 0.])
# Y = 1 - Al - Cu

data_ = np.array(list(solutions_ternary.keys()))

Al = data_[:, 0].flatten()  # Cloudy
Cu = data_[:, 1].flatten()  # Sunny
Y = data_[:, 2].flatten()  # Dark

# https://plotly.com/python/ternary-contour/
# synthetic data for mixing enthalpy
# See https://pycalphad.org/docs/latest/examples/TernaryExamples.html
# enthalpy = 2.e6 * (Al - 0.01) * Cu * (Al - 0.52) * (Cu - 0.48) * (Y - 1)**2 - 5000
enthalpy = np.array(list(solutions_ternary.values()))

min_max_scaler = preprocessing.MinMaxScaler()
enthalpy_minmax = min_max_scaler.fit_transform(enthalpy.reshape(-1, 1)).flatten()

# enthalpy = ((enthalpy-1.05)/1.05)*100
fig = ff.create_ternary_contour(np.array([Al, Y, Cu]), enthalpy,
                                pole_labels=['Cloudy', 'Overcast', 'Sunny'],
                                # interp_mode='ilr',
                                interp_mode='cartesian',
                                ncontours=40,
                                colorscale='Electric',
                                # coloring='lines',
                                showscale=True,
                                title=f'Max. voltage ternary weather. Load: {load}, PV: {pv}',
                                showmarkers=True,
                                vmincb=1.015,
                                vmaxcb=1.050)
fig.show()

mixture_frame = pd.DataFrame(data_, columns=["overcast", "sunny", "dark"])
max_voltage_frame = pd.DataFrame(enthalpy, columns=["voltage"])
ternary_data = pd.concat([mixture_frame, max_voltage_frame], axis=1)
ternary_data.to_csv(rf"ternary_plot\ternary_data\data_load_{int(load*100)}_pv_{int(pv*100)}.csv", index=False)


x1 = ternary_data["sunny"].values
x2 = ternary_data["overcast"].values
x3 = ternary_data["dark"].values
y = ternary_data["voltage"].values

# Highlithg
# (cloudy, sunny, dark)
mixture_highlight = (0.2, 0.6, 0.2)
idx = (ternary_data["overcast"] == mixture_highlight[0]) & (ternary_data["sunny"] == mixture_highlight[1]) & (ternary_data["dark"] == mixture_highlight[2])
pos = ternary_data[idx.values].index.values[0]

fig, ax = plt.subplots(1,3, figsize=(10, 3))
plt.subplots_adjust(wspace=0.3)
ax[0].scatter(x1,y, marker="o", facecolors='none', edgecolors='b', label="sunny")
ax[1].scatter(x2,y, marker="o", facecolors='none', edgecolors='r', label="overcast")
ax[2].scatter(x3,y, marker="o", facecolors='none', edgecolors='g', label="dark")

# Highlight:
ax[0].scatter(x1[pos],y[pos], marker="o", facecolors='black', edgecolors='b', label="comb. sunny")
ax[1].scatter(x2[pos],y[pos], marker="o", facecolors='black', edgecolors='r', label="comb. overcast")
ax[2].scatter(x3[pos],y[pos], marker="o", facecolors='black', edgecolors='g', label="comb. dark")

for ax_ in ax:
    ax_.set_ylim((1.015, 1.055))
    ax_.hlines(y=1.045, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], linestyles="--", linewidth=0.4, label="max. volt.")
    ax_.legend(fontsize="x-small")

ax[1].set_title(f"Load: {load}: pv: {pv} per:{quant_name}")
print(f"Tracked voltage: {y[pos]}")

#%%
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax1.scatter(x1, x2, y, marker="o", facecolors='none', edgecolors='r')
ax1.set_xlabel('sunny', fontsize="small")
ax1.set_ylabel('overcast', fontsize="small")
ax1.set_zlabel('voltage', fontsize="small")
ax1.tick_params(axis='both', which='major', labelsize="small")

ax2.scatter(x2, x3, y, marker="o", facecolors='none', edgecolors='g')
ax2.set_xlabel('overcast')
ax2.set_ylabel('dark')
ax2.set_zlabel('voltage')

ax3.scatter(x1, x3, y, marker="o", facecolors='none', edgecolors='b')
ax3.set_xlabel('sunny')
ax3.set_ylabel('dark')
ax3.set_zlabel('voltage')


#%%
import matplotlib as mpl

norm_individual = mpl.colors.Normalize(vmin=y.min(), vmax=y.max())
log_norm_individual = mpl.colors.LogNorm(vmin=y.min(), vmax=y.max())
sym_log_norm_individual = mpl.colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                              vmin=y.min(), vmax=y.max(), base=10)


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x1, x2, marker="o", facecolors=plt.cm.get_cmap('viridis')(norm_individual(y)), edgecolors="none")
ax[1].scatter(x2, x3, marker="o", facecolors=plt.cm.get_cmap('viridis')(norm_individual(y)), edgecolors="none")
ax[0].set_xlabel("sunny")
ax[0].set_ylabel("overcast")

ax[1].set_xlabel("overcast")
ax[1].set_ylabel("dark")

#%%
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline, make_interp_spline
from itertools import product

ternary_data = pd.concat([mixture_frame, max_voltage_frame], axis=1)

# Steps for the model
# 1. Fill nan values via interpolation
# 2. Fit a spline and tune it

method = "cubic"

# Create possible combinations for the mixture
res_type_days = 10
n_types_of_days = 3
percentages_mixtures = np.linspace(0, 1.0, res_type_days + 1).round(1)
mixture_combinations = [mixture for mixture in product(percentages_mixtures,
                                                       repeat=n_types_of_days) if np.isclose(1.0, sum(mixture))]

# Check if all the combinations exists, otherwise fill the column with a nan value.
new_rows = []
for mixture_value in mixture_combinations:
    sunny, overcast, dark = mixture_value
    idx = (ternary_data["sunny"] == sunny) \
          & (ternary_data["overcast"] == overcast) \
          & (ternary_data["dark"] == dark)
    if not idx.sum():
        new_row = pd.DataFrame.from_dict({"sunny": [sunny], "overcast": [overcast], "dark": [dark], "voltage": [np.nan]})
        new_rows.append(new_row)
new_rows = pd.concat(new_rows, axis=0, ignore_index=True)

ternary_data_full = pd.concat([ternary_data, new_rows], axis=0, ignore_index=True)
ternary_data_full.to_csv("ternary_plot/ternary_data/full_data.csv", index=False)
ternary_data.to_csv("ternary_plot/ternary_data/clean_data.csv", index=False)
# Full grid that will also contain the missing values
# x1_all = ternary_data_full["sunny"].values
# x2_all = ternary_data_full["overcast"].values
X, Y = np.meshgrid(percentages_mixtures, percentages_mixtures)

percentages_mixtures_high_res = np.linspace(0, 1.0, 200)
X_high_res, Y_high_res = np.meshgrid(percentages_mixtures_high_res, percentages_mixtures_high_res)

# Get only the valid values
x1 = ternary_data["sunny"].values
x2 = ternary_data["overcast"].values
y = ternary_data["voltage"].values
Ti = griddata((x1, x2), y, (X, Y), method=method)
Ti_high_res = griddata((x1, x2), y, (X_high_res, Y_high_res), method=method)

x_recovered = X.reshape(-1, 1).squeeze()
y_recovered = Y.reshape(-1, 1).squeeze()
z_recovered = Ti.reshape(-1, 1).squeeze()

x_recovered_high_res = X_high_res.reshape(-1, 1).squeeze()
y_recovered_high_res = Y_high_res.reshape(-1, 1).squeeze()
z_recovered_high_res = Ti_high_res.reshape(-1, 1).squeeze()

fig, ax = plt.subplots(1 , 2,figsize=(10,4))
ax[0].contourf(X, Y, Ti)
ax[0].set_title(f"method = {method}")
ax[1].contourf(X_high_res, Y_high_res, Ti_high_res, norm=norm_individual)
ax[1].set_title("method = cubic - High res (200 points)")

#%%
fig = plt.figure(figsize=(22, 5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax1.scatter(x1, x2, marker="o", facecolors=plt.cm.get_cmap('viridis')(norm_individual(y)), edgecolors="none")
ax2.scatter(x_recovered, y_recovered, marker="o", facecolors=plt.cm.get_cmap('viridis')(norm_individual(z_recovered)), edgecolors="none")
ax3.scatter(x_recovered_high_res, y_recovered_high_res, marker="o", s=1, facecolors=plt.cm.get_cmap('viridis')(norm_individual(z_recovered_high_res)), edgecolors="none")

ax = fig.add_subplot(1, 4, 4, projection="3d")
ax.scatter(x1, x2, y, marker="o", facecolors='none', edgecolors='r', label="Original")
ax.scatter(x_recovered, y_recovered, z_recovered, marker="x", color="b", label="Interpolated")
ax.set_xlabel('sunny', fontsize="small")
ax.set_ylabel('overcast', fontsize="small")
ax.set_zlabel('voltage', fontsize="small")
ax.tick_params(axis='both', which='major', labelsize="small")
ax.legend(fontsize="x-small")

#%% Wire frame of the original and the high res
fig, ax = plt.subplots(nrows=1, ncols=3, subplot_kw={'projection': '3d'}, figsize=(24, 8))
ax[0].plot_wireframe(X, Y, Ti, color='r', linewidth=0.4)
ax[1].plot_wireframe(X_high_res, Y_high_res, Ti_high_res, color='b', linewidth=0.4)
ax[2].plot_wireframe(X, Y, Ti, color='r', linewidth=1.0)
ax[2].plot_wireframe(X_high_res, Y_high_res, Ti_high_res, color='b', linewidth=0.4)

#%% Contourplot
fig, ax = plt.subplots(1, 1,figsize=(5, 5))
ax.plot([0,1], [1,0], color='k')
# cs = ax.contour(X_high_res, Y_high_res, Ti_high_res, cmap='Greys', levels=[1.015, 1.025, 1.035, 1.040, 1.042, 1.043])
cs = ax.contour(X_high_res, Y_high_res, Ti_high_res, cmap='Greys', levels=20)
ax.clabel(cs, inline=True, fontsize=7, colors ='k')

#%%
# ============================================================================================================================================================================================================================

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# ax1.scatter(x_recovered_high_res, y_recovered_high_res, z_recovered_high_res, marker="o", s=1, facecolors='none', edgecolors='r')
ax1.plot_wireframe(X_high_res, Y_high_res, Ti_high_res, color='b', linewidth=0.4, alpha=0.8)
cs = ax1.contour(X_high_res, Y_high_res, Ti_high_res, levels=[1.015, 1.025, 1.035, 1.040, 1.042, 1.043], zdir='z', offset=1.010)
ax1.clabel(cs, inline=True, fontsize=10, colors ='k')
ax1.set_xlabel('x', fontsize="small")
ax1.set_ylabel('y', fontsize="small")
ax1.set_zlabel('voltage', fontsize="small")
ax1.tick_params(axis='both', which='major', labelsize="small")
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.grid(False)
z_min_ = np.nanmin(Ti_high_res)


ax1.plot([0,1], [1,0], 1.010, linewidth=1.0, color='k')
ax1.plot([1,0], [0,0], 1.010, linewidth=1.0, color='r')
ax1.plot([0,0], [1,0], 1.010, linewidth=1.0, color='b')

# Format of points: x=[x_0, y_0], y=[x_1, y_1]  ax.plot(x, y, z=1.01)


#%%

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(x1, x2, y, marker="o", facecolors='none', edgecolors='r')
ax1.contourf(X, Y, Ti, levels=10)
ax1.set_xlabel('x', fontsize="small")
ax1.set_ylabel('y', fontsize="small")
ax1.set_zlabel('voltage', fontsize="small")
ax1.tick_params(axis='both', which='major', labelsize="small")

fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
ax.plot_wireframe(X, Y, Ti, color='k')

















# ax[1].plot_wireframe(X2, Y2, Z2, color='k')
#%% A different interpolation

rara = ternary_data[["sunny", "overcast", "voltage"]].sort_values(by=["sunny", "overcast"])
x1r = rara["sunny"].unique()
x2r = x1r.copy()
X, Y = np.meshgrid(x1r, x2r)
Z = np.zeros_like(X)

nx = X.shape[0]
ny = X.shape[1]

y2 = np.zeros_like(x1r)
for x_coord in range(nx):  # "Sunny"
    for y_coord in range(ny):  # "Overcast"
        idx = (ternary_data["sunny"] == x1r[x_coord]) & (ternary_data["overcast"] == x2r[y_coord])
        if idx.sum():
            Z[x_coord, y_coord] = ternary_data["voltage"][idx]
            y2 = ternary_data["voltage"][idx]
        else:
            Z[x_coord, y_coord] = np.nan

interp_spline = RectBivariateSpline(x1r, x2r, Z)


Ti2 = griddata((x1r, x2r), Z, (X, Y), method="cubic")




# Regularly-spaced, fine grid
dx2, dy2 = 0.1, 0.1
x2 = np.arange(x1r.min(), x1r.max() + dx2, dx2)
y2 = np.arange(x2r.min(), x2r.max() + dy2, dy2)
X2, Y2 = np.meshgrid(x2,y2)
Z2 = interp_spline(y2, x2)

#%%
fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
ax[0].plot_wireframe(X, Y, Z, color='k')
ax[1].plot_wireframe(X2, Y2, Z2, color='k')




#%%



fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot the model function and the randomly selected sample points
# ax[0,0].contourf(X, Y, T)
ax[0,0].scatter(x1, x2, c='k', alpha=0.2, marker='.')
ax[0,0].set_title('Sample points on f(X,Y)')

# Interpolate using three different methods and plot
for i, method in enumerate(('nearest', 'linear', 'cubic')):
    Ti = griddata((x1, x2), y, (X, Y), method=method)
    r, c = (i+1) // 2, (i+1) % 2
    ax[r,c].contourf(X, Y, Ti)
    ax[r,c].set_title("method = '{}'".format(method))

plt.tight_layout()
plt.show()

#%% Fit the data of the triangle to surface
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100)
y =  np.linspace(-1,1,100)
X, Y = np.meshgrid(x,y)

def f(x, y):
    s = np.hypot(x, y)
    phi = np.arctan2(y, x)
    tau = s + s*(1-s)/5 * np.sin(6*phi)
    return 5*(1-tau) + tau

T = f(X, Y)
# Choose npts random point from the discrete domain of our model function
npts = 400
px, py = np.random.choice(x, npts), np.random.choice(y, npts)

fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot the model function and the randomly selected sample points
ax[0,0].contourf(X, Y, T)
ax[0,0].scatter(px, py, c='k', alpha=0.2, marker='.')
ax[0,0].set_title('Sample points on f(X,Y)')

# Interpolate using three different methods and plot
for i, method in enumerate(('nearest', 'linear', 'cubic')):
    Ti = griddata((px, py), f(px,py), (X, Y), method=method)
    r, c = (i+1) // 2, (i+1) % 2
    ax[r,c].contourf(X, Y, Ti)
    ax[r,c].set_title("method = '{}'".format(method))

plt.tight_layout()
plt.show()









#%%
SQRT3 = np.sqrt(3)
SQRT3OVER2 = SQRT3 / 2.

def permute_point(p, permutation=None):
    """
    Permutes the point according to the permutation keyword argument. The
    default permutation is "012" which does not change the order of the
    coordinate. To rotate counterclockwise, use "120" and to rotate clockwise
    use "201"."""
    if not permutation:
        return p
    return [p[int(permutation[i])] for i in range(len(p))]


def project_point(p, permutation=None):
    """
    Maps (x,y,z) coordinates to planar simplex.

    Parameters
    ----------
    p: 3-tuple
        The point to be projected p = (x, y, z)
    permutation: string, None, equivalent to "012"
        The order of the coordinates, counterclockwise from the origin
    """
    permuted = permute_point(p, permutation=permutation)
    a = permuted[0]
    b = permuted[1]
    x = a + b/2.
    y = SQRT3OVER2 * b
    return np.array([x, y])

#%%

x1 = ternary_data["sunny"].values
x2 = ternary_data["overcast"].values
x3 = ternary_data["dark"].values
y = ternary_data["voltage"].values


print(project_point((0.5,0.5,0.5)))

xx= []
el_voltaje = []
for rr, the_row in ternary_data.iterrows():
    xx.append(project_point([the_row["sunny"], the_row["overcast"], the_row["dark"]]))
    el_voltaje.append(the_row["voltage"])
xx = np.array(xx)
el_voltaje = np.array(el_voltaje)


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(xx[:,0], xx[:,1], marker="o", facecolors=plt.cm.get_cmap('viridis')(norm_individual(el_voltaje)), edgecolors="none")

#%%
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.scatter(xx[:,0], xx[:,1], el_voltaje, marker="o", facecolors='none', edgecolors='r')
ax1.set_xlabel('x', fontsize="small")
ax1.set_ylabel('y', fontsize="small")
ax1.set_zlabel('voltage', fontsize="small")
ax1.tick_params(axis='both', which='major', labelsize="small")



#%%
import numpy as np
import matplotlib.pyplot as plt
import ternary
import math
import itertools
from skimage import measure

def shannon_entropy(p):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1. * s

scale = 20
level = [0.25, 0.5, 0.8, 0.9]       # values for contours

# === prepare coordinate list for contours
x_range = np.arange(0, 1.01, 0.01)   # ensure that grid spacing is small enough to get smooth contours
coordinate_list = np.asarray(list(itertools.product(x_range, repeat=2)))
coordinate_list = np.append(coordinate_list, (1 - coordinate_list[:, 0] - coordinate_list[:, 1]).reshape(-1, 1), axis=1)

# === calculate data with coordinate list
data_list = []
for point in coordinate_list:
    data_list.append(shannon_entropy(point))  # SHANNON ENTROPY COMPUTATION
data_list = np.asarray(data_list)
data_list[np.sum(coordinate_list[:, 0:2], axis=1) > 1] = np.nan  # remove data outside triangle

# === reshape coordinates and data for use with pyplot contour function
x = coordinate_list[:, 0].reshape(x_range.shape[0], -1)
y = coordinate_list[:, 1].reshape(x_range.shape[0], -1)

h = data_list.reshape(x_range.shape[0], -1)

# === use pyplot to calculate contours
contours = plt.contour(x, y, h, level)  # this needs to be BEFORE figure definition
# cs = ax.contour(X_high_res, Y_high_res, Ti_high_res, cmap='Greys', levels=20)
plt.clabel(contours, inline=True, fontsize=7, colors ='k')
# plt.clf()  # makes sure that contours are not plotted in carthesian plot


# #%%
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# for level_ in level:
#     contours_skimage = measure.find_contours(h, level_, fully_connected="low")
#     for contour in contours_skimage:
#         ax.plot(contour[:, 1]/100, contour[:, 0]/100, linewidth=2)
# ax.set_xlim((0, 1))
# ax.set_ylim((0, 1))


#%%

fig, tax = ternary.figure(scale=scale)

# === plot contour lines
for ii, contour in enumerate(contours.allsegs):
    for jj, seg in enumerate(contour):
        tax.plot(seg[:, 0:2] * scale, color='r')

plt.clabel(contours, inline=True, fontsize=7, colors ='k')

# === plot regular data
# tax.heatmapf(shannon_entropy, boundary=True, style='hexagonal', colorbar=True)
# tax.clear_matplotlib_ticks()
# tax.get_axes().axis('off')
# tax.boundary()
tax.boundary(linewidth=2.0)
# tax.gridlines(color="black", multiple=5)
# tax.gridlines(color="blue", multiple=1, linewidth=0.5)
# tax.left_axis_label("Left label $\\alpha^2$", fontsize=7)
# tax.right_axis_label("Right label $\\beta^2$", fontsize=7)
# tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=7)

plt.show()



#%%
scale = 1
data_mia = {}
points_mia = []
for _, (aa, bb, cc, dd) in ternary_data.iterrows():
    data_mia[(aa,bb,cc)] = dd
    points_mia.append((aa,bb,cc))
#%%
scale = 1.0
figure, tax = ternary.figure(scale=scale)
tax.heatmap(data_mia, style="triangular", use_rgba=False, colorbar=True, scale=0.1, vmin=min(data_mia.values()), vmax=max(data_mia.values()))

#%%
import numpy.random as rand

def random_points(num_points=25):
    points = []
    for i in range(num_points):
        x = rand.uniform()
        y = rand.uniform(low=0, high=1-x)
        z = 1 - x - y
        points.append((x, y, z))
    return points

figure = plt.figure(figsize=(16, 7))
fig = figure.add_subplot(121)

fig, tax = ternary.figure(scale=1)
fig.set_size_inches(8, 7)
tax.boundary(linewidth=2.0)
tax.gridlines(multiple=.1, color="blue")
points = random_points(30)
tax.scatter(points_mia, marker='s', color='red', label="Red Squares")
# points = random_points(30)
# tax.scatter(points, marker='D', color='green', label="Green Diamonds")
tax.legend(fontsize=15)
# tax.ticks(axis='lbr', linewidth=1, multiple=.1)
# tax.clear_matplotlib_ticks()
# tax._redraw_labels()

