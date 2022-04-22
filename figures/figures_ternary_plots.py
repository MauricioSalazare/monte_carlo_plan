import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from glob import glob
import os
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

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



path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
quant_name = "90"
file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quant_name}.pkl"
with open(path_file_parent / file_name_solutions_dictionary, "rb") as pickle_file:
    solutions_dict = pickle.load(pickle_file)

#%%
load = 0.2
pv = 0.4

solutions_ternary = {}
for solution_key in solutions_dict:
    print(solution_key)
    mixture_, load_, pv_ = solution_key
    if (load == load_) and (pv == pv_):
        solutions_ternary[mixture_] = solutions_dict[solution_key]["max_q_90"].max()


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

# synthetic data for mixing enthalpy
# See https://pycalphad.org/docs/latest/examples/TernaryExamples.html
# enthalpy = 2.e6 * (Al - 0.01) * Cu * (Al - 0.52) * (Cu - 0.48) * (Y - 1)**2 - 5000
enthalpy = np.array(list(solutions_ternary.values()))
# enthalpy = ((enthalpy-1.05)/1.05)*100
fig = ff.create_ternary_contour(np.array([Al, Y, Cu]), enthalpy,
                                pole_labels=['Cloudy', 'Dark', 'Sunny'],
                                # interp_mode='ilr',
                                interp_mode='cartesian',
                                ncontours=40,
                                colorscale='Viridis',
                                # coloring='lines',
                                showscale=True,
                                title=f'Max. voltage ternary weather. Load: {load}, PV: {pv}',
                                showmarkers=True)
fig.show()

#%%
mixture_frame = pd.DataFrame(data_, columns=["overcast", "sunny", "dark"])
max_voltage_frame = pd.DataFrame(enthalpy, columns=["voltage"])
ternary_data = pd.concat([mixture_frame, max_voltage_frame], axis=1)
ternary_data.to_csv(r"ternary_plot\ternary_data\data_load_20_pv_40.csv", index=False)

#%%
import numpy as np
import matplotlib.pyplot as plt
import ternary
import math
import itertools


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
    data_list.append(shannon_entropy(point))
data_list = np.asarray(data_list)
data_list[np.sum(coordinate_list[:, 0:2], axis=1) > 1] = np.nan  # remove data outside triangle

# === reshape coordinates and data for use with pyplot contour function
x = coordinate_list[:, 0].reshape(x_range.shape[0], -1)
y = coordinate_list[:, 1].reshape(x_range.shape[0], -1)

h = data_list.reshape(x_range.shape[0], -1)

# === use pyplot to calculate contours
contours = plt.contour(x, y, h, level)  # this needs to be BEFORE figure definition
plt.clf()  # makes sure that contours are not plotted in carthesian plot

fig, tax = ternary.figure(scale=scale)

# === plot contour lines
for ii, contour in enumerate(contours.allsegs):
    for jj, seg in enumerate(contour):
        tax.plot(seg[:, 0:2] * scale, color='r')

# === plot regular data
tax.heatmapf(shannon_entropy, boundary=True, style='hexagonal', colorbar=True)

plt.show()

#%%
import matplotlib.pyplot as plt

def color_point(x, y, z, scale):
    w = 255
    x_color = x * w / float(scale)
    y_color = y * w / float(scale)
    z_color = z * w / float(scale)
    r = math.fabs(w - y_color) / w
    g = math.fabs(w - x_color) / w
    b = math.fabs(w - z_color) / w
    return (r, g, b, 1.)


def generate_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, scale)
    return d

scale = 90
data = generate_heatmap_data(scale)
figure, tax = ternary.figure(scale=scale)
tax.heatmap(data, style="hexagonal", use_rgba=True, colorbar=True)
# Remove default Matplotlib Axes
# tax.clear_matplotlib_ticks()
# tax.get_axes().axis('off')
# tax.boundary()
tax.set_title("RGBA Heatmap")
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

