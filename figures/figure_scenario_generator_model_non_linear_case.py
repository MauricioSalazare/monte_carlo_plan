"""
Create the figure for the load growths for the linear and non-linear case.
Also add the plot for the PV growth that it is lineal.
"""


import pickle
import pandas as pd
from core.scenarios import ScenarioGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import collections  as mc
from scipy.interpolate import UnivariateSpline, interp1d, pchip_interpolate



file_name_load_models = r"../models/copula_model_load.pkl"
with open(file_name_load_models, "rb") as pickle_file:
    copula_load = pickle.load(pickle_file)

file_name_irradiance_models = r"../models/copula_model_irradiance.pkl"
with open(file_name_irradiance_models, "rb") as pickle_file:
    mixture_model_irradiance = pickle.load(pickle_file)

file_name_network = r"../data/processed_data/network_data/Nodes_34.csv"
grid = pd.read_csv(file_name_network)
grid_info = grid.iloc[1:, :]  # Drop the slack node


#%%
#======================================================================================================================
# CREATE MODEL FOR THE LINEAL CASE
#======================================================================================================================
scenario_generator = ScenarioGenerator(copula_load=copula_load,
                                       mixture_model_irradiance=mixture_model_irradiance,
                                       grid_info=grid_info,
                                       n_levels_load_growth=10,
                                       n_levels_pv_growth=10,
                                       n_levels_mixtures=10)

#======================================================================================================================
# CREATE MODEL FOR THE NON LINEAL CASE
#======================================================================================================================

#%% Spline function
# Control points for the spline
points = np.array([(0.0, 0.0),
                   (0.3, 0.4),
                   (0.5, 0.5),
                   (0.7, 0.5),
                   (1.0, 1.0)])

x_ = np.linspace(0.0, 1.0, 100)
x = np.linspace(0.0, 1.0, 11)
f = UnivariateSpline(points[:,0], points[:, 1],k=3)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.scatter(points[:,0], points[:,1], s=20, zorder=3)
ax.scatter(x, f(x), s=20)
ax.plot(x_, f(x_), linewidth=0.8, color="C0")
ax.grid("both")
ax.set_title("Normalized non-linear growth")

#%% Square root
f = lambda x: np.sqrt(x)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.scatter(x, f(x), s=20)
ax.plot(x_, f(x_), linewidth=0.8, color="C0")
ax.grid("both")
ax.set_title("Normalized non-linear growth")

#%%
scenario_generator_non_lineal = ScenarioGenerator(copula_load=copula_load,
                                                  mixture_model_irradiance=mixture_model_irradiance,
                                                  grid_info=grid_info,
                                                  n_levels_load_growth=10,
                                                  n_levels_pv_growth=10,
                                                  n_levels_mixtures=10,
                                                  lineal_growth=False,
                                                  load_growth_function=f)


#%%
# *********************************************************************************************************************
# PLOTTING SECTION
# *********************************************************************************************************************

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
plt.subplots_adjust(left=0.05, wspace=0.95, right=0.9, bottom=0.2)

# =====================================================================================================================
# LINEAL CASE
# =====================================================================================================================
#  Load growth

# The mapper_load_growth() exists, because each cluster growth is bounded by the max and min energy consumption of the
# cluster. Remember that copula can not extrapolate.

# Count the nodes with the same number of clusters
nodes_per_cluster = scenario_generator.nodes_per_cluster
x_ = np.arange(0, 11)

growth_curve_cluster = []
for cluster_, nodes_ in nodes_per_cluster.iteritems():
    y = np.array(list(scenario_generator.mapper_load_growth[cluster_].values()))
    n_nodes = nodes_per_cluster[cluster_]
    ax[0].plot(x_,y * nodes_, label=f"Cluster {cluster_}")
    growth_curve_cluster.append(y * n_nodes)  # Total energy increased by the cluster
growth_curve_cluster = np.array(growth_curve_cluster)
total_grid_energy_growth = growth_curve_cluster.sum(axis=0)

ax_ = ax[0].twinx()
ax_.plot(x_, total_grid_energy_growth, color="red", linewidth=3.0, label="Total grid energy")

max_ = total_grid_energy_growth.max()
min_ = total_grid_energy_growth.min()

f_x_y = lambda x: ( (x - min_)/(max_ - min_) ) * 100  #  y = (x-xmin)/(xmax - xmin)   : Function
f_y_x = lambda x: ( x * (max_ - min_) + min_ ) * 100  #  x = y * (xmax - xmin) + xmin : Inverse function

ax2 = ax_.secondary_yaxis(1.2, functions=(f_x_y, f_y_x))

ax[0].set_ylim(0, 20.0)
ax[0].set_ylabel("Annual energy consumption per cluster[GWh/Year]", fontsize="small")
ax[0].legend(loc="upper left", fontsize="x-small", title="Aggregated energy", title_fontsize="small")
ax[0].set_xlabel("index - Growth increment step", fontsize="small")
ax[0].set_title("Load growth - Lineal case", fontsize="small")
ax[0].xaxis.set_major_locator(mticker.MultipleLocator(1))
ax[0].tick_params(axis='both', which='major', labelsize="small")
ax[0].tick_params(axis='both', which='minor', labelsize="small")

ax_.set_ylim(2,40.0)
ax_.set_ylabel("Total grid annual Energy consumption [GWh/Year]", fontsize="small")
ax_.legend(loc="lower right", fontsize="small")
ax_.tick_params(axis='both', which='major', labelsize="small")
ax_.tick_params(axis='both', which='minor', labelsize="small")

ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylabel("Annual energy consumption growth", fontsize="small")
ax2.tick_params(axis='both', which='major', labelsize="small")
ax2.tick_params(axis='both', which='minor', labelsize="small")



# =====================================================================================================================
# =====================================================================================================================
# NON-LINEAL CASE
# =====================================================================================================================
# =====================================================================================================================
nodes_per_cluster = scenario_generator_non_lineal.nodes_per_cluster

total_grid_energy_growth_nl = []
for cluster_, nodes_ in nodes_per_cluster.iteritems():
    x = np.array(list(scenario_generator_non_lineal.mapper_load_growth[cluster_].keys()))
    y = np.array(list(scenario_generator_non_lineal.mapper_load_growth[cluster_].values()))
    ax[1].plot(x * 10, y * nodes_, label=f"Cluster {cluster_}")
    total_grid_energy_growth_nl.append(y * nodes_)
total_grid_energy_growth_nl = np.array(total_grid_energy_growth_nl).sum(axis=0)

# x = np.linspace(0, 10, 11)
ax_ = ax[1].twinx()
ax_.plot(x_, total_grid_energy_growth_nl, color="red", linewidth=3.0, label="Total grid energy")

max_ = total_grid_energy_growth_nl.max()
min_ = total_grid_energy_growth_nl.min()

f_x_y = lambda x: ( (x - min_)/(max_ - min_) ) * 100  #  y = (x-xmin)/(xmax - xmin)   : Function
f_y_x = lambda x: ( x * (max_ - min_) + min_ ) * 100  #  x = y * (xmax - xmin) + xmin : Inverse function

ax2 = ax_.secondary_yaxis(1.2, functions=(f_x_y, f_y_x))

ax[1].set_ylim(0, 20.0)
ax[1].set_ylabel("Annual energy consumption per cluster [GWh/Year]", fontsize="small")
ax[1].legend(loc="upper left", fontsize="x-small", title="Total energy", title_fontsize="small")
ax[1].set_xlabel("index - Growth increment step", fontsize="small")
ax[1].set_title("Load growth - Non-Lineal case", fontsize="small")
ax[1].xaxis.set_major_locator(mticker.MultipleLocator(1))
ax[1].tick_params(axis='both', which='major', labelsize="small")
ax[1].tick_params(axis='both', which='minor', labelsize="small")

ax_.set_ylim(2,40.0)
ax_.set_ylabel("Total grid annual Energy consumption [GWh/Year]", fontsize="small")
ax_.legend(loc="lower right", fontsize="small")
ax_.tick_params(axis='both', which='major', labelsize="small")
ax_.tick_params(axis='both', which='minor', labelsize="small")

ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylabel("Annual energy consumption growth", fontsize="small")
ax2.tick_params(axis='both', which='major', labelsize="small")
ax2.tick_params(axis='both', which='minor', labelsize="small")


# =====================================================================================================================
# =====================================================================================================================
# PV GROWTH LINEAL CASE
# =====================================================================================================================
# =====================================================================================================================

panel_instalations = grid_info["kwp"].value_counts().sort_index().reset_index(name="counts")
panel_instalations.rename(columns={"index":"kwp"}, inplace=True)
panel_instalations = panel_instalations[~(panel_instalations["kwp"] == 0)]
panel_instalations.reset_index(inplace=True, drop=True)

total_kwp_grid_base = (panel_instalations.kwp * panel_instalations.counts).sum()

# fig, ax = plt.subplots(1,1, figsize=(7,5))
# plt.subplots_adjust(left=0.15, right=0.75)

x_2 = np.arange(0, len(x_))
total_mwp_grid_curve = ((1 + x_2 / 10) * total_kwp_grid_base) / 1000

pv_growth_curve_pv_with_same_kwp = []

for idx, (kwp_, count_) in panel_instalations.iterrows():
    panel_mwp_grid_curve = ((1 + x_2 / 10) * kwp_) / 1000
    ax[2].plot(x_2, panel_mwp_grid_curve, linewidth=1.0)  # Total installed capacity in (MWp)

ax_ = ax[2].twinx()
ax_.plot(x_2, total_mwp_grid_curve, color="red", linewidth=3.0, label="Total PV grid capacity")  # Total installed capacity in (MWp)

min_kwp = total_mwp_grid_curve.min()
max_kwp = total_mwp_grid_curve.max()

f_x_y = lambda x: ( (x - min_kwp)/(max_kwp - min_kwp) ) * 100  #  y = (x-xmin)/(xmax - xmin)   : Function
f_y_x = lambda x: ( x * (max_kwp - min_kwp) + min_kwp ) * 100  #  x = y * (xmax - xmin) + xmin : Inverse function

ax2 = ax_.secondary_yaxis(1.3, functions=(f_x_y, f_y_x))
ax[2].xaxis.set_major_locator(mticker.MultipleLocator(1))
ax[2].set_ylim((0.0, 1.2))
ax[2].set_xlabel("index - Growth increment step", fontsize="small")
ax[2].set_ylabel("Installed capacity per node group [MWp/Year]", fontsize="small")
ax[2].set_title("PV installed capacity growth", fontsize="small")
ax[2].tick_params(axis='both', which='major', labelsize="small")
ax[2].tick_params(axis='both', which='minor', labelsize="small")

ax_.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax_.set_ylabel("Total Grid Installed Capacity [MWp/Year]", fontsize="small")
ax_.legend(fontsize="small")
ax_.tick_params(axis='both', which='major', labelsize="small")
ax_.tick_params(axis='both', which='minor', labelsize="small")

ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylabel("PV Installed capacity growth", fontsize="small")
ax2.tick_params(axis='both', which='major', labelsize="small")
ax2.tick_params(axis='both', which='minor', labelsize="small")


# #%% Analysis of the growths
# #%%
# def normalizer(min_, max_):
#     return lambda x: ( (x - min_)/(max_ - min_) ) * 100  #  y = (x-xmin)/(xmax - xmin)   : Function

#%%

def segments(x_data, z_data):
    # Collect the line segments between the data and its projections.
    # This is to replicate the projection line from R:
    # https://www.r-bloggers.com/2016/04/principal-curves-example-elements-of-statistical-learning/
    # Code of line collection for matplotlib taken from:
    # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
    #
    # Arguments:
    # ----------
    #   x_data: np.array: Real data (n_samples, dimension)
    #   z_data: np.array: Projected data (n_samples, dimension)
    #
    #     Notes: It is assumed that each row of x_data and z_data forms the line segment
    #
    # Returns:
    #   line_collections: np.array: Collection of lines that connects the real and projected data
    #       the dimension of the array is (n_samples, dimension, dimension)
    #       e.g.,
    #       Matrix for each line must be :
    #               [[x_1, x_2],  -> Real point coordinates (x_1, x_2)
    #                [z_1, z_2]]  -> Projected point coordinates (z_1, z_2)
    #       All small matrices of 2 x 2 are collected together in a 3D matrix called line_collections:
    #       So, e.g. for sample 10,        x[10,:] == line_collections[10, 0,:]
    #                             z1_vectors[10,:] == line_collections[10, 1,:]

    stacked = np.dstack([x_data, z_data])
    line_collections = np.array([stacked[i].T for i in range(len(stacked))])  # (n_samples, d, d)

    return line_collections



def normalize_and_inverse(min_, max_):
    """
    Creates a normalizer function and its inverse that maps a value in GWh/Year to fraction of
    percentage, i.e., [0,1]

    Parameters:
    ----------
        min_: float: Minimum annual energy consumption per year of the whole grid [GWh/year]
        max_: float: Maximum annual energy consumption per year of the whole grid [GWh/year]
    """

    f =      lambda x: np.round(np.clip(( (x - min_)/(max_ - min_) ), 0 , 1), 4)  #  y = (x-xmin)/(xmax - xmin)   : Function  [0.0 - 1.0]
    f_inv =  lambda x: ( x * (max_ - min_) + min_ )  #  x = y*(xmax-xmin) + xmin   : Inverse Function [0-1] -> [min, max]

    return f, f_inv

def growth_curve_normalized(x, y, k=3, curve="spline"):
    """
    Create a growth function with support [0,1] and range [0,1].

    Parameters:
    -----------
        x: np.ndarray: Control points for the interpolation that corresponds to the x-axis
        y: np.ndarray: Control points for the interpolation that corresponds to the y-axis
        k: int: Degree of the spline to fit.
        curve: str: Type of interpolation for the control points.
            The options are: ["spline", "lineal", "root", "square"]

    Returns:
    --------
        g_normalized: Function g() that maps any value between [0,1] to a range of [0,1]
            Example: to use the function just type g(vale).

    """


    if curve == "spline":
        assert k is not None, "Spline degree must be defined."
        g_normalized = UnivariateSpline(x, y, k=k)
    elif curve == "lineal":
        cp = np.array([(0,0),
                       (1,1)])
        g_normalized = interp1d(cp[:,0], cp[:,1])
    elif curve == "root":
        g_normalized = lambda x: np.sqrt(x)

    elif curve == "square":
        g_normalized = lambda x: x ** 2

    else:
        raise NotImplementedError

    assert np.allclose(g_normalized(0), 0, atol=0.01), "The non-lineal function should map closed to f(0) == 0.0"
    assert np.allclose(g_normalized(1), 1, rtol=0.01), "The non-lineal function should map closed to f(1) == 1.0"

    return g_normalized

def growth_curve(g_normalized, min_, max_):
    """
    Computes the de-normalization of the normalized growth curve g() to values in GWh/year, also computes a
    """

    g = lambda x: g_normalized(x) * (max_ - min_) + min_

    x = np.linspace(0.0, 1.0, 10000)
    y = g(x)  # Values in kw

    # Nearest value procedure: The function returns a value between [0,1]
    g_inv = lambda w: np.round(x[[np.argmin(np.abs(y - w_)) for w_ in w]], 3)  # w: Values in GWh/year.

    return g, g_inv

#%%
def growth_plot(ax, minimum, maximum, control_points,
                corrected_curve=None, growth="load", type="spline", highlight_point=None):

    if highlight_point is not None:
        assert isinstance(highlight_point, int), "Highlighted point must be a integer value."

    if growth == "load":
        title = "Load growth network - w(l)"
        x_label = "Step increment (l) - [years]"
        y1_label = "Energy consumption [GWh/year] -> g(n)"
        y2_label = "Annual Energy consumption growth [%] -> f(g(n))"

        label1 = "g(m): [GWh/year]: l-step"
        label2 = "g(n): [GWh/year]: l-step - Interpolated"
        label3 = "f(g(m)): Energy[%]"
        label4 = "f(g(m)): Corrected path [%]"

    elif growth == "pv":
        title = "PV growth network - alpha(l)"
        x_label = "Step increment (l) - [years]"
        y1_label = "PV installed capacity [MWp] -> g(n)"
        y2_label = "PV installed capacity growth [%] -> f(g(n))"

        label1 = "g(m): [MWp]: l-step"
        label2 = "g(n): [MWp]: l-step - Interpolated"
        label3 = "f(g(m)): PV Ints. Capacity [%]"
        label4 = "f(g(m)): Corrected path [%]"

    else:
        raise NotImplementedError

    m = np.linspace(0, 1, 11)
    if corrected_curve is not None:
        x_corrected = np.linspace(min(m), max(m), num=100)
        y_corrected_load = pchip_interpolate(m, corrected_curve, x_corrected)


    f, f_inv = normalize_and_inverse(minimum, maximum)
    g_normalized = growth_curve_normalized(control_points[:, 0], control_points[:, 1], curve=type)
    g, g_inv = growth_curve(g_normalized, minimum, maximum)


    to_integer = 10  # Conversion of decimal points of the x axis to integer numbers

    m_segments = False  # Integer values  (x-values)
    n_segments = True  # Interpolated x-values.

    m = np.linspace(0, 1, 11)
    x_1, y_1_kw, y_1_per = (m, g(m), f(g(m)))  # index, GWh/year, % Energy of GWh/year

    n = g_inv(f_inv(m))
    x_2, y_2_kw, y_2_per = n, g(n), f(g(n))  # index, GWh/year, % Energy of GWh/year

    # Set of line segments for the interpolated values (n - index)
    s1 = np.array([np.zeros(len(n)), y_2_kw]).T  # Points of the y-axis on the left (Energy  GWh/year)
    s2 = np.array([n * to_integer, y_2_kw]).T  # Points in the middle of the figure (Energy GWh/year)

    s3 = np.array([n * to_integer, y_2_per]).T  # Points in the middle of the figure (Energy GWh/year)
    s4 = np.array([np.ones(len(n)) * to_integer, y_2_per]).T  # Points of the y-axis on the right (% Energy)

    s5 = np.array([n * to_integer, np.zeros(len(n))]).T  # Points over the x-axis
    s6 = np.array([n * to_integer, y_2_per]).T  # Points in the middle of the figure (% Energy)

    # Set of line segments for the integer values as index (m - index)
    w1 = np.array([np.zeros(len(m)), y_1_kw]).T  # Points of the y-axis on the left (Energy  GWh/year)
    w2 = np.array([m * to_integer, y_1_kw]).T  # Points in the middle of the figure (Energy GWh/year)

    w3 = np.array([m * to_integer, y_1_per]).T  # Points in the middle of the figure (% Energy)
    w4 = np.array([np.ones(len(m)) * to_integer, y_1_per]).T  # Points of the y-axis on the right (% Energy)

    w5 = np.array([m * to_integer, np.zeros(len(m))]).T  # Points over the x-axis
    w6 = np.array([m * to_integer, y_1_per]).T  # Points in the middle of the figure (% Energy)

    segments_kw_s = segments(x_data=s1, z_data=s2)
    segments_per_s = segments(x_data=s3, z_data=s4)
    segments_ver_s = segments(x_data=s5, z_data=s6)

    segments_kw_w = segments(x_data=w1, z_data=w2)
    segments_per_w = segments(x_data=w3, z_data=w4)
    segments_ver_w = segments(x_data=w5, z_data=w6)

    lc1_s = mc.LineCollection(segments_kw_s, colors="C1", linewidths=0.4, linestyles="-.")
    lc2_s = mc.LineCollection(segments_per_s, colors="C1", linewidths=0.4, linestyles="-.")
    lc3_s = mc.LineCollection(segments_ver_s, colors="C1", linewidths=0.4, linestyles="-.")

    lc1_w = mc.LineCollection(segments_kw_w, colors="C2", linewidths=0.4, linestyles="-.")
    lc2_w = mc.LineCollection(segments_per_w, colors="C2", linewidths=0.4, linestyles="-.")
    lc3_w = mc.LineCollection(segments_ver_w, colors="C2", linewidths=0.4, linestyles="-.")

    ax.scatter(m * to_integer, y_1_kw, s=30, marker="o", color="C2", label=label1, zorder=10)
    ax.plot(np.linspace(0, 1, 100) * to_integer, g(np.linspace(0, 1, 100)), "-", color="C2", zorder=1)

    # Percentage axis
    ax2 = ax.twinx()
    x_values_high_res = np.linspace(0, 1, 100)
    y_values_high_res = f(g(x_values_high_res))
    ax2.plot(x_values_high_res * to_integer, y_values_high_res, "-", color="C2", label=label3, zorder=1)

    if corrected_curve is not None:
        ax2.plot(x_corrected * to_integer, y_corrected_load, "-", color="C0", label=label4, zorder=1)
        ax2.scatter(m * to_integer, corrected_curve, s=30, marker="o", color="C0", zorder=10,
                    label=label4)

    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    if n_segments:
        ax.add_collection(lc1_s)
        ax2.add_collection(lc2_s)
        ax2.add_collection(lc3_s)
        # ax.set_xlabel("n", color="C1")
        ax.set_xlabel(x_label)

    if m_segments:
        ax.add_collection(lc1_w)
        ax2.add_collection(lc2_w)
        ax2.add_collection(lc3_w)
        # ax.set_xlabel("m", color="C2")
        ax.set_xlabel(x_label)

    if highlight_point is not None:
        ax2.scatter(m[highlight_point] * to_integer, y_1_per[highlight_point],
                   s=200, marker="+", color="C4", zorder=10)

    ax.scatter(x_2 * to_integer, y_2_kw, s=10, color="C1", marker="o", zorder=0,
               label=label2)
    ax.legend(fontsize="x-small")
    ax2.legend(fontsize="x-small", loc="lower right")
    ax.set_ylabel(y1_label)
    ax2.set_ylabel(y2_label)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(title)

    return y_values_high_res, y_1_per  # Growth values in high resolution and low res (every l-steps)


# Borders from the scrip: figures_static_regions.py ~line:1426:
borders = pd.read_csv("borders_danger_caution_90_percentile.csv", index_col=0)
borders_fill_undervoltage = pd.read_csv("borders_undervoltage_90_percentile.csv", index_col=0)
x_undervoltage = borders_fill_undervoltage["under_volt_fill_x"]
y_undervoltage = borders_fill_undervoltage["under_volt_fill_y"]

danger_border = borders["danger"].values
caution_border = borders["caution"].values

# Option 1: Reduce PV only
corrected_curve = np.array([np.linspace(0.0, 1.0, 11),
                            np.linspace(0.0, 1.0, 11)]).T
corrected_curve[:,1] = np.minimum(corrected_curve[:,1], caution_border * 0.9)

# Option 2: Increase consumption only
corrected_curve = np.array([np.linspace(0.0, 1.0, 11),
                            np.linspace(0.0, 1.0, 11)]).T
corrected_curve[:,1] = np.minimum(corrected_curve[:,1], caution_border)
corrected_curve[1:-1,0] = corrected_curve[1:-1,0] * 1.1

# Option 3: Decrease PV and truncate the growth of everything
corrected_curve = np.array([np.linspace(0.0, 1.0, 11),
                            np.linspace(0.0, 1.0, 11)]).T
corrected_curve[:,1] = np.minimum(corrected_curve[:,1], caution_border * 0.95)

idx_greater = corrected_curve[:,0] > x_undervoltage[0]
idx_less = corrected_curve[:,0] <= x_undervoltage[0]

corrected_curve[idx_greater, 0] = corrected_curve[idx_less, 0][-1] + 0.001
corrected_curve[idx_greater, 1] = corrected_curve[idx_less, 1][-1] + 0.001

# corrected_curve[1:-1,0] = corrected_curve[1:-1,0] * 1.1

# Option 4: Draw the thing by hand.
corrected_curve = np.array([[0.0, 0.0],
                            [0.2, 0.1],
                            [0.3, 0.2],
                            [0.45, 0.25],
                            [0.6, 0.3],
                            [0.7, 0.4],
                            [0.75, 0.45],
                            [0.8, 0.5],
                            [0.85, 0.6],
                            [0.9, 0.7],
                            [0.90001, 0.70001]])


# For the spline:
control_points = np.array([(0.0, 0.0),
                           (0.3, 0.4),
                           (0.5, 0.5),
                           (0.7, 0.5),
                           (1.0, 1.0)])

highlight_point = 7

# Compute the functions for the load growth
clusters = scenario_generator.cluster_labels
nodes_per_cluster = scenario_generator.nodes_per_cluster
minimum = []
maximum = []
for cluster_ in clusters:
    nodes = nodes_per_cluster[cluster_]
    minimum.append(scenario_generator.min_max_annual_energy_per_cluster[cluster_]["min"] * nodes)
    maximum.append(scenario_generator.min_max_annual_energy_per_cluster[cluster_]["max"] * nodes)
minimum = np.sum(minimum)
maximum = np.sum(maximum)

# Compute the minimum and maximum for the PV growth
minimum_pv = scenario_generator.grid_info["kwp"].divide(1000).sum()  # Installed capacotu values in MWp
maximum_pv = minimum_pv * 2

# Corrected points curve in fraction (percentage)
x_corrected = np.linspace(min(corrected_curve[:,0]), max(corrected_curve[:,0]), num=100)
y_corrected = pchip_interpolate(corrected_curve[:,0], corrected_curve[:,1], x_corrected)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
plt.subplots_adjust(right=0.95, wspace=0.5, left=0.05)
load_high_res, load_per = growth_plot(ax[0], minimum, maximum, control_points,
                                      corrected_curve[:,0],
                                      highlight_point=highlight_point,
                                      growth="load", type="spline")
pv_high_res, pv_per = growth_plot(ax[1], minimum_pv, maximum_pv, control_points,
                                  corrected_curve[:, 1],
                                  highlight_point=highlight_point,

                                  growth="pv", type="root")

ax[2].plot(load_high_res, pv_high_res, linewidth=2.0, color="C2", label="Growth path")
ax[2].scatter(load_per, pv_per, s=30, marker="o", color="C2", label="g(m): [GWh/year]: m: integer", zorder=10)

ax[2].scatter(corrected_curve[:,0], corrected_curve[:,1], s=30, marker="o", color="C0", label="Corrected curve")
ax[2].plot(x_corrected, y_corrected, color="C0")

ax[2].grid(linestyle="--", color="C1", linewidth=0.4, alpha=0.9)
ax[2].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax[2].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax[2].xaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax[2].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax[2].set_xlabel("Annual Energy consumption growth [%]")
ax[2].set_ylabel("PV Installed capacity growth [%]")
ax[2].set_title("Static regions")
ax[2].legend(fontsize="x-small")
ax[2].plot(x, danger_border, color="r", linewidth=1.0, alpha=1.0)
ax[2].plot(x, caution_border, color="#FFA500", linewidth=1.0, alpha=1.0,)
ax[2].fill_between(x, 0, caution_border, color="green", alpha=0.3, zorder=0,
                label="Safe")
ax[2].fill_between(x, caution_border,
                danger_border, color="#FFA500", alpha=0.3, zorder=0,
                label="Caution")
ax[2].fill_between(x, danger_border, 1, color="red", alpha=0.5, zorder=0,
                label="Overvoltage")
ax[2].vlines(x=x_undervoltage[0], ymin=0, ymax=y_undervoltage[0], color="b", linewidth=1.0, alpha=1.0, zorder=1)
ax[2].fill_between(x_undervoltage, 0, y_undervoltage, color="blue", alpha=0.5, zorder=0, label="Undervoltage")
ax[2].scatter(load_per[highlight_point], pv_per[highlight_point],
                   s=200, marker="+", color="C4", zorder=10)
ax[2].set_xlim(0,1)
ax[2].set_ylim(0,1)
#%%



