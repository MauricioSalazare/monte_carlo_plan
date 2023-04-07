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
from matplotlib.patches import Rectangle, Circle
from scipy.interpolate import UnivariateSpline, interp1d, pchip_interpolate
from core.figure_utils import set_figure_art
import matplotlib.pyplot as plt
import matplotlib

set_figure_art()
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# matplotlib.verbose.level = 'debug-annoying'

class OOMFormatter(mticker.ScalarFormatter):
    def __init__(self, order=0, fformat='%1.1f', offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        mticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def retrieve_energy_limits(scenario_generator):
    nodes_per_cluster = scenario_generator.nodes_per_cluster

    minimums_cluster = []
    maximus_cluster = []

    for ii in range(scenario_generator.n_clusters):
        minimums_cluster = minimums_cluster + \
                           [min(scenario_generator.mapper_load_growth[ii].values()) * nodes_per_cluster.loc[ii]]
        maximus_cluster = maximus_cluster + \
                          [max(scenario_generator.mapper_load_growth[ii].values()) * nodes_per_cluster.loc[ii]]

    data_frame = {"type_bus": list(range(scenario_generator.n_clusters)),
                  "minimums": minimums_cluster,
                  "maximums": maximus_cluster}

    energy_limits = pd.DataFrame(data_frame)
    energy_limits.set_index('type_bus', inplace=True)

    return energy_limits


def normalize_and_inverse(min_, max_):
    """
    Creates a normalizer function and its inverse that maps a value in GWh/Year to fraction of
    percentage, i.e., [0,1]

    Parameters:
    ----------
        min_: float: Minimum annual energy consumption per year of the whole grid [GWh/year]
        max_: float: Maximum annual energy consumption per year of the whole grid [GWh/year]
    """

    # f =      lambda x: np.round(np.clip(( (x - min_)/(max_ - min_) ), 0 , 1), 20)  #  y = (x-xmin)/(xmax - xmin)   : Function  [0.0 - 1.0]
    f = lambda x: np.round(np.clip(((x - min_) / (max_ - min_)),
                                   -0.5, 1.5),
                           20)  # y = (x-xmin)/(xmax - xmin)   : Function  [0.0 - 1.0]
    f_inv =  lambda x: ( x * (max_ - min_) + min_ )  #  x = y*(xmax-xmin) + xmin   : Inverse Function [0-1] -> [min, max]

    return f, f_inv

def growth_curve_normalized(x=None, y=None, k=3, curve="spline"):
    """
    Create a growth function with support [0,1] and range [0,1].

    Parameters:
    -----------
        x: np.ndarray: Control points for the interpolation that corresponds to the x-axis if curve=="spline"
        y: np.ndarray: Control points for the interpolation that corresponds to the y-axis if curve=="spline"
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

    elif curve == "corrected":
        g_normalized = lambda w: np.clip(pchip_interpolate(x, y, w), 0, 1)

    else:
        raise NotImplementedError

    if curve != "corrected":
        assert np.allclose(g_normalized(0), 0, atol=0.01), "The non-lineal function should map closed to f(0) == 0.0"
        assert np.allclose(g_normalized(1), 1, rtol=0.01), "The non-lineal function should map closed to f(1) == 1.0"

    return g_normalized

def growth_curve(g_normalized, min_, max_):
    """
    Computes the de-normalization of the normalized growth curve g() to values in GWh/year, also computes a
    """

    g = lambda x: g_normalized(x) * (max_ - min_) + min_

    x = np.linspace(0.0, 1.0, 100000)
    y = g(x)  # Values in kw

    # Nearest value procedure: The function returns a value between [0,1]
    g_inv = lambda w: np.round(x[[np.argmin(np.abs(y - w_)) for w_ in w]], 20)  # w: Values in GWh/year.

    return g, g_inv



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
ax.set_title("Normalized spline non-linear growth")


#%%
# *********************************************************************************************************************
# PLOTTING SECTION
# *********************************************************************************************************************

fig, ax = plt.subplots(3, 1, figsize=(4, 7))
plt.subplots_adjust(top=0.95, left=0.15, wspace=1.05, right=0.7, bottom=0.05)
font_size = 10
highlight_step = 6
highlight_power = 25

max_ylim_node = 25
to_integer = 10  # Conversion of decimal points of the x axis to integer numbers
x_solutions_years = []
# =====================================================================================================================
# LINEAL CASE
# =====================================================================================================================
#  Load growth

# The mapper_load_growth() exists, because each cluster growth is bounded by the max and min energy consumption of the
# cluster. Remember that copula can not extrapolate.

# Count the nodes with the same number of clusters

n_clusters = scenario_generator.n_clusters
x_ = np.arange(0, 11)
labels_ = ["Commercial", "Residential", "Mixed activities"]
colors = ["C0", "C1", "C2"]

# From the real data
energy_limits = retrieve_energy_limits(scenario_generator)
energy_limits.loc["total"] = energy_limits.sum(axis=0, numeric_only=True)

# Made up by myself
energy_limits = pd.DataFrame({"type_bus": [0, 1, 2],
                              "minimums": [1.566978, 3.0, 1.212650],
                              "maximums": [15.507259, 9.076219, 12.0]})
energy_limits.set_index("type_bus", inplace=True)
energy_limits.loc["total"] = energy_limits.sum(axis=0, numeric_only=True)

# m = np.linspace(0, 1, 11)
m = np.linspace(0, 1, 1000)

ax_ = ax[0].twinx()
f, f_inv = normalize_and_inverse(energy_limits["minimums"]["total"], energy_limits["maximums"]["total"])  # From GWh to [% Energy]
g_normalized = growth_curve_normalized(curve="lineal")
# g_normalized = growth_curve_normalized(x=points[:, 0], y=points[:, 1], curve="spline")
g, g_inv = growth_curve(g_normalized, energy_limits["minimums"]["total"], energy_limits["maximums"]["total"])
x_1, y_1_kw, y_1_per = (m, g(m), f(g(m)))  # index, GWh/year, % Energy of GWh/year
ax_.plot(m * to_integer, y_1_kw, color="C3", linewidth=1.8, zorder=0)

ax2 = ax_.secondary_yaxis(1.2, functions=(f, f_inv))
x_highlighted = g_inv(np.array([highlight_power]))
x_solutions_years.append(np.round(x_highlighted*10,2))
lgt = ax_.scatter(x_highlighted * to_integer, highlight_power, s=20, color="C3", edgecolors="k", zorder=3,
                  label=r"$w_{\mathrm{A}}(l_{\mathrm{A}})=\sum_{i=1}^{3}\bar{w}_{i,\mathrm{A}}(l_{\mathrm{A}})$")
lgt_label = lgt.get_label()
ax_.hlines(y=highlight_power, xmin=x_highlighted * to_integer, xmax=1.0 * to_integer, color="C3", linewidths=0.5, linestyles="-.", alpha=0.8)
ax_.vlines(x=x_highlighted * to_integer, ymin=0, ymax=highlight_power, color="C3", linewidths=0.5, linestyles="-.", alpha=0.8)


y_lineal = []
lgds = []
lgds_label = []
for ii in range(n_clusters):
    f, f_inv = normalize_and_inverse(energy_limits["minimums"][ii], energy_limits["maximums"][ii])  # From GWh to [% Energy]
    g_normalized = growth_curve_normalized(curve="lineal")
    # g_normalized = growth_curve_normalized(x=points[:,0], y=points[:,1], curve="spline")
    g, g_inv = growth_curve(g_normalized, energy_limits["minimums"][ii], energy_limits["maximums"][ii])
    x_1, y_1_kw, y_1_per = (m, g(m), f(g(m)))  # index, GWh/year, % Energy of GWh/year
    ax[0].plot(m * to_integer, y_1_kw)
    lgx = ax[0].scatter(x_highlighted * to_integer, g(x_highlighted), s=20,
                        color=colors[ii], edgecolors="k",
                        label=r"$\bar{w}_{\mathrm{" + f"{ii + 1}" + r",\mathrm{A}}}(l_{\mathrm{A}})$")
    lgds_label.append(lgx.get_label())
    ax[0].hlines(y=g(x_highlighted), xmin=0, xmax=x_highlighted*to_integer, color=colors[ii], linewidths=0.5, linestyles="-.", alpha=0.8)

    lgds.append(lgx)
    y_lineal.append(g(x_highlighted))


ax[0].set_ylim((0,max_ylim_node))
ax_.set_ylim((energy_limits["minimums"]["total"], 40))
ax_.set_ylabel("Total grid Energy consump. [GWh/Year]")
ax_.spines['right'].set_color('red')
ax_.yaxis.label.set_color('red')
ax_.tick_params(axis='y', colors='red')

ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel("Annual energy\nconsumption growth [\%]", fontsize=font_size)
ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
ax2.tick_params(axis='both', which='minor', labelsize=font_size-1)
ax[0].set_ylabel("Annual energy consumption\nper node [GWh/Year]")
ax[0].set_title("(a)")
lgds = [lgt] + lgds
lgds_label = [lgt_label] + lgds_label
ax[0].legend(lgds, lgds_label,
             loc="upper left",
             handletextpad=-0.1,
             handlelength=2.0)
ax[0].xaxis.set_major_locator(mticker.MultipleLocator(1.0))

# =====================================================================================================================
# SPLINE CASE
# =====================================================================================================================
#  Load growth

# The mapper_load_growth() exists, because each cluster growth is bounded by the max and min energy consumption of the
# cluster. Remember that copula can not extrapolate.

ax_ = ax[1].twinx()
f, f_inv = normalize_and_inverse(energy_limits["minimums"]["total"], energy_limits["maximums"]["total"])  # From GWh to [% Energy]
# g_normalized = growth_curve_normalized(curve="lineal")
g_normalized = growth_curve_normalized(x=points[:, 0], y=points[:, 1], curve="spline")
g, g_inv = growth_curve(g_normalized, energy_limits["minimums"]["total"], energy_limits["maximums"]["total"])
x_1, y_1_kw, y_1_per = (m, g(m), f(g(m)))  # index, GWh/year, % Energy of GWh/year
ax_.plot(m * to_integer, y_1_kw, color="C3", linewidth=1.8, zorder=0)

ax2 = ax_.secondary_yaxis(1.2, functions=(f, f_inv))
x_highlighted = g_inv(np.array([highlight_power]))
x_solutions_years.append(np.round(x_highlighted*10,2))
lgt = ax_.scatter(x_highlighted * to_integer, highlight_power, s=20, color="C3", edgecolors="k", zorder=3,
                  label=r"$w_{\mathrm{B}}(l_{\mathrm{B}})=\sum_{i=1}^{3}\bar{w}_{i,\mathrm{B}}(l_{\mathrm{B}})$")
lgt_label = lgt.get_label()
ax_.hlines(y=highlight_power, xmin=x_highlighted * to_integer, xmax=1.0 * to_integer, color="C3", linewidths=0.5, linestyles="-.", alpha=0.8)
ax_.vlines(x=x_highlighted * to_integer, ymin=0, ymax=highlight_power, color="C3", linewidths=0.5, linestyles="-.", alpha=0.8)

# Count the nodes with the same number of clusters
energy_highlighted = []
y_spline = []
lgds = []
lgds_label = []
for ii in range(n_clusters):
    f, f_inv = normalize_and_inverse(energy_limits["minimums"][ii], energy_limits["maximums"][ii])  # From GWh to [% Energy]
    # g_normalized = growth_curve_normalized(curve="lineal")
    g_normalized = growth_curve_normalized(x=points[:,0], y=points[:,1], curve="spline")
    g, g_inv = growth_curve(g_normalized, energy_limits["minimums"][ii], energy_limits["maximums"][ii])
    x_1, y_1_kw, y_1_per = (m, g(m), f(g(m)))  # index, GWh/year, % Energy of GWh/year
    ax[1].plot(m * to_integer, y_1_kw)
    lgx = ax[1].scatter(x_highlighted * to_integer, g(x_highlighted), s=20,
                        color=colors[ii], edgecolors="k", label=r"$\bar{w}_{\mathrm{" + f"{ii+1}" + r",B}}(l_{\mathrm{B}})$")
    lgds_label.append(lgx.get_label())
    ax[1].hlines(y=g(x_highlighted), xmin=0, xmax=x_highlighted * to_integer, color=colors[ii], linewidths=0.5, linestyles="-.", alpha=0.8)

    lgds.append(lgx)

    y_spline.append(g(x_highlighted))

ax[1].set_ylim((0,max_ylim_node))
ax_.set_ylim((energy_limits["minimums"]["total"], 40))
ax_.set_ylabel("Total grid Energy consump. [GWh/Year]")
ax_.spines['right'].set_color('red')
ax_.yaxis.label.set_color('red')
ax_.tick_params(axis='y', colors='red')

ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel("Annual energy\nconsumption growth [\%]", fontsize=font_size)
ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
ax2.tick_params(axis='both', which='minor', labelsize=font_size-1)
ax[1].set_ylabel("Annual energy consumption\nper node [GWh/Year]")
ax[1].set_title("(b)")
lgds = [lgt] + lgds
lgds_label = [lgt_label] + lgds_label
ax[1].legend(lgds, lgds_label,
             loc="upper left",
             handletextpad=-0.1,
             handlelength=2.0)
ax[1].xaxis.set_major_locator(mticker.MultipleLocator(1.0))

# =====================================================================================================================
# ROOT CASE
# =====================================================================================================================
#  Load growth

# The mapper_load_growth() exists, because each cluster growth is bounded by the max and min energy consumption of the
# cluster. Remember that copula can not extrapolate.

ax_ = ax[2].twinx()
f, f_inv = normalize_and_inverse(energy_limits["minimums"]["total"], energy_limits["maximums"]["total"])  # From GWh to [% Energy]
# g_normalized = growth_curve_normalized(curve="lineal")
g_normalized = growth_curve_normalized(x=points[:, 0], y=points[:, 1], curve="root")
g, g_inv = growth_curve(g_normalized, energy_limits["minimums"]["total"], energy_limits["maximums"]["total"])
x_1, y_1_kw, y_1_per = (m, g(m), f(g(m)))  # index, GWh/year, % Energy of GWh/year
ax_.plot(m * to_integer, y_1_kw, color="C3", linewidth=1.8, zorder=0)

# ax_.arrow(x=0.9, y=0, dx=0, dy=25,
#           length_includes_head=True,
#           head_width=0.04,
#           head_length=3,
#           shape="full",
#           overhang=1,
#           color="C3")
# ax_.text(x=0.82,y=10,s=r"$\bar{\mathrm{A}}$", fontsize="x-large", color="C3")

ax2 = ax_.secondary_yaxis(1.2, functions=(f, f_inv))
x_highlighted = g_inv(np.array([highlight_power]))
x_solutions_years.append(np.round(x_highlighted*10,2))
lgt = ax_.scatter(x_highlighted * to_integer, highlight_power, s=20, color="C3", edgecolors="k", zorder=3,
                  label=r"$w_{\mathrm{C}}(l_{\mathrm{C}})=\sum_{i=1}^{3}\bar{w}_{i,\mathrm{C}}(l_{\mathrm{C}})$")
lgt_label = lgt.get_label()
ax_.hlines(y=highlight_power, xmin=x_highlighted * to_integer, xmax=1.0 * to_integer, color="C3", linewidths=0.5, linestyles="-.", alpha=0.8)
ax_.vlines(x=x_highlighted * to_integer, ymin=0, ymax=highlight_power, color="C3", linewidths=0.5, linestyles="-.", alpha=0.8)

# Count the nodes with the same number of clusters
energy_highlighted = []
y_root = []

x_arrow_position = [0.25, 0.05, 0.15]
arrow_names = ["B", "C", "D"]
lgds = []
lgds_label = []
for ii in range(n_clusters):
    f, f_inv = normalize_and_inverse(energy_limits["minimums"][ii], energy_limits["maximums"][ii])  # From GWh to [% Energy]
    # g_normalized = growth_curve_normalized(curve="lineal")
    g_normalized = growth_curve_normalized(x=points[:,0], y=points[:,1], curve="root")
    g, g_inv = growth_curve(g_normalized, energy_limits["minimums"][ii], energy_limits["maximums"][ii])
    x_1, y_1_kw, y_1_per = (m, g(m), f(g(m)))  # index, GWh/year, % Energy of GWh/year
    ax[2].plot(m * to_integer, y_1_kw, color=colors[ii])
    lgx = ax[2].scatter(x_highlighted * to_integer, g(x_highlighted), s=20,
                        color=colors[ii], edgecolors="k",
                        label=r"$\bar{w}_{\mathrm{" + f"{ii+1}" + r",\mathrm{C}}}(l_{\mathrm{C}})$" )
    lgds_label.append(lgx.get_label())
    ax[2].hlines(y=g(x_highlighted), xmin=0, xmax=x_highlighted * to_integer, color=colors[ii], linewidths=0.5, linestyles="-.", alpha=0.8)

    lgds.append(lgx)
    # ax[2].arrow(x=x_arrow_position[ii], y=0, dx=0, dy=g(x_highlighted)[0],
    #             length_includes_head=True,
    #             head_width=0.04,
    #             head_length=1.5,
    #             shape="full",
    #             overhang=1,
    #             color=colors[ii])
    # ax[2].text(x=x_arrow_position[ii] + 0.0125, y=1.25,
    #            s=r"$\bar{\mathrm{"+f"{arrow_names[ii]}"+ "}}$", fontsize="x-large", color=colors[ii])
    y_root.append(g(x_highlighted))

ax[2].set_ylim((0,max_ylim_node))
ax_.set_ylim((energy_limits["minimums"]["total"], 40))
ax_.set_ylabel("Total grid Energy consump. [GWh/Year]")
ax_.spines['right'].set_color('red')
ax_.yaxis.label.set_color('red')
ax_.tick_params(axis='y', colors='red')

ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax2.set_ylabel("Annual energy\nconsumption growth [\%]", fontsize=font_size)
ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
ax2.tick_params(axis='both', which='minor', labelsize=font_size-1)

ax[2].set_ylabel("Annual energy consumption\nper node [GWh/Year]")
ax[2].set_title("(c)")
lgds = [lgt] + lgds
lgds_label = [lgt_label] + lgds_label
ax[2].legend(lgds, lgds_label,
             loc="upper left",
             handletextpad=-0.1,
             handlelength=2.0)
ax[2].xaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax[2].set_xlabel("Step increment ($l$) - [years]")

comparison = pd.DataFrame(np.concatenate([y_lineal, y_root, y_spline], axis=1).T,
                          columns=["w1", "w2", "w3"],
                          index=["lineal", "root", "spline"])
print(comparison)
print(f"Years: {x_solutions_years}")
plt.savefig(".\path_growths\paths_appendix.pdf", dpi=700, bbox_inches='tight' )
#%%
