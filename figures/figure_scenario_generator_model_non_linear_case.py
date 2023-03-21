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
from scipy.interpolate import UnivariateSpline


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
