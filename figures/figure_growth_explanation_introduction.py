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
set_figure_art()


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

fig, ax = plt.subplots(1, 2, figsize=(7, 4))
plt.subplots_adjust(top=0.85, left=0.09, wspace=1.05, right=0.83, bottom=0.45)
font_size = 10
highlight_step = 6
# =====================================================================================================================
# LINEAL CASE
# =====================================================================================================================
#  Load growth

# The mapper_load_growth() exists, because each cluster growth is bounded by the max and min energy consumption of the
# cluster. Remember that copula can not extrapolate.

# Count the nodes with the same number of clusters
nodes_per_cluster = scenario_generator.nodes_per_cluster
x_ = np.arange(0, 11)
labels_ = ["Commercial", "Residential", "Mixed activities"]

minimums_cluster = []
for ii in range(scenario_generator_non_lineal.n_clusters):
    minimums_cluster = minimums_cluster + list(scenario_generator.mapper_load_growth[ii].values())
minimums_cluster = min(minimums_cluster)

colors_ = ["C0", "C1", "C2"]
growth_curve_cluster = []
for cluster_, nodes_ in nodes_per_cluster.iteritems():
    y = np.array(list(scenario_generator.mapper_load_growth[cluster_].values()))
    n_nodes = nodes_per_cluster[cluster_]
    ax[0].plot(x_, y * nodes_,
               label=labels_[cluster_] + ": " + r"$\tilde{w}_{\phi \in \Omega_{\gamma=" + f"{cluster_ + 1}" + "}}(l)$" +  " (" + r"$\gamma" + fr"={cluster_ + 1}$" + ")",
               color=colors_[cluster_])
    ax[0].scatter(x_[highlight_step], y[highlight_step] * nodes_, s=20, marker="o", color=colors_[cluster_])
    ax[0].vlines(x=x_[highlight_step], ymin=0.0, ymax=y[highlight_step]* nodes_,
               linestyles="-.", linewidth=0.9, zorder=0, color=colors_[cluster_])
    ax[0].hlines(y=y[highlight_step]* nodes_, xmin=x_.min(), xmax=x_[highlight_step],
               linestyles="-.", linewidth=0.9, zorder=0, color=colors_[cluster_])


    growth_curve_cluster.append(y * n_nodes)  # Total energy increased by the cluster
growth_curve_cluster = np.array(growth_curve_cluster)
total_grid_energy_growth = growth_curve_cluster.sum(axis=0)

# circle = Circle((5, 15), radius=1, color='yellow')
# ax[0].add_patch(circle)

ax[0].plot(5, 13.75,'bo', fillstyle='none', markeredgecolor="r",markersize=15)  # Big circle around A.
ax[0].text(5, 13.75, "A", horizontalalignment='center', verticalalignment='center', fontsize="x-large")
ax[0].text(7, 17, "$w(l)$", horizontalalignment='center', verticalalignment='center', fontsize="xx-large", color="red")

ax_ = ax[0].twinx()
ax_.scatter(x_[highlight_step], total_grid_energy_growth[highlight_step], color="r", s=60, marker="o")  # Point highlighted
ax_.vlines(x=x_[highlight_step], ymin=total_grid_energy_growth.min(), ymax=total_grid_energy_growth[highlight_step],
           linestyles="-.", colors="C3", linewidth=0.9, zorder=0)
ax_.hlines(y=total_grid_energy_growth[highlight_step], xmin=x_[highlight_step], xmax=x_.max(),
           linestyles="-.", colors="C3", linewidth=0.9, zorder=0)
ax_.spines['right'].set_color('red')
ax_.yaxis.label.set_color('red')
ax_.tick_params(axis='y', colors='red')
ax_.plot(x_, total_grid_energy_growth, "-o", markersize=3.5,
         color="red", linewidth=2.0,
         label="Total grid energy\n" + r"$w(l) = \sum_{\phi \in \Omega_{\gamma}} \tilde{w}_{\phi}(l) \enspace \forall \enspace \gamma=\{1,2,3\}$")

axlines, axlabels = ax[0].get_legend_handles_labels()
ax_line, ax_label = ax_.get_legend_handles_labels()

ax_lines_ = axlines + ax_line
ax_label_ = axlabels + ax_label

# legend_list.append(lgx[0])
# labs = [l.get_label() for l in legend_list]

max_ = total_grid_energy_growth.max()
min_ = total_grid_energy_growth.min()

f_x_y = lambda x: ( (x - min_)/(max_ - min_) ) * 100  #  y = (x-xmin)/(xmax - xmin)   : Function
f_y_x = lambda x: ( x * (max_ - min_) + min_ ) * 100  #  x = y * (xmax - xmin) + xmin : Inverse function

ax2 = ax_.secondary_yaxis(1.3, functions=(f_x_y, f_y_x))

ax[0].set_ylim(0, 20.0)
ax[0].set_ylabel("Annual energy consumption\naggregated per cluster [GWh/Year]", fontsize=font_size)
ax[0].legend(ax_lines_, ax_label_,
             fontsize=font_size,
             title_fontsize="small",
             loc='upper center',
             handlelength=1,
             bbox_to_anchor=(0.5, -0.28),)
ax[0].set_xlabel("Step increment ($l$) - [years]", fontsize=font_size)
ax[0].set_title("(c)\nAnnual energy\ngrowth -" + r" $w(l)$", fontsize=font_size)
ax[0].xaxis.set_major_locator(mticker.MultipleLocator(1))
ax[0].tick_params(axis='both', which='major', labelsize=font_size-1)
ax[0].tick_params(axis='both', which='minor', labelsize=font_size-1)

ax_.set_ylim(2,40.0)
ax_.set_ylabel("Total grid ann. energy cons. [GWh/Year]", fontsize=font_size)
# ax_.legend(loc="lower right", fontsize=font_size)
ax_.tick_params(axis='both', which='major', labelsize=font_size-1)
ax_.tick_params(axis='both', which='minor', labelsize=font_size-1)
ax_.yaxis.set_major_locator(mticker.MultipleLocator(5))

ax[0].yaxis.set_major_locator(mticker.MultipleLocator(2.5))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylabel("Annual energy consumption growth [\%]", fontsize=font_size)
ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
ax2.tick_params(axis='both', which='minor', labelsize=font_size-1)

# ax.tick_params(axis='both', which='major', labelsize=10)
# ax.tick_params(axis='both', which='minor', labelsize=8)


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

colors_ = ["C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]
for idx, (kwp_, count_) in panel_instalations.iterrows():
    panel_mwp_grid_curve = ((1 + x_2 / 10) * kwp_) / 1000

    if idx == len(panel_instalations) - 1:
        label_text = r"Bus $B$: $\tilde{\alpha}_{B}(l)$"
    else:
        label_text = f"Bus {idx+1}: " + r"$\tilde{\alpha}_{" + f"{idx+1}" + "}(l)$"

    ax[1].plot(x_2, panel_mwp_grid_curve, linewidth=1.0, label=label_text, color=colors_[idx])  # Total installed capacity in (MWp)
    ax[1].scatter(x_2[highlight_step], panel_mwp_grid_curve[highlight_step], s=20, marker="o", color=colors_[idx])
    ax[1].vlines(x=x_2[highlight_step], ymin=0.0, ymax=panel_mwp_grid_curve[highlight_step],
                 linestyles="-.", linewidth=0.9, zorder=0, color=colors_[idx])
    ax[1].hlines(y=panel_mwp_grid_curve[highlight_step], xmin=x_2.min(), xmax=x_2[highlight_step],
                 linestyles="-.", linewidth=0.9, zorder=0, color=colors_[idx])

ax[1].plot(5, 0.8,'bo', fillstyle='none', markeredgecolor="r",markersize=15)  # Big circle around the letter A.
ax[1].text(5, 0.8, "A", horizontalalignment='center', verticalalignment='center', fontsize="x-large")
ax[1].text(7, 1.01, r"$\alpha(l)$", horizontalalignment='center', verticalalignment='center', fontsize="xx-large", color="red")

ax_ = ax[1].twinx()
ax_.plot(x_2, total_mwp_grid_curve, "-o", markersize=3.5,
         color="red", linewidth=2.0,
         label="Total PV grid capacity: " + r"$\alpha(l) = \sum_{i=1}^{B} \tilde{\alpha}_{i}(l)$")  # Total installed capacity in (MWp)
ax_.scatter(x_[highlight_step], total_mwp_grid_curve[highlight_step], color="r", s=60, marker="o")
ax_.vlines(x=x_[highlight_step], ymin=total_mwp_grid_curve.min(), ymax=total_mwp_grid_curve[highlight_step],
           linestyles="-.", colors="C3", linewidth=0.9, zorder=0)
ax_.hlines(y=total_mwp_grid_curve[highlight_step], xmin=x_[highlight_step], xmax=x_.max(),
           linestyles="-.", colors="C3", linewidth=0.9, zorder=0)
ax_.spines['right'].set_color('red')
ax_.yaxis.label.set_color('red')
ax_.tick_params(axis='y', colors='red')

axlines, axlabels = ax[1].get_legend_handles_labels()
ax_line, ax_label = ax_.get_legend_handles_labels()

r = Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
ax_lines_ = axlines[:2] + [r] + [axlines[-1]] + ax_line
ax_label_ = axlabels[:2] + [r"$\vdots$"] + [axlabels[-1]] + ax_label

min_kwp = total_mwp_grid_curve.min()
max_kwp = total_mwp_grid_curve.max()

f_x_y = lambda x: ( (x - min_kwp)/(max_kwp - min_kwp) ) * 100  #  y = (x-xmin)/(xmax - xmin)   : Function
f_y_x = lambda x: ( x * (max_kwp - min_kwp) + min_kwp ) * 100  #  x = y * (xmax - xmin) + xmin : Inverse function

ax2 = ax_.secondary_yaxis(1.35, functions=(f_x_y, f_y_x))
ax[1].xaxis.set_major_locator(mticker.MultipleLocator(1))
ax[1].set_ylim((0.0, 1.2))
ax[1].set_xlabel("Step increment ($l$) - [years]", fontsize=font_size)
ax[1].set_ylabel("PV Installed capacity per node [MWp]", fontsize=font_size)
ax[1].legend(ax_lines_, ax_label_,
             fontsize=font_size,
             title_fontsize="small",
             loc='upper center',
             handlelength=1,
             labelspacing=0.3,
             bbox_to_anchor=(0.5, -0.28))

ax[1].set_title("(d)\nPV installed capacity\ngrowth - " + r"$\alpha (l)$", fontsize=font_size)
ax[1].tick_params(axis='both', which='major', labelsize=font_size-1)
ax[1].tick_params(axis='both', which='minor', labelsize=font_size-1)

ax_.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax_.set_ylabel("Total grid PV installed capacity [MWp]", fontsize=font_size)
# ax_.legend(fontsize=font_size)
ax_.tick_params(axis='both', which='major', labelsize=font_size-1)
ax_.tick_params(axis='both', which='minor', labelsize=font_size-1)

ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylabel("PV Installed capacity growth [\%]", fontsize=font_size)
ax2.tick_params(axis='both', which='major', labelsize=font_size-1)
ax2.tick_params(axis='both', which='minor', labelsize=font_size-1)

plt.savefig('introduction_figure/load_and_pv_growth_introduction.pdf', dpi=700, bbox_inches='tight')