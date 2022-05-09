import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import warnings
from pathlib import Path
from scipy import interpolate
from core.figure_utils import set_figure_art
from typing import List, Tuple
import numpy as np
from scipy.interpolate import griddata
import pickle
set_figure_art()
mpl.rc('text', usetex=False)


def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file_:
        scenario_generator = pickle.load(pickle_file_)
    return scenario_generator


def load_solutions(file_name_solutions):
    with open(file_name_solutions, "rb") as pickle_file_:
        solutions = pickle.load(pickle_file_)
    return solutions


def plot_quantile_mixture_case(solution_quantile,
                               ax,
                               max_quantile_highlight: str = None,
                               min_quantile_highlight: str = None,
                               plot_min_voltages = True
                               ):
    """Plots the quantiles of the daily voltage profiles"""

    # max_quantile_keys = ["95", "90", "75", "50", "25", "10", "05"]
    max_quantile_keys = ["95", "90",  "75", "50"]  # Max and min quantile keys must have the same length
    min_quantile_keys = ["05", "10", "25", "50"]
    linestyles = ["dashdot", "dashed", "dotted", "solid"]
    colors = ["r", "b", "g", "r", "g", "b", "r"]

    if max_quantile_highlight is None:
        max_quant_enhance = "90"
    else:
        max_quant_enhance = max_quantile_highlight

    if min_quantile_highlight is None:
        min_quant_enhance = "05"
    else:
        min_quant_enhance = min_quantile_highlight

    for max_quantile_, min_quantile_, linestyle_, color_ in zip(max_quantile_keys, min_quantile_keys, linestyles, colors):
        if max_quantile_ == max_quant_enhance:
            ax.plot(solution_quantile["max_q_" + max_quantile_], linestyle=linestyle_, color="k", linewidth=1.1,
                    label=max_quantile_ + r" \%", marker="o", markersize=1)
        else:
            ax.plot(solution_quantile["max_q_" + max_quantile_], linestyle=linestyle_, color="k", linewidth=0.8,
                    label=max_quantile_ + r" \%")

        if plot_min_voltages:
            if min_quantile_ == min_quant_enhance:
                ax.plot(solution_quantile["min_q_" + min_quantile_], linestyle=linestyle_, color="grey", linewidth=1.1,
                        label=min_quantile_ + r" \%", marker="o", markersize=1)
            else:
                ax.plot(solution_quantile["min_q_" + min_quantile_], linestyle=linestyle_, color="grey", linewidth=0.8,
                        label=min_quantile_ + r" \%")



def get_coords_max_values(profile):
    """Find the maximum voltage and the position in the profile vector"""
    x_max = np.argmax(profile)
    y_max = profile[x_max]

    return x_max, y_max

def get_coords_min_values(profile):
    """Find the minimum voltage and the position in the profile vector"""
    x_min = np.argmin(profile)
    y_min = profile[x_min]

    return x_min, y_min

def get_solutions_dictionary(path_file_solutions, quantiles: List[str]) -> dict:
    """
    Read pickle file with all the processed quantile solution from the HPC simulation

    Returns:
    -------
        solution_dict: dict: Nested dictionary with quantiles of the HPC simulation. The structure of the dictionary is:

            solution_dict[((mixture), load, pv)][quantile] -> (1-D) np.ndarray (N,) where N is the nodes in the grid

            The values of the array is the max or min quantile value of all scenarios for each node.

            quantile can be: "max_q_{%}" of "min_q_{%}", where % can be numbers: [50, 75, 90, 95]


        quantiles: list: can be numbers: [50, 75, 90, 95]
    """

    solutions_dict = {}

    for quant_name in quantiles:
        file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quant_name}.pkl"
        with open(path_file_solutions / file_name_solutions_dictionary, "rb") as pickle_file:
            file_solutions_dict = pickle.load(pickle_file)
            solutions_dict[quant_name] = file_solutions_dict

    return solutions_dict

def get_matrices_critical_quantiles(quantile: str,
                                    all_mixtures_irradiance_only: list,
                                    solutions_dict: dict,
                                    pv_growth_percentiles: np.ndarray,
                                    load_growth_percentiles: np.ndarray,
                                    process_max: bool = True) -> List[np.ndarray]:
    """
    Computes the max or min voltage of all grid's nodes and all time steps for the all the mixture combinations and
    combinations of pv and load growth.

    Parameter:
    ----------
        quantile: str: Quantile to compute. Can be ["5", "10", "25", "50", "75", "90", "95"]
        all_mixtures: set: Set of tuples with the combinations of mixture of type of day e.g., overcast, sunny, dark.
            The order of the tuple is (overcast, sunny, dark).
        solutions_dict: dict: Dictionary with all the solutions from the HPC simulation.
            Check get_solutions_dictionary() method for more information.
        pv_growth_percentiles: np.ndarray: (1-D) Vector with all the increments steps of the PV growth.
        load_growth_percentiles: np.ndarray: (1-D) Vector with all the increments steps of the load growth.
        process_max: bool: True if compute the maximum voltages. False for the minimum voltages.

    Return:
    -------
        list_matrices: list: Each item in the list is a (2-D) matrix, each one of the mixture combinations for one
            particular quantile specified by parameter quantile.
            e.g., list_matrices[0] -> for mixture_case_0 -> np.ndarray(np.ndarray of max( (load_0, pv_0) ),
                                                                       np.ndarray of max( (load_1, pv_0) ),
                                                                       ...
                                                                       np.ndarray of max( (load_x, pv_x) )]

                  list_matrices[1] -> for mixture_case_1 -> np.ndarray(np.ndarray of max( (load_0, pv_0) ),
                                                                       np.ndarray of max( (load_1, pv_0) ),
                                                                       ...
                                                                       np.ndarray of max( (load_x, pv_x) )]

    """

    list_matrices = []
    matrix_voltages = np.zeros((len(pv_growth_percentiles), len(load_growth_percentiles)))
    for mixture_case in all_mixtures_irradiance_only:
        for i, pv in enumerate(pv_growth_percentiles):  # x == pv growth percentiles
            for j, load in enumerate(load_growth_percentiles):  # y == load growth percentiles
                case = (mixture_case, load, pv)
                if process_max:
                    matrix_voltages[i, j] = np.max(solutions_dict[case]["max_q_" + quantile])
                else:
                    matrix_voltages[i, j] = np.min(solutions_dict[case]["min_q_" + quantile])
        list_matrices.append(matrix_voltages.copy())

    return list_matrices


def plot_contour_levels_irradiance_days(list_matrices,
                                        contour_level,
                                        linecolor_contour,
                                        linecolor_quantile,
                                        pv_growth_percentiles,
                                        load_growth_percentiles,
                                        ax,
                                        alpha_line_quantile=1.0,
                                        contour_linewidths=0.2,
                                        quantile_linewdith=1.0,
                                        plot_quantiles=True) -> Tuple[np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray]:

    y_case = []
    for matrix_ in list_matrices:  # Each matrix has a different irradiance day combination
        try:
            cs = ax.contour(pv_growth_percentiles, load_growth_percentiles, matrix_,
                            colors=linecolor_contour, levels=[contour_level], linewidths=contour_linewidths)
            p = cs.collections[0].get_paths()[0]
            v = p.vertices
            x_ = v[:, 0]
            y_ = v[:, 1]
            f = interpolate.interp1d(x_, y_, fill_value='extrapolate')
            ynew = f(pv_growth_percentiles)

        except UserWarning:
            print("No contour warning")
            ynew = np.empty(len(pv_growth_percentiles))
            ynew[:] = np.nan

        except RuntimeWarning:
            print("Divide by zero")
            ynew = np.empty(len(pv_growth_percentiles))
            ynew[:] = np.nan

        y_case.append(ynew)

    y_case = np.array(y_case)

    q_05 = np.nanquantile(y_case, q=0.05, axis=0)
    q_50 = np.nanquantile(y_case, q=0.50, axis=0)
    q_95 = np.nanquantile(y_case, q=0.95, axis=0)

    if plot_quantiles:
        ax.plot(pv_growth_percentiles, q_50, color=linecolor_quantile, linewidth=quantile_linewdith,
                alpha=alpha_line_quantile)
        ax.plot(pv_growth_percentiles, q_95, color=linecolor_quantile, linewidth=quantile_linewdith, linestyle="--",
                alpha=alpha_line_quantile)
        ax.plot(pv_growth_percentiles, q_05, color=linecolor_quantile, linewidth=quantile_linewdith, linestyle="-.",
                alpha=alpha_line_quantile)

    return q_05, q_50, q_95, y_case

def get_high_res_plane(quantile: str,
                       mixture_irradiance: list,
                       solutions_dictx: dict,
                       x:np.ndarray,
                       y: np.ndarray,
                       process_max=True):

    list_matrices = get_matrices_critical_quantiles(quantile,
                                                    all_mixtures_irradiance_only=mixture_irradiance,
                                                    solutions_dict=solutions_dictx[quantile],
                                                    pv_growth_percentiles=x,
                                                    load_growth_percentiles=y,
                                                    process_max=process_max)
    data_z = list_matrices[0]
    X, Y = np.meshgrid(x, y)

    Z_ = data_z.reshape(1, -1).squeeze()
    X_ = X.reshape(1, -1).squeeze()
    Y_ = Y.reshape(1, -1).squeeze()

    N = 150
    gr_x = np.linspace(x.min(), x.max(), N)
    gr_y = np.linspace(y.min(), y.max(), N)
    grid_x, grid_y = np.meshgrid(gr_x, gr_y)

    Ti = griddata((X_, Y_), Z_, (X, Y), method="cubic")
    Ti_high_res = griddata((X_, Y_), Z_, (grid_x, grid_y), method="cubic")

    return Ti, Ti_high_res, (X, Y), (grid_x, grid_y)


file_name_scenario_generator_model = "../models/scenario_generator_model_new_AWS.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
cases_combinations_scenario_generator = scenario_generator.cases_combinations
x = scenario_generator.percentages_pv_growth
y = scenario_generator.percentages_load_growth


#%%
critical_quantiles = ["05", "10", "15", "25", "50", "75", "90", "95"]
path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
solutions_dictx_new = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)

# Re-arrange dictionary, so the code in the first figure can work
cases_combinations = list(set(solutions_dictx_new["50"].keys()))
cases_combinations_irradiance_only_tuples = list(set([irradiance_mixture for irradiance_mixture, _, _ in cases_combinations]))
solutions_dict = {}
for mixture_comb in cases_combinations:
    solutions_dict[mixture_comb] = {}
    for quantile_name in critical_quantiles:
        solutions_dict[mixture_comb][f"max_q_{quantile_name}"] = solutions_dictx_new[quantile_name][mixture_comb][f"max_q_{quantile_name}"]
        solutions_dict[mixture_comb][f"min_q_{quantile_name}"] = solutions_dictx_new[quantile_name][mixture_comb][f"min_q_{quantile_name}"]


#%%
QUANTILE = "95"  # Quantile to plot in the first figure
MIN_QUANTILE_HIGHLIGHT = "05"

# (cloudy, sunny, overcast)
mixture_cases_plot = [(0.4, 0.0, 0.6),
                      (0.2, 0.6, 0.2),  # Normal case (Original)
                      # (0.3, 0.3, 0.4),
                      (0.0, 1.0, 0.0)]

list_matrices_plot = []
list_of_tuple_cases = []
matrix_voltages = np.zeros((len(x), len(y)))
for mixture_case in mixture_cases_plot:
    for i, pv in enumerate(x):
        for j, load in enumerate(y):
            case = (mixture_case, load, pv)
            list_of_tuple_cases.append(case)
            matrix_voltages[i, j] = np.max(solutions_dict[case]["max_q_" + QUANTILE])
    list_matrices_plot.append(matrix_voltages.copy())
warnings.filterwarnings("default")

# =====================================================================================================================
# FIGURE 1: Heat map with one contour plot and quantiles of daily voltage profile
# =====================================================================================================================

ax_ = np.empty((5, 3), dtype=object)
fig = plt.figure(figsize=(7, 8.5))
gs0 = gridspec.GridSpec(2, 1, figure=fig, wspace=0.2, hspace=0.08, left=0.11, bottom=0.11, right=0.98, top=0.94,
                        height_ratios=[1, 2.2], width_ratios=[1])  # Two columns to hold the different plots
gs00 = gs0[0].subgridspec(1, 3, wspace=0.35)  # Colormap
gs01 = gs0[1].subgridspec(4, 3, wspace=0.35, hspace=0.08)  # Voltage profiles

for ii in range(3):
    ax_[0, ii] = fig.add_subplot(gs00[ii])

kk = 0
for ii in range(4):
    for jj in range(3):
        ax_[ii + 1, jj] = fig.add_subplot(gs01[kk])
        kk += 1

ax_0_row = ax_[0, :].flatten()
ax_1_row = ax_[1, :].flatten()
ax_2_row = ax_[2, :].flatten()
ax_3_row = ax_[3, :].flatten()
ax_4_row = ax_[4, :].flatten()

# Combinations of load and pv that lies in the contour line
special_load_pv_comb = [(0.5, 0.5),
                        (0.5, 0.4),
                        (0.8, 0.6)]

special_load_pv_comb = [(0.5, 0.5),
                        (0.5, 0.5),
                        (0.5, 0.5)]

special_rectangles = [] # (x0, y0) cooridnates of the purple boxes
for (special_load, special_pv) in special_load_pv_comb:
    special_rectangles.append((np.round(special_load - 0.05, 3),
                               np.round(special_pv - 0.05, 3)))

min_voltage_cbar, max_voltage_cbar = 0.998, 1.095

max_technical_voltage_green = 1.040
max_technical_voltage_caution = 1.045
max_technical_voltage_danger = 1.05

min_technical_voltage = 0.96

titles_list = ["(a)", "(b)", "(c)"]
norm_individual = mpl.colors.Normalize(vmin=min_voltage_cbar, vmax=max_voltage_cbar)

for ii, (ax_0, ax_1, ax_2, ax_3, ax_4, mixture_case) in enumerate(zip(ax_0_row,
                                                                      ax_1_row,
                                                                      ax_2_row,
                                                                      ax_3_row,
                                                                      ax_4_row,
                                                                      mixture_cases_plot)):
    # Explanation of the variables
    # mixture_case = (overcast, sunny, dark)
    # x, y combination (Load, PV)
    data_mixture_case = [solutions_dict[(mixture_case, 0.0, 1.0)],
                         solutions_dict[(mixture_case, 1.0, 0.0)],
                         solutions_dict[(mixture_case, 0.0, 0.0)],
                         solutions_dict[(mixture_case, *special_load_pv_comb[ii])]]  # Case over contour line



    coords_max = []
    coords_min = []

    for data_mixture_ in data_mixture_case:
        coords_max.append(get_coords_max_values(data_mixture_["max_q_" + QUANTILE]))
        coords_min.append(get_coords_min_values(data_mixture_["min_q_" + MIN_QUANTILE_HIGHLIGHT]))

    colors_ = ["r", "g", "b", "purple"]

    ax_0.pcolor(x, y, list_matrices_plot[ii], shading='auto', vmin=min_voltage_cbar, vmax=max_voltage_cbar)
    ax_0.set_xlabel("Load growth", fontsize="x-large")
    ax_0.set_ylabel("PV growth", fontsize="x-large")
    ax_0.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('viridis')), ax=ax_0,
                        location="bottom", pad=0.18)
    cbar.ax.set_xlabel("Max. grid voltage", fontsize="x-large")

    for color_, (_, y_max_) in zip(colors_, coords_max):
        cbar.ax.vlines(x=y_max_, ymin=0, ymax=10, linewidth=2, color=color_)

    cs1 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_danger], linewidths=1.5,
                       linestyles="solid")
    cs2 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_caution], linewidths=1.5,
                       linestyles="dashed")
    cs3 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_green], linewidths=1.5,
                       linestyles="dashdot")

    ax_0.clabel(cs1, inline=True, fontsize=7, colors='k')
    ax_0.clabel(cs2, inline=True, fontsize=7, colors='k')
    ax_0.clabel(cs3, inline=True, fontsize=7, colors='k')



    ax_0.set_title(f"{titles_list[ii]}\n"
                   r"$(\pi_1=" + f"{mixture_case[0]}" + r"," +
                   r"\pi_2=" + f"{mixture_case[1]}" + r"," +
                   r"\pi_3=" + f"{mixture_case[2]}" + r")$",
                   fontsize="x-large")

    rect_upper_left = patches.Rectangle((-0.05, 0.95), width=0.1, height=0.1, linewidth=2, edgecolor='r',
                                        facecolor='none', linestyle="-")
    rect_low_right = patches.Rectangle((0.95, -0.05), width=0.1, height=0.1, linewidth=2, edgecolor='g',
                                       facecolor='none', linestyle="-")
    rect_low_left = patches.Rectangle((-0.05, -0.05), width=0.1, height=0.1, linewidth=2, edgecolor='b',
                                      facecolor='none', linestyle="-")
    rect_contour = patches.Rectangle(special_rectangles[ii], width=0.1, height=0.1, linewidth=2, edgecolor='purple',
                                     facecolor='none', linestyle="-")
    rect_set = [rect_low_left, rect_low_right, rect_upper_left, rect_contour]

    for rect in rect_set:
        ax_0.add_patch(rect)

    # =================================================================================================================
    # Lower subplots (Profiles with quantiles)
    # =================================================================================================================

    ax_row = [ax_1, ax_2, ax_3, ax_4]  # Rows

    for ax_col, color_, data_mixture_, (x_max_, y_max_), (x_min_, y_min_) in zip(ax_row, colors_, data_mixture_case, coords_max, coords_min):
        plot_quantile_mixture_case(data_mixture_,
                                   ax=ax_col,
                                   max_quantile_highlight=QUANTILE,
                                   min_quantile_highlight=QUANTILE,
                                   plot_min_voltages=True)

        ax_col.text(x=x_max_ / 2,
                    y=y_max_ * 1.008,
                    s=str(round(y_max_, 3)) + " [p.u]", ha="center", va="center", fontsize="large",
                    color=color_)
        ax_col.hlines(y=y_max_, xmin=0, xmax=x_max_, color="k", linewidth=0.8, linestyles="--")

        x_right_limit = ax_col.get_xlim()[1]

        ax_col.text(x=x_min_ + (x_right_limit - x_min_) / 2 - 5,
                    y=y_min_ * (2 - 1.008),
                    s=str(round(y_min_, 3)) + " [p.u]", ha="center", va="center", fontsize="large",
                    color=color_)
        ax_col.hlines(y=y_min_, xmin=x_min_, xmax=x_right_limit, color="k", linewidth=0.8, linestyles="--")

        for spines in ax_col.spines.values():
            spines.set_color(color_)
            spines.set_linewidth(1.0)

    ax_1.set_xticklabels(labels=[])
    ax_2.set_xticklabels(labels=[])
    ax_3.set_xticklabels(labels=[])
    ax_4.set_xlabel("Time step")

    if ii == 0:  # ii Stands for columns in the plot
        ax_1.set_ylabel(f"Load: 0.0\nPV: 1.0" + "\nVoltage [p.u.]", fontsize="large")
        ax_2.set_ylabel(f"Load: 1.0\nPV: 0.0" + "\nVoltage [p.u.]", fontsize="large")
        ax_3.set_ylabel(f"Load: 0.0\nPV: 0.0" + "\nVoltage [p.u.]", fontsize="large")
        ax_4.set_ylabel(f"Load: 0.5\nPV: 0.5" + "\nVoltage [p.u.]", fontsize="large")
        # ax_4.set_ylabel(f"Voltage [p.u.]", fontsize="large")

        # Plot legend of maximum quantiles
        lh, ll = ax_1.get_legend_handles_labels()
        idx_max_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 0]  # Even number

        max_lh = [lh[ii] for ii in idx_max_values]
        max_ll = [ll[ii] for ii in idx_max_values]

        leg = ax_4.legend(max_lh,
                          max_ll,
                          fontsize="large",
                          bbox_to_anchor=(0.75, -0.25),
                          loc="upper left",
                          ncol=2,
                          title="Max. voltage percentiles",
                          title_fontsize="large",
                          handlelength=1.5)

        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)




    if ii == 2:
        lh, ll = ax_4.get_legend_handles_labels()
        idx_min_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 1]  # Odd number

        min_lh = [lh[ii] for ii in idx_min_values]
        min_ll = [ll[ii] for ii in idx_min_values]

        leg = ax_4.legend(min_lh,
                          min_ll,
                          fontsize="large",
                          bbox_to_anchor=(-0.75, -0.25),
                          loc="upper left",
                          ncol=2,
                          title="Min. voltage percentiles",
                          title_fontsize="large",
                          # handleheight=10,
                          handlelength=1.5)

        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)

    # # Limits to see only maximum voltages correctly:
    # ax_1.set_ylim((0.99, 1.12))
    # ax_2.set_ylim((0.98, 1.05))
    # ax_3.set_ylim((0.98, 1.12))
    # ax_4.set_ylim((0.98, 1.12))

    # # Limits to see correctly the minimum voltages
    # ax_1.set_ylim((0.991, 1.01))
    # ax_2.set_ylim((0.95, 1.01))
    # ax_3.set_ylim((0.991, 1.01))
    # ax_4.set_ylim((0.97, 1.01))

    # Limits to see minimum and maximum voltages correctly
    ax_1.set_ylim((0.968, 1.115))
    ax_2.set_ylim((0.935, 1.028))
    ax_3.set_ylim((0.981, 1.065))
    ax_4.set_ylim((0.96, 1.08))



lh, ll = ax_4.get_legend_handles_labels()
idx_max_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 0]  # Even number
idx_min_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 1]  # Odd number

max_lh = [lh[ii] for ii in idx_max_values]
max_ll = [ll[ii] for ii in idx_max_values]

min_lh = [lh[ii] for ii in idx_min_values]
min_ll = [ll[ii] for ii in idx_min_values]


plt.savefig('static_regions/static_region_border.pdf', dpi=700, bbox_inches='tight')

#%%
# =====================================================================================================================
# FIGURE 2: Static operating zones
# =====================================================================================================================

max_technical_voltage_danger = 1.05
warnings.filterwarnings("error")
critical_quantile_max = ["75", "90", "95"]
critical_quantile_min = ["25", "10", "05"]
titles_list = ["(a)", "(b)", "(c)"]
min_quant_span = [0.90, 0.88, 0.86]  # This is calculated zooming into the figure and see the vertical contours
critical_quantiles = list(set(critical_quantile_min + critical_quantile_max))
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)

fig, ax_t = plt.subplots(1, 3, figsize=(7, 2.8))
plt.subplots_adjust(left=0.08, right=0.95, top=0.86, bottom=0.3, wspace=0.35)

# Iterate through columns
for ii, ax in enumerate(ax_t):  # Iterate through columns
    quant_to_process_min = critical_quantile_min[ii]
    list_matrices = get_matrices_critical_quantiles(quant_to_process_min,
                                                    all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                    solutions_dict=solutions_dictx[quant_to_process_min],
                                                    pv_growth_percentiles=x,
                                                    load_growth_percentiles=y,
                                                    process_max=False)
    q_05_min, q_50_min, q_95_min, _ = plot_contour_levels_irradiance_days(list_matrices,
                                                                          contour_level=min_technical_voltage,
                                                                          ax=ax,
                                                                          pv_growth_percentiles=x,
                                                                          load_growth_percentiles=y,
                                                                          linecolor_contour="b",
                                                                          linecolor_quantile="b",
                                                                          alpha_line_quantile=0.0)

    quant_to_process_max = critical_quantile_max[ii]
    list_matrices = get_matrices_critical_quantiles(quant_to_process_max,
                                                    all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                    solutions_dict=solutions_dictx[quant_to_process_max],
                                                    pv_growth_percentiles=x,
                                                    load_growth_percentiles=y,
                                                    process_max=True)

    q_05_max, q_50_max, q_95_max, _ = plot_contour_levels_irradiance_days(list_matrices,
                                                                          contour_level=max_technical_voltage_danger,
                                                                          # linecolor_contour="grey",
                                                                          linecolor_contour="#9E5400",
                                                                          linecolor_quantile="r",
                                                                          pv_growth_percentiles=x,
                                                                          load_growth_percentiles=y,
                                                                          ax=ax)
    ax.fill_between(x, 0, q_05_max, color="green", alpha=0.2, zorder=0, label="Safe")
    ax.fill_between(x, q_05_max, q_95_max, color="orange", alpha=0.2, zorder=0, label="Caution")
    ax.fill_between(x, q_95_max, 1, color="red", alpha=0.2, zorder=0, label="Overvoltage")
    # ax.fill_between(x, 1, y_case_min.mean(axis=0), color="blue", alpha=0.2)

    f_quant = interpolate.interp1d(x, q_05_max, fill_value='extrapolate')
    x_new = np.linspace(0,1,100)
    y_new = f_quant(x_new)

    idx = x_new > min_quant_span[ii]
    ax.fill_between(x_new[idx], 0, y_new[idx], color="blue", alpha=0.2, zorder=0, label="Undervoltage")
    # ax.axvspan(min_quant_span[ii], 1, 0.0, 1.0, alpha=0.2, color='blue')

    ax.grid()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Load Growth", fontsize="large")
    ax.set_ylabel("PV Growth", fontsize="large")
    ax.set_title(titles_list[ii] + "\n" f"Percentile {quant_to_process_max}" + r"\%", fontsize="x-large")
    ax.grid(which="both", linestyle=":")

    if ii==1:
        ax.legend(fontsize="x-large",
                  bbox_to_anchor=(-0.87, -0.18),
                  loc="upper left",
                  ncol=4,
                  title="Static Operating Zones:",
                  title_fontsize="x-large",
                  handlelength=1.5)

# plt.savefig('static_regions/static_contour_plots.pdf', dpi=700, bbox_inches='tight')


#%%
# =====================================================================================================================
# FIGURE 3: Collect all the lines for the 1.05 and 1.045 limits (One big plot attached to ternary plots)
# =====================================================================================================================

max_technical_voltage_danger = 1.05
warnings.filterwarnings("error")
critical_quantile_max = ["90"]
# critical_quantile_min = ["25", "10", "05"]
titles_list = ["(a)", "(b)", "(c)"]
min_quant_span = [0.90, 0.88, 0.86]  # This is calculated zooming into the figure and see the vertical contours
critical_quantiles = list(set(critical_quantile_min + critical_quantile_max))
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)

fig, ax = plt.subplots(1, 1, figsize=(7 / 2.54, 7 / 2.54))
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

ii = 0
quant_to_process_min = critical_quantile_min[ii]
list_matrices = get_matrices_critical_quantiles(quant_to_process_min,
                                                all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                solutions_dict=solutions_dictx[quant_to_process_min],
                                                pv_growth_percentiles=x,
                                                load_growth_percentiles=y,
                                                process_max=False)
q_05_min, q_50_min, q_95_min, _ = plot_contour_levels_irradiance_days(list_matrices,
                                                                      contour_level=min_technical_voltage,
                                                                      ax=ax,
                                                                      pv_growth_percentiles=x,
                                                                      load_growth_percentiles=y,
                                                                      linecolor_contour="b",
                                                                      linecolor_quantile="b",
                                                                      alpha_line_quantile=0.0,
                                                                      contour_linewidths=0.2)


quant_to_process_max = critical_quantile_max[ii]
list_matrices = get_matrices_critical_quantiles(quant_to_process_max,
                                                all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                solutions_dict=solutions_dictx[quant_to_process_max],
                                                pv_growth_percentiles=x,
                                                load_growth_percentiles=y,
                                                process_max=True)

(q_05_max,
 q_50_max,
 q_95_max,
 contours_max_danger) = plot_contour_levels_irradiance_days(list_matrices,
                                                            contour_level=1.05,
                                                            # linecolor_contour="grey",
                                                            linecolor_contour="grey",
                                                            linecolor_quantile="r",
                                                            pv_growth_percentiles=x,
                                                            load_growth_percentiles=y,
                                                            ax=ax,
                                                            plot_quantiles=False,
                                                            contour_linewidths=0.2)
(q_05_max_caution,
 q_50_max_caution,
 q_95_max_caution,
 contours_max_caution) = plot_contour_levels_irradiance_days(list_matrices,
                                                             contour_level=1.045,
                                                             # linecolor_contour="grey",
                                                             linecolor_contour="grey",
                                                             linecolor_quantile="b",
                                                             pv_growth_percentiles=x,
                                                             load_growth_percentiles=y,
                                                             ax=ax,
                                                             plot_quantiles=False,
                                                             contour_linewidths=0.2)

contours_max_caution_area = np.vstack([contours_max_danger, contours_max_caution])
q_05 = np.nanquantile(contours_max_caution_area, q=0.05, axis=0)
q_50 = np.nanquantile(contours_max_caution_area, q=0.50, axis=0)
q_90 = np.nanquantile(contours_max_caution_area, q=0.90, axis=0)
pv_growth_percentiles = x

# Soft the quantile curves
from scipy.interpolate import make_interp_spline
b_90 = make_interp_spline(pv_growth_percentiles, q_90)
b_50 = make_interp_spline(pv_growth_percentiles, q_50)
b_05 = make_interp_spline(pv_growth_percentiles, q_05)

x_new_spline = np.linspace(pv_growth_percentiles.min(), pv_growth_percentiles.max(), 100)
y_new_90 = b_90(x_new_spline)
y_new_50 = b_50(x_new_spline)
y_new_05 = b_05(x_new_spline)

alpha_line_quantile = 1.0
linecolor_quantile = "r"
ax.plot(x_new_spline, y_new_90, linestyle="--", color=linecolor_quantile, alpha=alpha_line_quantile)
ax.plot(x_new_spline, y_new_50, linestyle="-", color=linecolor_quantile, alpha=alpha_line_quantile)
ax.plot(x_new_spline, y_new_05, linestyle="-.", color=linecolor_quantile, alpha=alpha_line_quantile)

ax.grid()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.set_xlabel("Load Growth", fontsize="large")
ax.set_ylabel("PV Growth", fontsize="large")

ax.fill_between(x_new_spline, 0, y_new_05, color="green", alpha=0.2, zorder=0, label="Safe")
ax.fill_between(x_new_spline, y_new_05, y_new_90, color="orange", alpha=0.2, zorder=0, label="Caution")
ax.fill_between(x_new_spline, y_new_90, 1, color="red", alpha=0.2, zorder=0, label="Overvoltage")
# ax.fill_between(x, 1, y_case_min.mean(axis=0), color="blue", alpha=0.2)

f_quant = interpolate.interp1d(x_new_spline, y_new_05, fill_value='extrapolate')
x_new = np.linspace(0, 1, 100)
y_new = f_quant(x_new)

idx = x_new > min_quant_span[ii]
ax.fill_between(x_new[idx], 0, y_new[idx], color="blue", alpha=0.2, zorder=0, label="Undervoltage")
ax.legend(fontsize="medium",
          # bbox_to_anchor=(0.015, -0.15),
          loc="upper left",
          ncol=2,
          title="Static Operating Zones:",
          title_fontsize="medium",
          handlelength=1)

plt.savefig('static_regions/static_contour_for_ternary.pdf', dpi=700, bbox_inches='tight')

#%%
# =====================================================================================================================
# FIGURE 4: See the planes in 3D to understand better
# =====================================================================================================================

Ti_1, Ti_high_res_1, (X, Y), (grid_x, grid_y) = get_high_res_plane("90", [(0.4, 0.0, 0.6)], solutions_dictx, x, y)
Ti_2, Ti_high_res_2, _, _ = get_high_res_plane("90", [(0.0, 1.0, 0.0)], solutions_dictx, x, y)
Ti_3, Ti_high_res_3, _, _ = get_high_res_plane("90", [(0.3, 0.3, 0.4)], solutions_dictx, x, y)

Ti_1_min, Ti_high_res_1_min, (X_min, Y_min), (grid_x_min, grid_y_min) = get_high_res_plane("10", [(0.4, 0.0, 0.6)],
                                                                                           solutions_dictx, x, y,
                                                                                           process_max=False)
Ti_2_min, Ti_high_res_2_min, _, _ = get_high_res_plane("10", [(0.0, 1.0, 0.0)], solutions_dictx, x, y,
                                                       process_max=False)
Ti_3_min, Ti_high_res_3_min, _, _ = get_high_res_plane("10", [(0.3, 0.3, 0.4)], solutions_dictx, x, y,
                                                       process_max=False)

fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection': '3d'}, figsize=(16, 12))
ax[0, 0].plot_wireframe(X, Y, Ti_1, color='r', linewidth=0.4)
ax[0, 1].plot_wireframe(grid_x, grid_y, Ti_high_res_1, color='b', linewidth=0.4, label="[(cloudy=0.4, sunny=0.0, overcast=0.6)]")
ax[0, 1].plot_wireframe(grid_x, grid_y, Ti_high_res_2, color='r', linewidth=0.4, label=" [(0.0, 1.0, 0.0)]")
ax[0, 1].plot_wireframe(grid_x, grid_y, Ti_high_res_3, color='orange', linewidth=0.4, label="[(0.3, 0.3, 0.4)]")
ax[0, 1].plot_wireframe(grid_x, grid_y, np.full_like(Ti_high_res_2, 1.05), color='k', linewidth=0.4, label="Max. Technical limit")
ax[0, 1].set_xlabel("Load growth [/%]")
ax[0, 1].set_ylabel("PV growth [/%]")
ax[0, 1].set_zlabel("Max. voltage [p.u]")
ax[0, 1].set_title("Maximum grid voltage Quantile 90")
ax[0, 0].set_title("Low resolution (Simulations)")
ax[0, 2].set_title("Simulation + 3D Cubic Spline")
ax[0, 1].legend(fontsize="large")
ax[0, 2].plot_wireframe(X, Y, Ti_1, color='r', linewidth=1.0, label="Simulated")
ax[0, 2].plot_wireframe(grid_x, grid_y, Ti_high_res_1, color='b', linewidth=0.4, label="Cubic spline")
ax[0, 2].legend(fontsize="small")

ax[1, 0].plot_wireframe(X, Y, Ti_1, color='r', linewidth=0.4)
ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_1_min, color='b', linewidth=0.4, label="[(cloudy=0.4, sunny=0.0, overcast=0.6)]")
ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_2_min, color='r', linewidth=0.4, label=" [(0.0, 1.0, 0.0)]")
ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_3_min, color='orange', linewidth=0.4, label="[(0.3, 0.3, 0.4)]")
ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, np.full_like(Ti_high_res_2_min, 0.96), color='k', linewidth=0.4, label="Min. Technical limit")
ax[1, 1].set_xlabel("Load growth [/%]")
ax[1, 1].set_ylabel("PV growth [/%]")
ax[1, 1].set_zlabel("Min. voltage [p.u]")
ax[1, 1].set_title("Minimum grid voltage Quantile 90")
ax[1, 0].set_title("Low resolution (Simulations)")
ax[1, 2].set_title("Simulation + 3D Cubic Spline")
ax[1, 1].legend(fontsize="large")

ax[1, 2].plot_wireframe(X_min, Y_min, Ti_1_min, color='r', linewidth=1.0, label="Simulated")
ax[1, 2].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_1_min, color='b', linewidth=0.4, label="Cubic spline")
ax[1, 2].legend(fontsize="small")

np.quantile([0.002, 1,1,2,3,3,3,3,4,3.4], q=0)


# plt.style.use("seaborn-white")
# plt.style.use("../../probnum.mplstyle")

