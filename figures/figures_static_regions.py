import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import warnings
from pathlib import Path
from scipy import interpolate
from core.figure_utils import set_figure_art
from typing import List, Tuple
set_figure_art()
# mpl.rc('text', usetex=False)


def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file_:
        scenario_generator = pickle.load(pickle_file_)
    return scenario_generator


def load_solutions(file_name_solutions):
    with open(file_name_solutions, "rb") as pickle_file_:
        solutions = pickle.load(pickle_file_)
    return solutions


def plot_quantile_mixture_case(solution_quantile, ax, quantile_highlight: str = None):
    """Plots the quantiles of the daily voltage profiles"""

    quantile_keys = ["95", "90", "75", "50", "25", "10", "05"]
    linestyles = ["--", "--", "--", "-", "-.", "-.", "-."]
    colors = ["r", "b", "g", "r", "g", "b", "r"]

    if quantile_highlight is None:
        enhance = "90"
    else:
        enhance = quantile_highlight

    for quantile_, linestyle_, color_ in zip(quantile_keys, linestyles, colors):
        if quantile_ == enhance:
            ax.plot(solution_quantile["max_q_" + quantile_], linestyle=linestyle_, color=color_, linewidth=1.1,
                    label=quantile_ + r" \%", marker="o", markersize=1)
        else:
            ax.plot(solution_quantile["max_q_" + quantile_], linestyle=linestyle_, color=color_, linewidth=0.8,
                    label=quantile_ + r" \%")


def get_coords_max_values(profile):
    """Find the maximum voltage and the position in the profile vector"""
    x_max = np.argmax(profile)
    y_max = profile[x_max]

    return x_max, y_max


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
                                    all_mixtures: set,
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
    for mixture_case in all_mixtures:
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
                                        alpha_line_quantile=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    y_case = []
    for matrix_ in list_matrices:
        try:
            cs = ax.contour(pv_growth_percentiles, load_growth_percentiles, matrix_,
                            colors=linecolor_contour, levels=[contour_level], linewidths=0.4)
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

    ax.plot(pv_growth_percentiles, q_50, color=linecolor_quantile, linewidth=1.8,
            alpha=alpha_line_quantile)
    ax.plot(pv_growth_percentiles, q_95, color=linecolor_quantile, linewidth=1.0, linestyle="--",
            alpha=alpha_line_quantile)
    ax.plot(pv_growth_percentiles, q_05, color=linecolor_quantile, linewidth=1.0, linestyle="-.",
            alpha=alpha_line_quantile)

    return q_05, q_50, q_95


file_name_scenario_generator_model = "../models/scenario_generator_model_new_AWS.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
cases_combinations = scenario_generator.cases_combinations

file_name_solutions_dictionary = "../HPC/solutions/solutions_dictionary_AWS.pkl"
solutions_dict = load_solutions(file_name_solutions_dictionary)

x = scenario_generator.percentages_pv_growth
y = scenario_generator.percentages_load_growth

matrix_voltages = np.zeros((len(x), len(y)))
# mixture_cases = [(0.0, 0.0, 1.0),
#                  (0.2, 0.0, 0.8),
#                  (0.4, 0.0, 0.6),
#                  (0.6, 0.0, 0.4),
#                  (0.8, 0.0, 0.2),
#                  (0.2, 0.6, 0.2),  # Normal case (Original)
#                  (0.0, 1.0, 0.0)]
#
# list_matrices = []
# list_of_tuple_cases = []
# for  mixture_case in mixture_cases:
#     for i, pv in enumerate(x):
#         for j, load  in enumerate(y):
#             case = (mixture_case, load, pv)
#             list_of_tuple_cases.append(case)
#             matrix_voltages[i, j] = np.max(solutions_dict[case]['max_q_90'])
#
#     list_matrices.append(matrix_voltages.copy())
# #TODO: There is a glitch always in the first matrix to plot!!!

#%%
QUANTILE = "90"  # Quantile to plot in the first figure
mixture_cases_plot = [(0.4, 0.0, 0.6),
                      (0.2, 0.6, 0.2),  # Normal case (Original)
                      (0.0, 1.0, 0.0)]
list_matrices_plot = []
list_of_tuple_cases = []
for mixture_case in mixture_cases_plot:
    for i, pv in enumerate(x):
        for j, load in enumerate(y):
            case = (mixture_case, load, pv)
            list_of_tuple_cases.append(case)
            matrix_voltages[i, j] = np.max(solutions_dict[case]["max_q_" + QUANTILE])

    list_matrices_plot.append(matrix_voltages.copy())
warnings.filterwarnings("default")

#%%
# =====================================================================================================================
# FIGURE 1: Heat map with one contour plot and quantiles of daily voltage profile
# =====================================================================================================================

ax_ = np.empty((5, 3), dtype=object)
fig = plt.figure(figsize=(7, 8))
gs0 = gridspec.GridSpec(2, 1, figure=fig, wspace=0.2, hspace=0.08, left=0.1, bottom=0.11, right=0.98, top=0.94,
                        height_ratios=[1, 2], width_ratios=[1])  # Two columns to hold the different plots
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

special_rectangles = [(0.45, 0.45),
                      (0.45, 0.35),
                      (0.75, 0.55)]

min_voltage_cbar, max_voltage_cbar = 0.998, 1.095
max_technical_voltage = 1.05

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
    coords_ = []
    for data_mixture_ in data_mixture_case:
        coords_.append(get_coords_max_values(data_mixture_["max_q_" + QUANTILE]))

    colors_ = ["r", "g", "b", "purple"]

    ax_0.pcolor(x, y, list_matrices_plot[ii], shading='auto', vmin=min_voltage_cbar, vmax=max_voltage_cbar)
    ax_0.set_xlabel("Load growth", fontsize="x-large")
    ax_0.set_ylabel("PV growth", fontsize="x-large")
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('viridis')), ax=ax_0,
                        location="bottom", pad=0.18)
    cbar.ax.set_xlabel("Max. grid voltage", fontsize="x-large")

    for color_, (_, y_max_) in zip(colors_, coords_):
        cbar.ax.vlines(x=y_max_, ymin=0, ymax=10, linewidth=2, color=color_)

    ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage], linewidths=1.5)
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

    ax_iter = [ax_1, ax_2, ax_3, ax_4]  # Rows

    for ax_fig, color_, data_mixture_, (x_max_, y_max_) in zip(ax_iter, colors_, data_mixture_case, coords_):
        plot_quantile_mixture_case(data_mixture_, ax=ax_fig, quantile_highlight=QUANTILE)
        ax_fig.text(x=x_max_ / 2,
                    y=y_max_ * 1.008,
                    s=str(round(y_max_, 3)) + " [p.u]", ha="center", va="center", fontsize="large",
                    color=color_)
        ax_fig.hlines(y=y_max_, xmin=0, xmax=x_max_, color="k", linewidth=0.8, linestyles="--")
        for spines in ax_fig.spines.values():
            spines.set_color(color_)
            spines.set_linewidth(1.0)

    ax_1.set_xticklabels(labels=[])
    ax_2.set_xticklabels(labels=[])
    ax_3.set_xticklabels(labels=[])
    ax_4.set_xlabel("Time step")

    if ii == 0:
        ax_1.set_ylabel(f"Load: 0.0\nPV: 1.0" + "\nVoltage [p.u.]", fontsize="large")
        ax_2.set_ylabel(f"Load: 1.0\nPV: 0.0" + "\nVoltage [p.u.]", fontsize="large")
        ax_3.set_ylabel(f"Load: 0.0\nPV: 0.0" + "\nVoltage [p.u.]", fontsize="large")
        ax_4.set_ylabel(f"Voltage [p.u.]", fontsize="large")

    if ii == 1:
        ax_4.legend(fontsize="large",
                    bbox_to_anchor=(-1.1, -0.3),
                    loc="upper left",
                    ncol=7,
                    title="Percentiles",
                    title_fontsize="large",
                    handlelength=1.5)

    ax_1.set_ylim((0.99, 1.12))
    ax_2.set_ylim((0.98, 1.05))
    ax_3.set_ylim((0.98, 1.12))
    ax_4.set_ylim((0.98, 1.12))

plt.savefig('static_regions/static_region_border.pdf', dpi=700, bbox_inches='tight')

#%%
# =====================================================================================================================
# FIGURE 2: Static operating zones
# =====================================================================================================================

warnings.filterwarnings("error")
critical_quantile_max = ["75", "90", "95"]
critical_quantile_min = ["25", "15", "05"]
titles_list = ["(a)", "(b)", "(c)"]
min_quant_span = [0.90, 0.88, 0.86]  # This is calculated zooming into the figure and see the vertical contours
critical_quantiles = list(set(critical_quantile_min + critical_quantile_max))

path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)
all_mixtures = set([case_mixture_part[0] for case_mixture_part in cases_combinations])

fig, ax_t = plt.subplots(1, 3, figsize=(7, 2.8))
plt.subplots_adjust(left=0.08, right=0.95, top=0.86, bottom=0.3, wspace=0.35)

# Iterate through columns
for ii, ax in enumerate(ax_t):  # Iterate through columns
    quant_to_process_min = critical_quantile_min[ii]
    list_matrices = get_matrices_critical_quantiles(quant_to_process_min,
                                                    all_mixtures=all_mixtures,
                                                    solutions_dict=solutions_dictx[quant_to_process_min],
                                                    pv_growth_percentiles=x,
                                                    load_growth_percentiles=y,
                                                    process_max=False)
    q_05_min, q_50_min, q_95_min = plot_contour_levels_irradiance_days(list_matrices,
                                                                       contour_level=0.96,
                                                                       ax=ax,
                                                                       pv_growth_percentiles=x,
                                                                       load_growth_percentiles=y,
                                                                       linecolor_contour="b",
                                                                       linecolor_quantile="b",
                                                                       alpha_line_quantile=0.0)

    quant_to_process_max = critical_quantile_max[ii]
    list_matrices = get_matrices_critical_quantiles(quant_to_process_max,
                                                    all_mixtures=all_mixtures,
                                                    solutions_dict=solutions_dictx[quant_to_process_max],
                                                    pv_growth_percentiles=x,
                                                    load_growth_percentiles=y,
                                                    process_max=True)

    q_05_max, q_50_max, q_95_max = plot_contour_levels_irradiance_days(list_matrices,
                                                                       contour_level=1.05,
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

plt.savefig('static_regions/static_contour_plots.pdf', dpi=700, bbox_inches='tight')

