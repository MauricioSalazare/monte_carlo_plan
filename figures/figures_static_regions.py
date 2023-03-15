# ===================================
# Generates the following figures:
# Fig 8. Heat maps with the contour lines, inset plots with the maximum and minimum voltage profiles
# Fig 11. Regions of the ternary plot
# ===================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import warnings
from pathlib import Path
from scipy import interpolate
from core.figure_utils import set_figure_art
from typing import List, Tuple
import numpy as np
from scipy.interpolate import griddata
import itertools
import matplotlib.ticker as ticker
import pandas as pd
import pickle
import plotly.express as px
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


def get_worst_max_scenario(solutions_dict: dict,
                           quantile: str,
                           worst_irradiance_mixture: Tuple,
                           pv_growth_percentiles,
                           load_growth_percentiles,
                           offset,
                           voltage_cut_level):

    # Retrieve data of the worst scenario
    list_matrices = get_matrices_critical_quantiles(quantile,
                                                    all_mixtures_irradiance_only=[worst_irradiance_mixture],
                                                    solutions_dict=solutions_dict[quantile],
                                                    pv_growth_percentiles=pv_growth_percentiles,
                                                    load_growth_percentiles=load_growth_percentiles,
                                                    process_max=True,
                                                    OFFSET=offset)

    figx, ax = plt.subplots(1,1, figsize=(5, 5))
    try:
        cs = ax.contour(pv_growth_percentiles, load_growth_percentiles, list_matrices[0], levels=[voltage_cut_level])
        p = cs.collections[0].get_paths()[0]
        v = p.vertices
        x_ = v[:, 0]
        y_ = v[:, 1]
        f = interpolate.interp1d(x_, y_, fill_value='extrapolate')
        ynew = f(x)

    except UserWarning:
        print("No contour warning")
        ynew = np.empty(len(x))
        ynew[:] = np.nan

    except RuntimeWarning:
        print("Divide by zero")
        ynew = np.empty(len(x))
        ynew[:] = np.nan

    plt.close(figx)

    return ynew



def get_worst_max_scenario_v2(solutions_dict: dict,
                           quantile: str,
                           all_tuple_things: list,
                           pv_growth_percentiles,
                           load_growth_percentiles,
                           offset,
                           voltage_cut_level):

    # Retrieve data of the worst scenario
    list_matrices = get_matrices_critical_quantiles(quantile,
                                                    all_mixtures_irradiance_only=all_tuple_things,
                                                    solutions_dict=solutions_dict[quantile],
                                                    pv_growth_percentiles=pv_growth_percentiles,
                                                    load_growth_percentiles=load_growth_percentiles,
                                                    process_max=True,
                                                    OFFSET=offset)

    figx, ax = plt.subplots(1, 1, figsize=(5, 5))
    _,_,_,y_data = plot_contour_levels_irradiance_days(list_matrices=list_matrices,
                                                        contour_level=voltage_cut_level,
                                                        # linecolor_contour="grey",
                                                        linecolor_contour="r",
                                                        linecolor_quantile="r",
                                                        pv_growth_percentiles=pv_growth_percentiles,
                                                        load_growth_percentiles=load_growth_percentiles,
                                                        # contour_linewidths=0.08,
                                                        contour_linewidths=1.00,
                                                        alpha_line_contour=0.4,
                                                        ax=ax,
                                                        plot_quantiles=False)
    plt.close(figx)

    ynew = np.nanmin(y_data["y_cases"], axis=0)

    return ynew


def plot_quantile_mixture_case(solution_quantile,
                               ax,
                               max_quantile_highlight: str = None,
                               min_quantile_highlight: str = None,
                               plot_min_voltages = True
                               ):
    """Plots the quantiles of the daily voltage profiles"""

    x_axis_ = pd.date_range(start="2021-11-01", periods=48, freq="30T")

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
            ax.plot(x_axis_, solution_quantile["max_q_" + max_quantile_], linestyle=linestyle_, color="k", linewidth=1.1,
                    label=max_quantile_ + r" \%", marker="o", markersize=1)
        else:
            ax.plot(x_axis_, solution_quantile["max_q_" + max_quantile_], linestyle=linestyle_, color="k", linewidth=0.8,
                    label=max_quantile_ + r" \%")

        if plot_min_voltages:
            if min_quantile_ == min_quant_enhance:
                ax.plot(x_axis_, solution_quantile["min_q_" + min_quantile_], linestyle=linestyle_, color="grey", linewidth=1.1,
                        label=min_quantile_ + r" \%", marker="o", markersize=1)
            else:
                ax.plot(x_axis_, solution_quantile["min_q_" + min_quantile_], linestyle=linestyle_, color="grey", linewidth=0.8,
                        label=min_quantile_ + r" \%")

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        # ax.set_xlim((x_axis_[0], x_axis_[-1] + pd.Timedelta(minutes=29)))



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
            solutions_dict[quant_name] = file_solutions_dict  # This is critical to group all mixtures in the same quant

    return solutions_dict

def get_matrices_critical_quantiles(quantile: str,
                                    all_mixtures_irradiance_only: list,
                                    solutions_dict: dict,
                                    pv_growth_percentiles: np.ndarray,
                                    load_growth_percentiles: np.ndarray,
                                    process_max: bool = True,
                                    OFFSET: float=0.0) -> List[np.ndarray]:
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
                    matrix_voltages[i, j] = np.max(solutions_dict[case]["max_q_" + quantile].astype(np.float64) - OFFSET)
                else:
                    matrix_voltages[i, j] = np.min(solutions_dict[case]["min_q_" + quantile].astype(np.float64) - OFFSET)
        list_matrices.append(matrix_voltages.copy())

    return list_matrices


def plot_contour_levels_irradiance_days(list_matrices,
                                        contour_level,
                                        linecolor_contour,
                                        linecolor_quantile,
                                        pv_growth_percentiles: np.ndarray,
                                        load_growth_percentiles: np.ndarray,
                                        ax,
                                        alpha_line_quantile=1.0,
                                        alpha_line_contour=0.5,
                                        contour_linewidths=0.2,
                                        quantile_linewdith=1.0,
                                        plot_quantiles=True) -> Tuple[np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray,
                                                                          dict]:

    y_case = []
    minimum_contour = np.inf

    for matrix_ in list_matrices:  # Each matrix has a different irradiance day combination
        try:
            cs = ax.contour(pv_growth_percentiles, load_growth_percentiles, matrix_,
                            colors=linecolor_contour,
                            alpha=alpha_line_contour,
                            levels=[contour_level],
                            linewidths=contour_linewidths)
            p = cs.collections[0].get_paths()[0]
            v = p.vertices



            x_ = v[:, 0]
            y_ = v[:, 1]

            if len(x_) == 1:
                continue

            f = interpolate.interp1d(x_, y_, fill_value='extrapolate')
            ynew = f(pv_growth_percentiles)

        except ValueError:
            print("Interpolation error")
            ynew = np.empty(len(pv_growth_percentiles))
            ynew[:] = np.nan

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

    # Get the lowest of all valid lines
    matrix_limits = y_case
    matrix_limits = matrix_limits[~np.isnan(matrix_limits).any(axis=1), :]  # Clear nans
    matrix_limits = matrix_limits[(matrix_limits > 0).all(axis=1), :]  # Clear negative
    idx = np.argmin(matrix_limits.mean(axis=1))
    lower_line_ = matrix_limits[idx, :]

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

    data = {"y_cases": y_case,
            "lower_limit": lower_line_}

    return q_05, q_50, q_95, data

def get_high_res_plane(quantile: str,
                       mixture_irradiance: list,
                       solutions_dictx: dict,
                       x:np.ndarray,
                       y: np.ndarray,
                       process_max=True,
                       offset: float=0.0):

    list_matrices = get_matrices_critical_quantiles(quantile,
                                                    all_mixtures_irradiance_only=mixture_irradiance,
                                                    solutions_dict=solutions_dictx[quantile],
                                                    pv_growth_percentiles=x,
                                                    load_growth_percentiles=y,
                                                    process_max=process_max,
                                                    OFFSET=offset)
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
OFFSET = 0.008  # TODO: WARNING: This value should be the same in the in the ternary plot scripts!!!
# max_technical_voltage_green = 1.040
max_technical_voltage_caution = 1.045
max_technical_voltage_danger = 1.05
min_technical_voltage = 0.95

critical_quantiles = ["05", "10", "15", "25", "50", "75", "90", "95", "100"]
path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
solutions_dictx_new = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)

# Re-arrange dictionary, so the code in the first figure can work
cases_combinations = list(set(solutions_dictx_new["50"].keys()))
cases_combinations_irradiance_only_tuples_ = list(set([irradiance_mixture for irradiance_mixture, _, _ in cases_combinations]))
outlier_set = {(0.0, 0.0, 1.0),
               (0.0, 0.1, 0.9),
               (0.1, 0.0, 0.9),
               (0.1, 0.2, 0.7),
               (0.4, 0.0, 0.6),
               (0.0, 0.2, 0.8),
               (0.2, 0.0, 0.8),
               (0.2, 0.1, 0.7),
               (0.1, 0.1, 0.8),
               (0.4, 0.4, 0.2),
               (0.7, 0.1, 0.2)}
cases_combinations_irradiance_only_tuples = list(set(cases_combinations_irradiance_only_tuples_) - outlier_set)
solutions_dict = {}
for mixture_comb in cases_combinations:
    solutions_dict[mixture_comb] = {}
    for quantile_name in critical_quantiles:
        solutions_dict[mixture_comb][f"max_q_{quantile_name}"] = solutions_dictx_new[quantile_name][mixture_comb][f"max_q_{quantile_name}"].astype(np.float64) - OFFSET
        solutions_dict[mixture_comb][f"min_q_{quantile_name}"] = solutions_dictx_new[quantile_name][mixture_comb][f"min_q_{quantile_name}"].astype(np.float64) - OFFSET


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
            matrix_voltages[i, j] = np.max(solutions_dict[case]["max_q_" + QUANTILE])  # Profile max
    list_matrices_plot.append(matrix_voltages.copy())
warnings.filterwarnings("default")

#%%
# =====================================================================================================================
# FIGURE 1 (ver 2): Heat map with one contour plot and quantiles of daily voltage profile
# =====================================================================================================================
x_axis = pd.date_range(start="2021-11-01", periods=48, freq="30T")

ax_ = np.empty((3, 5), dtype=object)
fig = plt.figure(figsize=(7.3*2, 3.8*2))
gs0 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, hspace=0.08, left=0.05, bottom=0.11, right=0.98, top=0.94,
                        height_ratios=[1], width_ratios=[1, 4])  # Two columns to hold the different plots
gs00 = gs0[0].subgridspec(3, 1, hspace=0.35)  # Colormap
gs01 = gs0[1].subgridspec(3, 4, wspace=0.45, hspace=0.55)  # Voltage profiles

for ii in range(3):
    ax_[ii, 0] = fig.add_subplot(gs00[ii])

kk = 0
for ii in range(3):
    for jj in range(4):
        ax_[ii, jj + 1] = fig.add_subplot(gs01[kk])
        kk += 1

ax_0_column = ax_[:, 0].flatten()
ax_1_column = ax_[:, 1].flatten()
ax_2_column = ax_[:, 2].flatten()
ax_3_column = ax_[:, 3].flatten()
ax_4_column = ax_[:, 4].flatten()


# Combinations of load and pv that lies in the contour line
special_load_pv_comb = [(0.5, 0.5),
                        (0.5, 0.4),
                        (0.8, 0.6)]

special_load_pv_comb = [(0.3, 0.5),
                        (0.3, 0.5),
                        (0.3, 0.5)]

special_rectangles = [] # (x0, y0) cooridnates of the purple boxes
for (special_load, special_pv) in special_load_pv_comb:
    special_rectangles.append((np.round(special_load - 0.05, 3),
                               np.round(special_pv - 0.05, 3)))

min_voltage_cbar, max_voltage_cbar = 0.998, 1.095



# titles_list = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)"]

titles_array = np.array([["(a)", "(b)", "(c)", "(d)", "(e)"],
                         ["(f)", "(g)", "(h)", "(i)", "(j)"],
                         ["(k)", "(l)", "(m)", "(n)", "(o)"]])


norm_individual = mpl.colors.Normalize(vmin=min_voltage_cbar, vmax=max_voltage_cbar)

# Create a legend that it will be showed in the heat map
lines_ = [Line2D([0], [0], color="k", linewidth=1.5, linestyle='-'),
          Line2D([0], [0], color="k", linewidth=1.5, linestyle='dashed')]
labels_ = [r"$\overline{V}$",
           r"$V_{\mathrm{caution}}$"]

# Iteration is through rows
for ii, (ax_0, ax_1, ax_2, ax_3, ax_4, title_row, mixture_case) in enumerate(zip(ax_0_column,
                                                                      ax_1_column,
                                                                      ax_2_column,
                                                                      ax_3_column,
                                                                      ax_4_column,
                                                                      titles_array,
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

    if ii == 2:
        ax_0.set_xlabel(f"Annual energy" + r" consumption growth [\%]" , fontsize="x-large")

    # if ii == 0:
    ax_0.set_ylabel("PV installed\n" + r"capacity growth [\%]", fontsize="x-large")

    ax_0.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('viridis')), ax=ax_0,
                        location="right")
    cbar.ax.set_ylabel("Max. grid voltage mag. [p.u.]", fontsize="x-large")

    for color_, (_, y_max_) in zip(colors_, coords_max):
        # cbar.ax.vlines(x=y_max_, ymin=0, ymax=10, linewidth=2, color=color_)
        cbar.ax.hlines(y=y_max_, xmin=0, xmax=10, linewidth=2, color=color_)

    cs1 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_danger], linewidths=1.5,
                       linestyles="solid")
    cs2 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_caution], linewidths=1.5,
                       linestyles="dashed")
    # cs3 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_green], linewidths=1.5,
    #                    linestyles="dashdot")

    ax_0.clabel(cs1, inline=True, fontsize=7, colors='k')
    ax_0.clabel(cs2, inline=True, fontsize=7, colors='k')
    # ax_0.clabel(cs3, inline=True, fontsize=7, colors='k')

    ax_0.set_title(f"{title_row[0]}\n"
                   r"$(\pi_1=" + f"{mixture_case[0]}" + r"," +  # Cloudy
                   r"\pi_2=" + f"{mixture_case[1]}" + r"," +  # Sunny
                   r"\pi_3=" + f"{mixture_case[2]}" + r")$",  # Overcast
                   fontsize="x-large")

    ax_0.set_aspect(1)

    if ii == 1:
        ax_0.legend(lines_, labels_, loc="upper center", handlelength=1.5, labelspacing=0.2, borderaxespad=0.1, fontsize="large")

    # WARNING: I changed the order, so it agrees with the clustering of one of the figures in the paper
    # In the paper the order is (cluster_1 = cloudy, cluster_2=overcast, cluster_3=sunny)
    # ax_0.set_title(f"{titles_list[ii]}\n"
    #                r"$(\pi_1=" + f"{mixture_case[0]}" + r"," +
    #                r"\pi_2=" + f"{mixture_case[2]}" + r"," +
    #                r"\pi_3=" + f"{mixture_case[1]}" + r")$",
    #                fontsize="x-large")

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

    # Change the axis to percentages
    valsy = ax_0.get_yticks()
    ax_0.yaxis.set_major_locator(ticker.FixedLocator(valsy))
    ax_0.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=r'\%', is_latex=True))

    valsx = ax_0.get_xticks()
    ax_0.xaxis.set_major_locator(ticker.FixedLocator(valsx))
    ax_0.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=r'\%', is_latex=True))

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

        ax_col.text(x=x_axis[np.ceil(x_max_/2).astype(int)],
                    y=y_max_ * 1.008,
                    s=str(round(y_max_, 3)) + " [p.u]", ha="center", va="center", fontsize="large",
                    color=color_)
        ax_col.hlines(y=y_max_, xmin=x_axis[0], xmax=x_axis[x_max_], color="k", linewidth=0.8, linestyles="--")
        # x_right_limit = ax_col.get_xlim()[1]

        ax_col.text(x=x_axis[x_min_-1],
                    y=y_min_ * (2 - 1.008),
                    s=str(round(y_min_, 3)) + " [p.u]", ha="center", va="center", fontsize="large",
                    color=color_)
        ax_col.hlines(y=y_min_, xmin=x_axis[x_min_], xmax=x_axis[-1], color="k", linewidth=0.8, linestyles="--")

        for spines in ax_col.spines.values():
            spines.set_color(color_)
            spines.set_linewidth(1.0)

    ax_1.set_xticklabels(labels=[])
    ax_2.set_xticklabels(labels=[])
    ax_3.set_xticklabels(labels=[])
    ax_4.set_xticklabels(labels=[])
    # ax_4.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    # ax_4.set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=29)))






    if ii == 2:

        # Plot legend of maximum quantiles
        lh, ll = ax_1.get_legend_handles_labels()
        idx_max_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 0]  # Even number

        max_lh = [lh[ii] for ii in idx_max_values]
        max_ll = [ll[ii] for ii in idx_max_values]

        leg = ax_1.legend(max_lh,
                          max_ll,
                          fontsize="large",
                          bbox_to_anchor=(0.55, -0.22),
                          loc="upper left",
                          ncol=4,
                          title="Max. voltage percentiles",
                          title_fontsize="large",
                          handlelength=1.5)

        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)


        # Plot legend of minimum quantiles

        lh, ll = ax_4.get_legend_handles_labels()
        idx_min_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 1]  # Odd number

        min_lh = [lh[ii] for ii in idx_min_values]
        min_ll = [ll[ii] for ii in idx_min_values]

        leg = ax_4.legend(min_lh,
                          min_ll,
                          fontsize="large",
                          bbox_to_anchor=(-1.05, -0.22),
                          loc="upper left",
                          ncol=4,
                          title="Min. voltage percentiles",
                          title_fontsize="large",
                          # handleheight=10,
                          handlelength=1.5)

        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)

    ax_1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax_2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax_3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax_4.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax_1.set_xlabel("Time of day")
    ax_2.set_xlabel("Time of day")
    ax_3.set_xlabel("Time of day")
    ax_4.set_xlabel("Time of day")

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

    ax_1.set_title(f"{title_row[1]}" + "\n" + r"Energy growth: 0\%" + "\n" + r"PV growth: 100\%", fontsize="large")
    ax_2.set_title(f"{title_row[2]}" + "\n" + r"Energy growth: 100\%" + "\n" + r"PV growth: 0\%", fontsize="large")
    ax_3.set_title(f"{title_row[3]}" + "\n" + r"Energy growth: 0\%" + "\n" + r"PV growth: 0\%", fontsize="large")
    ax_4.set_title(f"{title_row[4]}" + "\n" + r"Energy growth: 50\%" + "\n" + r"PV growth: 50\%", fontsize="large")

    ax_1.set_ylabel(r"Voltage mag. [p.u]")
    ax_2.set_ylabel(r"Voltage mag. [p.u]")
    ax_3.set_ylabel(r"Voltage mag. [p.u]")
    ax_4.set_ylabel(r"Voltage mag. [p.u]")

    ax_1.yaxis.set_ticks(np.arange(0.96, 1.115 + 0.02, 0.02))
    ax_2.yaxis.set_ticks(np.arange(0.930, 1.032 + 0.01, 0.01))
    ax_3.yaxis.set_ticks(np.arange(0.970, 1.058 + 0.01, 0.01))
    ax_4.yaxis.set_ticks(np.arange(0.95, 1.08 + 0.01, 0.01))

    ax_1.set_ylim((0.96, 1.115))
    ax_2.set_ylim((0.930, 1.032))
    ax_3.set_ylim((0.970, 1.058))
    ax_4.set_ylim((0.955, 1.08))



lh, ll = ax_4.get_legend_handles_labels()
idx_max_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 0]  # Even number
idx_min_values = [ii for ii in np.arange(len(lh)) if ii % 2 == 1]  # Odd number

max_lh = [lh[ii] for ii in idx_max_values]
max_ll = [ll[ii] for ii in idx_max_values]

min_lh = [lh[ii] for ii in idx_min_values]
min_ll = [ll[ii] for ii in idx_min_values]

plt.savefig('static_regions/static_region_border_ver2.pdf', dpi=700, bbox_inches='tight')























#%%
# =====================================================================================================================
# FIGURE 1: Heat map with one contour plot and quantiles of daily voltage profile
# =====================================================================================================================

x_axis = pd.date_range(start="2021-11-01", periods=48, freq="30T")

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

special_load_pv_comb = [(0.3, 0.5),
                        (0.3, 0.5),
                        (0.3, 0.5)]

special_rectangles = [] # (x0, y0) cooridnates of the purple boxes
for (special_load, special_pv) in special_load_pv_comb:
    special_rectangles.append((np.round(special_load - 0.05, 3),
                               np.round(special_pv - 0.05, 3)))

min_voltage_cbar, max_voltage_cbar = 0.998, 1.095



titles_list = ["(a)", "(b)", "(c)"]
norm_individual = mpl.colors.Normalize(vmin=min_voltage_cbar, vmax=max_voltage_cbar)

# Create a legend that it will be showed in the heat map
lines_ = [Line2D([0], [0], color="k", linewidth=1.5, linestyle='-'),
          Line2D([0], [0], color="k", linewidth=1.5, linestyle='dashed')]
labels_ = [r"$\overline{V}$",
           r"$V_{\mathrm{caution}}$"]

# Iteration is through columns
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

    if ii == 1:
        ax_0.set_xlabel(f"Annual energy" + r" consumption growth [\%]" , fontsize="x-large")

    if ii == 0:
        ax_0.set_ylabel("PV installed\n" + r"capacity growth [\%]", fontsize="x-large")

    ax_0.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('viridis')), ax=ax_0,
                        location="bottom", pad=0.18)
    cbar.ax.set_xlabel("Max. grid voltage mag. [p.u.]", fontsize="x-large")

    for color_, (_, y_max_) in zip(colors_, coords_max):
        cbar.ax.vlines(x=y_max_, ymin=0, ymax=10, linewidth=2, color=color_)

    cs1 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_danger], linewidths=1.5,
                       linestyles="solid")
    cs2 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_caution], linewidths=1.5,
                       linestyles="dashed")
    # cs3 = ax_0.contour(x, y, list_matrices_plot[ii], colors="k", levels=[max_technical_voltage_green], linewidths=1.5,
    #                    linestyles="dashdot")

    ax_0.clabel(cs1, inline=True, fontsize=7, colors='k')
    ax_0.clabel(cs2, inline=True, fontsize=7, colors='k')
    # ax_0.clabel(cs3, inline=True, fontsize=7, colors='k')

    ax_0.set_title(f"{titles_list[ii]}\n"
                   r"$(\pi_1=" + f"{mixture_case[0]}" + r"," +  # Cloudy
                   r"\pi_2=" + f"{mixture_case[1]}" + r"," +  # Sunny
                   r"\pi_3=" + f"{mixture_case[2]}" + r")$",  # Overcast
                   fontsize="x-large")

    if ii == 1:
        ax_0.legend(lines_, labels_, loc="upper center", handlelength=1.5, labelspacing=0.2, borderaxespad=0.1, fontsize="large")

    # WARNING: I changed the order, so it agrees with the clustering of one of the figures in the paper
    # In the paper the order is (cluster_1 = cloudy, cluster_2=overcast, cluster_3=sunny)
    # ax_0.set_title(f"{titles_list[ii]}\n"
    #                r"$(\pi_1=" + f"{mixture_case[0]}" + r"," +
    #                r"\pi_2=" + f"{mixture_case[2]}" + r"," +
    #                r"\pi_3=" + f"{mixture_case[1]}" + r")$",
    #                fontsize="x-large")

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

    # Change the axis to percentages
    valsy = ax_0.get_yticks()
    ax_0.yaxis.set_major_locator(ticker.FixedLocator(valsy))
    ax_0.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=r'\%', is_latex=True))

    valsx = ax_0.get_xticks()
    ax_0.xaxis.set_major_locator(ticker.FixedLocator(valsx))
    ax_0.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=r'\%', is_latex=True))

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

        ax_col.text(x=x_axis[np.ceil(x_max_/2).astype(int)],
                    y=y_max_ * 1.008,
                    s=str(round(y_max_, 3)) + " [p.u]", ha="center", va="center", fontsize="large",
                    color=color_)
        ax_col.hlines(y=y_max_, xmin=x_axis[0], xmax=x_axis[x_max_], color="k", linewidth=0.8, linestyles="--")
        # x_right_limit = ax_col.get_xlim()[1]

        ax_col.text(x=x_axis[x_min_-1],
                    y=y_min_ * (2 - 1.008),
                    s=str(round(y_min_, 3)) + " [p.u]", ha="center", va="center", fontsize="large",
                    color=color_)
        ax_col.hlines(y=y_min_, xmin=x_axis[x_min_], xmax=x_axis[-1], color="k", linewidth=0.8, linestyles="--")

        for spines in ax_col.spines.values():
            spines.set_color(color_)
            spines.set_linewidth(1.0)

    ax_1.set_xticklabels(labels=[])
    ax_2.set_xticklabels(labels=[])
    ax_3.set_xticklabels(labels=[])
    ax_4.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    # ax_4.set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=29)))
    ax_4.set_xlabel("Time of day")

    if ii == 0:  # ii Stands for columns in the plot
        ax_1.set_ylabel(r"Energy growth: 0\%" + "\n" + r"PV growth: 100\%" + "\nVoltage mag. [p.u.]", fontsize="large")
        ax_2.set_ylabel(r"Energy growth: 100\%" + "\n" + r"PV growth: 0\%" + "\nVoltage mag. [p.u.]", fontsize="large")
        ax_3.set_ylabel(r"Energy growth: 0\%" + "\n" + r"PV growth: 0\%" + "\nVoltage mag. [p.u.]", fontsize="large")
        ax_4.set_ylabel(r"Energy growth: 50\%" + "\n" + r"PV growth: 50\%" + "\nVoltage mag. [p.u.]", fontsize="large")
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
    ax_1.set_ylim((0.96, 1.115))
    ax_2.set_ylim((0.930, 1.032))
    ax_3.set_ylim((0.970, 1.058))
    ax_4.set_ylim((0.955, 1.08))



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

critical_quantile_max = ["75", "80", "85", "90", "95", "975", "98", "99", "100"]
voltages_levels= {"danger": max_technical_voltage_danger,
                  "caution": max_technical_voltage_caution}
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantile_max)


# warnings.filterwarnings("default")
border_limit_region = {}
for keys_ in voltages_levels:
    max_technical_voltage = voltages_levels[keys_]
    border_danger = {}
    for quantile_max in critical_quantile_max:
        worst_line = get_worst_max_scenario_v2(solutions_dict=solutions_dictx,
                                               quantile=quantile_max,
                                               all_tuple_things=cases_combinations_irradiance_only_tuples,
                                               pv_growth_percentiles=x,
                                               load_growth_percentiles=y,
                                               voltage_cut_level=max_technical_voltage,
                                               offset=OFFSET)
        border_danger[quantile_max] = worst_line

    border_limit_region[keys_] = border_danger

# max_technical_voltage_danger = 1.04
warnings.filterwarnings("error")
critical_quantile_max = ["100", "75", "99"]
critical_quantile_min = ["05", "10", "05"]
titles_list = ["(a)", "(b)", "(c)"]
min_quant_span = [0.9114, 0.909, 0.86]  # This is calculated zooming into the figure and see the vertical contours
lambda_loads = [0.4, 0.6]  # Where do i want to put the lambdas for the load
critical_quantiles = list(set(critical_quantile_min + critical_quantile_max))
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles + ["100"])

fig, ax_t = plt.subplots(1, 3, figsize=(7, 2.8))
plt.subplots_adjust(left=0.08, right=0.95, top=0.86, bottom=0.3, wspace=0.35)

# Iterate through columns
for ii, ax in enumerate(ax_t):  # Iterate through columns

    if ii == 2:
        loading_levels = x
        # risk_mapping = {"75": 0.25, "80": 0.2, "85": 0.15, "90": 0.1, "95": 0.05, "98": 0.02, "99": 0.01, "100": 0.0}
        risk_mapping = {"85": 0.15, "90": 0.1, "95": 0.05, "98": 0.02, "99": 0.01, "100": 0.0}
        pvhc_loading_values = {}
        for loading_value in [0.4, 0.6]:
            pvhc_loading = []
            risk_values = []
            idx_loading = np.where(np.isclose(loading_levels, loading_value))[0][0]
            for key_ in risk_mapping:
                pvhc_loading.append(border_limit_region["danger"][key_][idx_loading])
                risk_values.append(risk_mapping[key_])

            pvhc_loading_values[loading_value] = {"pvhc": pvhc_loading,
                                                  "risks": risk_values}

        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        colors = ['blue', 'grey', 'green']
        cc = itertools.cycle(colors)

        for loading_value, sign_ in zip(pvhc_loading_values, [-0.05, 0.05]):
            the_color = next(cc)

            risks_x_axis = np.array(pvhc_loading_values[loading_value]["risks"]) * 100
            ax.plot(risks_x_axis,
                    pvhc_loading_values[loading_value]["pvhc"],
                    "-o",
                    fillstyle='none',
                    color=the_color,
                    label=r"$\lambda_{" + f"{loading_value}" + r"}$")

            idx_loading_10 = np.where(np.isclose(risks_x_axis, 10))[0][0]
            idx_loading_5 = np.where(np.isclose(risks_x_axis, 5))[0][0]
            idx_loading_0 = np.where(np.isclose(risks_x_axis, 0))[0][0]

            pvhc_y_10 = pvhc_loading_values[loading_value]["pvhc"][idx_loading_10]
            pvhc_y_5 = pvhc_loading_values[loading_value]["pvhc"][idx_loading_5]
            pvhc_y_0 = pvhc_loading_values[loading_value]["pvhc"][idx_loading_0]

            ax.text(x=10 , y=pvhc_y_10 + sign_,
                    s=r"$\lambda_{" + f"{loading_value}" + r"}^{" + f"{int(10)}" + r"\%}$",
                    ha="center", va="center", fontsize=10, color=the_color)

            ax.text(x=5 , y=pvhc_y_5 + sign_,
                    s=r"$\lambda_{" + f"{loading_value}" + r"}^{" + f"{int(5)}" + r"\%}$",
                    ha="center", va="center", fontsize=10, color=the_color)

            ax.text(x=-1, y=pvhc_y_0 + sign_,
                    s=r"$\lambda_{" + f"{loading_value}" + r"}^{" + f"{int(0)}" + r"\%}$",
                    ha="center", va="center", fontsize=10, color=the_color)



        ax.set_ylim((0.2, 0.7))
        ax.set_xlim((-2.5, 16))
        ax.set_xlabel("Risk")
        ax.set_ylabel("PV Growth")

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        ax.grid("both")
        ax.legend(fontsize="medium",
                  loc="lower right",
                  title="Load growth")
        ax.set_title(titles_list[ii] + "\n" + "Installed PV capacity", fontsize="x-large")

    else:
        quant_to_process_min = critical_quantile_min[ii]
        # MINIMUM contour plots
        list_matrices = get_matrices_critical_quantiles(quant_to_process_min,
                                                        all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                        solutions_dict=solutions_dictx[quant_to_process_min],
                                                        pv_growth_percentiles=x,
                                                        load_growth_percentiles=y,
                                                        process_max=False,
                                                        OFFSET=OFFSET)
        q_05_min, q_50_min, q_95_min, _ = plot_contour_levels_irradiance_days(list_matrices,
                                                                              contour_level=min_technical_voltage,
                                                                              ax=ax,
                                                                              alpha_line_contour=0.2,
                                                                              pv_growth_percentiles=x,
                                                                              load_growth_percentiles=y,
                                                                              linecolor_contour="b",
                                                                              linecolor_quantile="b",
                                                                              alpha_line_quantile=0.0,
                                                                              plot_quantiles=False
                                                                              )

        # MAXIMUM contour plots
        quant_to_process_max = critical_quantile_max[ii]
        list_matrices = get_matrices_critical_quantiles(quant_to_process_max,
                                                        all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                        solutions_dict=solutions_dictx[quant_to_process_max],
                                                        pv_growth_percentiles=x,
                                                        load_growth_percentiles=y,
                                                        process_max=True,
                                                        OFFSET=OFFSET)

        # Caution
        plot_contour_levels_irradiance_days(list_matrices,
                                            contour_level=max_technical_voltage_caution,
                                            # linecolor_contour="grey",
                                            linecolor_contour="#FFA500",
                                            linecolor_quantile="#FFA500",
                                            pv_growth_percentiles=x,
                                            load_growth_percentiles=y,
                                            contour_linewidths=1.00,
                                            alpha_line_contour=0.4,
                                            ax=ax,
                                            plot_quantiles=False)

        # Danger contour lines
        plot_contour_levels_irradiance_days(list_matrices,
                                            contour_level=max_technical_voltage_danger,
                                            # linecolor_contour="grey",
                                            linecolor_contour="r",
                                            linecolor_quantile="r",
                                            pv_growth_percentiles=x,
                                            load_growth_percentiles=y,
                                            # contour_linewidths=0.08,
                                            contour_linewidths=1.00,
                                            alpha_line_contour=0.4,
                                            ax=ax,
                                            plot_quantiles=False)



        ax.plot(x, border_limit_region["danger"]["100"], color="purple", linewidth=1.5, label=r"$\overline{V}:$" + r" 0\% Risk", zorder=3)
        ax.plot(x, border_limit_region["danger"][quant_to_process_max], color="r", linewidth=1.0, alpha=1.0,
                label=r"$\overline{V}:$" +  f" {100 - int(quant_to_process_max)}" + r"\% Risk")
        ax.plot(x, border_limit_region["caution"][quant_to_process_max], color="#FFA500", linewidth=1.0, alpha=1.0,
                label=r"$V_{\mathrm{caution}}:$" +  f" {100 - int(quant_to_process_max)}" + r"\% Risk")

        loading_levels = x
        loading_value_plot = lambda_loads[0]

        idx_loading_0 = np.where(np.isclose(loading_levels, lambda_loads[0]))[0][0]
        idx_loading_1 = np.where(np.isclose(loading_levels, lambda_loads[1]))[0][0]


        lambda_risk_0 = border_limit_region["danger"][quant_to_process_max][idx_loading_0]
        lambda_0_risk_0 = border_limit_region["danger"]["100"][idx_loading_0]

        ax.scatter([loading_value_plot], [lambda_risk_0], s=18, color="#660000", zorder=3)
        ax.scatter([loading_value_plot], [lambda_0_risk_0], s=18, color="purple", zorder=3)

        ax.text(x=loading_value_plot - 0.1, y=lambda_risk_0*1.15,
                s=r"$\lambda_{"+ f"{loading_value_plot}" + r"}^{" + f"{100 - int(quant_to_process_max)}" + r"\%}$",
                ha="center", va="center", fontsize=10, color="#660000")
        ax.text(x=loading_value_plot + 0.13, y=lambda_0_risk_0,
                s=r"$\lambda_{" + f"{loading_value_plot}" + r"}^{0\%}$",
                ha="center", va="center", fontsize=10, color="purple")

        loading_value_plot = lambda_loads[1]
        lambda_risk_1 = border_limit_region["danger"][quant_to_process_max][idx_loading_1]
        lambda_0_risk_1 = border_limit_region["danger"]["100"][idx_loading_1]

        ax.scatter([loading_value_plot], [lambda_risk_1], s=18, color="#660000", zorder=3)
        ax.scatter([loading_value_plot], [lambda_0_risk_1], s=18, color="purple", zorder=3)

        ax.text(x=loading_value_plot - 0.1, y=lambda_risk_1 * 1.15,
                s=r"$\lambda_{" + f"{loading_value_plot}" + r"}^{" + f"{100 - int(quant_to_process_max)}" + r"\%}$",
                ha="center", va="center", fontsize=10, color="#660000")
        ax.text(x=loading_value_plot + 0.13, y=lambda_0_risk_1,
                s=r"$\lambda_{" + f"{loading_value_plot}" + r"}^{0\%}$",
                ha="center", va="center", fontsize=10, color="purple")



        ax.fill_between(x, 0, border_limit_region["caution"][quant_to_process_max], color="green", alpha=0.3, zorder=0, label="Safe")
        ax.fill_between(x, border_limit_region["caution"][quant_to_process_max],
                        border_limit_region["danger"][quant_to_process_max], color="#FFA500", alpha=0.3, zorder=0, label="Caution")
        ax.fill_between(x,  border_limit_region["danger"][quant_to_process_max], 1, color="red", alpha=0.5, zorder=0, label="Overvoltage")
        # ax.fill_between(x, 1, y_case_min.mean(axis=0), color="blue", alpha=0.2)

        f_quant = interpolate.interp1d(x, border_limit_region["caution"][quant_to_process_max], fill_value='extrapolate')
        x_new = np.linspace(0,1,100)
        y_new = f_quant(x_new)

        idx = x_new > min_quant_span[ii]
        ax.fill_between(x_new[idx], 0, y_new[idx], color="blue", alpha=0.2, zorder=0, label="Undervoltage")
        # ax.axvspan(min_quant_span[ii], 1, 0.0, 1.0, alpha=0.2, color='blue')

        lh, ll = ax.get_legend_handles_labels()
        lh_lines = lh[0:3]
        ll_lines = ll[0:3]

        lh_regions = lh[3:]
        ll_regions = ll[3:]

        ax.grid()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_xlabel("Load Growth", fontsize="large")
        ax.set_ylabel("PV Growth", fontsize="large")
        ax.set_title(titles_list[ii] + "\n" + f"Risk {100 - int(quant_to_process_max)}" + r"\%", fontsize="x-large")
        ax.grid(which="both", linestyle=":")

    if ii==0:
        first_legend = ax.legend(lh_lines,
                                 ll_lines,
                                 handlelength=1.5, labelspacing=0.2, borderaxespad=0.1,
                                 fontsize="medium",
                                 loc="upper left")
        ax.add_artist(first_legend)
        ax.legend(lh_regions,
                  ll_regions,
                  fontsize="x-large",
                  bbox_to_anchor=(0.5, -0.18),
                  loc="upper left",
                  ncol=4,
                  title="Static Operating Regions:",
                  title_fontsize="x-large",
                  handlelength=1.5)
    elif ii==1:
        ax.legend(lh_lines,
                  ll_lines,
                  handlelength = 1.5, labelspacing = 0.2, borderaxespad = 0.1,
                  fontsize="medium",
                  loc="upper left")

plt.savefig('static_regions/static_contour_plots.pdf', dpi=700, bbox_inches='tight')


#%%
# =====================================================================================================================
# FIGURE 3: Collect all the lines for the 1.05 and 1.045 limits (One big plot attached to ternary plots)
# =====================================================================================================================

max_technical_voltage_danger = 1.05
warnings.filterwarnings("error")
critical_quantile_max = ["90"]
quant_to_process_max = "90"
# critical_quantile_min = ["25", "10", "05"]
titles_list = ["(a)", "(b)", "(c)"]
min_quant_span = [0.90, 0.88, 0.86]  # This is calculated zooming into the figure and see the vertical contours
critical_quantiles = list(set(critical_quantile_min + critical_quantile_max))
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)

fig, ax = plt.subplots(1, 1, figsize=(7. / 2.54, 7 / 2.54))
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
#
# ii = 0
# quant_to_process_min = critical_quantile_min[ii]
# list_matrices = get_matrices_critical_quantiles(quant_to_process_min,
#                                                 all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
#                                                 solutions_dict=solutions_dictx[quant_to_process_min],
#                                                 pv_growth_percentiles=x,
#                                                 load_growth_percentiles=y,
#                                                 process_max=False,
#                                                 OFFSET=OFFSET)
# q_05_min, q_50_min, q_95_min, _ = plot_contour_levels_irradiance_days(list_matrices,
#                                                                       contour_level=min_technical_voltage,
#                                                                       ax=ax,
#                                                                       pv_growth_percentiles=x,
#                                                                       load_growth_percentiles=y,
#                                                                       linecolor_contour="b",
#                                                                       linecolor_quantile="b",
#                                                                       alpha_line_quantile=0.0,
#                                                                       contour_linewidths=0.1)
#
#
# quant_to_process_max = critical_quantile_max[ii]
list_matrices = get_matrices_critical_quantiles(quant_to_process_max,
                                                all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                solutions_dict=solutions_dictx[quant_to_process_max],
                                                pv_growth_percentiles=x,
                                                load_growth_percentiles=y,
                                                process_max=True,
                                                OFFSET=OFFSET)
#
(q_05_max,
 q_50_max,
 q_95_max,
 contours_max_danger) = plot_contour_levels_irradiance_days(list_matrices,
                                                            contour_level=1.05,
                                                            # linecolor_contour="grey",
                                                            linecolor_contour="r",
                                                            linecolor_quantile="r",
                                                            pv_growth_percentiles=x,
                                                            load_growth_percentiles=y,
                                                            ax=ax,
                                                            plot_quantiles=False,
                                                            contour_linewidths=0.1)
(q_05_max_caution,
 q_50_max_caution,
 q_95_max_caution,
 contours_max_caution) = plot_contour_levels_irradiance_days(list_matrices,
                                                             contour_level=1.045,
                                                             # linecolor_contour="grey",
                                                             linecolor_contour="#FFA500",
                                                             linecolor_quantile="b",
                                                             pv_growth_percentiles=x,
                                                             load_growth_percentiles=y,
                                                             ax=ax,
                                                             plot_quantiles=False,
                                                             contour_linewidths=0.1)
#
# contours_max_caution_area = np.vstack([contours_max_danger["y_cases"], contours_max_caution["y_cases"]])
# q_05 = np.nanquantile(contours_max_caution_area, q=0.05, axis=0)
# q_50 = np.nanquantile(contours_max_caution_area, q=0.50, axis=0)
# q_90 = np.nanquantile(contours_max_caution_area, q=0.90, axis=0)

ax.plot(x, border_limit_region["danger"][quant_to_process_max], color="r", linewidth=1.0, alpha=1.0)
ax.plot(x, border_limit_region["caution"][quant_to_process_max], color="#FFA500", linewidth=1.0, alpha=1.0,)


ax.fill_between(x, 0, border_limit_region["caution"][quant_to_process_max], color="green", alpha=0.3, zorder=0,
                label="Safe")
ax.fill_between(x, border_limit_region["caution"][quant_to_process_max],
                border_limit_region["danger"][quant_to_process_max], color="#FFA500", alpha=0.3, zorder=0,
                label="Caution")
ax.fill_between(x, border_limit_region["danger"][quant_to_process_max], 1, color="red", alpha=0.5, zorder=0,
                label="Overvoltage")


f_quant = interpolate.interp1d(x, border_limit_region["caution"][quant_to_process_max], fill_value='extrapolate')
x_new = np.linspace(0,1,100)
y_new = f_quant(x_new)

idx = x_new > min_quant_span[0]
ax.vlines(x=x_new[idx][0], ymin=0, ymax=y_new[idx][0], color="b", linewidth=1.0, alpha=1.0, zorder=1)
ax.fill_between(x_new[idx], 0, y_new[idx], color="blue", alpha=0.5, zorder=0, label="Undervoltage")

ax.legend(fontsize="medium",
          # bbox_to_anchor=(0.015, -0.15),
          loc="upper left",
          ncol=2,
          title="Static Operating Regions:",
          title_fontsize="medium",
          handlelength=1,
          fancybox=True, framealpha=1.0)
ax.grid()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.set_xlabel(f"Annual energy" + r" consumption growth [\%]" , fontsize="large")
ax.set_ylabel("PV installed" + r" capacity growth [\%]", fontsize="large")
# Change the axis to percentages

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.20))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=r'\%', is_latex=True))

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0, symbol=r'\%', is_latex=True))

plt.savefig('static_regions/static_contour_for_ternary.pdf', dpi=700, bbox_inches='tight')
#%%

#
#
# pv_growth_percentiles = x
#
# # Soft the quantile curves
# from scipy.interpolate import make_interp_spline
# b_90 = make_interp_spline(pv_growth_percentiles, q_90)
# b_50 = make_interp_spline(pv_growth_percentiles, q_50)
# b_05 = make_interp_spline(pv_growth_percentiles, q_05)
#
# x_new_spline = np.linspace(pv_growth_percentiles.min(), pv_growth_percentiles.max(), 100)
# y_new_90 = b_90(x_new_spline)
# y_new_50 = b_50(x_new_spline)
# y_new_05 = b_05(x_new_spline)
#
# alpha_line_quantile = 1.0
# linecolor_quantile = "r"
# ax.plot(x_new_spline, y_new_90, linestyle="--", color=linecolor_quantile, alpha=alpha_line_quantile)
# ax.plot(x_new_spline, y_new_50, linestyle="-", color=linecolor_quantile, alpha=alpha_line_quantile)
# ax.plot(x_new_spline, y_new_05, linestyle="-.", color=linecolor_quantile, alpha=alpha_line_quantile)
#
# ax.grid()
# ax.set_xlim((0, 1))
# ax.set_ylim((0, 1))
# ax.set_xlabel("Load Growth", fontsize="large")
# ax.set_ylabel("PV Growth", fontsize="large")
#
# ax.fill_between(x_new_spline, 0, y_new_05, color="green", alpha=0.2, zorder=0, label="Safe")
# ax.fill_between(x_new_spline, y_new_05, y_new_90, color="orange", alpha=0.2, zorder=0, label="Caution")
# ax.fill_between(x_new_spline, y_new_90, 1, color="red", alpha=0.2, zorder=0, label="Overvoltage")
# # ax.fill_between(x, 1, y_case_min.mean(axis=0), color="blue", alpha=0.2)
#
# f_quant = interpolate.interp1d(x_new_spline, y_new_05, fill_value='extrapolate')
# x_new = np.linspace(0, 1, 100)
# y_new = f_quant(x_new)
#
# idx = x_new > min_quant_span[ii]
# ax.fill_between(x_new[idx], 0, y_new[idx], color="blue", alpha=0.2, zorder=0, label="Undervoltage")
# ax.legend(fontsize="medium",
#           # bbox_to_anchor=(0.015, -0.15),
#           loc="upper left",
#           ncol=2,
#           title="Static Operating Regions:",
#           title_fontsize="medium",
#           handlelength=1)
#
# plt.savefig('static_regions/static_contour_for_ternary.pdf', dpi=700, bbox_inches='tight')
#
# #%%
# # =====================================================================================================================
# # FIGURE 4: See the planes in 3D to understand better
# # =====================================================================================================================
#
# Ti_1, Ti_high_res_1, (X, Y), (grid_x, grid_y) = get_high_res_plane("90", [(0.4, 0.0, 0.6)], solutions_dictx, x, y, offset=OFFSET)
# Ti_2, Ti_high_res_2, _, _ = get_high_res_plane("90", [(0.0, 1.0, 0.0)], solutions_dictx, x, y, offset=OFFSET)
# Ti_3, Ti_high_res_3, _, _ = get_high_res_plane("90", [(0.3, 0.3, 0.4)], solutions_dictx, x, y, offset=OFFSET)
#
# Ti_1_min, Ti_high_res_1_min, (X_min, Y_min), (grid_x_min, grid_y_min) = get_high_res_plane("10", [(0.4, 0.0, 0.6)],
#                                                                                            solutions_dictx, x, y,
#                                                                                            process_max=False,
#                                                                                            offset=OFFSET)
# Ti_2_min, Ti_high_res_2_min, _, _ = get_high_res_plane("10", [(0.0, 1.0, 0.0)], solutions_dictx, x, y,
#                                                        process_max=False, offset=OFFSET)
# Ti_3_min, Ti_high_res_3_min, _, _ = get_high_res_plane("10", [(0.3, 0.3, 0.4)], solutions_dictx, x, y,
#                                                        process_max=False, offset=OFFSET)
#
# fig, ax = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection': '3d'}, figsize=(16, 12))
# ax[0, 0].plot_wireframe(X, Y, Ti_1, color='r', linewidth=0.4)
# ax[0, 1].plot_wireframe(grid_x, grid_y, Ti_high_res_1, color='b', linewidth=0.4, label="[(cloudy=0.4, sunny=0.0, overcast=0.6)]")
# ax[0, 1].plot_wireframe(grid_x, grid_y, Ti_high_res_2, color='r', linewidth=0.4, label=" [(0.0, 1.0, 0.0)]")
# ax[0, 1].plot_wireframe(grid_x, grid_y, Ti_high_res_3, color='orange', linewidth=0.4, label="[(0.3, 0.3, 0.4)]")
# ax[0, 1].plot_wireframe(grid_x, grid_y, np.full_like(Ti_high_res_2, 1.05), color='k', linewidth=0.4, label="Max. Technical limit")
# ax[0, 1].set_xlabel("Load growth [/%]")
# ax[0, 1].set_ylabel("PV growth [/%]")
# ax[0, 1].set_zlabel("Max. voltage [p.u]")
# ax[0, 1].set_title("Maximum grid voltage Quantile 90")
# ax[0, 0].set_title("Low resolution (Simulations)")
# ax[0, 2].set_title("Simulation + 3D Cubic Spline")
# ax[0, 1].legend(fontsize="large")
# ax[0, 2].plot_wireframe(X, Y, Ti_1, color='r', linewidth=1.0, label="Simulated")
# ax[0, 2].plot_wireframe(grid_x, grid_y, Ti_high_res_1, color='b', linewidth=0.4, label="Cubic spline")
# ax[0, 2].legend(fontsize="small")
#
# ax[1, 0].plot_wireframe(X, Y, Ti_1, color='r', linewidth=0.4)
# ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_1_min, color='b', linewidth=0.4, label="[(cloudy=0.4, sunny=0.0, overcast=0.6)]")
# ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_2_min, color='r', linewidth=0.4, label=" [(0.0, 1.0, 0.0)]")
# ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_3_min, color='orange', linewidth=0.4, label="[(0.3, 0.3, 0.4)]")
# ax[1, 1].plot_wireframe(grid_x_min, grid_y_min, np.full_like(Ti_high_res_2_min, 0.96), color='k', linewidth=0.4, label="Min. Technical limit")
# ax[1, 1].set_xlabel("Load growth [/%]")
# ax[1, 1].set_ylabel("PV growth [/%]")
# ax[1, 1].set_zlabel("Min. voltage [p.u]")
# ax[1, 1].set_title("Minimum grid voltage Quantile 90")
# ax[1, 0].set_title("Low resolution (Simulations)")
# ax[1, 2].set_title("Simulation + 3D Cubic Spline")
# ax[1, 1].legend(fontsize="large")
#
# ax[1, 2].plot_wireframe(X_min, Y_min, Ti_1_min, color='r', linewidth=1.0, label="Simulated")
# ax[1, 2].plot_wireframe(grid_x_min, grid_y_min, Ti_high_res_1_min, color='b', linewidth=0.4, label="Cubic spline")
# ax[1, 2].legend(fontsize="small")
#
# np.quantile([0.002, 1,1,2,3,3,3,3,4,3.4], q=0)
#
#
# # plt.style.use("seaborn-white")
# plt.style.use("../../probnum.mplstyle")


#%%
critical_quantiles = ["05", "10", "15", "25", "50", "75", "80", "85", "90", "95", "975","98","99","100"]
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)

outlier_set = {(0.0, 0.0, 1.0),
               (0.0, 0.1, 0.9),
               (0.1, 0.0, 0.9),
               (0.1, 0.2, 0.7),
               (0.4, 0.0, 0.6),
               (0.0, 0.2, 0.8),
               (0.2, 0.0, 0.8),
               (0.2, 0.1, 0.7),
               (0.1, 0.1, 0.8),
               (0.4, 0.4, 0.2),
               (0.7, 0.1, 0.2)}
cases_combinations_irradiance_only_tuples_ = list(set([irradiance_mixture for irradiance_mixture, _, _ in cases_combinations]))
cases_combinations_irradiance_only_tuples = list(set(cases_combinations_irradiance_only_tuples_) - outlier_set)


CUANTILE = "100"
fig, ax = plt.subplots(1,1, figsize=(5,5))
list_matrices = get_matrices_critical_quantiles(CUANTILE,
                                                all_mixtures_irradiance_only=cases_combinations_irradiance_only_tuples,
                                                solutions_dict=solutions_dictx[CUANTILE],
                                                pv_growth_percentiles=x,
                                                load_growth_percentiles=y,
                                                process_max=True,
                                                OFFSET=OFFSET)
_, _, _, y_data = plot_contour_levels_irradiance_days(list_matrices,
                                                      contour_level=1.05,
                                                      ax=ax,
                                                      alpha_line_contour=0.2,
                                                      pv_growth_percentiles=x,
                                                      load_growth_percentiles=y,
                                                      linecolor_contour="b",
                                                      linecolor_quantile="b",
                                                      alpha_line_quantile=0.0,
                                                      plot_quantiles=False
                                                      )

# Build frame
data_to_plotly = []
for ii in range(len(cases_combinations_irradiance_only_tuples)):
    data_to_plotly.append(pd.DataFrame(dict(x_values=x,
                          y_values=y_data["y_cases"][ii],
                          case_irr_mix=[cases_combinations_irradiance_only_tuples[ii]]*len(x))))
# Attach the worst ever
data_to_plotly.append(pd.DataFrame(dict(x_values=x,
                                          y_values=np.min(y_data["y_cases"], axis=0),
                                          case_irr_mix=["worst"]*len(x))))

data_to_plotly = pd.concat(data_to_plotly, axis=0, ignore_index=True)
# pd.concat(data_to_plotly, axis=0, ignore_index=True).to_csv("static_regions/frontiers.csv", index=False)

# ===============================
# import pandas as pd


# data = pd.read_csv("static_regions/frontiers.csv")
fig = px.line(data_to_plotly, x='x_values', y='y_values', color='case_irr_mix', symbol="case_irr_mix", width=800, height=800)
fig.write_html('static_regions/frontiers.html', auto_open=True)
