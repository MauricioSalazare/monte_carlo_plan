import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.ticker as ticker
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


def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file_:
        scenario_generator = pickle.load(pickle_file_)
    return scenario_generator


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



OFFSET = 0.008  # TODO: WARNING: This value should be the same in the in the ternary plot scripts!!!
# max_technical_voltage_green = 1.040
max_technical_voltage_caution = 1.045
max_technical_voltage_danger = 1.05
min_technical_voltage = 0.95

path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
critical_quantiles = ["05", "10", "15", "25", "50", "75", "90", "95", "100"]
solutions_dictx_new = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles)
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

# file_name_scenario_generator_model = "../models/scenario_generator_model_new_AWS.pkl"
# scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
# cases_combinations_scenario_generator = scenario_generator.cases_combinations
# x = scenario_generator.percentages_pv_growth
# y = scenario_generator.percentages_load_growth

x = np.linspace(0, 1.0, 11).round(1)
y = np.linspace(0, 1.0, 11).round(1)

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

#%%
# max_technical_voltage_danger = 1.04
warnings.filterwarnings("error")
critical_quantile_max = ["100", "90", "75"]
critical_quantile_min = ["05", "10", "25"]
titles_list = ["(a)", "(b)", "(c)"]
min_quant_span = [0.9119, 0.9218, 0.9392]  # This is calculated zooming into the figure and see the vertical contours
lambda_loads = [0.4, 0.6]  # Where do i want to put the lambdas for the load
critical_quantiles = list(set(critical_quantile_min + critical_quantile_max))
solutions_dictx = get_solutions_dictionary(path_file_parent, quantiles=critical_quantiles + ["100"])

#%%
set_figure_art(fontsize=8)
perc_offset_y = [0.7, 0.82, 0.85]
fig, ax_t = plt.subplots(1, 3, figsize=(7, 2.8))
plt.subplots_adjust(left=0.08, right=0.95, top=0.86, bottom=0.3, wspace=0.35)

# Iterate through columns
for ii, ax in enumerate(ax_t):  # Iterate through columns
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


    # Load 1
    lambda_risk_0 = border_limit_region["danger"][quant_to_process_max][idx_loading_0]
    lambda_0_risk_0 = border_limit_region["danger"]["100"][idx_loading_0]

    ax.scatter([loading_value_plot], [lambda_risk_0], s=18, color="#660000", zorder=3)
    ax.scatter([loading_value_plot], [lambda_0_risk_0], s=18, color="purple", zorder=3)


    ax.text(x=loading_value_plot, y=lambda_risk_0 * perc_offset_y[ii],
            s=r"$\lambda_{"+ f"{loading_value_plot}" + r"}^{" + f"{100 - int(quant_to_process_max)}" + r"\%}$",
            ha="center", va="center", fontsize=10, color="#660000")
    if ii != 0:
        ax.text(x=loading_value_plot + 0.13, y=lambda_0_risk_0,
                s=r"$\lambda_{" + f"{loading_value_plot}" + r"}^{0\%}$",
                ha="center", va="center", fontsize=10, color="purple")

    # Load 2
    loading_value_plot = lambda_loads[1]
    lambda_risk_1 = border_limit_region["danger"][quant_to_process_max][idx_loading_1]
    lambda_0_risk_1 = border_limit_region["danger"]["100"][idx_loading_1]

    ax.scatter([loading_value_plot], [lambda_risk_1], s=18, color="#660000", zorder=3)
    ax.scatter([loading_value_plot], [lambda_0_risk_1], s=18, color="purple", zorder=3)

    ax.text(x=loading_value_plot, y=lambda_risk_1 * perc_offset_y[ii],
            s=r"$\lambda_{" + f"{loading_value_plot}" + r"}^{" + f"{100 - int(quant_to_process_max)}" + r"\%}$",
            ha="center", va="center", fontsize=10, color="#660000")

    if ii != 0:
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

    first_legend = ax.legend(lh_lines,
                             ll_lines,
                             handlelength=1.5, labelspacing=0.2, borderaxespad=0.1,
                             fontsize="medium",
                             loc="upper left")

    if ii==0:
        ax.add_artist(first_legend)
        ax.legend(lh_regions,
                  ll_regions,
                  fontsize="large",
                  bbox_to_anchor=(0.45, -0.18),
                  loc="upper left",
                  ncol=4,
                  title="Static Operating Zones:",
                  title_fontsize="large",
                  handlelength=1.5)

plt.savefig('static_regions/static_contour_plots.pdf', dpi=700, bbox_inches='tight')

#%%
# =====================================================================================================================
# FIGURE 1: Static operating zones
# =====================================================================================================================

set_figure_art(fontsize=8)
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
plt.subplots_adjust(top=0.95, right=0.95, left=0.15, bottom=0.1)
# plt.tight_layout()
loading_levels = x
# risk_mapping = {"75": 0.25, "80": 0.2, "85": 0.15, "90": 0.1, "95": 0.05, "98": 0.02, "99": 0.01, "100": 0.0}
risk_mapping = {"85": 0.15, "90": 0.1, "95": 0.05, "98": 0.02, "99": 0.01, "100": 0.0}
pvhc_loading_values = {}
for loading_value in [0.0, 0.2, 0.4, 0.6, 0.8]:
    pvhc_loading = []
    risk_values = []
    idx_loading = np.where(np.isclose(loading_levels, loading_value))[0][0]
    for key_ in risk_mapping:
        pvhc_loading.append(border_limit_region["danger"][key_][idx_loading])
        risk_values.append(risk_mapping[key_])

    pvhc_loading_values[loading_value] = {"pvhc": pvhc_loading,
                                          "risks": risk_values}

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
colors = ['blue', 'grey', 'green', 'orange', 'olive']
cc = itertools.cycle(colors)

for loading_value, sign_ in zip(pvhc_loading_values, [-0.05, -0.05, 0.05, 0.05, 0.05]):
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
            ha="center", va="center", fontsize=9, color=the_color)

    ax.text(x=5 , y=pvhc_y_5 + sign_,
            s=r"$\lambda_{" + f"{loading_value}" + r"}^{" + f"{int(5)}" + r"\%}$",
            ha="center", va="center", fontsize=9, color=the_color)

    ax.text(x=-1, y=pvhc_y_0 + sign_,
            s=r"$\lambda_{" + f"{loading_value}" + r"}^{" + f"{int(0)}" + r"\%}$",
            ha="center", va="center", fontsize=9, color=the_color)

ax.set_ylim((0.0, 1.0))
ax.set_xlim((-3.5, 16))
ax.set_xlabel("Risk")
ax.set_ylabel("PV Growth")

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
ax.grid("both")
ax.legend(loc="upper left",
          fontsize="medium",
          title="Load growth",
          handlelength=1.5, labelspacing=0.2, borderaxespad=0.1)
plt.savefig('static_regions/risk_plots.pdf', dpi=700, bbox_inches='tight')
# ax.set_title(titles_list[ii] + "\n" + "Installed PV capacity", fontsize="x-large")

