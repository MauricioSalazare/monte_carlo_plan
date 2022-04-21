import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from glob import glob
import os
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import interpolate
import warnings


def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator

def load_solutions(file_name_solutions):
    with open(file_name_solutions, "rb") as pickle_file:
        solutions = pickle.load(pickle_file)

    return solutions

file_name_scenario_generator_model = "../../models/scenario_generator_model_new_AWS.pkl"
# file_name_solutions = "solutions.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
# solutions = load_solutions(file_name_solutions)
cases_combinations = scenario_generator.cases_combinations

# assert len(cases_combinations) == len(solutions), "The labeling could be wrong"

# # Assign the case combination to the solutions.
# solutions_dict = {}
# for case_, solution in zip(cases_combinations, solutions):
#     solutions_dict[case_] = solution

file_name_solutions_dictionary = "solutions_dictionary_AWS.pkl"
with open(file_name_solutions_dictionary, "rb") as pickle_file:
    solutions_dict = pickle.load(pickle_file)

x = scenario_generator.percentages_pv_growth
y = scenario_generator.percentages_load_growth

#%%
matrix_voltages = np.zeros((len(x), len(y)))



# mixture_case = (0.0, 0.0, 1.0)
# mixture_case = (0.2, 0.0, 0.8)
# mixture_case = (0.4, 0.0, 0.6)
# mixture_case = (0.6, 0.0, 0.4)
# mixture_case = (0.8, 0.0, 0.2)
# mixture_case = (0.6, 0.4, 0.0)
# mixture_case = (0.4, 0.2, 0.4)
# mixture_case = (0.0, 1.0, 0.0)
#%%
mixture_cases = [(0.0, 0.0, 1.0),
                 (0.2, 0.0, 0.8),
                 (0.4, 0.0, 0.6),
                 (0.6, 0.0, 0.4),
                 (0.8, 0.0, 0.2),
                 (0.2, 0.6, 0.2),  # Normal case (Original)
                 (0.0, 1.0, 0.0)]

list_matrices = []
list_of_tuple_cases = []
for  mixture_case in mixture_cases:
    for i, pv in enumerate(x):
        for j, load  in enumerate(y):
            case = (mixture_case, load, pv)
            list_of_tuple_cases.append(case)
            matrix_voltages[i, j] = np.max(solutions_dict[case]['max_q_90'])

    list_matrices.append(matrix_voltages.copy())

#%%
warnings.filterwarnings("default")
fig, ax_ = plt.subplots(4, len(mixture_cases), figsize=(25, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.05, right=0.98, top=0.90)
ax_0_row = ax_[0,:].flatten()
ax_1_row = ax_[1,:].flatten()
ax_2_row = ax_[2,:].flatten()
ax_3_row = ax_[3,:].flatten()

for ii, (ax_0, ax_1, ax_2, ax_3, mixture_case) in enumerate(zip(ax_0_row,
                                                                ax_1_row,
                                                                ax_2_row,
                                                                ax_3_row,
                                                                mixture_cases)):

    ax_0.pcolor(x, y, list_matrices[ii], shading='auto', vmin=0.998, vmax=1.095)
    ax_0.set_xlabel("Load growth")
    ax_0.set_ylabel("PV growth")


    cs = ax_0.contour(x, y, list_matrices[ii], colors="k", levels=[1.05])
    ax_0.set_title(f"Mixture: {mixture_case}")
    ax_0.axis("square")

    for i, load_ in enumerate(x):
        for j, pv_ in enumerate(y):
            text = ax_0.text(pv_, load_, list_matrices[ii][i, j].round(3),
                           ha="center", va="center", color="w", fontsize="x-small")

    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["max_q_95"], linestyle="--", color="r", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["max_q_90"], linestyle="--", color="b", linewidth=0.5, )
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["max_q_75"], linestyle="--", color="g", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["max_q_50"], linestyle="-", color="r", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["max_q_25"], linestyle="-.", color="g", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["max_q_10"], linestyle="-.", color="b", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["max_q_05"], linestyle="-.", color="r", linewidth=0.5)

    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["max_q_95"], linestyle="--", color="r", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["max_q_90"], linestyle="--", color="b", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["max_q_75"], linestyle="--", color="g", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["max_q_50"], linestyle="-", color="r", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["max_q_25"], linestyle="-.", color="g", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["max_q_10"], linestyle="-.", color="b", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["max_q_05"], linestyle="-.", color="r", linewidth=0.5)

    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["max_q_95"], linestyle="--", color="r", linewidth=0.5, label="q_95")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["max_q_90"], linestyle="--", color="b", linewidth=0.5, label="q_90")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["max_q_75"], linestyle="--", color="g", linewidth=0.5, label="q_75")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["max_q_50"], linestyle="-", color="r", linewidth=0.5, label="q_50")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["max_q_25"], linestyle="-.", color="g", linewidth=0.5, label="q_25")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["max_q_10"], linestyle="-.", color="b", linewidth=0.5, label="q_10")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["max_q_05"], linestyle="-.", color="r", linewidth=0.5, label="q_05")

    if ii == 0:
        ax_1.set_ylabel(f"Load: 0.0 - PV: 1.0" + "\nVoltage [p.u.]", fontsize=13)
        ax_2.set_ylabel(f"Load: 1.0 - PV: 0.0" + "\nVoltage [p.u.]", fontsize=13)
        ax_3.set_ylabel(f"Load: 0.0 - PV: 0.0" + "\nVoltage [p.u.]", fontsize=13)

    if ii == 6:
        ax_3.legend(fontsize="xx-small")

    ax_1.set_ylim((0.99, 1.1))
    ax_2.set_ylim((0.98, 1.05))
    ax_3.set_ylim((0.98, 1.1))
fig.suptitle("0.9 percentile cut", fontsize=15)


#%% Network X
import networkx as nx
grid_info = pd.read_csv("../../data/processed_data/network_data/Lines_34.csv")

# case_ = ((0.2, 0.6, 0.2), 0.8, 0.4)  # Normal case below the critical boundary
# case_ = ((0.0, 1.0, 0.0), 0.8, 0.4)  # Normal case sunny
case_ = ((0.0, 1.0, 0.0), 0.6, 0.6)  # Normal case sunny above critical boundary

max_by_time = solutions_dict[case_]["tensor_voltage"].max(axis=2)  # Maximum voltages over the day for all scenarios and all nodes
scenario_ = 300
v_slack = 1.0

# voltages_grid = np.append(v_slack, max_by_time[:, scenario_])
voltages_grid = np.append(1.0, np.nanquantile(max_by_time, q=0.95, axis=1))

G = nx.Graph()
for _, (from_, to_, r) in grid_info[['FROM', 'TO', 'R']].iterrows():
    G.add_edge(int(from_), int(to_), length= r * 100)

label_str = [str(node) + ": " + str(voltage) for node, voltage in zip(range(1, G.nodes.__len__() + 1), voltages_grid.round(3))]
node_labels = dict(zip(range(1, G.nodes.__len__() + 1), label_str))

node_labels_2 = dict(zip(range(1, G.nodes.__len__() + 1), voltages_grid.round(3)))
nx.set_node_attributes(G, node_labels_2, "voltages")
labels = nx.get_node_attributes(G, "voltages")


fig, ax = plt.subplots(1,1,figsize=(6, 6))
pos = nx.kamada_kawai_layout(G)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, node_color=list(node_labels_2.values()),
                            node_size=140, cmap=plt.cm.seismic, vmin=0.95, vmax=1.095, ax=ax)
for key_, value_ in pos.items():
    ax.text(x=value_[0], y=value_[1], s=key_, fontsize=6, ha="center", va="center")
    ax.text(x=value_[0] + 0.035, y=value_[1], s=node_labels_2[key_], fontsize=6, ha="left")
ax.set_title(f"Mixutre: {case_[0]} - Load: {case_[1]} - PV: {case_[2]}", fontsize=10)
ax.axis("off")
plt.colorbar(nc, ax=ax)

#%%
# Get all the critical lines
warnings.filterwarnings("error")

critical_quantile = ["max_q_50", "max_q_75", "max_q_90", "max_q_95"]
quantile_file = ["50", "75", "90", "95"]

path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")

solutions_dict = []
for quant_name in quantile_file:
    file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quant_name}.pkl"
    with open(path_file_parent / file_name_solutions_dictionary, "rb") as pickle_file:
        file_solutions_dict = pickle.load(pickle_file)
        solutions_dict.append(file_solutions_dict)


all_mixtures = set([case_mixture_part[0] for case_mixture_part in cases_combinations])

fig, ax_t = plt.subplots(2, len(critical_quantile), figsize=(20, 10))
ax_ = ax_t[0, :].flatten()
ay_ = ax_t[1, :].flatten()

x_quantiles = []
y_quantiles =[]
for ax, ay, critical_quantile_, solutions_dict_ in zip(ax_, ay_, critical_quantile, solutions_dict):
    list_matrices = []
    for  mixture_case in all_mixtures:
        for i, pv in enumerate(x):  # x == pv growth percentiles
            for j, load  in enumerate(y):  # y == load growth percentiles
                case = (mixture_case, load, pv)
                matrix_voltages[i, j] = np.max(solutions_dict_[case][critical_quantile_])

        list_matrices.append(matrix_voltages.copy())

    x_case = []
    y_case = []

    for matrix_ in list_matrices:
        try:
            cs = ax.contour(x, y, matrix_, colors="k", levels=[1.05])
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

        x_case.append(x)
        y_case.append(ynew)

    x_case = np.array(x_case)
    y_case = np.array(y_case)

    ay.plot(x_case.T, y_case.T, color="grey", linewidth=0.7)
    ay.plot(x_case[0,:], np.nanquantile(y_case, q=0.5, axis=0), color="r", linewidth=1.8)
    ay.grid()
    ay.set_xlim((0, 1))
    ay.set_ylim((0, 1))
    ay.set_xlabel("Load Growth")
    ay.set_ylabel("PV Growth")

    x_quantiles.append(x_case)
    y_quantiles.append(y_case)

    ax.set_title(f"Quant: {critical_quantile_}", fontsize=9)
    ax.set_xlabel("Load Growth")
    ax.set_ylabel("PV Growth")

    ax.grid()


























#%%
# im = ax.imshow(matrix_voltages, interpolation="spline16")
# ax.set_xlabel("load")
# ax.set_ylabel("pv")
#
# for i in range(len(x)):
#     for j in range(len(y)):
#         text = ax.text(j, i, matrix_voltages[i, j].round(3),
#                        ha="center", va="center", color="w", fontsize="x-small")
# # ax.invert_xaxis()
#
# ax.invert_yaxis()
# # ax.set_xticks(np.arange(len(x)), labels=x.astype(str))
# # ax.set_yticks(np.arange(len(y)), labels=y.astype(str))
#
# fig, ax = plt.subplots()
# for i, load_ in enumerate(x):
#     for j, pv_ in enumerate(y):
#         text = ax.text(pv_, load_, matrix_voltages[i, j].round(3),
#                        ha="center", va="center", color="w", fontsize="x-small")
# cs = ax.contour(x,y, matrix_voltages, colors="k", levels=[0.9])
# ax.set_title(f"Mixture: {mixture_case}")
# ax.axis("square")





#%%


#
#
#
# fig, ax = plt.subplots()
# im = ax.imshow(matrix_voltages, interpolation="spline16",
#                vmin=0.998, vmax=1.095)
# ax.set_xlabel("load")
# ax.set_ylabel("pv")
#
# for i in range(len(x)):
#     for j in range(len(y)):
#         text = ax.text(j, i, matrix_voltages[i, j].round(3),
#                        ha="center", va="center", color="w", fontsize="x-small")
# # ax.invert_xaxis()
#
# ax.invert_yaxis()
# # ax.set_xticks(np.arange(len(x)), labels=x.astype(str))
# # ax.set_yticks(np.arange(len(y)), labels=y.astype(str))
#
# fig, ax = plt.subplots()
# ax.pcolor(x, y, matrix_voltages, shading='auto', vmin=0.998, vmax=1.095)
# ax.set_xlabel("loads")
# ax.set_ylabel("pv")
#
# for i, load_ in enumerate(x):
#     for j, pv_ in enumerate(y):
#         text = ax.text(pv_, load_, matrix_voltages[i, j].round(3),
#                        ha="center", va="center", color="w", fontsize="x-small")
# cs = ax.contour(x,y, matrix_voltages, colors="k", levels=[1.05])
# ax.set_title(f"Mixture: {mixture_case}")
# ax.axis("square")
#
#
#
#
#
# #%%
# for i, pv in enumerate(x):
#     for j, load  in enumerate(y):
#         case = (mixture_case, load, pv)
#         matrix_voltages[i, j] = np.min(solutions_dict[case]['min_q_05'])
#
# fig, ax = plt.subplots()
# im = ax.imshow(matrix_voltages, interpolation="spline16"
#               )
# ax.set_xlabel("load")
# ax.set_ylabel("pv")
#
# for i in range(len(x)):
#     for j in range(len(y)):
#         text = ax.text(j, i, matrix_voltages[i, j].round(3),
#                        ha="center", va="center", color="w", fontsize="x-small")
# # ax.invert_xaxis()
#
# ax.invert_yaxis()
# # ax.set_xticks(np.arange(len(x)), labels=x.astype(str))
# # ax.set_yticks(np.arange(len(y)), labels=y.astype(str))
#
# fig, ax = plt.subplots()
# ax.pcolor(x, y, matrix_voltages, shading='auto')
# ax.set_xlabel("loads")
# ax.set_ylabel("pv")
#
# for i, load_ in enumerate(x):
#     for j, pv_ in enumerate(y):
#         text = ax.text(pv_, load_, matrix_voltages[i, j].round(3),
#                        ha="center", va="center", color="w", fontsize="x-small")
# cs = ax.contour(x,y, matrix_voltages, colors="k", levels=[0.9])
# ax.set_title(f"Mixture: {mixture_case}")
# ax.axis("square")
