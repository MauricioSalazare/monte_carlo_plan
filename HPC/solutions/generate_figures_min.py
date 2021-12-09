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

file_name_scenario_generator_model = "../../models/scenario_generator_model.pkl"
file_name_solutions = "solutions.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
solutions = load_solutions(file_name_solutions)
cases_combinations = scenario_generator.cases_combinations

assert len(cases_combinations) == len(solutions), "The labeling could be wrong"

# Assign the case combination to the solutions.
solutions_dict = {}
for case_, solution in zip(cases_combinations, solutions):
    solutions_dict[case_] = solution

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
for  mixture_case in mixture_cases:
    for i, pv in enumerate(x):
        for j, load  in enumerate(y):
            case = (mixture_case, load, pv)
            matrix_voltages[i, j] = np.min(solutions_dict[case]['min_q_05'])

    list_matrices.append(matrix_voltages.copy())

#%%

fig, ax_ = plt.subplots(4, len(mixture_cases), figsize=(25, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.05, right=0.98, top=0.95)
ax_0_row = ax_[0,:].flatten()
ax_1_row = ax_[1,:].flatten()
ax_2_row = ax_[2,:].flatten()
ax_3_row = ax_[3,:].flatten()

for ii, (ax_0, ax_1, ax_2, ax_3, mixture_case) in enumerate(zip(ax_0_row,
                                                                ax_1_row,
                                                                ax_2_row,
                                                                ax_3_row,
                                                                mixture_cases)):

    ax_0.pcolor(x, y, list_matrices[ii], shading='auto', vmin=0.95, vmax=1.006)
    ax_0.set_xlabel("Load growth")
    ax_0.set_ylabel("PV growth")


    cs = ax_0.contour(x, y, list_matrices[ii], colors="k", levels=[0.96])
    ax_0.set_title(f"Mixture: {mixture_case}")
    ax_0.axis("square")

    for i, load_ in enumerate(x):
        for j, pv_ in enumerate(y):
            text = ax_0.text(pv_, load_, list_matrices[ii][i, j].round(3),
                           ha="center", va="center", color="w", fontsize="x-small")

    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["min_q_95"], linestyle="--", color="r", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["min_q_90"], linestyle="--", color="b", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["min_q_75"], linestyle="--", color="g", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["min_q_50"], linestyle="-", color="r", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["min_q_25"], linestyle="-.", color="g", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["min_q_10"], linestyle="-.", color="b", linewidth=0.5)
    ax_1.plot(solutions_dict[(mixture_case, 0.0, 1.0)]["min_q_05"], linestyle="-.", color="r", linewidth=0.5)

    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["min_q_95"], linestyle="--", color="r", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["min_q_90"], linestyle="--", color="b", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["min_q_75"], linestyle="--", color="g", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["min_q_50"], linestyle="-", color="r", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["min_q_25"], linestyle="-.", color="g", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["min_q_10"], linestyle="-.", color="b", linewidth=0.5)
    ax_2.plot(solutions_dict[(mixture_case, 1.0, 0.0)]["min_q_05"], linestyle="-.", color="r", linewidth=0.5)

    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["min_q_95"], linestyle="--", color="r", linewidth=0.5, label="q_95")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["min_q_90"], linestyle="--", color="b", linewidth=0.5, label="q_90")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["min_q_75"], linestyle="--", color="g", linewidth=0.5, label="q_75")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["min_q_50"], linestyle="-", color="r", linewidth=0.5, label="q_50")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["min_q_25"], linestyle="-.", color="g", linewidth=0.5, label="q_25")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["min_q_10"], linestyle="-.", color="b", linewidth=0.5, label="q_10")
    ax_3.plot(solutions_dict[(mixture_case, 0.0, 0.0)]["min_q_05"], linestyle="-.", color="r", linewidth=0.5, label="q_05")

    if ii == 0:
        ax_1.set_ylabel(f"Load: 0.0 - PV: 1.0" + "\nVoltage [p.u.]", fontsize=13)
        ax_2.set_ylabel(f"Load: 1.0 - PV: 0.0" + "\nVoltage [p.u.]", fontsize=13)
        ax_3.set_ylabel(f"Load: 0.0 - PV: 0.0" + "\nVoltage [p.u.]", fontsize=13)

    if ii == 6:
        ax_3.legend(fontsize="xx-small")

    # ax_1.set_ylim((0.99, 1.1))
    # ax_2.set_ylim((0.98, 1.05))
    # ax_3.set_ylim((0.98, 1.1))



#%% Network X
import networkx as nx
grid_info = pd.read_csv("../../data/processed_data/network_data/Lines_34.csv")

# case_ = ((0.2, 0.6, 0.2), 0.8, 0.4)  # Normal case below the critical boundary
# case_ = ((0.0, 1.0, 0.0), 0.8, 0.4)  # Normal case sunny
case_ = ((0.0, 0.0, 1.0), 1.0, 0.0)  # Normal case sunny above critical boundary

min_by_time = solutions_dict[case_]["tensor_voltage"].min(axis=2)  # Maximum voltages over the day for all scenarios and all nodes
scenario_ = 300
v_slack = 1.0

# voltages_grid = np.append(v_slack, max_by_time[:, scenario_])
voltages_grid = np.append(1.0, np.nanquantile(min_by_time, q=0.95, axis=1))

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
