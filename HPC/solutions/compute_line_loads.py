import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator



file_name_scenario_generator_model = "../../models/scenario_generator_model_new_AWS.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
cases_combinations = scenario_generator.cases_combinations

scenario_number = list(range(len(cases_combinations)))
scenario_to_mixture = dict(zip(scenario_number, cases_combinations))
mixture_to_scenario = dict(zip(cases_combinations, scenario_number))

s_base = 1000
v_base = 11
z_base = (v_base ** 2 * 1000) / s_base
i_base = s_base / (np.sqrt(3) * v_base)


node_file_path = r"../../data/processed_data/network_data/Nodes_34.csv"
lines_file_path = r"../../data/processed_data/network_data/Lines_34.csv"

branch_info = pd.read_csv(lines_file_path)
bus_info = pd.read_csv(node_file_path)

# Get info first line (connected to transformer)
from_node = branch_info.loc[1]["FROM"].astype(int)
to_node = branch_info.loc[1]["TO"].astype(int)
z_imp_pu = (branch_info.loc[1]["R"] + 1j * branch_info.loc[1]["X"]) / z_base


#%%
# ((cloudy, sunny, dark) Load, PV)

super_high_sun_base = mixture_to_scenario[((0.0, 1.0, 0.0), 0.0, 0.0)]
super_high_sun_high_pv = mixture_to_scenario[((0.0, 1.0, 0.0), 0.0, 1.0)]
super_dark_high_load = mixture_to_scenario[((0.0, 0.0, 1.0), 1.0, 0.0)]
super_dark_medium_load = mixture_to_scenario[((0.0, 0.0, 1.0), 0.5, 0.0)]
super_sunny_medium_load_medium_pv = mixture_to_scenario[((0.0, 1.0, 0.0), 0.5, 0.5)]
medium_sunny_medium_load_medium_pv = mixture_to_scenario[((0.2, 0.5, 0.3), 0.5, 0.5)]
ideal_sunny_medium_load_medium_pv = mixture_to_scenario[((0.0, 1.0, 0.0), 0.5, 0.3)]

high_sunny_high_load_medium_pv = mixture_to_scenario[((0.0, 1.0, 0.0), 0.8, 0.3)]
high_sunny_high_load_medium_pv = mixture_to_scenario[((0.2, 0.2, 0.6), 0.9, 0.8)]


# Structure on dictionary
# temp_dict[(n_scenario, n_time_step)]
case_number = super_high_sun_base
path_file_parent = Path(r"D:\monte_carlo_solutions_AWS")
file_name = f"voltage_dict_case_{case_number}_scenarios_500.pkl"

with open(path_file_parent / file_name, "rb") as pickle_file:
    temp_dict = pickle.load(pickle_file)


line_current_amp = []
line_angle_deg = []
voltage_diff = []
for scenario in range(500):
    line_current_scenario = []
    line_angle_scenario = []
    voltage_diff_scenario = []
    for time_step in range(48):
        vs = temp_dict[(scenario, time_step)]["v"][from_node - 2]
        vr = temp_dict[(scenario, time_step)]["v"][to_node - 2]
        line_current_scenario.append(abs((vs - vr) / z_imp_pu ) * i_base)
        line_angle_scenario.append(np.rad2deg(
                                              np.angle((vs - vr))
                                              )
                                  )
        voltage_diff_scenario.append((vs - vr))
    line_current_amp.append(line_current_scenario)
    line_angle_deg.append(line_angle_scenario)
    voltage_diff.append(voltage_diff_scenario)


# fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
# for line_current_ in line_current_amp:
#     ax.plot(line_current_, color="grey", linewidth=0.2)

line_data_scenario = np.array(line_current_amp)
q_95 = np.nanquantile(line_data_scenario, q=0.95, axis=0)
q_90 = np.nanquantile(line_data_scenario, q=0.90, axis=0)
q_75 = np.nanquantile(line_data_scenario, q=0.75, axis=0)
q_50 = np.nanquantile(line_data_scenario, q=0.50, axis=0)
q_25 = np.nanquantile(line_data_scenario, q=0.25, axis=0)
q_10 = np.nanquantile(line_data_scenario, q=0.10, axis=0)
q_05 = np.nanquantile(line_data_scenario, q=0.05, axis=0)
all_quantile = [q_95, q_90, q_75, q_50, q_25, q_10, q_05]

A_MAX = 230
A_MAX16 = np.round(A_MAX * 1.16, 3)

idx_violations = (line_data_scenario > A_MAX)
amps_violations = np.zeros_like(line_data_scenario)
amps_violations[idx_violations] = line_data_scenario[idx_violations]
counts_violation = idx_violations.sum(axis=1)

times_above, counts_above = np.unique(idx_violations.sum(axis=1), return_counts=True)


total_violations = pd.DataFrame({"times": times_above.tolist(), "counts": counts_above.tolist()})
total_violations.drop(0, inplace=True)

total_violations["perc"] = round(total_violations["counts"] / total_violations["counts"].sum(), 3)

print(total_violations)


fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
for quant_line in all_quantile:
    ax.plot(quant_line, color="grey", linewidth=0.4)
ax.hlines(y=A_MAX, xmin=0, xmax=ax.get_xlim()[1], color="r", linewidth=0.8)
ax.hlines(y=A_MAX16, xmin=0, xmax=ax.get_xlim()[1], color="r", linewidth=0.8)

violations_values = line_data_scenario[line_data_scenario > A_MAX]

#%%

# mixture_irradiance = (0.2, 0.2, 0.6)
mixture_irradiance = (0.0, 1.0, 0.0)
max_current = np.zeros((11, 11))
percentages_pv = np.linspace(0, 1, 11).round(1)
percentages_load = np.linspace(0, 1, 11).round(1)

for ii, load_step in enumerate(tqdm(percentages_pv)):
    for jj, pv_step in enumerate(percentages_load):
        scenario_idx = mixture_to_scenario[(mixture_irradiance, load_step, pv_step)]
        case_number = scenario_idx
        path_file_parent = Path(r"D:\monte_carlo_solutions_AWS")
        file_name = f"voltage_dict_case_{case_number}_scenarios_500.pkl"

        with open(path_file_parent / file_name, "rb") as pickle_file:
            temp_dict = pickle.load(pickle_file)

        line_current_amp = []
        line_angle_deg = []
        voltage_diff = []
        for scenario in range(500):
            line_current_scenario = []
            # line_angle_scenario = []
            # voltage_diff_scenario = []
            for time_step in range(48):
                vs = temp_dict[(scenario, time_step)]["v"][from_node - 2]
                vr = temp_dict[(scenario, time_step)]["v"][to_node - 2]
                line_current_scenario.append(abs((vs - vr) / z_imp_pu) * i_base)
                # line_angle_scenario.append(np.rad2deg(
                #     np.angle((vs - vr))
                # )
                # )
                # voltage_diff_scenario.append((vs - vr))
            line_current_amp.append(line_current_scenario)
            # line_angle_deg.append(line_angle_scenario)
            # voltage_diff.append(voltage_diff_scenario)

        line_data_scenario = np.array(line_current_amp)
        q_95 = np.nanquantile(line_data_scenario, q=0.95, axis=0)

        max_current[ii, jj] = q_95.max()

#%%
X, Y = np.meshgrid(percentages_pv, percentages_load)
# Y = np.flip(Y)
fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'}, figsize=(8, 8))
ax.plot_wireframe(X, Y, max_current.T, color='r', linewidth=0.4)


#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.pcolor(percentages_pv, percentages_load, max_current.T, shading="auto")
cs1 = ax.contour(percentages_pv, percentages_load, max_current.T, colors="k", levels=[A_MAX], linewidths=1.5,
                       linestyles="solid")
cs2 = ax.contour(percentages_pv, percentages_load, max_current.T, colors="k", levels=[A_MAX16], linewidths=1.5,
                       linestyles="dashed")
ax.clabel(cs1, inline=True, fontsize=10, colors='k')
ax.clabel(cs2, inline=True, fontsize=10, colors='k')
ax.set_title(f"clody")
ax.set_xlabel("Load growth")
ax.set_ylabel("PV growth")



