"""
Create the load and irradiance scenarios at the same time.


"""


import pickle
import pandas as pd
from core.scenarios import ScenarioGenerator
import numpy as np
import matplotlib.pyplot as plt

file_name_load_models = "models/copula_model_load.pkl"
with open(file_name_load_models, "rb") as pickle_file:
    copula_load = pickle.load(pickle_file)

file_name_irradiance_models = "models/copula_model_irradiance.pkl"
with open(file_name_irradiance_models, "rb") as pickle_file:
    mixture_model_irradiance = pickle.load(pickle_file)

file_name_network = r"data/processed_data/network_data/Nodes_34.csv"
grid = pd.read_csv(file_name_network)
grid_info = grid.iloc[1:, :]  # Drop the slack node

scenario_generator = ScenarioGenerator(copula_load=copula_load,
                                       mixture_model_irradiance=mixture_model_irradiance,
                                       grid_info=grid_info,
                                       n_levels_load_growth=10,
                                       n_levels_pv_growth=10,
                                       n_levels_mixtures=10)

cases = scenario_generator.cases_combinations
# mapper = dict(zip(set(cases), range(len(set(cases)))))  # This operation does not preserve order

#%% Worst loading scenario
n_scenarios = 100
# TODO: The scenario generator is hardcoded for 48 time steps, 30-min resolution daily.

# Worst case scenario for loading
## pi_mixture = (0.0, 0.0, 1.0)  # Mixture of (cloudy, sunny and dark days)
# load_growth, pv_growth = 1.0, 0.0
# case = (pi_mixture,load_growth, pv_growth)

# Worst case scenario for PV generation
pi_mixture = (0.0, 1.0, 0.0)  # Mixture of (cloudy, sunny and dark days)
load_growth, pv_growth = 0.0, 1.0
case = (pi_mixture, load_growth, pv_growth)


np.random.seed(123)
case_dictionary = scenario_generator.create_case_scenarios(case=case,
                                                           n_scenarios=n_scenarios)
np.random.seed(123)

grid_loading = np.zeros((n_scenarios, 48))
for ii in range(n_scenarios): # Scenario
    for jj in range(48):  # Timestep
        # Summ all the power in the grid
        grid_loading[ii, jj] = case_dictionary[(ii, jj)]["P"].sum()

print(f"Worst Avg. loading: {grid_loading.mean(axis=0).max()}")

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(grid_loading.mean(axis=0), color="b", linewidth=0.5)
ax.set_ylim((-8000, 6200))
ax.set_title("Total Grid loading")
ax.set_ylabel("kW")
ax.set_xlabel("Time step [30 min]")
ax.grid()

#%%
node_ = 6
# WARNING: Grid info has an offset of 2 (start node count from 1 and also the first node is slack)
# So, -1 for the non-starting at 0, and -1 for elimination of slack node. That's -2.
# WARNING: Also notice that the index of grid_info starts from 1, so, USE .iloc and NOT .loc, as the last one uses
# the index instead of the positioning.

idx = node_ - 2
grid_loading = np.zeros((n_scenarios, 48))
for ii in range(n_scenarios): # Scenario
    for jj in range(48):  # Timestep
        grid_loading[ii, jj] = case_dictionary[(ii, jj)]["P"][idx]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(scenario_generator._last_active_power_stack[idx,...].mean(axis=0),
        color="b", linewidth=0.5, label="without pv")
ax.plot(grid_loading.mean(axis=0), color="r", linewidth=0.5, label="with pv" )
ax.set_title(f"Load node: {scenario_generator.grid_info.iloc[idx]['NODES']}\n"
             f"PV Installed: {scenario_generator.grid_info.iloc[idx]['kwp']}")
ax.legend()
ax.set_ylabel("kW")
ax.set_xlabel("Time step [30 min]")
# ax.set_ylim((0, 150))
ax.grid()

print(scenario_generator.grid_info)

# file_name_scenario_generator_model = "models/scenario_generator_model_new_AWS.pkl"
# with open(file_name_scenario_generator_model, "wb") as pickle_file:
#     pickle.dump(scenario_generator, pickle_file)

