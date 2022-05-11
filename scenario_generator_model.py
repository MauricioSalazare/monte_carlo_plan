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
mapper = dict(zip(set(cases), range(len(set(cases)))))

#%% Worst loading scenario
case = ((0.0, 0.0, 1.0), 1.0, 0.0)
worst_case = mapper[case]
np.random.seed(123)
case_dictionary = scenario_generator.create_case_scenarios(case=cases[worst_case],
                                                           n_scenarios=500)
np.random.seed(123)

grid_loading = np.zeros((500, 48))
for ii in range(500): # Scenario
    for jj in range(48):  # Timestep
        grid_loading[ii, jj] = case_dictionary[(ii, jj)]["P"].sum()

print(f"Worst Avg. loading: {grid_loading.mean(axis=0).max()}")

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(grid_loading.mean(axis=0), color="b", linewidth=0.5)
ax.set_ylim((-3000, 4200))
ax.set_title("Total Grid loading")
ax.set_ylabel("kW")
ax.set_xlabel("Time step [30 min]")
ax.grid()

#%%
node = 7
grid_loading = np.zeros((500, 48))
for ii in range(500): # Scenario
    for jj in range(48):  # Timestep
        grid_loading[ii, jj] = case_dictionary[(ii, jj)]["P"][node]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(scenario_generator._last_active_power_stack[node,...].mean(axis=0), color="b", linewidth=0.5, label="without pv")
ax.plot(grid_loading.mean(axis=0), color="r", linewidth=0.5, label="with pv" )
ax.set_title(f"Load node: {node}")
ax.legend()
ax.set_ylabel("kW")
ax.set_xlabel("Time step [30 min]")
# ax.set_ylim((0, 150))
ax.grid()



# file_name_scenario_generator_model = "models/scenario_generator_model_new_AWS.pkl"
# with open(file_name_scenario_generator_model, "wb") as pickle_file:
#     pickle.dump(scenario_generator, pickle_file)

