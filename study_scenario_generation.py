from core.power_flow import Grid
import pickle
from time import perf_counter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: WARNING: If you change something in the grid file, you need to create the scenario generator model again

file_name_scenario_generator_model = "models/scenario_generator_model.pkl"
with open(file_name_scenario_generator_model, "rb") as pickle_file:
    scenario_generator = pickle.load(pickle_file)

node_file_path = r"data/processed_data/network_data/Nodes_34.csv"
lines_file_path = r"data/processed_data/network_data/Lines_34.csv"

ieee_grid = Grid(node_file_path=node_file_path,
                 lines_file_path=lines_file_path)

cases = scenario_generator.cases_combinations

N_SCENARIOS = 100

# case_dictionary = scenario_generator.create_case_scenarios(case=cases[350],
#                                                            n_scenarios=N_SCENARIOS)

# Case: (mixture components of irradiance, % load growth, % pv_growth)
# Mixture irradiance= (cloudy, sunny, dark)

worst_case_sunny_high_pv = ((0.0, 1.0, 0.0), 0.0, 1.0)
worst_case_dark_high_pv = ((0.0, 0.0, 1.0), 0.0, 1.0)
best_case_dark_low_pv = ((0.0, 0.0, 1.0), 0.0, 0.0)
high_load_dark_case = ((0.0, 0.0, 1.0), 1.0, 0.0)
high_load_high_pv_case = ((0.0, 1.0, 0.0), 1.0, 1.0)
normal_case = ((1/3, 1/3, 1/3), 0.4, 0.0)

give_me_the_case = high_load_dark_case

case_dictionary = scenario_generator.create_case_scenarios(case=give_me_the_case,
                                                           n_scenarios=N_SCENARIOS)

#%%
voltage_solutions = {}
start = perf_counter()
for ii in tqdm(range(N_SCENARIOS)):
    for jj in range(48):
        active_power = case_dictionary[(ii, jj)]["P"]
        reactive_power = case_dictionary[(ii, jj)]["Q"]

        voltage = ieee_grid.run_pf(active_power=active_power,
                                   reactive_power=reactive_power)
        voltage_magnitude = np.abs(voltage)

        voltage_solutions[(ii, jj)] = {"v": voltage,
                                       "v_mag": voltage_magnitude,
                                       "v_max": voltage_magnitude.max(),
                                       "v_min": voltage_magnitude.min()}
end = perf_counter()
print(f"Total time one case: {end-start}")

#%% Create Tensor for the case
volt_mag_matrix_grid = np.zeros((33, N_SCENARIOS, 48))
max_volt_grid = np.zeros((N_SCENARIOS, 48))
min_volt_grid = np.zeros((N_SCENARIOS, 48))

for ii in tqdm(range(33)):
    for jj in range(N_SCENARIOS):
        for kk in range(48):
            volt_mag_matrix_grid[ii, jj, kk] = voltage_solutions[(jj, kk)]["v_mag"][ii]
            max_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_max"]
            min_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_min"]

case_processed = {}
case_processed["max_q_05"] = np.nanquantile(max_volt_grid, q=0.05, axis=0)
case_processed["max_q_50"] = np.nanquantile(max_volt_grid, q=0.50, axis=0)
case_processed["max_q_95"] = np.nanquantile(max_volt_grid, q=0.95, axis=0)

case_processed["min_q_05"] = np.nanquantile(min_volt_grid, q=0.05, axis=0)
case_processed["min_q_50"] = np.nanquantile(min_volt_grid, q=0.50, axis=0)
case_processed["min_q_95"] = np.nanquantile(min_volt_grid, q=0.95, axis=0)


#%%
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
ax.plot(max_volt_grid.T, linewidth=0.3, color="gray")
ax.plot(np.nanquantile(max_volt_grid, q=0.05, axis=0), color="red", linestyle="--")
ax.plot(np.nanquantile(max_volt_grid, q=0.50, axis=0), color="red", linestyle="-")
ax.plot(np.nanquantile(max_volt_grid, q=0.95, axis=0), color="red", linestyle="-.")
ax.set_title("Profiles Max. voltages on the grid")
ax.set_ylabel("Voltage [p.u.]")
ax.set_xlabel("Time step [30 min.]")
ax.set_ylim((0.99, 1.1))

#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(min_volt_grid.T, linewidth=0.3, color="gray")
ax.plot(np.nanquantile(min_volt_grid, q=0.05, axis=0), color="red", linestyle="--")
ax.plot(np.nanquantile(min_volt_grid, q=0.50, axis=0), color="red", linestyle="-")
ax.plot(np.nanquantile(min_volt_grid, q=0.95, axis=0), color="red", linestyle="-.")
ax.set_title("Min. voltage distribution")


#%%
from matplotlib import ticker

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.kdeplot(x=volt_mag_matrix_grid.mean(axis=1).flatten(), cumulative=False, ax=ax[0])
sns.kdeplot(x=volt_mag_matrix_grid.mean(axis=1).flatten(), cumulative=True, ax=ax[1])
ax[0].set_title("PDF")
ax[0].set_xlim((0.975, 1.085))
ax[0].yaxis.set_major_locator(ticker.NullLocator())

ax[1].set_title("CDF")
ax[1].set_xlim((0.975, 1.1))
ax[1].set_ylim((0, 1))
ax[1].set_ylabel("Probability")

y_lim = ax[1].get_ylim()
x_lim = ax[1].get_xlim()

all_volts = volt_mag_matrix_grid.mean(axis=1).flatten()
all_volts.sort()
ecdf = all_volts.cumsum() / all_volts.cumsum().max()
volt_quant = all_volts[np.argmin(np.abs(ecdf - 0.95))]
ecdf_quant = ecdf[np.argmin(np.abs(ecdf - 0.95))]

ax[1].hlines(y=ecdf_quant, xmin=x_lim[0], xmax=x_lim[1], color="red")
ax[1].vlines(x=volt_quant, ymin=y_lim[0], ymax=y_lim[1], color="red")

fig.suptitle(f"Volt all grid/ all day (Mean over scenarios) - Volt 0.95 perc. : {volt_quant.__round__(4)}")

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.plot(all_volts, ecdf)
# ax.set_ylim(y_lim)
# ax.set_xlim(x_lim)
# ax.set_title("Manual ECDF")

#%%
# Compute quantiles
scenario_sweep = np.array(list(range(1, N_SCENARIOS + 1)))
q_90 = []
for ii in scenario_sweep:
    q_90.append(np.nanquantile(max_volt_grid[:ii, :], q=0.90, axis=0))
q_90 = np.array(q_90)

#%%
fig, ax = plt.subplots(1, 1, figsize=(5,3))
plt.subplots_adjust(left=0.2, top=0.8, bottom=0.2)
ax.plot(scenario_sweep, q_90[:,23])
ax.plot(scenario_sweep, q_90[:,23].cumsum()/scenario_sweep)
ax.set_title("Max. volt stabilizing plot\n0.9 quantile of time step 23")
ax.set_ylabel("Voltage [p.u.]")
ax.set_xlabel("Number scenarios")
