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
mapper = dict(zip(set(cases), range(len(set(cases)))))

mixture_prob, load_growth, pv_growth = cases

#%%
growth_curves_gwh = []
for ii in range(3):
    growth_curves_gwh.append(list(scenario_generator.mapper_load_growth[ii].values()))
growth_curves_gwh = np.array(growth_curves_gwh)

#%%

fig, ax = plt.subplots(1,1, figsize=(5,5))
x_ = list(scenario_generator.mapper_load_growth[0].keys())
for ii, label in enumerate(["Cluster 0", "Cluster 1", "Cluster 2"]):
    ax.plot(x_, growth_curves_gwh[ii,:], label=label)
ax_ = ax.twinx()
ax_.plot(x_, growth_curves_gwh.sum(axis=0), color="red", linewidth=3.0)
ax.set_ylim(0, 2.0)
ax_.set_ylim(0,4.0)
ax.set_ylabel("Annual Energy value GWh/Year")
ax.legend(loc="upper left", fontsize="x-small")