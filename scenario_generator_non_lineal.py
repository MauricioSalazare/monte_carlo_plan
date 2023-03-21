"""
Create the load and irradiance scenarios at the same time.


"""


import pickle
import pandas as pd
from core.scenarios import ScenarioGenerator
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


file_name_load_models = "models/copula_model_load.pkl"
with open(file_name_load_models, "rb") as pickle_file:
    copula_load = pickle.load(pickle_file)

file_name_irradiance_models = "models/copula_model_irradiance.pkl"
with open(file_name_irradiance_models, "rb") as pickle_file:
    mixture_model_irradiance = pickle.load(pickle_file)

file_name_network = r"data/processed_data/network_data/Nodes_34.csv"
grid = pd.read_csv(file_name_network)
grid_info = grid.iloc[1:, :]  # Drop the slack node

#======================================================================================================================
# CREATE MODEL FOR THE NON LINEAL CASE
#======================================================================================================================

#%% Spline function
# Control points for the spline
points = np.array([(0.0, 0.0),
                   (0.3, 0.4),
                   (0.5, 0.5),
                   (0.7, 0.5),
                   (1.0, 1.0)])

x_ = np.linspace(0.0, 1.0, 100)
x = np.linspace(0.0, 1.0, 11)
f = UnivariateSpline(points[:,0], points[:, 1],k=3)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.scatter(points[:,0], points[:,1], s=20, zorder=3)
ax.scatter(x, f(x), s=20)
ax.plot(x_, f(x_), linewidth=0.8, color="C0")
ax.grid("both")
ax.set_title("Normalized non-linear growth")

#%% Square root
f = lambda x: np.sqrt(x)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.scatter(x, f(x), s=20)
ax.plot(x_, f(x_), linewidth=0.8, color="C0")
ax.grid("both")
ax.set_title("Normalized non-linear growth")

#%%
scenario_generator_non_lineal = ScenarioGenerator(copula_load=copula_load,
                                                  mixture_model_irradiance=mixture_model_irradiance,
                                                  grid_info=grid_info,
                                                  n_levels_load_growth=10,
                                                  n_levels_pv_growth=10,
                                                  n_levels_mixtures=10,
                                                  lineal_growth=False,
                                                  load_growth_function=f)
sgn = scenario_generator_non_lineal

# Example
case_dict = sgn.create_case_scenarios(case=((0.0,0.0,1.0),1.0,0.0),
                                      n_scenarios=100)
active, reactive = sgn.create_case_scenarios_tensorpoweflow(case=((0.0,0.0,1.0),1.0,0.0),
                                                            n_scenarios=100)


#%%
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.plot(active[0,:,14])
