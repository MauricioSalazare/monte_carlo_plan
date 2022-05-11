import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List

def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator


def get_solutions_dictionary_load(path_file_solutions, quantiles: List[str]) -> dict:
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
        file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_load_{quant_name}.pkl"
        with open(path_file_solutions / file_name_solutions_dictionary, "rb") as pickle_file:
            file_solutions_dict = pickle.load(pickle_file)
            solutions_dict[quant_name] = file_solutions_dict  # This is critical to group all mixtures in the same quant

    return solutions_dict


#%%
# ((cloudy, sunny, dark) Load, PV)

A_MAX = 230
A_MAX16 = np.round(A_MAX * 1.16, 3)

critical_quantiles = ["95"]
path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles_loading")
solutions_dictx_new = get_solutions_dictionary_load(path_file_parent, quantiles=critical_quantiles)

mixture_irradiance = (0.0, 1.0, 0.0)
max_current = np.zeros((11, 11))
percentages_pv = np.linspace(0, 1, 11).round(1)
percentages_load = np.linspace(0, 1, 11).round(1)

for ii, load_step in enumerate(tqdm(percentages_pv)):
    for jj, pv_step in enumerate(percentages_load):
        look_mixture = (mixture_irradiance, load_step, pv_step)
        q_95 = solutions_dictx_new["95"][look_mixture]["max_q_95"]
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




