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

file_name_scenario_generator_model = "../../models/scenario_generator_model_new_AWS.pkl"
# file_name_solutions = "solutions.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
# solutions = load_solutions(file_name_solutions)
cases_combinations = scenario_generator.cases_combinations

# assert len(cases_combinations) == len(solutions), "The labeling could be wrong"

# # Assign the case combination to the solutions.
# solutions_dict = {}
# for case_, solution in zip(cases_combinations, solutions):  # The order of the combinations is the same as the solutions
#     solutions_dict[case_] = solution

x = scenario_generator.percentages_pv_growth
y = scenario_generator.percentages_load_growth



path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
quant_name = "90"
file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quant_name}.pkl"
with open(path_file_parent / file_name_solutions_dictionary, "rb") as pickle_file:
    solutions_dict = pickle.load(pickle_file)

#%%
load = 0.2
pv = 0.4

solutions_ternary = {}
for solution_key in solutions_dict:
    mixture_, load_, pv_ = solution_key
    if (load == load_) and (pv == pv_):
        solutions_ternary[mixture_] = solutions_dict[solution_key]["max_q_90"].max()


import plotly.figure_factory as ff
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

# Al = np.array([0. , 0. , 0., 0., 1./3, 1./3, 1./3, 2./3, 2./3, 1.])
# Cu = np.array([0., 1./3, 2./3, 1., 0., 1./3, 2./3, 0., 1./3, 0.])
# Y = 1 - Al - Cu

data_ = np.array(list(solutions_ternary.keys()))

Al = data_[:, 0].flatten()  # Cloudy
Cu = data_[:, 1].flatten()  # Sunny
Y = data_[:, 2].flatten()  # Dark

# synthetic data for mixing enthalpy
# See https://pycalphad.org/docs/latest/examples/TernaryExamples.html
# enthalpy = 2.e6 * (Al - 0.01) * Cu * (Al - 0.52) * (Cu - 0.48) * (Y - 1)**2 - 5000
enthalpy = np.array(list(solutions_ternary.values()))
# enthalpy = ((enthalpy-1.05)/1.05)*100
fig = ff.create_ternary_contour(np.array([Al, Y, Cu]), enthalpy,
                                pole_labels=['Cloudy', 'Dark', 'Sunny'],
                                # interp_mode='ilr',
                                interp_mode='cartesian',
                                ncontours=40,
                                colorscale='Viridis',
                                # coloring='lines',
                                showscale=True,
                                title=f'Max. voltage ternary weather. Load: {load}, PV: {pv}',
                                showmarkers=True)
fig.show()