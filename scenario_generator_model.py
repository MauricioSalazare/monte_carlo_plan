import pickle
import pandas as pd
from core.scenarios import ScenarioGenerator

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
case_dictionary = scenario_generator.create_case_scenarios(case=cases[350],
                                                           n_scenarios=500)

file_name_scenario_generator_model = "models/scenario_generator_model_new_AWS.pkl"
with open(file_name_scenario_generator_model, "wb") as pickle_file:
    pickle.dump(scenario_generator, pickle_file)

