"""
Loads in memory the case combinations that is specifically requested.
This avoids the overloading in RAM of unnecessary cases that will no be plotted/used.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

QUANTILE_DICT = {
                 "05": {"max_tag": "max_q_05",
                        "min_tag": "min_q_05",
                        "quantile": 0.05},
                 "10": {"max_tag": "max_q_10",
                        "min_tag": "min_q_10",
                        "quantile": 0.10},
                 "25": {"max_tag": "max_q_25",
                        "min_tag": "min_q_25",
                        "quantile": 0.25},
                 "50": {"max_tag": "max_q_50",
                        "min_tag": "min_q_50",
                        "quantile": 0.50},
                 "75": {"max_tag": "max_q_75",
                        "min_tag": "min_q_75",
                        "quantile": 0.75},
                 "90": {"max_tag": "max_q_90",
                        "min_tag": "min_q_90",
                        "quantile": 0.90},
                 "95": {"max_tag": "max_q_95",
                        "min_tag": "min_q_95",
                        "quantile": 0.95}
                }

def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator

def max_min_quantiles(voltage_solutions, *, n_scenarios):
    # %% Create Tensor for the case
    volt_mag_matrix_grid = np.zeros((33, n_scenarios, 48))
    max_volt_grid = np.zeros((n_scenarios, 48))
    min_volt_grid = np.zeros((n_scenarios, 48))

    for ii in range(33):  # Number of nodes
        for jj in range(n_scenarios):
            for kk in range(48):  # Time steps
                volt_mag_matrix_grid[ii, jj, kk] = voltage_solutions[(jj, kk)]["v_mag"][ii]
                max_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_max"]
                min_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_min"]

    case_processed = {}
    case_processed["tensor_voltage"] = volt_mag_matrix_grid.astype(np.float32).copy()

    case_processed["max_q_05"] = np.nanquantile(max_volt_grid, q=0.05, axis=0).astype(np.float32)
    case_processed["max_q_10"] = np.nanquantile(max_volt_grid, q=0.10, axis=0).astype(np.float32)
    case_processed["max_q_25"] = np.nanquantile(max_volt_grid, q=0.25, axis=0).astype(np.float32)
    case_processed["max_q_50"] = np.nanquantile(max_volt_grid, q=0.50, axis=0).astype(np.float32)
    case_processed["max_q_75"] = np.nanquantile(max_volt_grid, q=0.75, axis=0).astype(np.float32)
    case_processed["max_q_90"] = np.nanquantile(max_volt_grid, q=0.90, axis=0).astype(np.float32)
    case_processed["max_q_95"] = np.nanquantile(max_volt_grid, q=0.95, axis=0).astype(np.float32)

    case_processed["min_q_05"] = np.nanquantile(min_volt_grid, q=0.05, axis=0).astype(np.float32)
    case_processed["min_q_10"] = np.nanquantile(min_volt_grid, q=0.90, axis=0).astype(np.float32)
    case_processed["min_q_25"] = np.nanquantile(min_volt_grid, q=0.25, axis=0).astype(np.float32)
    case_processed["min_q_50"] = np.nanquantile(min_volt_grid, q=0.50, axis=0).astype(np.float32)
    case_processed["min_q_75"] = np.nanquantile(min_volt_grid, q=0.75, axis=0).astype(np.float32)
    case_processed["min_q_90"] = np.nanquantile(min_volt_grid, q=0.90, axis=0).astype(np.float32)
    case_processed["min_q_95"] = np.nanquantile(min_volt_grid, q=0.95, axis=0).astype(np.float32)

    return case_processed

def process_file(case_number):
    print(f"Case number: {case_number}")
    path_file_parent = Path(r"D:\monte_carlo_solutions_AWS")
    # path_file_parent = Path(r"C:\Users\20175334\Documents\PycharmProjects\monte_carlo_plan\HPC\solutions\AWS")
    file_name = f"voltage_dict_case_{case_number}_scenarios_500.pkl"

    with open(path_file_parent / file_name, "rb") as pickle_file:
        temp_dict = pickle.load(pickle_file)
        solution_case = max_min_quantiles(temp_dict, n_scenarios=500)

    return solution_case

def process_cases(cases_combinations: list, list_requested_cases: list):
    """ Process only the selected cases from all the possible cases """
    case_number_list = [cases_combinations.index(index_value) for index_value in list_requested_cases]

    solutions_dict = {}
    total_cases = len(case_number_list)
    for ii, (tuple_case, case_index) in enumerate(zip(list_of_tuple_cases, case_number_list)):
        print(f"iteration: {ii} of {total_cases}")
        solutions_dict[tuple_case] = process_file(case_index)

    path_file_parent = Path(r"C:\Users\20175334\Documents\PycharmProjects\monte_carlo_plan\HPC\solutions")
    file_name_solutions_dictionary = "solutions_dictionary_AWS.pkl"

    with open(path_file_parent / file_name_solutions_dictionary, "wb") as pickle_file:
        pickle.dump(solutions_dict, pickle_file)


def max_min_quantiles_individual(voltage_solutions, quantile: str, *, n_scenarios, only_tensor_voltage=False):
    # %% Create Tensor for the case
    assert quantile in set(QUANTILE_DICT.keys()), "Wrong quantile selected"

    volt_mag_matrix_grid = np.zeros((33, n_scenarios, 48))
    max_volt_grid = np.zeros((n_scenarios, 48)).astype(np.float32)
    min_volt_grid = np.zeros((n_scenarios, 48)).astype(np.float32)

    for ii in range(33):  # Number of nodes
        for jj in range(n_scenarios):
            for kk in range(48):  # Time steps
                volt_mag_matrix_grid[ii, jj, kk] = voltage_solutions[(jj, kk)]["v_mag"][ii].astype(np.float32)
                max_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_max"].astype(np.float32)
                min_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_min"].astype(np.float32)

    case_processed = {}

    if only_tensor_voltage:
        case_processed["tensor_voltage"] = volt_mag_matrix_grid.astype(np.float32).copy()
    else:
        case_processed[QUANTILE_DICT[quantile]["max_tag"]] = np.nanquantile(max_volt_grid, q=QUANTILE_DICT[quantile]["quantile"], axis=0).astype(np.float32)
        case_processed[QUANTILE_DICT[quantile]["min_tag"]] = np.nanquantile(min_volt_grid, q=QUANTILE_DICT[quantile]["quantile"], axis=0).astype(np.float32)

    return case_processed

def process_quantiles(case_number: int, quantile: str):
    print(f"Case number: {case_number}")
    path_file_parent = Path(r"D:\monte_carlo_solutions_AWS")
    # path_file_parent = Path(r"C:\Users\20175334\Documents\PycharmProjects\monte_carlo_plan\HPC\solutions\AWS")
    file_name = f"voltage_dict_case_{case_number}_scenarios_500.pkl"

    with open(path_file_parent / file_name, "rb") as pickle_file:
        temp_dict = pickle.load(pickle_file)
        solution_case = max_min_quantiles_individual(temp_dict, quantile=quantile, n_scenarios=500)

    return solution_case

def generate_quantile_files(cases_combinations: list, quantile: str):
    assert quantile in set(QUANTILE_DICT.keys()), "Wrong quantile selected"
    print(f"============ Quantile: {quantile} ================")

    solutions_dict = {}
    total_cases = len(cases_combinations)
    # quantile = "05"
    for ii, tuple_case in tqdm(enumerate(cases_combinations)):
        print(f"iteration: {ii + 1} of {total_cases}")
        solutions_dict[tuple_case] = process_quantiles(ii, quantile=quantile)

    path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles")
    file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_{quantile}.pkl"

    with open(path_file_parent / file_name_solutions_dictionary, "wb") as pickle_file:
        pickle.dump(solutions_dict, pickle_file)

if __name__ == "__main__":
    # file_name_scenario_generator_model = "../../models/scenario_generator_model_new.pkl"
    file_name_scenario_generator_model = "../../models/scenario_generator_model_new_AWS.pkl"
    scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
    cases_combinations = scenario_generator.cases_combinations

    #%% Check some combinations
    all_mixtures = []
    for mixture_, load_, pv_ in cases_combinations:
        print(mixture_)
        all_mixtures.append(mixture_)



    # # -----------------------------
    # # Requested scenarios list:
    # mixture_cases = [(0.0, 0.0, 1.0),
    #                  (0.2, 0.0, 0.8),
    #                  (0.4, 0.0, 0.6),
    #                  (0.6, 0.0, 0.4),
    #                  (0.8, 0.0, 0.2),
    #                  (0.2, 0.6, 0.2),  # Normal case (Original)
    #                  (0.0, 1.0, 0.0)]
    # x = scenario_generator.percentages_pv_growth
    # y = scenario_generator.percentages_load_growth
    #
    # list_of_tuple_cases = []
    # for  mixture_case in mixture_cases:
    #     for i, pv in enumerate(x):
    #         for j, load  in enumerate(y):
    #             list_of_tuple_cases.append((mixture_case, load, pv))
    # # -----------------------------

    # case_number_list = [cases_combinations.index(index_value) for index_value in list_of_tuple_cases]


    # # Process the combinations of the irradiance scenarios for all loads and pv generation: WARNING: TAKE TOO LONG!
    # process_cases(cases_combinations, list_of_tuple_cases)

    # # Process the quantiles of all the data
    # generate_quantile_files(cases_combinations, "05")

    # TODO: If you do it all at once, memory will overflow and stop working
    list_quantiles = [
                      # (cases_combinations, "05"),
                      (cases_combinations, "10"),
                      # (cases_combinations, "25")
                      # (cases_combinations, "50"),
                      # (cases_combinations, "75"),
                      # (cases_combinations, "90"),
                      # (cases_combinations, "95")
    ]

    print(f"Total cpus to use: {mp.cpu_count()}")

    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        pool.starmap(generate_quantile_files, list_quantiles)

    # for quant in ["75", "90", "95"]:
    #     print(f"============ Quantile: {quant} ================")
    #     generate_quantile_files(cases_combinations, quant)