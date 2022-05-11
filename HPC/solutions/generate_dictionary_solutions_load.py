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

QUANTILE_DICT = {"00": {"max_tag": "max_q_00",
                        "min_tag": "min_q_00",
                        "quantile": 0.0},
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
                        "quantile": 0.95},
                 "100": {"max_tag": "max_q_100",
                        "min_tag": "min_q_100",
                        "quantile": 1.0}
                }

def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator

def load_grid_parameters(lines_file_path: str) -> dict:
    s_base = 1000
    v_base = 11
    z_base = (v_base ** 2 * 1000) / s_base
    i_base = s_base / (np.sqrt(3) * v_base)

    branch_info = pd.read_csv(lines_file_path)

    # Get info first line (connected to transformer)
    from_node = branch_info.loc[1]["FROM"].astype(int)
    to_node = branch_info.loc[1]["TO"].astype(int)
    z_imp_pu = (branch_info.loc[1]["R"] + 1j * branch_info.loc[1]["X"]) / z_base

    grid_parameters = {}
    grid_parameters["from_node"] = from_node
    grid_parameters["to_node"] = to_node
    grid_parameters["z_imp_pu"] = z_imp_pu
    grid_parameters["i_base"] = i_base

    return grid_parameters


def max_min_quantiles_loading(voltage_solutions, grid_param: dict, quantile: str, *, n_scenarios: int):
    line_current_amp = []
    for scenario in range(n_scenarios):
        line_current_scenario = []
        for time_step in range(48):
            vs = voltage_solutions[(scenario, time_step)]["v"][grid_param["from_node"] - 2] # Line connected to Transformer
            vr = voltage_solutions[(scenario, time_step)]["v"][grid_param["to_node"] - 2] # End of line
            line_current_scenario.append(abs((vs - vr) / grid_param["z_imp_pu"]) * grid_param["i_base"])
        line_current_amp.append(line_current_scenario)
    line_data_scenario = np.array(line_current_amp)
    # quantile_profile = np.nanquantile(line_data_scenario, q=QUANTILE_DICT[quantile]["quantile"], axis=0)

    case_processed = {}
    case_processed[QUANTILE_DICT[quantile]["max_tag"]] = np.nanquantile(line_data_scenario,
                                                                        q=QUANTILE_DICT[quantile]["quantile"],
                                                                        axis=0).astype(np.float32)
    return case_processed


def process_quantiles_loading(case_number: int, grid_param: dict,  quantile: str):
    print(f"Case number: {case_number}")
    path_file_parent = Path(r"D:\monte_carlo_solutions_AWS")
    # path_file_parent = Path(r"C:\Users\20175334\Documents\PycharmProjects\monte_carlo_plan\HPC\solutions\AWS")
    file_name = f"voltage_dict_case_{case_number}_scenarios_500.pkl"

    with open(path_file_parent / file_name, "rb") as pickle_file:
        temp_dict = pickle.load(pickle_file)

    solution_case = max_min_quantiles_loading(temp_dict, grid_param, quantile=quantile, n_scenarios=500)

    return solution_case


if __name__ == "__main__":
    QUANTILE = "10"
    file_name_scenario_generator_model = "../../models/scenario_generator_model_new_AWS.pkl"
    scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
    lines_file_path = r"../../data/processed_data/network_data/Lines_34.csv"
    grid_param = load_grid_parameters(lines_file_path)
    cases_combinations = scenario_generator.cases_combinations
    total_cases = len(cases_combinations)

    # Parallel
    # dict(zip(cases_combinations, solution_parallel_shit))
    case_ = list(range(total_cases))
    grid_param_ = [grid_param] * total_cases
    quant_ = [QUANTILE] * total_cases

    star_input = list(tuple(zip(case_, grid_param_, quant_)))

    print(f"Total cpus to use: {mp.cpu_count()}")
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        solution_parallel_shit = pool.starmap(process_quantiles_loading, star_input)

    solutions_dict = dict(zip(cases_combinations, solution_parallel_shit))

    path_file_parent = Path(r"D:\monte_carlo_solutions_AWS_quantiles_loading")
    file_name_solutions_dictionary = f"solutions_dictionary_AWS_quantile_load_{QUANTILE}.pkl"

    with open(path_file_parent / file_name_solutions_dictionary, "wb") as pickle_file:
        pickle.dump(solutions_dict, pickle_file)

