# import os
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

from core.power_flow import Grid
from time import perf_counter
from tqdm import tqdm
import numpy as np
import pickle
import argparse
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['C:\\Users\\20175334\\Documents\\PycharmProjects\\monte_carlo_plan', 'C:/Users/20175334/Documents/PycharmProjects/monte_carlo_plan'])

def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator

def solve_case(case_dictionary, ieee_grid, n_scenarios, time_steps):
    voltage_solutions = {}
    start = perf_counter()
    for ii in tqdm(range(n_scenarios)):
        for jj in range(time_steps):
            active_power = case_dictionary[(ii, jj)]["P"]
            reactive_power = case_dictionary[(ii, jj)]["Q"]

            voltage = ieee_grid.run_pf(active_power=active_power,
                                       reactive_power=reactive_power)
            voltage_magnitude = np.abs(voltage).astype(np.float32)

            voltage_solutions[(ii, jj)] = {"v": voltage.astype(np.csingle),
                                           "v_mag": voltage_magnitude.astype(np.float32),
                                           "v_max": voltage_magnitude.max(),
                                           "v_min": voltage_magnitude.min()}
    end = perf_counter()
    print(f"Total time case: {(end - start).__round__(2)} sec")

    return voltage_solutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case', required=False, type=int, default=10,
                        help="Case number to solve")
    parser.add_argument('-n', '--n_scenarios', required=False, type=int, default=500,
                        help='Number of scenarios to simulate.')
    args, unknown = parser.parse_known_args()

    print(f"Processing case: {args.case}")

    file_name_scenario_generator_model = "models/scenario_generator_model_new_AWS.pkl"
    node_file_path = r"data/processed_data/network_data/Nodes_34.csv"
    lines_file_path = r"data/processed_data/network_data/Lines_34.csv"

    np.random.seed()  # Critical for linux
    ieee_grid = Grid(node_file_path=node_file_path,
                     lines_file_path=lines_file_path)

    scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
    cases = scenario_generator.cases_combinations

    assert args.case < len(cases), "Input case number is higher than the total expected cases."

    case_dictionary = scenario_generator.create_case_scenarios(case=cases[args.case],
                                                               n_scenarios=args.n_scenarios)

    voltage_solutions = solve_case(case_dictionary=case_dictionary,
                                   ieee_grid=ieee_grid,
                                   n_scenarios=args.n_scenarios,
                                   time_steps=48)

    # Save solution dictionary in pickle file:
    file_name_load_models = f"HPC/solutions/AWS/voltage_dict_case_{args.case}_scenarios_{args.n_scenarios}.pkl"
    with open(file_name_load_models, "wb") as pickle_file:
        pickle.dump(voltage_solutions, pickle_file)
    #
