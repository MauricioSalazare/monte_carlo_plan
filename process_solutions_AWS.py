import numpy as np
import pickle
from pathlib import Path
import multiprocessing as mp

def process_file(case_number):
    print(f"Case number: {case_number}")
    # path_file_parent = Path(r"F:\monte_carlo_solutions")
    path_file_parent = Path(r"HPC/solutions/AWS")
    file_name = f"voltage_dict_case_{case_number}_scenarios_500.pkl"

    with open(path_file_parent / file_name, "rb") as pickle_file:
        temp_dict = pickle.load(pickle_file)
        solution_case = max_min_quantiles(temp_dict, n_scenarios=500)

    return solution_case

def max_min_quantiles(voltage_solutions, *, n_scenarios):
    # %% Create Tensor for the case
    volt_mag_matrix_grid = np.zeros((33, n_scenarios, 48))
    max_volt_grid = np.zeros((n_scenarios, 48))
    min_volt_grid = np.zeros((n_scenarios, 48))

    for ii in range(33):
        for jj in range(n_scenarios):
            for kk in range(48):
                volt_mag_matrix_grid[ii, jj, kk] = voltage_solutions[(jj, kk)]["v_mag"][ii]
                max_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_max"]
                min_volt_grid[jj, kk] = voltage_solutions[(jj, kk)]["v_min"]

    case_processed = {}
    case_processed["tensor_voltage"] = volt_mag_matrix_grid.astype(np.float32).copy()

    case_processed["max_q_05"] = np.nanquantile(max_volt_grid, q=0.05, axis=0).astype(np.float32)
    case_processed["max_q_10"] = np.nanquantile(max_volt_grid, q=0.10, axis=0).astype(np.float32)
    case_processed["max_q_25"] = np.nanquantile(max_volt_grid, q=0.25, axis=0).astype(np.float32)
    case_processed["ma:x_q_50"] = np.nanquantile(max_volt_grid, q=0.50, axis=0).astype(np.float32)
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


if __name__ == "__main__":
    # solution = process_file(case_number=10)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        solutions = pool.map(process_file, range(7502))

    with open("HPC/solutions/solutions_AWS.pkl", "wb") as pickle_file:
        pickle.dump(solutions, pickle_file)
