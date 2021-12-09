import pickle

def load_scenarios_model(file_name_scenario_generator_model):
    with open(file_name_scenario_generator_model, "rb") as pickle_file:
        scenario_generator = pickle.load(pickle_file)

    return scenario_generator

file_name_scenario_generator_model = "../models/scenario_generator_model.pkl"
scenario_generator = load_scenarios_model(file_name_scenario_generator_model)
cases = scenario_generator.cases_combinations
N_SCENARIOS = 500

for case in range(len(cases)):
    code_lines = ["#!/bin/bash",
                  "#SBATCH --nodes=1",
                  "#SBATCH --ntasks=1",
                  "#SBATCH --partition=elec.default.q",
                  "#SBATCH --error=slurm-%j.err",
                  "#SBATCH --output=slurm-%j.out",
                  "#SBATCH --time=00:06:00",
                  "",
                  "source /cm/shared/apps/Anaconda/2020.07/pth3.8/etc/profile.d/conda.sh >> log.txt",
                  "conda activate /home/tue/20175334/.conda/envs/montecarlo >> log.txt",
                  f"python main.py -c {case} -n {N_SCENARIOS} >> log.txt",
                  "conda deactivate"]

    with open(f'slurm_files/file{case}.run', 'w', newline='\n') as f:
        f.write('\n'.join(code_lines))



