import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import os

path_file_parent = Path(r"F:\monte_carlo_solutions")

assert path_file_parent.is_dir(), "Incorrect folder name"

files = os.listdir(path_file_parent)
cases = [file.split("case_")[1] for file in files]
numbers = [int(case.split("_scenarios")[0]) for case in cases]

desired = set(range(756))
missing = desired - set(numbers)


N_SCENARIOS = 500
for ii, case in enumerate(sorted(missing)):
    code_lines = ["#!/bin/bash",
                  "#SBATCH --nodes=1",
                  "#SBATCH --ntasks=1",
                  "#SBATCH --partition=elec.default.q",
                  "#SBATCH --error=slurm-%j.err",
                  "#SBATCH --output=slurm-%j.out",
                  "#SBATCH --mail-type=end",
                  "#SBATCH --mail-user=e.m.salazar.duque@tue.nl",
                  "#SBATCH --time=00:06:00",
                  "",
                  "source /cm/shared/apps/Anaconda/2020.07/pth3.8/etc/profile.d/conda.sh >> log.txt",
                  "conda activate /home/tue/20175334/.conda/envs/montecarlo >> log.txt",
                  f"python main.py -c {case} -n {N_SCENARIOS} >> log.txt",
                  "conda deactivate"]

    with open(f'slurm_files_remain/file{ii}.run', 'w', newline='\n') as f:
        f.write('\n'.join(code_lines))
