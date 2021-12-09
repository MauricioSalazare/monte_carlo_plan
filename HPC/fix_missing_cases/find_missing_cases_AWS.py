import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import os

# path_file_parent = Path(r"F:\monte_carlo_solutions")
path_file_parent = Path(r"../solutions/AWS")

assert path_file_parent.is_dir(), "Incorrect folder name"

files = os.listdir(path_file_parent)
cases = [file.split("case_")[1] for file in files]
numbers = [int(case.split("_scenarios")[0]) for case in cases]

desired = set(range(7502))
missing = desired - set(numbers)


N_SCENARIOS = 500
for ii, case in enumerate(sorted(missing)):
    code_lines = ["#!/bin/bash",
                  "# SBATCH --job-name=multithread",
                  "# SBATCH --nodes=1",
                  "# SBATCH --ntasks=1",
                  "# SBATCH --cpus-per-task=1",
                  "# SBATCH --time=0:15:00",
                  "# SBATCH --out=python_job-%j.log",
                  "# SBATCH --error=slurm-%j.err",
                  "# SBATCH --output=slurm-%j.out",
                  "# SBATCH --mail-type=begin",
                  "# SBATCH --mail-type=end",
                  "# SBATCH --mail-user=e.m.salazar.duque@tue.nl",
                  "",
                  """echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID""",
                  """echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID""",
                  """echo "My SLURM_CPUS_PER_TASK is $SLURM_CPUS_PER_TASK""",
                  """echo "Executing on the machine:" $(hostname)""",
                  "",
                  "export OMP_NUM_THREADS =$SLURM_CPUS_PER_TASK",
                  "export OPENBLAS_NUM_THREADS =$SLURM_CPUS_PER_TASK",
                  "export MKL_NUM_THREADS =$SLURM_CPUS_PER_TASK",
                  "export VECLIB_MAXIMUM_THREADS =$SLURM_CPUS_PER_TASK",
                  "export NUMEXPR_NUM_THREADS =$SLURM_CPUS_PER_TASK",
                  "",
                  "spack load miniconda3",
                  "source activate montecarlo",
                  "",
                  f"python main.py -c {case} -n {N_SCENARIOS}",
                  "conda deactivate"]

    with open(f'slurm_files_remain/file{ii}.run', 'w', newline='\n') as f:
        f.write('\n'.join(code_lines))
