#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
#SBATCH --out=python_job-%j.log
#SBATCH --partition=elec.default.q
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --array=0-4

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module load anaconda/2020.07-pth38
source activate dask_test

python my_script.py
source deactivate