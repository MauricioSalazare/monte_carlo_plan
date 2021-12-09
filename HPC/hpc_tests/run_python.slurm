#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
#SBATCH --out=python_job-%j.log
#SBATCH --partition=elec.default.q
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out

module load anaconda/2020.07-pth38
source activate dask_test

# run the python script, given as command line argument
python $1
source deactivate