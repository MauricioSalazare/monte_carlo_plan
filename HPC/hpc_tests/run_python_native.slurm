#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
#SBATCH --out=python_job-%j.log
#SBATCH --partition=elec.default.q
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out

# run the python script, given as command line argument
python3 $1
