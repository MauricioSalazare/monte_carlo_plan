#!/bin/bash
#SBATCH --job-name=multithread
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --out=python_job-%j.log
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=e.m.salazar.duque@tue.nl
#SBATCH --array=0-5

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID"
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "My SLURM_CPUS_PER_TASK is $SLURM_CPUS_PER_TASK"
echo "Executing on the machine:" $(hostname)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

spack load miniconda3
conda activate montecarlo

python main.py -c $SLURM_ARRAY_TASK_ID
conda deactivate