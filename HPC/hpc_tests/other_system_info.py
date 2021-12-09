import multiprocessing as mp
import os


print(f"n_cpus_assigned: {os.environ['SLURM_NTASKS_PER_NODE']}")
print(f"n_cpus_node: {mp.cpu_count()}")