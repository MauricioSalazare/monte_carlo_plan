import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
parameters = [2.5, 5.0, 7.5, 10.0, 12.5]
myparam = parameters[idx]

print(f"---USING PARAMETER: {myparam}", flush=True)