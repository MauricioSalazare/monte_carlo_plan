import os
os.environ["OMP_NUM_THREADS"] = "64" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "64" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "64" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "64" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "64" # export NUMEXPR_NUM_THREADS=6


import numpy as np
import time


start_time = time.time()

a = np.random.randn(2000, 20000)
b = np.random.randn(20000, 2000)
ran_time = time.time() - start_time
print("time to complete random matrix generation was %s seconds" % ran_time)


np.dot(a, b) # this line is multithreaded
print("time to complete dot was %s seconds" % (time.time() - start_time - ran_time))