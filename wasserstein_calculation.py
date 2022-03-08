import ot
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt

cov1 = [[1, 0.8],
        [0.8, 1]]

cov2 = [[1, 0.8],
        [0.8, 1]]

mean1 = [10, 10]
mean2 = [13, 13]

#%%
wd_test = []
for ii in trange(5000):
    samples1 = multivariate_normal.rvs(mean=mean1, cov=cov1, size=200)
    samples2 = multivariate_normal.rvs(mean=mean2, cov=cov2, size=200)

    M = ot.dist(x1=samples1, x2=samples2)  # x1 (n1,d) n1: samples d: dimension
    Wd = ot.emd2(a=[], b=[], M=M) # exact linear program
    wd_test.append(Wd)

#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 5) )
sns.kdeplot(x= wd_test)

# ot.emd
# ot.lp.emd2()
