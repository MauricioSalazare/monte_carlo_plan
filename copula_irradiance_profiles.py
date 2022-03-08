from core.copula import MixtureCopulaIrradiance
from core.copula import create_copula_model_irradiance_all_clusters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import seaborn as sns

#%% Irradiance models
np.random.seed(12)
file_time_series_irradiance = r"data/processed_data/consumption_weather/knmi_15min_qg.csv"
file_irradiance_clusters = r"data/processed_data/consumption_weather/clustering_irradiance.csv"
freq = "30T"

copula_models_irradiance, mixture_prob = create_copula_model_irradiance_all_clusters(file_time_series_irradiance,
                                                                                     file_irradiance_clusters,
                                                                                     freq="30T")

mixture_model = MixtureCopulaIrradiance(copula_models_irradiance, mixture_prob)

# file_name_load_models = "models/copula_model_irradiance.pkl"
# with open(file_name_load_models, "wb") as pickle_file:
#     pickle.dump(mixture_model, pickle_file)

#%%
from matplotlib import ticker
titles = {0: "Cloudy day",
          1: "Sunny day",
          2: "Dark/Rainy day"}


fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for k_cluster in tqdm(range(3)):
    original_data_filled = mixture_model.get_original_data_cluster(k_model=k_cluster)
    sampled_data_filled = mixture_model.sample_model(n_samples=200, k_model=k_cluster)

    ax[0, k_cluster].plot(original_data_filled.T.values, linewidth=0.5, color="gray")
    ax[0, k_cluster].plot(np.nanmean(original_data_filled.T, axis=1), linewidth=0.5, color='k', label='mean')
    ax[0, k_cluster].plot(np.nanquantile(original_data_filled.T, q=0.05, axis=1),
            color='r', linewidth=0.5, linestyle='--', label='q0.05')
    ax[0, k_cluster].plot(np.nanquantile(original_data_filled.T, q=0.95, axis=1),
            color='r', linewidth=0.5, linestyle='-', label='q0.95')
    ax[0, k_cluster].set_ylim((0, 1000))
    ax[0, k_cluster].xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax[0, k_cluster].set_title(titles[k_cluster])
    ax[0, k_cluster].set_xlim((0, 48))


    ax[1, k_cluster].plot(sampled_data_filled.T.values, linewidth=0.5, color="gray")
    ax[1, k_cluster].plot(np.nanmean(sampled_data_filled.T, axis=1), linewidth=0.5, color='k', label='mean')
    ax[1, k_cluster].plot(np.nanquantile(sampled_data_filled.T, q=0.05, axis=1),
            color='r', linewidth=0.5, linestyle='--', label='q0.05')
    ax[1, k_cluster].plot(np.nanquantile(sampled_data_filled.T, q=0.95, axis=1),
            color='r', linewidth=0.5, linestyle='-', label='q0.95')
    ax[1, k_cluster].set_ylim((0, 1000))
    ax[1, k_cluster].xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax[1, k_cluster].set_xlabel("Time step [30 min]")
    ax[1, k_cluster].set_xlim((0, 48))

    if k_cluster == 0:
        ax[0, k_cluster].set_ylabel("Global irradiance [W/m^2]")
        ax[1, k_cluster].set_ylabel("Global irradiance [W/m^2]")

fig.suptitle(f"Copula modelling Irradiance\nUpper row original data - Lower row sampled from Copula")

#%%  Sample from separate components and put everything together.
original_set_frame = []
sampled_set_frame = []

x_time = "q_22_qg"
y_time = "q_23_qg"

for k_cluster in tqdm(range(3)):
    original_set = mixture_model.get_original_data_cluster(k_model=k_cluster)
    n_samples = len(original_set)
    sampled_set = mixture_model.sample_model(n_samples=n_samples, k_model=k_cluster)

    original_set["k_component"] = np.full(n_samples, k_cluster)
    sampled_set["k_component"] = np.full(n_samples, k_cluster)

    original_set_frame.append(original_set)
    sampled_set_frame.append(sampled_set)

original_set_frame = pd.concat(original_set_frame, axis=0, ignore_index=True)
sampled_set_frame = pd.concat(sampled_set_frame, axis=0, ignore_index=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.kdeplot(data=original_set_frame, x=x_time,y=y_time, hue="k_component", fill=False, linewidths=0.4, zorder=-1, ax=ax[0], palette="tab10")
sns.scatterplot(data=original_set_frame, x=x_time,y=y_time, hue="k_component", s=20, marker="o", zorder=1, ax=ax[0], palette="tab10")
ax[0].set_title("Original")

sns.kdeplot(data=sampled_set_frame, x=x_time,y=y_time, hue="k_component", fill=False, linewidths=0.4, zorder=-1, ax=ax[1])
sns.scatterplot(data=sampled_set_frame, x=x_time,y=y_time, hue="k_component", s=20, marker="o", zorder=1, ax=ax[1], palette="tab10")
ax[1].set_title("Sampled")

#%%
original_mixture_set = mixture_model.get_original_mixture_data()
sampled_mixture_set = mixture_model.sample_mixture(n_samples=len(original_mixture_set))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.kdeplot(data=original_mixture_set, x=x_time,y=y_time, fill=False, linewidths=0.4, zorder=-1, colors="red", ax=ax[0])
sns.scatterplot(data=original_mixture_set, x=x_time,y=y_time, color="red", s=10, marker="o", zorder=1, ax=ax[0])
ax[0].set_title("Original")

sns.kdeplot(data=sampled_mixture_set, x=x_time,y=y_time, fill=False, linewidths=0.4, zorder=-1, colors="red", ax=ax[1])
sns.scatterplot(data=sampled_mixture_set, x=x_time,y=y_time, color="red", s=10, marker="o", zorder=1, ax=ax[1])
ax[1].set_title("Sampled")
fig.suptitle("Mixture model sampled")