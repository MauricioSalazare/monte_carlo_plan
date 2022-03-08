"""
This script tests the
After irradiance clustering, this script creates a k-means classifier to label the data of the irradiance
for days from 2011 - 2021

"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from core.clustering import (ClusteringRLP,
                             plot_scores,
                             rlp_irradiance,
                             load_time_series,
                             unique_filename,
                             plotly_plot_solutions)
from pvlib import clearsky, atmosphere, solarposition
from core.copula import EllipticalCopula
from pvlib.location import Location
from tqdm import trange
from scipy.stats import multivariate_normal
from scipy.stats import norm, ks_2samp, wasserstein_distance
import seaborn as sns
import ot
from tqdm import tqdm, trange
from scipy import interpolate


# titles = {0: "Cloudy day",
#           1: "Sunny day",
#           2: "Dark/Rainy day"}


def helper_plot(data_values, idx_cluster, ax):
    data_plot = data_values[idx_cluster].iloc[:, :-2].values
    ci_daily = data_values[idx_cluster].loc[:, ["ci_daily"]].values.ravel()
    for data_plot_, ci_daily_ in zip(data_plot, ci_daily):
        ax.plot(data_plot_.ravel(), linewidth=0.3, marker='.', markersize=2,
                  markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
                  color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))

#%%
def wasserstein_distance_d(x1: np.array, x2: np.ndarray, factor: float = 1):
    """
    Computes Wasserstein metric as a optimal transportation problem.
    x1 is a sample from the one distribution x2 is a sample from another distribution.

    Parameters:
    ----------
        x1: np.array of dimension (n1,d) n1: samples d: dimension
        x2: np.array of dimension (n1,d) n1: samples d: dimension


    Returns:
    --------
        W_d: Wasserstein distance metric which is the loss function of the optimization problem

    """
    assert x1.shape[1] == x2.shape[1], "Both samples must have the same dimensions"

    M = ot.dist(x1=(x1 / factor), x2=(x2 / factor))
    wass_dist = ot.emd2(a=[], b=[], M=M)  # exact linear program

    return wass_dist

def create_models(data):
    """
    Creates models instances for Multivariate Elliptical Copula, Multivariate Gaussian Distribution, and
    Normal distributions (for each one of the variables).

    Input:
    ------
        data: np.array : (d x n) matrix which d: dimension and n: samples

    Returns:
    --------
        models_dict: dict: dictionary with all models and important information for sampling

    """

    models_dict = {}

    # Multivariate t-Student Copula (MVT) model
    copula_cluster = EllipticalCopula(data)
    copula_cluster.fit()
    models_dict["MVT"] = copula_cluster

    # Multivariate Gaussian Model (MVG) model
    correlation_pearson = pd.DataFrame(data).T.corr(method='pearson').values
    mean = data_irradiance_copula.mean(axis=1)
    std_dev = data_irradiance_copula.std(axis=1)
    mvn_frozen = multivariate_normal(mean=np.zeros(data.shape[0]),
                                     cov=correlation_pearson,
                                     allow_singular=True)
    models_dict["MVG"] = {"model": mvn_frozen,
                          "std": std_dev,
                          "mean": mean}

    # Univariate normal distribution (Normal):
    normal_frozen_models = []
    for mean_, std_dev_ in zip(mean, std_dev):
        normal_frozen_models.append(norm(mean_, std_dev_))

    models_dict["Normal"] = normal_frozen_models

    return models_dict


def sample_models_per_cluster(models_per_cluster: dict, n_samples: int = None):
    """
    Sample all the models PER CLUSTER to create synthetic profiles.
    n_samples is the number of samples per cluster to return. If not specified, the number of samples is the count
    of the profiles of the original dataset to create the models.
    """

    samples_per_cluster = {}
    for cluster_, cluster_models in tqdm(models_per_cluster.items(), desc="Sampling models"):
        if n_samples is None:
            samples_per_cluster[cluster_] = sample_models(cluster_models["models"],
                                                          n_samples=cluster_models["n_profiles_original"])
        else:
            samples_per_cluster[cluster_] = sample_models(cluster_models["models"],
                                                          n_samples=n_samples)

    return samples_per_cluster



def sample_models(models_dict: dict, n_samples: int = 100, clear_negative_ci_indexes: bool = True):
    """
    Sample each models to create synthetic profiles.
    i.e. a models_dict is the collection of models for ONE cluster only.

    Parameters:
    -----------
        models_dict: dict: Dictionary with the key with the name of the model and element with the model
        n_samples: int: Number of samples that will be computed on all models.
        clear_negative_ci_indexes: Bool: (default True): The CI (Clear-Sky-Index), Should be typically values
            between 0 - 1. But never below 0. With this flag, all the samples less than 0 will be clipped to 0.

    Returns:
    --------
        samples_dict: dict: Key name of the model and pair the array with the samples.

    """

    samples_dict = {}
    for model_type in models_dict:
        if model_type == "MVT":
            copula_samples = copula_cluster.sample(n_samples, drop_inf=True)
            samples_dict[model_type] = copula_samples

        elif model_type == "MVG":
            multivariate_normal_samples = mvn_frozen.rvs(n_samples)
            mvg_samples = np.matmul(multivariate_normal_samples,
                                    np.diag(models_dict[model_type]["std"])) + models_dict[model_type]["mean"]
            mvg_samples = mvg_samples.T

            if clear_negative_ci_indexes:
                mvg_samples[mvg_samples < 0] = 0.0

            samples_dict[model_type] = mvg_samples

        elif model_type == "Normal":
            normal_samples = []
            for normal_model in normal_frozen_models:
                normal_samples.append(normal_model.rvs(size=n_samples))
            normal_samples = np.array(normal_samples)

            if clear_negative_ci_indexes:
                normal_samples[normal_samples < 0.0] = 0.0  # Clean negative indexes

            samples_dict[model_type] = normal_samples

        else:
            raise ValueError

    return samples_dict

def sub_sample_dataset(samples_dict: dict, n_subsamples: int = 100):

    sub_samples_dict = {}
    for model_name, samples_model in samples_dict.items():
        number_of_columns = samples_model.shape[1]  # Columns are samples
        random_indices = np.random.choice(number_of_columns, size=n_subsamples, replace=False)
        random_samples = samples_model[:, random_indices]
        sub_samples_dict[model_name] = random_samples

    return sub_samples_dict

def compute_metrics(original_data: np.ndarray, samples_dict: dict):
    """
    Compute 1-D Wasserstein distance metric and p-values for each one of the samples in the samples dictionary
    """

    original_data_flatten = original_data.flatten()
    prob_distance_metrics = {}

    for model_name, sample_data in samples_dict.items():
        ks_metric, p_value = ks_2samp(original_data_flatten, sample_data.flatten())  # 1-D
        wd_metric = wasserstein_distance(original_data_flatten, sample_data.flatten())  # 1-D
        emd_metric = wasserstein_distance_d(original_data.T, sample_data.T)  # Multi dimensional

        prob_distance_metrics[model_name] = {"KS": ks_metric,
                                             "KS-p": p_value,
                                             "WD": wd_metric,
                                             "EMD": emd_metric}


    # return pd.DataFrame.from_dict(wasserstein_metrics)
    return pd.DataFrame.from_dict(prob_distance_metrics)


def pivot_dataframe(work_frame):
    "Pivots data frame so the rows are day profiles and columsn are time steps of the profile."

    work_frame["day_of_year"] = work_frame.index.dayofyear
    work_frame['quarterIndex'] = work_frame.index.hour * 100 + work_frame.index.minute
    pivoted_frame = work_frame.pivot(index='quarterIndex', columns='day_of_year')
    pivoted_frame.columns = pivoted_frame.columns.droplevel()
    pivoted_frame_ = pivoted_frame.transpose()

    return pivoted_frame_

#%%
RESAMPLE = "10T"
START_DATE = "2021-03-01 00:00:00"
END_DATE = "2021-09-30 23:50:00"

#%% Load irradiance
file_irradiance_full_data = r"data/raw_data/knmi_data/10_min_resolution/knmi_data_enschede_all_2021.csv"
knmi_10min_w_m2 = pd.read_csv(file_irradiance_full_data,
                               parse_dates=True,
                               index_col='time',
                               date_parser=lambda col: pd.to_datetime(col, utc=True))

#%%
# Data for the Twente climatological station.
latitude = 52.27314817052
# longitude = 6.8908745111116 + 8  # Offset to align sunrise and sunset (equivalent of shifting the time)
longitude = 6.8908745111116  # Offset to align sunrise and sunset (equivalent of shifting the time)
elevation = 33.2232  # Meters

twente = Location(latitude=latitude, longitude=longitude, tz='UTC', altitude=elevation, name='Twente')
# times = pd.date_range(start='20-07-01', end='2016-07-04', freq='1min', tz=twente.tz)
times = pd.date_range(start='2021-01-01', end='2021-12-31 23:50:00', freq='10T', tz=twente.tz)
cs_simplified_solis = twente.get_clearsky(times, model="simplified_solis")
cs_haurwitz = twente.get_clearsky(times, model="haurwitz")
cs_ineichen = twente.get_clearsky(times, model="ineichen")

cs_simplified_solis.rename(columns={"ghi": "ghi_solis"}, inplace=True)
cs_haurwitz.rename(columns={"ghi": "ghi_haurwitz"}, inplace=True)  # Best model
cs_ineichen.rename(columns={"ghi": "ghi_ineichen"}, inplace=True)

#%%
data_aligned = pd.concat([knmi_10min_w_m2[["qg"]], cs_haurwitz, cs_simplified_solis[["ghi_solis"]], cs_ineichen[["ghi_ineichen"]]], axis=1)
data_aligned.plot()
data_aligned.resample(RESAMPLE).mean()[START_DATE:END_DATE].plot()
data_aligned_sliced = data_aligned.resample(RESAMPLE).mean()[START_DATE:END_DATE].copy()

#%% Process the daylight times with the solar model
IRR_THRESHOLD = 20  # W/m^2

irradiance_days = pivot_dataframe(data_aligned_sliced[["ghi_haurwitz"]])
max_sunlight_time = (irradiance_days > 20).sum(axis=1).max()
irradiance_days_real = pivot_dataframe(data_aligned_sliced[["qg"]])
irradiance_days_real_clean = irradiance_days_real.dropna(axis=0)

fig, ax = plt.subplots(1,1, figsize=(15, 8))
ax.plot(irradiance_days.transpose())

#%% Stretching the time axis for the SOLAR model and REAL data.
# day_mapping = {}
# for day_ in irradiance_days.index:
#     max_irradiance = irradiance_days.loc[day_].max()
#     idx = irradiance_days.loc[day_] > IRR_THRESHOLD
#     y_ = irradiance_days.loc[day_][idx].values
#     x_ = np.linspace(1, max_sunlight_time, idx.sum())
#     f = interpolate.interp1d(x_, y_, fill_value='extrapolate')
#     x_prime = np.linspace(1, max_sunlight_time, max_sunlight_time)
#     y_prime = f(x_prime)
#     day_mapping[day_] = {"max_q": max_irradiance,
#                          "idx": idx.values,
#                          "f": f,
#                          "x_": x_,
#                          "y_": x_,
#                          "x_prime": x_prime,
#                          "y_prime": y_prime,
#                          "GHI": y_.mean()}
#
#     if day_ in irradiance_days_real_clean.index:
#         idx_day = day_mapping[day_]["idx"]
#         y_real = irradiance_days_real_clean.loc[day_][idx_day].values
#         x_real = np.linspace(1, max_sunlight_time, idx_day.sum())
#         f_real = interpolate.interp1d(x_real, y_real, fill_value='extrapolate')
#         x_real_prime = np.linspace(1, max_sunlight_time, max_sunlight_time)  # new x-values
#         y_real_prime = f_real(x_real_prime)  # Stretched values in x
#         # sunlight_time_real.append(y_real_prime)
#         # sunlight_time_real_scaled.append(y_real_prime / day_mapping[day_]["max_q"])
#         # sunlight_time_real_ci.append(y_real_prime / day_mapping[day_]["y_prime"])
#         # clear_index.append(y_real.mean() / day_mapping[day_]["GHI"])
#
#         day_mapping[day_]["y_real_prime"] = y_real_prime  # IMPORTANT! SAVED FOR STATISTICAL COMPARISONS
#         day_mapping[day_]["y_real_prime_scaled"] = y_real_prime / day_mapping[day_]["max_q"]
#         day_mapping[day_]["y_real_prime_ci"] = y_real_prime / day_mapping[day_]["y_prime"]
#         day_mapping[day_]["CIS_daily"] = y_real.mean() / day_mapping[day_]["GHI"]
#
#






#%% Stretching the time axis for the SOLAR model
day_mapping = {}
for day_ in irradiance_days.index:
    max_irradiance = irradiance_days.loc[day_].max()
    idx = irradiance_days.loc[day_] > IRR_THRESHOLD
    y_ = irradiance_days.loc[day_][idx].values
    x_ = np.linspace(1, max_sunlight_time, idx.sum())
    f = interpolate.interp1d(x_, y_, fill_value='extrapolate')
    x_prime = np.linspace(1, max_sunlight_time, max_sunlight_time)
    y_prime = f(x_prime)
    day_mapping[day_] = {"max_q": max_irradiance,
                         "idx": idx.values,
                         "f": f,
                         "x_": x_,
                         "y_": x_,
                         "x_prime": x_prime,
                         "y_prime": y_prime,
                         "GHI": y_.mean()}

fig, ax = plt.subplots(1,1, figsize=(15, 8))
ax.plot(np.array(pd.DataFrame.from_dict(day_mapping).transpose()["y_prime"].to_list()).T)

#%% Compare 2 days to see the difference in sunglight times
# fig, ax = plt.subplots(1,1, figsize=(10,5))
# ax.plot(irradiance_days.loc[75], "-o")
# ax.plot(irradiance_days.loc[150], "-x")
# ax.set_title("Difference between July and September")


#%% Plot only sunlight_hours with the real data
sunlight_time_real = []
sunlight_time_real_ci = []
sunlight_time_real_scaled = []
clear_index = []

for day_ in irradiance_days_real_clean.index:
    idx_day = day_mapping[day_]["idx"]
    y_real = irradiance_days_real_clean.loc[day_][idx_day].values
    x_real = np.linspace(1, max_sunlight_time, idx_day.sum())
    f_real = interpolate.interp1d(x_real, y_real, fill_value='extrapolate')
    x_real_prime = np.linspace(1, max_sunlight_time, max_sunlight_time)  # new x-values
    y_real_prime = f_real(x_real_prime)  # Stretched values in x
    sunlight_time_real.append(y_real_prime)
    sunlight_time_real_scaled.append(y_real_prime/ day_mapping[day_]["max_q"])
    sunlight_time_real_ci.append(y_real_prime / day_mapping[day_]["y_prime"])
    clear_index.append(y_real.mean() / day_mapping[day_]["GHI"])

    day_mapping[day_]["y_real_prime"] = y_real_prime  # IMPORTANT! SAVED FOR STATISTICAL COMPARISONS

sunlight_time_real = np.array(sunlight_time_real)
sunlight_time_real_scaled = np.array(sunlight_time_real_scaled)
sunlight_time_real_ci = np.array(sunlight_time_real_ci)
clear_index = np.array(clear_index)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(sunlight_time_real_ci.T, color="grey", marker=".", markersize=0.2, linewidth=0.4)
ax.set_title("Clear Sky Index (CSI)")
ax.set_xlabel("Time step")
ax.set_ylabel("CSI")

#%% Histogram
fig, ax = plt.subplots(1, 1, figsize=(5,5))
# plt.subplots_adjust(left=0.2)
n, bins, patches = ax.hist(sunlight_time_real_ci.ravel(), 50, density=True, facecolor='g', alpha=0.75)
ax.set_xlim((0, 2))
ax.set_xlabel("Clear index (CI)")
ax.set_ylabel("Counts")

#%%
data_set = sunlight_time_real_ci
cluster_ap = ClusteringRLP()
cluster_ap.cluster_dataset(data_set, plot_solutions=False, end=20, std_scale=False)
score_values_ap = cluster_ap.get_clustering_scores()
plot_scores(score_values_ap["CDI"],
            score_values_ap["MDI"],
            score_values_ap["DBI"],
            score_values_ap["CHI"])
ap_cluster_solutions = cluster_ap.get_cluster_labels()
#%%
N_CLUSTERS = 6
ALGORITHM = "HC(Ward)"
fig, ax = plt.subplots(4, N_CLUSTERS, figsize=(20, 15))
ax_ = ax[0,:].flatten()
ax2_ = ax[1,:].flatten()
ax3_ = ax[2,:].flatten()
ax4_ = ax[3,:].flatten()


# norm_individual = mpl.colors.Normalize(vmin=clear_index.min(),
#                                        vmax=clear_index.max())
norm_individual = mpl.colors.Normalize(vmin=0,
                                       vmax=1)

labeled_data_year_ci = pd.concat([pd.DataFrame(sunlight_time_real_ci),
                               # pd.DataFrame(labels, columns=["cluster"])],
                                  pd.DataFrame(ap_cluster_solutions.loc[ALGORITHM, N_CLUSTERS], columns=["cluster"]),
                                  pd.DataFrame(clear_index, columns=["ci_daily"])],
                                  axis=1)

labeled_data_year = pd.concat([pd.DataFrame(sunlight_time_real),
                               # pd.DataFrame(labels, columns=["cluster"])],
                               pd.DataFrame(ap_cluster_solutions.loc[ALGORITHM, N_CLUSTERS], columns=["cluster"]),
                               pd.DataFrame(clear_index, columns=["ci_daily"])],
                               axis=1)

labeled_data_year_scaled = pd.concat([pd.DataFrame(sunlight_time_real_scaled),
                               # pd.DataFrame(labels, columns=["cluster"])],
                               pd.DataFrame(ap_cluster_solutions.loc[ALGORITHM, N_CLUSTERS], columns=["cluster"]),
                                pd.DataFrame(clear_index, columns=["ci_daily"])],
                               axis=1)

labeled_data_year_original = pd.concat([irradiance_days_real_clean,
                               # pd.DataFrame(labels, columns=["cluster"])],
                               pd.DataFrame(ap_cluster_solutions.loc[ALGORITHM, N_CLUSTERS], columns=["cluster"], index= irradiance_days_real_clean.index),
                                        pd.DataFrame(clear_index, columns=["ci_daily"], index= irradiance_days_real_clean.index)],
                               axis=1, ignore_index=False)

for ax_i, ax_j, ax_z, ax_w, cluster_ in zip(ax_, ax2_, ax3_, ax4_, np.linspace(0, N_CLUSTERS-1, N_CLUSTERS)):
    idx_cluster = labeled_data_year["cluster"] == cluster_


    helper_plot(labeled_data_year_ci, idx_cluster=idx_cluster, ax=ax_i)
    ax_i.set_ylim((0, 2.0))

    helper_plot(labeled_data_year, idx_cluster=idx_cluster, ax=ax_j)
    ax_j.set_ylim((0, 1000))

    helper_plot(labeled_data_year_scaled, idx_cluster=idx_cluster, ax=ax_z)
    ax_z.set_ylim((0, 1.1))

    helper_plot(labeled_data_year_original, idx_cluster=idx_cluster.values, ax=ax_w)
    ax_w.set_ylim((0, 1000))

fig.suptitle(f"Algorith: {ALGORITHM}")

#%% Plot original data with the clear index
fig, ax = plt.subplots(1,1, figsize=(6, 4))
for ii,(_, data_plot_) in enumerate(labeled_data_year_original.iloc[:, :-2].iterrows()):
    ci_daily_ = labeled_data_year_original.loc[:, ["ci_daily"]].iloc[ii,:]
    ax.plot(data_plot_.ravel(), linewidth=0.3, marker='.', markersize=2,
            markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
            color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))
ax.set_title("Original load profiles")
cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('plasma')), ax=ax)
cbar_2.ax.set_ylabel('Daily clear index [$K_d$]')


#%% Create models on the clear-index-sky domain data
cluster_labels = np.sort(labeled_data_year_ci["cluster"].unique())
models_per_cluster = {}
data_per_cluster = {}
for cluster_ in cluster_labels:
    idx = labeled_data_year_ci["cluster"] == cluster_
    data_irradiance_ci = labeled_data_year_ci[idx].iloc[:, :-2].values.T
    data_per_cluster[cluster_] = data_irradiance_ci  # Original dataset to compute the metrics
    models_per_cluster[cluster_] = {"n_profiles_original": np.sum(idx),
                                    "models": create_models(data_irradiance_ci)}

#%% Sample the models in the clear-sky-domain
samples_per_cluster = sample_models_per_cluster(models_per_cluster, n_samples=4000)

#%% Compute metrics on the clear-sky-domain
bootstrap_size = 200
metrics_per_cluster = {}
for cluster_, samples_per_model in samples_per_cluster.items():
    original_ci_cluster = data_per_cluster[cluster_]
    n_samples = original_ci_cluster.shape[1]
    metrics_per_test = []
    for test_n in trange(bootstrap_size, desc=f"Bootstrapping cluster: {cluster_}"):
        metrics_per_test.append(
                                compute_metrics(original_ci_cluster,
                                                sub_sample_dataset(samples_per_model, n_subsamples=n_samples))
        )
    metrics_per_cluster[cluster_] = metrics_per_test

#%% Gather all the bootstrapped statistics per cluster:
stats_metric_per_cluster = {}
for cluster_, metric_bootstrap_set in tqdm(metrics_per_cluster.items()):
    bootstrap_size_ = len(metric_bootstrap_set)
    stats_metric_per_cluster[cluster_] = pd.concat(metric_bootstrap_set, keys=np.arange(bootstrap_size_))

#%% Compute the mean and standard deviations for the boostratped statistics
metric_names = stats_metric_per_cluster[0].xs(0, level=0).index
merged_metrics_per_cluster = []
idx_multi = pd.IndexSlice

metric_mean_per_cluster = {}
for cluster_, stats_metric in stats_metric_per_cluster.items():
    merged_metrics = []
    for metric_name in metric_names:
        merged_metrics.append(stats_metric.loc[idx_multi[:, metric_name], :].mean().to_frame(name=f"{metric_name}_mean"))
    metric_mean_per_cluster[cluster_] = pd.concat(merged_metrics, ignore_index=False, axis=1).transpose()

#%% Create the profiles using the sampled data from the clear-sky-domain:
solar_model_clustered = pd.concat([irradiance_days_real_clean,
                                   labeled_data_year_original["cluster"]], join="inner", ignore_index=False, axis=1)








#%% Copulas
cluster_labels = np.sort(labeled_data_year_ci["cluster"].unique())
original_data = []
samples_copulas = []
samples_mvg = []
samples_normal = []

solar_model_clustered = pd.concat([irradiance_days_real_clean, labeled_data_year_original["cluster"]], join="inner", ignore_index=False, axis=1)

profiles_original = []
profiles_copula = []
profiles_mvg = []
profiles_normal = []

profiles_daylight_original = []
diff_profiles_daylight_original = []
profiles_daylight_copula = []
diff_profiles_daylight_copula = []
profiles_daylight_mvg = []
diff_profiles_daylight_mvg = []
profiles_daylight_normal = []
diff_profiles_daylight_normal = []

y_hat_prime_all_original = []
y_hat_prime_all_copula = []
y_hat_prime_all_mvg = []
y_hat_prime_all_normal = []


for cluster_ in cluster_labels:
    idx = labeled_data_year_ci["cluster"] == cluster_
    data_irradiance_copula = labeled_data_year_ci[idx].iloc[:, :-2].values.T
    n_samples = data_irradiance_copula.shape[1]

    models_dict = create_models(data_irradiance_copula)

    # ===============================================================================================================
    # Copula model
    copula_cluster = EllipticalCopula(data_irradiance_copula)
    copula_cluster.fit()

    # ===============================================================================================================
    # Multivariate gaussian model
    correlation_pearson = pd.DataFrame(data_irradiance_copula).T.corr(method='pearson').values
    mean = data_irradiance_copula.mean(axis=1)
    std_dev = data_irradiance_copula.std(axis=1)
    mvn_frozen = multivariate_normal(mean=np.zeros(data_irradiance_copula.shape[0]),
                                     cov=correlation_pearson,
                                     allow_singular=True)
    # ===============================================================================================================
    # Uni-variate gaussian model
    normal_frozen_models = []
    for mean_, std_dev_ in zip(mean, std_dev):
        normal_frozen_models.append(norm(mean_, std_dev_))

    # ===============================================================================================================
    # =========================================== SAMPLING ==========================================================
    # ===============================================================================================================

    bootstrap_iterations = 1

    data_ks = []
    data_p = []

    copula_ks = []
    copula_p = []

    mvg_ks = []
    mvg_p = []

    normal_ks = []
    normal_p = []

    mvg_samples = []
    copula_samples = []
    normal_samples= []

    for ii in trange(bootstrap_iterations):

        data_irradiance_frame = pd.DataFrame(data_irradiance_copula.T)
        chunk_a = data_irradiance_frame.sample(frac=.5)
        chunk_b = data_irradiance_frame.drop(chunk_a.index)

        chunk_a_array = chunk_a.to_numpy().ravel()
        chunk_b_array = chunk_b.to_numpy().ravel()

        # n_samples = chunk_a.shape[0]
        # n_samples = data_irradiance_frame.shape[0]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # MVG sample
        new_samples = mvn_frozen.rvs(n_samples)
        mvg_samples = np.matmul(new_samples, np.diag(std_dev)) + mean
        mvg_samples = mvg_samples.T
        mvg_samples[mvg_samples < 0] = 0  # Clean negative indexes

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Copula sample
        copula_samples = copula_cluster.sample(n_samples, drop_inf=True)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Uni-variate gaussian sample
        normal_samples = []
        for normal_model in normal_frozen_models:
            normal_samples.append(normal_model.rvs(size=n_samples))
        normal_samples = np.array(normal_samples)
        normal_samples[normal_samples < 0] = 0  # Clean negative indexes


        w, z = ks_2samp(chunk_a_array, chunk_b_array)
        a, b = ks_2samp(chunk_a_array.ravel(), copula_samples.ravel())
        c, d = ks_2samp(chunk_a_array.ravel(), mvg_samples.ravel())
        e, f = ks_2samp(chunk_a_array.ravel(), normal_samples.ravel())

        data_ks.append(w)
        data_p.append(z)

        copula_ks.append(a)
        copula_p.append(b)

        mvg_ks.append(c)
        mvg_p.append(d)

        normal_ks.append(e)
        normal_p.append(f)

    # fig, ay = plt.subplots(1, 1, figsize=(8, 8))
    # sns.kdeplot(x=np.array(data_p).ravel(), ax=ay, color="grey", label="original")
    # sns.kdeplot(x=np.array(copula_p).ravel(), ax=ay, color="C0", label="copula")
    # sns.kdeplot(x=np.array(normal_p).ravel(), ax=ay, color="C1", label="normal")
    # sns.kdeplot(x=np.array(mvg_p).ravel(), ax=ay, color="C2", label="mvg")
    # ay.set_title(f"Cluster: {cluster_} -- p-values")
    # ay.legend()
    #
    # fig, ay = plt.subplots(1, 1, figsize=(8, 8))
    # sns.kdeplot(x=np.array(data_ks).ravel(), ax=ay, color="grey", label="original")
    # sns.kdeplot(x=np.array(copula_ks).ravel(), ax=ay, color="C0", label="copula")
    # sns.kdeplot(x=np.array(normal_ks).ravel(), ax=ay, color="C1", label="normal")
    # sns.kdeplot(x=np.array(mvg_ks).ravel(), ax=ay, color="C2", label="mvg")
    # ay.set_title(f"Cluster: {cluster_} -- KS-values")
    # ay.legend()

    original_data.append(data_irradiance_copula)
    samples_copulas.append(copula_samples)
    samples_mvg.append(mvg_samples)
    samples_normal.append(normal_samples)

    a_was = wasserstein_distance(data_irradiance_copula.ravel(), copula_samples.ravel())
    c_was = wasserstein_distance(data_irradiance_copula.ravel(), mvg_samples.ravel())
    e_was = wasserstein_distance(data_irradiance_copula.ravel(), normal_samples.ravel())

    a, b = ks_2samp(data_irradiance_copula.ravel(), copula_samples.ravel())
    c, d = ks_2samp(data_irradiance_copula.ravel(), mvg_samples.ravel())
    e, f = ks_2samp(data_irradiance_copula.ravel(), normal_samples.ravel())

    print(f"Cluster: {cluster_} ")
    print(f"KS:: copula: {a} -- mvg: {b} -- normal: {e}")
    print(f"p-values:: copula: {b} -- mvg: {d} -- normal: {f}")
    print(f"Wasserstein:: copula: {a_was} -- mvg: {c_was} -- normal: {e_was}")

    # ===============================================================================================================
    # Transform the sample data into the real irradiance
    # ===============================================================================================================

    idx = solar_model_clustered["cluster"] == cluster_

    profiles_cluster_original = []
    profiles_cluster_copula = []
    profiles_cluster_mvg = []
    profiles_cluster_normal = []

    profiles_daylight_cluster_original = []
    diff_profiles_daylight_cluster_original = []
    profiles_daylight_cluster_copula = []
    diff_profiles_daylight_cluster_copula = []
    profiles_daylight_cluster_mvg = []
    diff_profiles_daylight_cluster_mvg = []
    profiles_daylight_cluster_normal = []
    diff_profiles_daylight_cluster_normal = []

    y_hat_prime_original_values = []
    y_hat_prime_copula_values = []
    y_hat_prime_mvg_values = []
    y_hat_prime_normal_values = []


    for ii, DAY_ in enumerate(solar_model_clustered[idx].index):
        y_hat_prime_original = day_mapping[DAY_]["y_real_prime"]
        y_hat_prime_copula = day_mapping[DAY_]["y_prime"] * copula_samples[:, ii]  # Samples GHI (x-stretched)
        y_hat_prime_mvg = day_mapping[DAY_]["y_prime"] * mvg_samples[:, ii]  # Samples GHI (x-stretched)
        y_hat_prime_normal = day_mapping[DAY_]["y_prime"] * normal_samples[:, ii]  # Samples GHI (x-stretched)

        # De-stretching
        f_hat_copula = interpolate.interp1d(day_mapping[DAY_]["x_prime"], y_hat_prime_copula)
        y_hat_copula = f_hat_copula(day_mapping[DAY_]["x_"])

        f_hat_mvg = interpolate.interp1d(day_mapping[DAY_]["x_prime"], y_hat_prime_mvg)
        y_hat_mvg = f_hat_mvg(day_mapping[DAY_]["x_"])

        f_hat_normal = interpolate.interp1d(day_mapping[DAY_]["x_prime"], y_hat_prime_normal)
        y_hat_normal = f_hat_normal(day_mapping[DAY_]["x_"])

        # Create the profiles
        profile_created_copula = np.zeros(day_mapping[DAY_]["idx"].shape[0])
        profile_created_copula[day_mapping[DAY_]["idx"]] = y_hat_copula

        profile_created_mvg = np.zeros(day_mapping[DAY_]["idx"].shape[0])
        profile_created_mvg[day_mapping[DAY_]["idx"]] = y_hat_mvg

        profile_created_normal = np.zeros(day_mapping[DAY_]["idx"].shape[0])
        profile_created_normal[day_mapping[DAY_]["idx"]] = y_hat_normal

        profiles_cluster_original.append(solar_model_clustered.loc[DAY_].drop("cluster").values)
        profiles_cluster_copula.append(profile_created_copula)
        profiles_cluster_mvg.append(profile_created_mvg)
        profiles_cluster_normal.append(profile_created_normal)

        # DENSITY FUNCTIONS FOR THE IRRADIANCE AND DELTA IRRADIANCE FOR FURTHER COMPARISON
        idx_ = solar_model_clustered.loc[DAY_].drop("cluster") > IRR_THRESHOLD
        y_hat_original = solar_model_clustered.loc[DAY_].drop("cluster")[idx_].values

        profiles_daylight_cluster_original.append(y_hat_original)
        diff_profiles_daylight_cluster_original.append(np.diff(y_hat_original))

        profiles_daylight_cluster_copula.append(y_hat_copula)
        diff_profiles_daylight_cluster_copula.append(np.diff(y_hat_copula))

        profiles_daylight_cluster_mvg.append(y_hat_mvg)
        diff_profiles_daylight_cluster_mvg.append(np.diff(y_hat_mvg))

        profiles_daylight_cluster_normal.append(y_hat_normal)
        diff_profiles_daylight_cluster_normal.append(np.diff(y_hat_normal))


        # Save the profiles with the x-axis stretched (y_hat)
        y_hat_prime_original_values.append(y_hat_prime_original)
        y_hat_prime_copula_values.append(y_hat_prime_copula)
        y_hat_prime_mvg_values.append(y_hat_prime_mvg)
        y_hat_prime_normal_values.append(y_hat_prime_normal)







    y_hat_prime_all_original.append(np.array(y_hat_prime_original_values))
    y_hat_prime_all_copula.append(np.array(y_hat_prime_copula_values))
    y_hat_prime_all_mvg.append(np.array(y_hat_prime_mvg_values))
    y_hat_prime_all_normal.append(np.array(y_hat_prime_normal_values))

# #%%
#     CLUSTERITO = 5
#
#
#     # samples1 = np.diff(y_hat_prime_all_original[CLUSTERITO], axis=1) / 1000
#     # samples2 = np.diff(y_hat_prime_all_copula[CLUSTERITO], axis=1) / 1000
#     # samples3 = np.diff(y_hat_prime_all_mvg[CLUSTERITO], axis=1) / 1000
#     # samples4 = np.diff(y_hat_prime_all_normal[CLUSTERITO], axis=1) / 1000
#
#     samples1 = original_data[CLUSTERITO].T
#     samples2 = samples_copulas[CLUSTERITO].T
#     samples3 = samples_mvg[CLUSTERITO].T
#     samples4 = samples_normal[CLUSTERITO].T
#
#     samples3[samples3 < 0] = 0
#     samples4[samples4 < 0] = 0
#
#     # samples1 = np.diff(original_data[CLUSTERITO].T, axis=1)
#     # samples2 = np.diff(samples_copulas[CLUSTERITO].T, axis=1)
#     # samples3 = np.diff(samples_mvg[CLUSTERITO].T, axis=1)
#     # samples4 = np.diff(samples_normal[CLUSTERITO].T, axis=1)
#
#     Wd1 = wasserstein_distance_d(samples1, samples2)
#     Wd2 = wasserstein_distance_d(samples1, samples3)
#     Wd3 = wasserstein_distance_d(samples1, samples4)
#
#     print(f"Copula: {Wd1} - MVG : {Wd2} - Normal: {Wd3}")
# #%%

    profiles_original.append(np.array(profiles_cluster_original))
    profiles_copula.append(np.array(profiles_cluster_copula))
    profiles_mvg.append(np.array(profiles_cluster_mvg))
    profiles_normal.append(np.array(profiles_cluster_normal))

    profiles_daylight_original.append(np.concatenate(profiles_daylight_cluster_original))
    diff_profiles_daylight_original.append(np.concatenate(diff_profiles_daylight_cluster_original))
    profiles_daylight_copula.append(np.concatenate(profiles_daylight_cluster_copula))
    diff_profiles_daylight_copula.append(np.concatenate(diff_profiles_daylight_cluster_copula))
    profiles_daylight_mvg.append(np.concatenate(profiles_daylight_cluster_mvg))
    diff_profiles_daylight_mvg.append(np.concatenate(diff_profiles_daylight_cluster_mvg))
    profiles_daylight_normal.append(np.concatenate(profiles_daylight_cluster_normal))
    diff_profiles_daylight_normal.append(np.concatenate(diff_profiles_daylight_cluster_normal))

    a_prof_was = wasserstein_distance(np.concatenate(profiles_daylight_cluster_original),
                                      np.concatenate(profiles_daylight_cluster_copula))
    c_prof_was = wasserstein_distance(np.concatenate(profiles_daylight_cluster_original),
                                      np.concatenate(profiles_daylight_cluster_mvg))
    e_prof_was = wasserstein_distance(np.concatenate(profiles_daylight_cluster_original),
                                      np.concatenate(profiles_daylight_cluster_normal))

    aa_prof_was = wasserstein_distance(np.abs(np.concatenate(diff_profiles_daylight_cluster_original)),
                                      np.abs(np.concatenate(diff_profiles_daylight_cluster_copula)))
    cc_prof_was = wasserstein_distance(np.abs(np.concatenate(diff_profiles_daylight_cluster_original)),
                                      np.abs(np.concatenate(diff_profiles_daylight_cluster_mvg)))
    ee_prof_was = wasserstein_distance(np.abs(np.concatenate(diff_profiles_daylight_cluster_original)),
                                      np.abs(np.concatenate(diff_profiles_daylight_cluster_normal)))

    # a, b = ks_2samp(data_irradiance_copula.ravel(), copula_samples.ravel())
    # c, d = ks_2samp(data_irradiance_copula.ravel(), mvg_samples.ravel())
    # e, f = ks_2samp(data_irradiance_copula.ravel(), normal_samples.ravel())
    print(f"============================== profiles =============================================== ")
    print(f"Cluster: {cluster_} ")
    # print(f"KS:: copula: {a} -- mvg: {b} -- normal: {e}")
    # print(f"p-values:: copula: {b} -- mvg: {d} -- normal: {f}")
    print(f"Wasserstein:: copula: {a_prof_was} -- mvg: {c_prof_was} -- normal: {e_prof_was}")

    print(f"============================== DIFF profiles =========================================== ")
    print(f"Cluster: {cluster_} ")
    # print(f"KS:: copula: {a} -- mvg: {b} -- normal: {e}")
    # print(f"p-values:: copula: {b} -- mvg: {d} -- normal: {f}")
    print(f"Wasserstein:: copula: {aa_prof_was} -- mvg: {cc_prof_was} -- normal: {ee_prof_was}")


#%%
fig, ax = plt.subplots(6, N_CLUSTERS, figsize=(15, 10))
ax_i = ax[0, :].flatten()
ax_j = ax[1, :].flatten()
ax_z = ax[2, :].flatten()
ax_w = ax[3, :].flatten()
ax_t = ax[4, :].flatten()
ax_k = ax[5, :].flatten()

for ii, (ax_i_, ax_j_, ax_z_, ax_w_, ax_t_, ax_k_) in enumerate(zip(ax_i, ax_j, ax_z, ax_w, ax_t, ax_k)):

    ax_i_.plot(original_data[ii], color="grey", linewidth=0.4, marker='.', markersize=2)
    ax_j_.plot(samples_copulas[ii], color="C0", linewidth=0.4, marker='.', markersize=2)
    ax_z_.plot(samples_normal[ii], color="C1", linewidth=0.4, marker='.', markersize=2)
    ax_w_.plot(samples_mvg[ii], color="C2", linewidth=0.4, marker='.', markersize=2)

    ax_i_.set_title(f"Samples: {original_data[ii].shape[1]}")

    sns.ecdfplot(data=original_data[ii].ravel(),  ax=ax_t_, color="grey", stat="proportion")
    sns.ecdfplot(data=samples_copulas[ii].ravel(),  ax=ax_t_, color="C0", stat="proportion")
    sns.ecdfplot(data=samples_normal[ii].ravel(), ax=ax_t_, color="C1", stat="proportion")
    sns.ecdfplot(data=samples_mvg[ii].ravel(), ax=ax_t_, color="C2", stat="proportion")
    ax_t_.set_ylabel("")

    sns.kdeplot(x=np.diff(original_data[ii], axis=0).ravel(), ax=ax_k_, color="grey")
    sns.kdeplot(x=np.diff(samples_copulas[ii], axis=0).ravel(), ax=ax_k_, color="C0")
    sns.kdeplot(x=np.diff(samples_normal[ii], axis=0).ravel(), ax=ax_k_, color="C1")
    sns.kdeplot(x=np.diff(samples_mvg[ii], axis=0).ravel(), ax=ax_k_, color="C2")
    ax_k_.set_ylabel("")
    ax_k_.set_xlim((-0.5, 0.5))

    ax_i_.set_ylim(0, 2)
    ax_j_.set_ylim(0, 2)
    ax_z_.set_ylim(0, 2)
    ax_w_.set_ylim(0, 2)

fig.suptitle("Sampling model of the copulas")


#%%
fig, ax = plt.subplots(6, N_CLUSTERS, figsize=(15, 10))
ax_i = ax[0, :].flatten()
ax_j = ax[1, :].flatten()
ax_z = ax[2, :].flatten()
ax_w = ax[3, :].flatten()
ax_t = ax[4, :].flatten()
ax_k = ax[5, :].flatten()

for ii, (ax_i_, ax_j_, ax_z_, ax_w_, ax_t_, ax_k_) in enumerate(zip(ax_i, ax_j, ax_z, ax_w, ax_t, ax_k)):
# for ii, (ax_i_, ax_j_, ax_z_, ax_w_) in enumerate(zip(ax_i, ax_j, ax_z, ax_w)):

    ax_i_.plot(profiles_original[ii].T, color="grey", linewidth=0.4, marker='.', markersize=2)
    ax_j_.plot(profiles_copula[ii].T, color="C0", linewidth=0.4, marker='.', markersize=2)
    ax_z_.plot(profiles_normal[ii].T, color="C1", linewidth=0.4, marker='.', markersize=2)
    ax_w_.plot(profiles_mvg[ii].T, color="C2", linewidth=0.4, marker='.', markersize=2)

    sns.ecdfplot(data=profiles_daylight_original[ii].ravel(),  ax=ax_t_, color="grey", stat="proportion")
    sns.ecdfplot(data=profiles_daylight_copula[ii].ravel(),  ax=ax_t_, color="C0", stat="proportion")
    sns.ecdfplot(data=profiles_daylight_normal[ii].ravel(), ax=ax_t_, color="C1", stat="proportion")
    sns.ecdfplot(data=profiles_daylight_mvg[ii].ravel(), ax=ax_t_, color="C2", stat="proportion")
    ax_t_.set_ylabel("")

    sns.kdeplot(x=np.abs(diff_profiles_daylight_original[ii]).ravel(), ax=ax_k_, color="grey", cut=0)
    sns.kdeplot(x=np.abs(diff_profiles_daylight_copula[ii]).ravel(), ax=ax_k_, color="C0", cut=0)
    sns.kdeplot(x=np.abs(diff_profiles_daylight_normal[ii]).ravel(), ax=ax_k_, color="C1", cut=0)
    sns.kdeplot(x=np.abs(diff_profiles_daylight_mvg[ii]).ravel(), ax=ax_k_, color="C2", cut=0)
    ax_k_.set_ylabel("")
    ax_k_.set_xlim((-10, 200))

    ax_i_.set_ylim(0, 1000)
    ax_j_.set_ylim(0, 1000)
    ax_z_.set_ylim(0, 1000)
    ax_w_.set_ylim(0, 1000)

fig.suptitle("Irradiance samples")

#%%
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.histplot(data=original_data[ii].ravel(), kde=True, ax=ax, color="grey", stat="density")
sns.histplot(data=samples_copulas[ii].ravel(), kde=True, ax=ax, color="C0", stat="density")

#%%
# ==================================================================================================================
# Fractal dimension
# ==================================================================================================================
#%%


fig, ax = plt.subplots(1, 31, figsize=(25, 4))
plt.subplots_adjust(left=0.01, right=0.99)
for ii, ax_ in enumerate(ax):
    sns.histplot(data=original_data[0][ii, :].ravel(), kde=True, ax=ax_, color="grey", stat="density")


# ==================================================================================================================
# Standard deviation of increments dimension
# ==================================================================================================================




# ==================================================================================================================
# Variability index
# ==================================================================================================================







# ==================================================================================================================
#
# ==================================================================================================================






# labeled_data_year = pd.concat([irr_data_low_res,
#                                pd.DataFrame(labels, index=irr_data_low_res.index, columns=["cluster"])],
#                                axis=1)
# #%%
# fig, ax = plt.subplots(1,1)
# ax.plot(data_set.T)
#
# #%%
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax_ = ax.flatten()
#
# for ax_i, cluster_ in zip(ax_, [0,1,2]):
#     idx_cluster = labeled_data_year["cluster"] == cluster_
#     ax_i.plot(labeled_data_year[idx_cluster].values.T)
#
#
#
#
#
# #%%
#
#
# irr_data_all, idx_mask = rlp_irradiance(data.resample("H").mean())
# labeled_data = pd.concat([irr_data_all, clusters_irradiance], axis=1)
# idx_nan = labeled_data.cluster.isna()
# labeled_data = labeled_data[~idx_nan.values]
# labeled_data.cluster = labeled_data.cluster.astype("int64")
#
#
# #%% Train classifier
# # Test scalers
# from sklearn.preprocessing import scale
# from sklearn.preprocessing import StandardScaler
#
# data_X = labeled_data.iloc[:, :-1].values.copy()
# y = labeled_data.iloc[:, -1].values.copy()
#
# X = scale(data_X, axis=1)
# mu_ = data_X.mean(axis=1)
# std_ = data_X.std(axis=1)
# scaled_X = (data_X - mu_[:,np.newaxis]) / std_[:, np.newaxis]
# f_x = lambda x: (x - mu_[:,np.newaxis]) / std_[:, np.newaxis]
#
#
# from sklearn.neighbors import KNeighborsClassifier
#
# neigh = KNeighborsClassifier(n_neighbors=1)
# neigh.fit(X, y)  # X == (n_samples, n_features)
# neigh.predict(X)
#
# #%%
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].plot(X.T)
#
# #%% Filter data and classify
# year_range = np.arange(1988, 2021, 1)
# month_range = np.arange(7, 10, 1)
# mixture_years = {}
# labeled_data_years = {}
# for year_ in year_range:
#     print(f"Year: {year_}")
#     idx_1 = knmi_hourly_w_m2.index.year == year_
#     # idx_2 = (knmi_hourly_w_m2.index.month == 7) | (knmi_hourly_w_m2.index.month == 8) | (knmi_hourly_w_m2.index.month == 9)
#     # idx_2 = (knmi_hourly_w_m2.index.month == 3) | (knmi_hourly_w_m2.index.month == 4)
#     for month_ in month_range:
#         idx_2 = (knmi_hourly_w_m2.index.month == month_)
#
#         idx_net = idx_1 & idx_2
#         rlp_year, _ = rlp_irradiance(knmi_hourly_w_m2[idx_net], idx_mask=idx_mask)
#         rlp_year_scaled = scale(rlp_year, axis=1)
#         label_year  = neigh.predict(rlp_year_scaled)
#         class_label, count_label = np.unique(label_year, return_counts=True)
#         mixture_years[str(year_) + "-" + str(month_)] = dict(zip( class_label, count_label / len(label_year) ))
#
#         labeled_data_year = pd.concat([rlp_year,
#                                        pd.DataFrame(label_year, index=rlp_year.index, columns=["cluster"])],
#                                        axis=1)
#         labeled_data_years[str(year_) + "-" + str(month_)] = labeled_data_year
#
# data_frame = pd.DataFrame.from_dict(mixture_years, orient="index")
# data_frame.columns = ["Cloudy", "Sunny", "Dark"]
#
# #%%
# YEAR = "1990-7"
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax_ = ax.flatten()
#
# for ax_i, cluster_ in zip(ax_, [0,1,2]):
#     idx_cluster = labeled_data_years[YEAR]["cluster"] == cluster_
#     ax_i.plot(labeled_data_years[YEAR][idx_cluster].values.T)
#
#
# #%%
# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "browser"
# fig = px.scatter_ternary(data_frame, a="Cloudy", b="Dark", c="Sunny")
# fig.show()
