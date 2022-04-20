"""
This script tests the
After irradiance clustering, this script creates a k-means classifier to label the data of the irradiance
for days from 2011 - 2021

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from core.clustering import (ClusteringRLP,
                             plot_scores)
from core.figure_utils import set_figure_art
from core.copula import EllipticalCopula
from pvlib.location import Location
from scipy.stats import multivariate_normal, norm, ks_2samp, wasserstein_distance, linregress
from scipy import interpolate
import seaborn as sns
import ot
from tqdm import tqdm, trange


def helper_plot(data_values: np.ndarray, ci_daily: np.ndarray, ax):
    norm_individual = mpl.colors.Normalize(vmin=0, vmax=1)
    for data_plot_, ci_daily_ in zip(data_values, ci_daily):
        ax.plot(data_plot_.ravel(), linewidth=0.3, marker='.', markersize=2,
                  markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
                  color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))

def wasserstein_distance_d(x1: np.array, x2: np.ndarray, factor: float = 1):
    """
    Computes Wasserstein metric (Earth Mover's distance - EMD) as a optimal transportation problem.
    x1 is a sample from the one distribution x2 is a sample from another distribution.

    Parameters:
    ----------
        x1: np.array of dimension (n1,d) n1: samples d: dimension
        x2: np.array of dimension (n1,d) n1: samples d: dimension


    Returns:
    --------
        W_d: Wasserstein distance metric which is the loss function of the optimization problem

    """
    assert x1.shape[1] == x2.shape[1], "Both samples must have the same dimensions (variables)"

    m = ot.dist(x1=(x1 / factor), x2=(x2 / factor))
    wass_dist = ot.emd2(a=[], b=[], M=m)  # exact linear program

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
    mean = data.mean(axis=1)
    std_dev = data.std(axis=1)
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
            samples_per_cluster_ = sample_models(cluster_models["models"],
                                                 n_samples=cluster_models["n_profiles_original"])
            samples_per_cluster_["Original"] = cluster_models["measured_data"]
            samples_per_cluster[cluster_] = samples_per_cluster_

        else:
            samples_per_cluster_ = sample_models(cluster_models["models"],
                                                 n_samples=n_samples)
            samples_per_cluster_["Original"] = cluster_models["measured_data"]
            samples_per_cluster[cluster_] = samples_per_cluster_

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
            copula_samples = models_dict[model_type].sample(n_samples, drop_inf=True)
            samples_dict[model_type] = copula_samples

        elif model_type == "MVG":
            multivariate_normal_samples = models_dict[model_type]["model"].rvs(n_samples)
            mvg_samples = np.matmul(multivariate_normal_samples,
                                    np.diag(models_dict[model_type]["std"])) + models_dict[model_type]["mean"]
            mvg_samples = mvg_samples.T

            if clear_negative_ci_indexes:
                mvg_samples[mvg_samples < 0] = 0.0

            samples_dict[model_type] = mvg_samples

        elif model_type == "Normal":
            normal_samples = []
            for normal_model in models_dict[model_type]:
                normal_samples.append(normal_model.rvs(size=n_samples))
            normal_samples = np.array(normal_samples)

            if clear_negative_ci_indexes:
                normal_samples[normal_samples < 0.0] = 0.0  # Clean negative indexes

            samples_dict[model_type] = normal_samples

        else:
            raise ValueError

    return samples_dict

def sub_sample_dataset(samples_dict: dict, n_subsamples: int = 100):
    "Reduce the number of samples per model in the samples_dict to a number of n_sub-samples"

    samples_dict_copy = samples_dict.copy()
    samples_dict_copy.pop("Original")  # Data set used to create the models. This is not sub-sampled.
    sub_samples_dict = {}
    for model_name, samples_model in samples_dict_copy.items():
        number_of_columns = samples_model.shape[1]  # Columns are samples
        random_indices = np.random.choice(number_of_columns, size=n_subsamples, replace=False)
        random_samples = samples_model[:, random_indices]
        sub_samples_dict[model_name] = random_samples

    sub_samples_dict["Original"] = samples_dict["Original"].copy()  # Re-atach the original data

    return sub_samples_dict

def sub_sample_clusters(samples_per_cluster: dict, n_subsamples: int = None):
    """
    Reduce the number of samples per model AT CLUSTER LEVEL.
    if n_subsamples is not provides, is assumed that the sampled dataset should be reduced to the size of the original
    dataset used to create the models
    """
    sub_samples_cluster_dict = {}
    for cluster_, samples_dict in samples_per_cluster.items():
        if n_subsamples is None:
            n_subsamples_ = samples_dict["Original"].shape[1]
        else:
            n_subsamples_ = n_subsamples

        sub_sampled_dataset = sub_sample_dataset(samples_dict, n_subsamples=n_subsamples_)
        sub_samples_cluster_dict[cluster_] = sub_sampled_dataset

    return sub_samples_cluster_dict

def pivot_dataframe(work_frame: pd.DataFrame) -> pd.DataFrame:
    "Pivots data frame so the rows are day profiles and columns are time steps of the profile."

    work_frame_copy = work_frame.copy()
    work_frame_copy["day_of_year"] = work_frame_copy.index.dayofyear
    work_frame_copy['quarterIndex'] = work_frame_copy.index.hour * 100 + work_frame_copy.index.minute
    pivoted_frame = work_frame_copy.pivot(index='quarterIndex', columns='day_of_year')
    pivoted_frame.columns = pivoted_frame.columns.droplevel()
    pivoted_frame_ = pivoted_frame.transpose()

    return pivoted_frame_

def process_profiles_from_samples(samples_per_cluster: dict, day_mapping: dict, clustered_days: pd.DataFrame):
    """
    Process the CSI samples from the probabilistic models.

    Transforms sampled data from Clear-Sky-Index (CSI) numbers to (W/m^2) using the solar model (GHI) values.
    Save the profiles at different conversion stages.

    Parameters:
    -----------
        samples_per_cluster: dict: It has the following structure:
                samples_per_cluster[CLUSTER][MODEL] -> np.ndarray (d x n1) (d: dimensions n1: samples (profiles))

        day_mapping: dict: Dictionary with all the data from the solar model and original measured profiles.
                For more information about this dictionary check the day_processing_mapper method.

        clustered_days: pd.DataFrame: Contains the day of the year and the labels created by the clustering algorithm.

    """

    profiles_per_cluster = {}
    n_clusters = len(samples_per_cluster)
    for cluster_ in range(n_clusters):
        idx_cluster = clustered_days["cluster"] == cluster_
        days_cluster = clustered_days[idx_cluster].index
        model_profile = {}
        for model_name, samples_model in samples_per_cluster[cluster_].items():
            profile_list = []
            profile_csi_list = []
            profile_y_prime_list = []
            daily_csi_list = []
            profile_y_list = []
            profile_y_solar_model_list = []
            profile_y_prime_solar_model_list = []

            profile_k_t_list = []

            for ii, day_ in enumerate(days_cluster):
                if model_name != "Original":  # Process from CSI to Watts/m^2
                    # Transform CSI to Watts/m^2  (y_prime is irradiance data from the solar model)
                    y_hat_prime = day_mapping[day_]["y_prime"] * samples_model[:, ii]
                    # De-stretch x-time
                    f_hat = interpolate.interp1d(day_mapping[day_]["x_prime"], y_hat_prime)
                    y_hat = f_hat(day_mapping[day_]["x_"])
                    # Build the full profile with night times
                    profile_created = np.zeros(day_mapping[day_]["idx"].shape[0])
                    profile_created[day_mapping[day_]["idx"]] = y_hat

                    k_t = y_hat / day_mapping[day_]["y_"]

                    profile_list.append(profile_created)
                    profile_csi_list.append(samples_model[:, ii])
                    profile_y_prime_list.append(y_hat_prime)
                    daily_csi_list.append(y_hat_prime.mean() / day_mapping[day_]["GHI_day"])
                    profile_y_list.append(y_hat)

                else: # Attach original data to compute the prob distance metrics
                    k_t = day_mapping[day_]["y_real"] / day_mapping[day_]["y_"]

                    profile_list.append(day_mapping[day_]["q_t_all"])
                    profile_csi_list.append(day_mapping[day_]["y_real_prime_ci"])
                    profile_y_prime_list.append(day_mapping[day_]["y_real_prime"])
                    daily_csi_list.append(day_mapping[day_]["CSI_day"])
                    profile_y_list.append(day_mapping[day_]["y_real"])

                profile_y_solar_model_list.append(day_mapping[day_]["y_"])
                profile_y_prime_solar_model_list.append(day_mapping[day_]["y_prime"])

                profile_k_t_list.append(k_t)

            y_diff = [np.diff(profile) for profile in profile_y_list]
            k_t_diff = [np.diff(profile) for profile in profile_k_t_list]
            y_solar_diff = [np.diff(profile) for profile in profile_y_solar_model_list]


            # TODO: Here I must assemble the SOLAR metrics, so in the outside you calculate the prob. distances


            model_profile[model_name] = {"profile": np.array(profile_list),  # Whole profile includes night time
                                         "ci": np.array(profile_csi_list),
                                         "ci_diff": np.diff(np.array(profile_csi_list), axis=1),  # Slopes of the CSI
                                         "y_": profile_y_list,  # Daylight time NOT stretched (also hat)
                                         "y_diff": y_diff,  # \delta GHI
                                         "y_solar": profile_y_solar_model_list, # Daylight time NOT stretched from SOLAR MODEL
                                         "y_solar_diff": y_solar_diff, # \delta y_solar
                                         "k_t": profile_k_t_list, # Instantaneous clear sky index NOT stretched (also hat)
                                         "k_t_diff": k_t_diff, # \delta k_t NOT stretched
                                         "y_prime": np.array(profile_y_prime_list), # Daylight time stretched (also hat)
                                         "y_prime_diff": np.diff(np.array(profile_y_prime_list), axis=1),  # Slopes of the created profiles
                                         "y_prime_solar": np.array(profile_y_prime_solar_model_list), # Daylight time stretched SOLAR MODEL
                                         "CSI_day": np.array(daily_csi_list)}

        profiles_per_cluster[cluster_] = model_profile

    return profiles_per_cluster

def day_processing_mapper(data_aligned_sliced: pd.DataFrame, IRR_THRESHOLD: int):
    """
    Process the measured and solar irradiance model data per day.
    Stretch x-axis and normalize the measured data using the SOLAR MODEL. The normalized metric is the
    Clear-Sky-Index (CSI)

    Guide: (x_, y_)          : Solar irradiance filtered by sunlight.
           (x_prime, y_prime):  Solar irradiance stretched in the x-axis (This is done using interpolation)

    Parameters:
    -----------
    data_aligned_sliced: pd.DataFrame: Contains the measured data ("qg"), and solar irradiance models. By default, the
            solar irradiance model used is "ghi_haurwitz".

    IRR_THRESHOLD: int: Value of irradiance in [W/m^2] which is considered to be the starting and ending of sunlight
            times. This is used to filter the profile to sunlight times.

    Returns:
    --------
     day_mapping: dict: Contains all the information necessary to do the conversion of the samples. It has the
                following structure:

                day_mapping[DAY][INFO] -> multiple formats.

                Where:
                    DAY: day of the year (It has only the range of days from the original dataset)
                    INFO:
                        "max_q": float:  Maximum irradiance per day in the SOLAR MODEL [W/m^2]
                        "idx": np.ndarray(bool):  Indices of time steps where is solar irradiance. The minimum solar
                                            irradiance was managed by the global variable THRESHOLD.
                        "f": scipy.interpolate: Function which associates the real (x_) time steps and real irradiance
                                            values denoted by (y_).
                        "x_": np.ndarray: x-axis with the measured time steps where there is solar irradiance
                        "y_": np.ndarray: y-values with the SOLAR MODEL irradiance values [W/m^2] (ONLY DAYLIGHT)
                        "x_prime":  np.ndarray: x-axis-stretched to the maximum number of time steps with solar
                                            irradiance during the year.
                        "y_prime":  np.ndarray: y-values-stretched or interpolated of irradiance values from the
                                            SOLAR MODEL!! [W/m^2] This has values of ONLY DAYLIGHT.
                        "GHI_t_all": np.ndarray: This is the irradiance values of the whole SOLAR MODEL profile which
                                            includes NIGHT times and DAYLIGHT. [W/m^2]
                        "GHI_day": float: Mean value of all the irradiance values during the day (SOLAR MODEL). [W/m^2]

                        -------------------------------------------------------------------------------
                        For some days you will have additional field, ONLY if the day has measured data.
                        -------------------------------------------------------------------------------
                        "y_real": np.ndarray: y-values with the MEASURED irradiance values [W/m^2] (ONLY DAYLIGHT)
                        "y_real_prime":  np.array: y-values-stretched or interpolated of irradiance values from the
                                            MEASURED DATA!! [W/m^2] This has values of ONLY DAYLIGHT.
                        "y_real_prime_ci": np.array: profile of the Clear-Sky-Index (CSI) for the measured profile.
                        "CSI_day": float: This is the ratio between the average measured data and average
                                            solar model irradiance.
                        "q_t_all": np.ndarray: This is the irradiance values of the whole MEASURED profile which includes
                                            NIGHT times and DAYLIGHT. [W/m^2]

    """


    irradiance_solar_model = pivot_dataframe(data_aligned_sliced[["ghi_haurwitz"]])
    max_sunlight_time = (irradiance_solar_model > IRR_THRESHOLD).sum(axis=1).max()
    irradiance_days_real = pivot_dataframe(data_aligned_sliced[["qg"]])
    irradiance_days_real_clean = irradiance_days_real.dropna(axis=0)

    day_mapping = {}
    for day_ in irradiance_solar_model.index:
        max_irradiance = irradiance_solar_model.loc[day_].max()
        idx = irradiance_solar_model.loc[day_] > IRR_THRESHOLD
        y_ = irradiance_solar_model.loc[day_][idx].values
        x_ = np.linspace(1, max_sunlight_time, idx.sum())
        f = interpolate.interp1d(x_, y_, fill_value='extrapolate')
        x_prime = np.linspace(1, max_sunlight_time, max_sunlight_time)
        y_prime = f(x_prime)
        # Data from the solar model
        day_mapping[day_] = {"max_q": max_irradiance,
                             "idx": idx.values,
                             "f": f,
                             "x_": x_,  # Daylight time
                             "y_": y_,  # Daylight time [W/m^2]
                             "x_prime": x_prime,  # Daylight time stretched
                             "y_prime": y_prime,
                             "GHI_t_all": irradiance_solar_model.loc[day_].values,
                             # Irradiance model pivoted (with night)
                             "GHI_day": y_.mean()}

        if day_ in irradiance_days_real_clean.index:
            idx_day = day_mapping[day_]["idx"]
            y_real = irradiance_days_real_clean.loc[day_][idx_day].values
            x_real = np.linspace(1, max_sunlight_time, idx_day.sum())
            f_real = interpolate.interp1d(x_real, y_real, fill_value='extrapolate')
            x_real_prime = np.linspace(1, max_sunlight_time, max_sunlight_time)  # new x-values
            y_real_prime = f_real(x_real_prime)  # Stretched values in x

            day_mapping[day_]["y_real"] = y_real
            day_mapping[day_]["y_real_prime"] = y_real_prime  # IMPORTANT! SAVED FOR STATISTICAL COMPARISONS
            day_mapping[day_]["y_real_prime_scaled"] = y_real_prime / day_mapping[day_]["max_q"]
            day_mapping[day_]["y_real_prime_ci"] = y_real_prime / day_mapping[day_]["y_prime"]
            day_mapping[day_]["CSI_day"] = y_real.mean() / day_mapping[day_]["GHI_day"]
            day_mapping[day_]["q_t_all"] = irradiance_days_real_clean.loc[day_]  # Irradiance model pivoted (with night)

            # Compute solar metrics:
            if data_aligned_sliced.index.freq == "30T":
                N_MAX_FRACTAL_DIMENSION = 2
            else:
                N_MAX_FRACTAL_DIMENSION = 6

            FD, reg_fd, x_fd, y_fd = fractal_dimension(y_real, N_MAX=N_MAX_FRACTAL_DIMENSION)
            fractal_dimension_metrics = {"FD_value": FD,
                                         "reg_model": reg_fd,
                                         "x_fd": x_fd,
                                         "y_fd": y_fd}

            variability_index_value = variability_index(y_real, y_)

            day_mapping[day_]["metrics"] = {"FD": fractal_dimension_metrics,
                                            "VI": variability_index_value}

    return day_mapping

def compute_metrics(profiles_per_cluster, metric_to_compute):
    """
    Compute the multiple metrics to assess the quality of the profiles

    Solar indexes:
    1. Standard Deviation of Increments (SDI)
    2. Mean of absolute GHI ramps (MI)
    3. Variability Index (VI)

    Distributions:
    1. Clear-Sky-Index (CSI)  = "ci"
    2. Delta-CSI (DK_t) = "y_prime_diff"
    3. Global Horizontal Irradiance (GHI) = "y_prime"

    Each ECDF of each metric is compared against the original dataset. The following probability distance metrics are
    used:
    1. Kolmogorov–Smirnov distance (KS)
    2. 1-D Wasserstein Distance
    3. p-D Wasserstein Distance (Earth Mover's distance) (EMD).

    """

    raise NotImplementedError

def compute_prob_distance_metrics(original_data: np.ndarray,
                                  sampled_data: np.ndarray,
                                  multivariate: bool = True,
                                  factor: float = None):
    """
        Compute 1-D Wasserstein distance metric and p-values for each one of the samples in the samples dictionary

        Parameters:
        -----------
            original_data: np.array of dimension (d, n1) d: dimension n1: samples
            sampled_data: np.array of dimension (d, n2) d: dimension n2: samples

    """
    ks_metric, p_value = ks_2samp(original_data.flatten(), sampled_data.flatten())  # 1-D
    wd_metric = wasserstein_distance(original_data.flatten(), sampled_data.flatten())  # 1-D
    if factor is not None:
        factor_ = factor
    else:
        factor_ = 1

    if multivariate:
        emd_metric = wasserstein_distance_d(original_data.T/ factor_, sampled_data.T / factor_)  # Multi dimensional
    else:
        emd_metric = np.nan

    prob_distance_metrics = {"KS": ks_metric,
                             "KS-p": p_value,
                             "WD": wd_metric,
                             "EMD": emd_metric}

    return prob_distance_metrics

def variability_index(delta_ghi: list, delta_ghi_solar_model: list, delta_t: int = 10):


    if (
            (isinstance(delta_ghi, np.ndarray) and isinstance(delta_ghi_solar_model, np.ndarray)) and
            ((len(delta_ghi.shape) == 1) and (len(delta_ghi_solar_model.shape) == 1))
    ):
        delta_ghi = delta_ghi[np.newaxis, :]
        delta_ghi_solar_model = delta_ghi_solar_model[np.newaxis, :]

    assert len(delta_ghi) == len(delta_ghi_solar_model), "Both datasets should have the same number of profiles"

    vi = []
    for delta_t_ghi_profile, delta_t_ghi_model in zip(delta_ghi, delta_ghi_solar_model):
        vi.append(
                    np.sum(np.sqrt(delta_t_ghi_profile ** 2 + delta_t ** 2))
                    / np.sum(np.sqrt(delta_t_ghi_model ** 2 + delta_t ** 2))
                  )
    vi = np.array(vi)

    return vi

def standard_deviation_index(delta_ci):
    """
    Compute standard deviation index (SDI) from the delta clear-sky-index (\delta CSI)

    Parameters:
    ----------
        delta_ci: iterable: if nd.array (n1 x d)  n1: samples (a.k.a. daylight profiles) d: dimensions (variables)
                            if list, then len(delta_ci) == # profiles, and each profile can have different number of
                            dimensions or time steps
    """

    if isinstance(delta_ci, np.ndarray) and (len(delta_ci.shape) == 1): # (1-D) Vector, Meaning one profile only
        delta_ci = delta_ci[np.newaxis, :].copy()

    sdi_list = []
    mi_list = []
    for profile in delta_ci:
        delta_ci_abs_ = np.abs(profile)
        MI_ = delta_ci_abs_.mean()
        sdi = np.sqrt(np.sum((delta_ci_abs_ - MI_) ** 2) / (len(delta_ci_abs_) - 1))

        sdi_list.append(sdi)
        mi_list.append(MI_)

    sdi_array = np.array(sdi_list)
    mi_array = np.array(mi_list)

    return sdi_array, mi_array

def fractal_dimension(profiles, N_MAX: int = 6):
    """
    Computes fractal dimension (FD) for the sunlight times of a irradiance profile.
    This is the implementation of paper [1]

    [1] S. Harrouni, A. Guessoum, and A. Maafi, “Classification of daily solar irradiation by fractional analysis of
        10-min-means of solar irradiance,” Theor. Appl. Climatol., vol. 80, no. 1, pp. 27–36, Feb. 2005.

    Parameters:
    ----------
        profiles: np.ndarray or list: time series profiles to process. In the case of numpy array, rows are samples and
            columns are the time steps i.e. (n_samples x Time_steps(sunlight)).
            In the case of a list, each item is the profile to process.

        N_MAX: int: Number of different numbers of \Delta \tau that will be processed.
            i.e. \Delta \tau_max = N/N_MAX, where N is the number of points in the profile.

            For more information check page 30 on paper [1] "Classification of daily solar irradiation by fractional
            analysis of 10-min-means of solar irradiance", S.Harrouni, A.Gessoum, and A. Maafi.

    Returns:
    --------
        FD_list: np.ndarray (1-D) : Fractal dimension (FD) value, for each one of the profiles in the parameters input.
            Each item in the vector corresponds for de FD of each profile.

        regression_fd_list: linregress (scipy.stats): Linear regression model, on which the slope is the FD

         x_axis_fd_list: np.ndarray : x-values for the linear regression model. Each vector of the of the matrix is a
            series of numbers that corresponds to the equation: np.log(1 / tau)

         y_axis_fd_list_list: np.ndarray : x-values for the linear regression model. Each vector of the of the matrix
            is a series of numbers that corresponds to the equation: np.log(np.sum(energy) / tau ** 2)

    """


    if isinstance(profiles, np.ndarray) and len(profiles.shape) == 1:  # (1-D) Vector
        profiles = profiles[np.newaxis, :].copy()

    FD_list = []
    regression_fd_list = []
    x_axis_fd_list = []
    y_axis_fd_list_list = []

    for profile_ in profiles:
        n_clear = len(profile_)
        MAX_ETA = np.floor(n_clear / N_MAX).astype(int)

        x_axis_fd = []
        y_axis_fd = []
        for tau in range(1, MAX_ETA + 1):
            q, mod = divmod(n_clear, tau)
            idx = list(range(0, n_clear, tau))

            if mod != 0:
                idx = idx + [idx[-1] + (mod-1)]

            ii = idx[:-1]
            jj = idx[1:]

            energy = []
            for ii_, jj_ in zip(ii, jj):
                energy.append(tau * (np.abs(profile_[jj_] - profile_[ii_])))

            y = np.log(np.sum(energy) / tau ** 2)
            x = np.log(1 / tau)

            x_axis_fd.append(x)
            y_axis_fd.append(y)
        x_axis_fd = np.array(x_axis_fd)
        y_axis_fd = np.array(y_axis_fd)

        regression_fd = linregress(x_axis_fd, y_axis_fd)
        FD = regression_fd.slope

        FD_list.append(FD)
        regression_fd_list.append(regression_fd)
        x_axis_fd_list.append(x_axis_fd)
        y_axis_fd_list_list.append(y_axis_fd)

    FD_list = np.array(FD_list)
    # regression_fd_list = np.array(regression_fd_list)

    return FD_list, regression_fd_list, x_axis_fd_list, y_axis_fd_list_list

def generate_profiles(pi_mixture, mvt_models: list, day_mapping, data_aligned_sliced, START_DATE, END_DATE):
    # mvt_models = []
    # pi_mixture = []
    # for cluster_ in range(n_clusters):
    #     mvt_models.append(models_per_cluster[cluster_]["models"]["MVT"])
    #     pi_mixture.append(models_per_cluster[cluster_]["n_profiles_original"])
    # pi_mixture = np.array(pi_mixture) / np.array(pi_mixture).sum()
    # pi_mixture = np.array([0.0, 1.0, 0.0])
    # pi_mixture = np.array([0.0, 0.0, 1.0])
    # TODO: pi_mixture could be different to sample the model

    n_samples = len(pd.date_range(start=START_DATE, end=END_DATE, freq="D"))
    sample_cluster_labels = np.random.choice(list(range(n_clusters)), n_samples, p=pi_mixture)
    label, counts = np.unique(sample_cluster_labels, return_counts=True)

    # %% Align the labels with solar model
    irr_solar_model = pivot_dataframe(data_aligned_sliced[["ghi_haurwitz"]])
    irr_solar_model_clustered = pd.concat([irr_solar_model,
                                           pd.DataFrame(sample_cluster_labels,
                                                        columns=["cluster"],
                                                        index=irr_solar_model.index)],
                                          axis=1)

    # %% Create the samples from the model in the CI domain and attach the day of the year for further conversion to W/m^2
    sample_set = []
    label_vector = []
    for label_, n_samples_ in zip(label, counts):
        samples_irradiance_full = mvt_models[label_].sample(n_samples=n_samples_, drop_inf=True)
        sample_set.append(pd.DataFrame(samples_irradiance_full.T))
        label_vector += [label_] * n_samples_

    label_vector_np = np.array(label_vector)
    sample_set_concat = pd.concat(sample_set, axis=0, ignore_index=True)
    sample_set_concat["cluster"] = label_vector_np
    # Shuffle, otherwise the sunny, cloudy and dark days will be next to each other
    sample_set_concat_shuffled = sample_set_concat.sample(frac=1).reset_index(drop=True)
    sample_set_concat_shuffled.index = irr_solar_model_clustered.index.copy()

    # %% Convert to W/m^2
    profile_list = []
    CSI_day = []
    for day_, ci_ in sample_set_concat_shuffled.iterrows():
        y_hat_prime = day_mapping[day_]["y_prime"] * ci_.drop("cluster").values
        # De-stretch x-time
        f_hat = interpolate.interp1d(day_mapping[day_]["x_prime"], y_hat_prime)
        y_hat = f_hat(day_mapping[day_]["x_"])
        # Build the full profile with night times
        profile_created = np.zeros(day_mapping[day_]["idx"].shape[0])
        profile_created[day_mapping[day_]["idx"]] = y_hat

        k_t = y_hat.mean() / day_mapping[day_]["y_"].mean()

        profile_list.append(pd.DataFrame(profile_created).T)
        CSI_day.append(k_t)

    profile_list_concat = pd.concat(profile_list, ignore_index=True)
    profile_list_concat.index = irr_solar_model.index.copy()
    profile_list_concat.columns = irr_solar_model.columns.copy()
    profile_list_concat["CSI_day"] = CSI_day


    # %%
    YEAR = 2021

    def get_part(x, part="minutes"):
        hour, minutes = divmod(x, 100)
        if part == "minutes":
            return minutes
        else:
            return hour

    profile_list_concat_new = profile_list_concat.drop(columns=["CSI_day"]).reset_index()
    key_quarterIndex = profile_list_concat.drop(columns=["CSI_day"]).columns.unique().values
    minutes_dict = dict(zip(key_quarterIndex, [get_part(x_, "minutes") for x_ in key_quarterIndex]))
    hours_dict = dict(zip(key_quarterIndex, [get_part(x_, "hours") for x_ in key_quarterIndex]))
    profile_series = pd.melt(profile_list_concat_new,
                             id_vars=['day_of_year'],
                             value_name='qg_hat').sort_values(by=['day_of_year', 'quarterIndex'])
    profile_series["hours"] = profile_series["quarterIndex"].apply(lambda x: hours_dict[x])
    profile_series["minutes"] = profile_series["quarterIndex"].apply(lambda x: minutes_dict[x])
    profile_series["seconds"] = ["00"] * len(profile_series["quarterIndex"])
    profile_series["year"] = [YEAR] * len(profile_series["quarterIndex"])
    profile_series["date"] = pd.to_datetime(profile_series["year"] * 1000 + profile_series["day_of_year"],
                                            format="%Y%j")
    # Assemble
    profile_series["datetime"] = profile_series["date"].astype(str) + " " + profile_series["hours"].astype(str) + ":" + \
                                 profile_series["minutes"].astype(str) + ":" + profile_series["seconds"].astype(str)
    profile_series["datetime"] = pd.to_datetime(profile_series["datetime"])
    profile_series = profile_series[["datetime", "qg_hat"]].set_index("datetime")
    profile_series = profile_series.tz_localize("UTC")

    # profile_series_concat = pd.concat([profile_series, data_aligned_sliced[["ghi_haurwitz", "qg"]]], axis=1)

    return profile_series, profile_list_concat



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
longitude = 6.8908745111116  # Offset to align sunrise and sunset (equivalent of shifting the time)
elevation = 33.2232  # Meters

twente = Location(latitude=latitude, longitude=longitude, tz='UTC', altitude=elevation, name='Twente')
times = pd.date_range(start='2021-01-01', end='2021-12-31 23:50:00', freq='10T', tz=twente.tz)
cs_simplified_solis = twente.get_clearsky(times, model="simplified_solis")
cs_haurwitz = twente.get_clearsky(times, model="haurwitz")
cs_ineichen = twente.get_clearsky(times, model="ineichen")

cs_simplified_solis.rename(columns={"ghi": "ghi_solis"}, inplace=True)
cs_haurwitz.rename(columns={"ghi": "ghi_haurwitz"}, inplace=True)  # Best model
cs_ineichen.rename(columns={"ghi": "ghi_ineichen"}, inplace=True)

#%%
data_aligned = pd.concat([knmi_10min_w_m2[["qg"]], cs_haurwitz, cs_simplified_solis[["ghi_solis"]], cs_ineichen[["ghi_ineichen"]]], axis=1)
data_aligned.resample(RESAMPLE).mean()[START_DATE:END_DATE].plot()
data_aligned_sliced = data_aligned.resample(RESAMPLE).mean()[START_DATE:END_DATE].copy()
day_mapping = day_processing_mapper(data_aligned_sliced, IRR_THRESHOLD=20)

day_mapping_frame = pd.DataFrame.from_dict(day_mapping).transpose()
data_list = [np.array(day_mapping_frame["GHI_t_all"].to_list()).T,
             np.array(day_mapping_frame["q_t_all"].dropna().to_list()).T,
             np.array(day_mapping_frame["y_prime"].to_list()).T,
             np.array(day_mapping_frame["y_real_prime"].dropna().to_list()).T,
             np.array(day_mapping_frame["y_real_prime_ci"].dropna().to_list()).T,
             np.array(day_mapping_frame["y_real_prime_ci"].dropna().to_list()).T]
csi_measured_profiles = data_list[-1].mean(axis=0)

#%% Visual check that everything was computed correctly
fig, ax = plt.subplots(1, 6, figsize=(20, 3.5))
plt.subplots_adjust(wspace=0.4, left=0.05, right=0.98, bottom=0.15)
ax = ax.flatten()

x_labels = ["Time step"] * 6
y_labels = ["Watts / m^2"] * 4 + ["CSI"] + ["CSI counts"]
titles = ["Global irradiance model", "Measure irradiance", "Global irr. stretch",
          "Measure irr. stretch", "Clear sky index", "Histogram CSI"]

norm_individual = mpl.colors.Normalize(vmin=0, vmax=1)
for ii, (ax_, data_, x_label, y_label, title) in enumerate(zip(ax, data_list, x_labels, y_labels, titles)):
    if ii != 5:
        if ii == 0 or ii == 2:
            ax_.plot(data_, linewidth=0.4, color="grey", marker=".", markersize=0.3)
        else:
            for ii, (data_plot_, ci_daily_) in enumerate(zip(data_.T, csi_measured_profiles)):
                ax_.plot(data_plot_, linewidth=0.3, marker='.', markersize=2,
                        markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
                        color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))
            ax_.set_title("Original load profiles")
            ax_.set_ylabel("W/m${}^2$")
            cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('plasma')), ax=ax_)
            cbar_2.ax.set_ylabel('Daily clear index [$K_d$]')

    else:
        n, bins, patches = ax_.hist(data_.ravel(), 50, density=True, facecolor='g', alpha=0.75)
    ax_.set_xlabel(x_label)
    ax_.set_ylabel(y_label)
    ax_.set_title(title)


#%%  Clustering the dataset based on CI
sunlight_time_real_ci = np.array(day_mapping_frame["y_real_prime_ci"].dropna().to_list())
data_set = sunlight_time_real_ci
cluster_ap = ClusteringRLP()
cluster_ap.cluster_dataset(data_set, plot_solutions=False, end=20, std_scale=False)
score_values_ap = cluster_ap.get_clustering_scores()
plot_scores(score_values_ap["CDI"],
            score_values_ap["MDI"],
            score_values_ap["DBI"],
            score_values_ap["CHI"])
ap_cluster_solutions = cluster_ap.get_cluster_labels()

#%% Merge cluster solutions with data
N_CLUSTERS = 6
ALGORITHM = "Birch"

# Merge cluster labels results with the data
clear_index_day_frame = day_mapping_frame[["CSI_day"]].dropna()
label_frame = pd.DataFrame(ap_cluster_solutions.loc[ALGORITHM, N_CLUSTERS], columns=["cluster"],
                           index=clear_index_day_frame.index)
data_list_raw = [np.array(day_mapping_frame["q_t_all"].dropna().to_list()),  # TODO: Repeated from data list
                 np.array(day_mapping_frame["y_real_prime"].dropna().to_list()),
                 np.array(day_mapping_frame["y_real_prime_scaled"].dropna().to_list()),
                 np.array(day_mapping_frame["y_real_prime_ci"].dropna().to_list())]
data_list_frame = [pd.DataFrame(data_, index=clear_index_day_frame.index) for data_ in data_list_raw]
data_list_concat = [pd.concat([data_, clear_index_day_frame, label_frame], axis=1) for data_ in data_list_frame]

#%% Prepare more data for future steps
irradiance_measured_dataset = data_list_concat[0]
irradiance_measured_y_prime = data_list_concat[1]
labeled_data_year_ci = data_list_concat[3]

#%% Plot original data with the clear index
norm_individual = mpl.colors.Normalize(vmin=0, vmax=1)
fig, ax = plt.subplots(1,1, figsize=(6, 4))
for ii,(_, data_plot_) in enumerate(irradiance_measured_dataset.iterrows()):
    ci_daily_ = data_plot_["CSI_day"]
    ax.plot(data_plot_.drop(["CSI_day", "cluster"]).ravel(), linewidth=0.3, marker='.', markersize=2,
            markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
            color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))
ax.set_title("Original load profiles")
ax.set_ylabel("W/m${}^2$")
cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('plasma')), ax=ax)
cbar_2.ax.set_ylabel('Daily clear index [$K_d$]')

#%%
irr_solar_model = pivot_dataframe(data_aligned_sliced[["ghi_haurwitz"]])
irr_measured = pivot_dataframe(data_aligned_sliced[["qg"]])
z_min, z_max = 0, 1000
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# ax.pcolormesh(X, Y, Z.T, shading='auto', cmap=plt.cm.get_cmap('plasma'), vmin=z_min, vmax=z_max)
b = ax[0].imshow(irr_measured.values.T,  cmap=plt.cm.get_cmap('plasma'), vmin=z_min, vmax=z_max, interpolation='nearest')
c = ax[1].imshow(irr_solar_model.values.T,  cmap=plt.cm.get_cmap('plasma'), vmin=z_min, vmax=z_max, interpolation='none')
ax[0].set_title("Measured data")
ax[1].set_title("Solar model")

ax[1].set_xlabel("Day of year")
ax[1].set_ylabel("Time step")

cbar_1 = plt.colorbar(b, ax=ax[0])
cbar_2 = plt.colorbar(c, ax=ax[1])
cbar_1.ax.set_ylabel('Global Irradiance [W/m${}^2$]')
cbar_2.ax.set_ylabel('Global Irradiance [W/m${}^2$]')


#%% Check clustering results
fig, ax = plt.subplots(4, N_CLUSTERS, figsize=(20, 12))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3)
ax_ = [ax[:, ii].flatten() for ii in range(N_CLUSTERS)]
y_lims = [(0, 1000.0)] * 2 + [(0, 1.5)] + [(0, 2.0)]
y_labels = ["W/m^2"] * 2 + ["Norm"] + ["CSI"]

if N_CLUSTERS < 4:
    x_labels = ["Time step"] * 4
else:
    x_labels = ["Time step"] * N_CLUSTERS

for jj, (ax_i, cluster_) in enumerate(zip(ax_, range(N_CLUSTERS))):  # Columns
    idx_cluster = label_frame["cluster"] == cluster_

    for ii, (ax_row, y_lim_, y_label_, x_label_) in enumerate(zip(ax_i, y_lims, y_labels, x_labels)):  # Rows
        n_days_cluster = data_list_concat[ii][idx_cluster].drop(columns=["CSI_day", "cluster"]).to_numpy().shape[0]
        helper_plot(data_list_concat[ii][idx_cluster].drop(columns=["CSI_day", "cluster"]).to_numpy(),
                    data_list_concat[ii][idx_cluster]["CSI_day"].to_numpy(), ax=ax_row)
        if ii == 0:
            ax_row.set_title(f"Samples: ({n_days_cluster})")
        if jj == 0:
            ax_row.set_ylabel(y_label_)
        if ii == 3:
            ax_row.set_xlabel(x_label_)
        if jj == N_CLUSTERS - 1:
            cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('plasma')), ax=ax_row)
            cbar_2.ax.set_ylabel('Daily clear index [$K_d$]')

        ax_row.set_ylim(y_lim_)

fig.suptitle(f"Algorith: {ALGORITHM}")

#%%
# ======================================================================================================================
# ====================================== SOLAR IRRADIANCE PROPOSAL =====================================================
# ======================================================================================================================
# Solar model
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

set_figure_art()
mpl.rc('text', usetex=False)

n_steps = data_list[0].shape[0]
x_axis = pd.date_range(start="2021-11-01", periods=n_steps, freq=RESAMPLE)

# fig = plt.figure(figsize=(5, 9.5))
# widths = [1]
# heights = [1, 2]
# outer = fig.add_gridspec(2, 1, wspace=0.1, hspace=0.15, left=0.15, bottom=0.05, right=0.85, top=0.97, width_ratios=widths, height_ratios=heights)
# sub_1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.1)  # 15min
# sub_2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1])  # 30min

# fig, ax = plt.subplots(2, 2, figsize=(4, 6))
# ax = ax.flatten()
# plt.subplots_adjust(wspace=0.5, hspace=0.5, right=0.99, top=0.95)

fig = plt.figure(figsize=(3.5, 4))
gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.35, left=0.15, bottom=0.05, right=0.99, top=0.95, width_ratios=[1, 1], height_ratios=[1, 1.5])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

ax1.plot(x_axis, data_list[0], linewidth=0.1, color="green", marker=".", markersize=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax1.set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=9)))
ax1.set_ylim((-10, 1100))
ax1.set_xlabel("Time of day", fontsize=6)
ax1.set_ylabel("[W/m${}^2$]", fontsize=6)
ax1.set_title("(a)")

ax2.plot(data_list[2], linewidth=0.4, color="green", marker=".", markersize=0.3)
ax2.set_ylim((-10, 1100))
ax2.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.set_xlabel("Common time frame - $x^{'}$", fontsize=6)
ax2.set_title("(b)")

csi_measured_profiles = data_list[-1].mean(axis=0)
for data_plot_, ci_daily_ in zip(data_list[1].T, csi_measured_profiles):
    ax3.plot(x_axis, data_plot_, linewidth=0.4, marker='.', markersize=2,
             markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
             color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax3.set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=9)))
ax3.set_ylim((-10, 1100))
ax3.set_xlabel("Time of day", fontsize=6)
ax3.set_ylabel("W/m${}^2$", fontsize=6)
ax3.set_title("(c)")

for data_plot_, ci_daily_ in zip(data_list[3].T, csi_measured_profiles):
    ax4.plot(data_plot_, linewidth=0.4, marker='.', markersize=2,
             markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
             color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))
cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('plasma')),
                      orientation="horizontal",
                      shrink=0.5,
                      pad=0.2,
                      ax=[ax3, ax4])

cbar_2.ax.set_xlabel('Daily clear sky index [$K_d$]')
ax4.set_ylim((-10, 1100))
ax4.set_xlabel("Common time frame - $x^{'}$", fontsize=6)
ax4.yaxis.set_major_formatter(ticker.NullFormatter())
ax4.set_title("(d)")
# plt.tight_layout()
plt.savefig('figures/solar_model_proposal/sampling_load_models.png', dpi=700, bbox_inches='tight')

#%%
fig, ax = plt.subplots(3, 2, figsize=(4, 6))
plt.subplots_adjust(wspace=0.65, hspace=0.65, left=0.1, right=0.99, top=0.93, bottom=0.06)
ax[0,0].axis("off")
ax[2,0].axis("off")
ax_ = [ax[0,1], ax[1,1], ax[2,1]]
titles_list = ["(f)\nCluster 1", "(g)\nCluster 2", "(h)\nCluster 3"]

for data_plot_, ci_daily_ in zip(data_list[4].T, csi_measured_profiles):
    ax[1, 0].plot(data_plot_, linewidth=0.4, marker='.', markersize=2,
             markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
             color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))
ax[1, 0].set_ylim((0, 2))
ax[1, 0].set_title("(e)", fontsize="x-large")
ax[1, 0].set_xlabel("Common time frame - $x^{'}$")
ax[1, 0].set_ylabel("Clear sky index - CSI - $K_t$")

for jj, (ax_i, cluster_, titles_) in enumerate(zip(ax_, range(N_CLUSTERS), titles_list)):  # Columns
    ax_i.set_title(titles_, fontsize="x-large")
    ax_i.set_ylabel("CSI")

    if jj == 2:
        ax_i.set_xlabel("Common time frame - $x^{'}$")
    idx_cluster = label_frame["cluster"] == cluster_
    n_days_cluster = data_list_concat[3][idx_cluster].drop(columns=["CSI_day", "cluster"]).to_numpy().shape[0]
    helper_plot(data_list_concat[3][idx_cluster].drop(columns=["CSI_day", "cluster"]).to_numpy(),
                data_list_concat[3][idx_cluster]["CSI_day"].to_numpy(), ax=ax_i)
    ax_i.set_ylim((0, 2))
plt.savefig('figures/solar_model_proposal/fig_2.png', dpi=700, bbox_inches='tight')

#%%
DAY_TO_PLOT = 150
DAY_TO_PLOT = 170

fig, ax = plt.subplots(1, 2, figsize=(4.5, 2))
plt.subplots_adjust(right=0.99, left=0.14, wspace=0.9, bottom=0.2)
ax[0].plot(day_mapping[DAY_TO_PLOT]["y_real_prime_ci"], linewidth=1.4, marker='.', markersize=2,
             markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(day_mapping[DAY_TO_PLOT]["CSI_day"])),
             color=plt.cm.get_cmap('plasma')(norm_individual(day_mapping[DAY_TO_PLOT]["CSI_day"])))
ax[0].set_ylim((0,2))
ax[0].set_xlabel("Common time frame - $x^{'}$")
ax[0].set_ylabel("CSI")
ax[0].set_title("(i)", fontsize="x-large")

ax[1].plot(day_mapping[DAY_TO_PLOT]["y_prime"], linewidth=1.4, color="green", marker=".", markersize=0.3)
ax[1].set_ylim((-10, 1100))
ax[1].set_xlabel("Common time frame - $x^{'}$")
ax[1].set_ylabel("W/m${}^2$")
ax[1].set_title("(j)", fontsize="x-large")
plt.savefig('figures/solar_model_proposal/fig_3.png', dpi=700, bbox_inches='tight')

#%%
fig, ax = plt.subplots(1, 2, figsize=(4.5, 2))
plt.subplots_adjust(right=0.99, left=0.14, wspace=0.9, bottom=0.2)
ax[0].plot(day_mapping[DAY_TO_PLOT]["y_real_prime"], linewidth=1.4, marker='.', markersize=2,
             markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(day_mapping[DAY_TO_PLOT]["CSI_day"])),
             color=plt.cm.get_cmap('plasma')(norm_individual(day_mapping[DAY_TO_PLOT]["CSI_day"])))
ax[0].set_ylim((-10, 1100))
ax[0].set_xlabel("Common time frame - $x^{'}$")
ax[0].set_ylabel("W/m${}^2$")
ax[0].set_title("(k)", fontsize="x-large")

ax[1].plot(x_axis, day_mapping[DAY_TO_PLOT]["q_t_all"], linewidth=1.4, marker=".", markersize=0.3,
           markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(day_mapping[DAY_TO_PLOT]["CSI_day"])),
           color=plt.cm.get_cmap('plasma')(norm_individual(day_mapping[DAY_TO_PLOT]["CSI_day"])))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax[1].set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=9)))
ax[1].set_ylim((-10, 1100))
ax[1].set_xlabel("Time of day")
ax[1].set_ylabel("W/m${}^2$")
ax[1].set_title("(l)", fontsize="x-large")
plt.savefig('figures/solar_model_proposal/fig_4.png', dpi=700, bbox_inches='tight')


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

#%% Create models on the clear-index-sky domain data
models_per_cluster = {}
for cluster_ in range(N_CLUSTERS):
    idx = labeled_data_year_ci["cluster"] == cluster_
    data_irradiance_ci = labeled_data_year_ci[idx].drop(columns=["CSI_day", "cluster"]).values.T
    models_per_cluster[cluster_] = {"n_profiles_original": np.sum(idx),
                                    "measured_data": data_irradiance_ci,
                                    "models": create_models(data_irradiance_ci)}

#%% Create the profiles using the sampled data from the clear-sky-domain:
samples_per_cluster = sample_models_per_cluster(models_per_cluster, n_samples=4000)

#%%
metric_names = {"VI": "Variability index (VI)",
                "SDI": "Standard Deviation of Increments index (SDI)",
                "MI": "Mean of absolute GHI ramps (MI)",
                "FD": "Fractal dimension",
                "ci": "CSI distribution (k_t)",
                "ci_diff": "delta CSI distribution (\delta k_t)",
                "y_prime_diff": "delta GHI distribution (\delta GHI_t)",
                "y_prime": "Daylight profile (GHI_t)"}

solar_metrics = {"VI", "SDI", "MI", "FD"}  # Requires extra processing before comparing the distributions

fields_to_compute = {"VI", "SDI", "MI", "FD", "ci", "ci_diff", "y_prime_diff", "y_prime"}
# fields_to_compute = {"ci_diff"}

field_metrics_means = []
field_metrics_std = []

for field_to_compute in fields_to_compute:
    n_clusters = len(samples_per_cluster.keys())
    bootstrap_size = 3
    bootstrap_metrics = []
    for _ in trange(bootstrap_size, desc=f"Metric: {metric_names[field_to_compute]}"):
        samples_per_cluster_ = sub_sample_clusters(samples_per_cluster)
        profiles_per_cluster = process_profiles_from_samples(samples_per_cluster_, day_mapping, label_frame)
        metrics_per_cluster = {}
        for cluster_ in range(n_clusters):
            models_names = set(profiles_per_cluster[cluster_].keys())
            metrics_per_model = {}
            if field_to_compute in solar_metrics:
                solar_metrics_per_model = {}
                for model_name in models_names:
                    if field_to_compute == "VI":
                        solar_metrics_per_model[model_name] = variability_index(
                            profiles_per_cluster[cluster_][model_name]["y_diff"],
                            profiles_per_cluster[cluster_][model_name]["y_solar_diff"])
                    elif field_to_compute == "SDI":
                        solar_metrics_per_model[model_name], _ = standard_deviation_index(
                            profiles_per_cluster[cluster_][model_name]["k_t_diff"])
                    elif field_to_compute == "MI":
                        _, solar_metrics_per_model[model_name] = standard_deviation_index(
                            profiles_per_cluster[cluster_][model_name]["k_t_diff"])
                    elif field_to_compute == "FD":
                        FD, _, _, _ = fractal_dimension(profiles_per_cluster[cluster_][model_name]["y_"])
                        solar_metrics_per_model[model_name] = FD

                models_names = models_names - {"Original"}

                for model_name in models_names:
                    if (field_to_compute == "y_prime_diff") or (field_to_compute == "y_prime"):
                        factor_ = 1000
                    else:
                        factor_ = None

                    metrics_per_model[model_name] = compute_prob_distance_metrics(
                        solar_metrics_per_model["Original"].ravel(),
                        solar_metrics_per_model[model_name].ravel(),
                        multivariate=False,
                        factor=factor_)
            else:
                models_names = models_names - {"Original"}

                for model_name in models_names:
                    metrics_per_model[model_name] = compute_prob_distance_metrics(
                        profiles_per_cluster[cluster_]["Original"][field_to_compute].T,
                        profiles_per_cluster[cluster_][model_name][field_to_compute].T,
                        multivariate=True,
                        factor=None)

            metrics_per_cluster[cluster_] = pd.DataFrame.from_dict(metrics_per_model).transpose()

        metrics_per_cluster_frame = pd.concat(metrics_per_cluster, axis=0).transpose()
        metrics_per_cluster_frame.columns.names = {"cluster", "model"}
        metrics_per_cluster_frame.index.name = "metric"
        bootstrap_metrics.append(metrics_per_cluster_frame.transpose())

    bootstrap_metrics_frames = pd.concat(bootstrap_metrics, axis=0, keys=range(bootstrap_size))
    bootstrap_metrics_frames.index.rename({None: "bootstrap", "model": "model", "cluster": "cluster"}, inplace=True)

    # Compute the mean and standard deviations for the bootstrapped statistics
    idx_multi = pd.IndexSlice
    metric_mean_per_cluster = []
    metric_std_per_cluster = []
    for cluster_ in range(n_clusters):
        mean_metrics = bootstrap_metrics_frames.loc[idx_multi[:, cluster_, :]].groupby(level=2).mean()
        std_metrics = bootstrap_metrics_frames.loc[idx_multi[:, cluster_, :]].groupby(level=2).std()
        metric_mean_per_cluster.append(mean_metrics)
        metric_std_per_cluster.append(std_metrics)

    metric_mean_per_cluster_frame = pd.concat(metric_mean_per_cluster, axis=0, keys=range(n_clusters))
    metric_mean_per_cluster_frame.index.rename({None: "cluster"}, inplace=True)

    metric_std_per_cluster_frame = pd.concat(metric_std_per_cluster, axis=0, keys=range(n_clusters))
    metric_std_per_cluster_frame.index.rename({None: "cluster"}, inplace=True)

    # print("Mean")
    # print(metric_mean_per_cluster_frame)
    #
    # print("Std")
    # print(metric_std_per_cluster_frame)

    if field_to_compute in solar_metrics:
        field_metrics_means.append(pd.concat([metric_mean_per_cluster_frame.drop(columns=["KS-p", "EMD"]).transpose()], keys=[field_to_compute], names=['metric']))
    else:
        field_metrics_means.append(pd.concat([metric_mean_per_cluster_frame.drop(columns=["KS-p"]).transpose()], keys=[field_to_compute], names=['metric']))

    field_metrics_std.append(pd.concat([metric_std_per_cluster_frame.drop(columns=["KS-p"]).transpose()], keys=[field_to_compute], names=['metric']))

# pd.concat(field_metrics_means).to_csv("metric_means_KMEANS_3clusters.csv")

#%% Plot CI
fig, ax = plt.subplots(6, N_CLUSTERS, figsize=(15, 10))
# plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3)
ax_ = [ax[:, ii].flatten() for ii in range(N_CLUSTERS)]
y_lims = [(0, 2.0)] * 4 + [(0, 1)] * 2
y_labels = ["CSI"] * 4 + ["ECDF"] + ["\delta Q"]
if N_CLUSTERS < 7:
    x_labels = ["Time step"] * 6
else:
    x_labels = ["Time step"] * N_CLUSTERS
colors = ["grey", "C0", "C1", "C2"]
model_names = ["Original", "MVT", "Normal", "MVG"]

for jj, (ax_i, cluster_) in enumerate(zip(ax_, range(N_CLUSTERS))):  # Columns
    for ii, (ax_row, y_lim_, y_label_, x_label_) in enumerate(zip(ax_i, y_lims, y_labels, x_labels)):  # Rows
        if ii < 4:
            ax_row.plot(profiles_per_cluster[cluster_][model_names[ii]]["ci"].T, color=colors[ii], linewidth=0.4, marker='.', markersize=2)
            ax_row.set_ylim(y_lim_)

            if ii ==0:
                ax_row.set_title(f"Samples: {profiles_per_cluster[cluster_][model_names[ii]]['ci'].T.shape[1]}")
        elif ii == 4:
            for model_name_, colors_ in zip(model_names, colors):
                sns.ecdfplot(data=profiles_per_cluster[cluster_][model_name_]["ci"].T.ravel(), ax=ax_row, color=colors_, stat="proportion")
                ax_row.set_ylabel("")
        elif ii == 5:
            for model_name_, colors_ in zip(model_names, colors):
                sns.kdeplot(data=np.diff(profiles_per_cluster[cluster_][model_name_]["ci"].T, axis=0).ravel(), ax=ax_row,
                             color=colors_)
                ax_row.set_ylabel("")
                ax_row.set_xlim((-0.5, 0.5))

fig.suptitle("Sampling from the models")

#%% Plot whole profile
fig, ax = plt.subplots(6, N_CLUSTERS, figsize=(15, 10))
# plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3)
ax_ = [ax[:, ii].flatten() for ii in range(N_CLUSTERS)]
y_lims = [(0, 1000)] * 4 + [(0, 1)] * 2
y_labels = ["CSI"] * 4 + ["ECDF"] + ["\delta Q"]
if N_CLUSTERS < 7:
    x_labels = ["Time step"] * 6
else:
    x_labels = ["Time step"] * N_CLUSTERS
colors = ["grey", "C0", "C1", "C2"]
model_names = ["Original", "MVT", "Normal", "MVG"]

for jj, (ax_i, cluster_) in enumerate(zip(ax_, range(N_CLUSTERS))):  # Columns
    for ii, (ax_row, y_lim_, y_label_, x_label_) in enumerate(zip(ax_i, y_lims, y_labels, x_labels)):  # Rows
        if ii < 4:
            ax_row.plot(profiles_per_cluster[cluster_][model_names[ii]]["profile"].T, color=colors[ii], linewidth=0.4, marker='.', markersize=2)
            ax_row.set_ylim(y_lim_)

            if ii ==0:
                ax_row.set_title(f"Samples: {profiles_per_cluster[cluster_][model_names[ii]]['profile'].T.shape[1]}")
        elif ii == 4:
            for model_name_, colors_ in zip(model_names, colors):
                sns.ecdfplot(data=profiles_per_cluster[cluster_][model_name_]["y_prime"].T.ravel(), ax=ax_row, color=colors_, stat="proportion")
                ax_row.set_ylabel("")
        elif ii == 5:
            for model_name_, colors_ in zip(model_names, colors):
                sns.kdeplot(data=np.diff(profiles_per_cluster[cluster_][model_name_]["y_prime"].T, axis=0).ravel(), ax=ax_row,
                             color=colors_)
                ax_row.set_ylabel("")
                ax_row.set_xlim((-500, 500))

fig.suptitle("Sampling from the models")

#%% Use birch and 10 min resolution an activate the commented code
CLEAR_DAY = 20  # Birch 6 clusters
CLOUDY_DAY = 13

profile_clear = profiles_per_cluster[3]["Original"]["y_"][20]  # Clear day
profile_cloudy = profiles_per_cluster[1]["Original"]["y_"][13]  # Cloudy  #10

profile_clear = profiles_per_cluster[3]["Original"]["y_"]
profile_cloudy = profiles_per_cluster[1]["Original"]["y_"]

profile_ = profile_cloudy
fd_clear, rg_fd_clear, x_t_clear, y_t_clear = fractal_dimension(profile_clear, N_MAX=4)
fd_cloudy, rg_fd_cloudy, x_t_cloudy, y_t_cloudy = fractal_dimension(profile_cloudy, N_MAX=4)

fd_clear = fd_clear[CLEAR_DAY]
rg_fd_clear = rg_fd_clear[CLEAR_DAY]
x_t_clear = x_t_clear[CLEAR_DAY]
y_t_clear = y_t_clear[CLEAR_DAY]

fd_cloudy = fd_cloudy[CLOUDY_DAY]
rg_fd_cloudy = rg_fd_cloudy[CLOUDY_DAY]
x_t_cloudy = x_t_cloudy[CLOUDY_DAY]
y_t_cloudy = y_t_cloudy[CLOUDY_DAY]


#%% Fractal dimension example
import matplotlib.patches as patches
from matplotlib.ticker import NullFormatter

mpl.rc('text', usetex=True)

fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
plt.subplots_adjust(right=0.99, left=0.11, bottom=0.14, top=0.89, wspace=0.3)

lg1 = ax[0].plot(profile_clear[CLEAR_DAY], color="red", label="Sunny day")
ax_2 = ax[0].twiny()
lg2 = ax_2.plot(profile_cloudy[CLOUDY_DAY], color="blue", label="Cloudy day")
ax_2.xaxis.set_major_formatter(NullFormatter())

lgs = lg1 + lg2
lgs_names = [l.get_label() for l in lgs]

# ax_2.set_xlabel("Time stamp - Cloudy day", fontsize="large")
ax[0].set_xlabel("Time step", fontsize="large")
ax[0].set_ylim(0, 1150)
ax[0].set_ylabel("W/m${}^2$", fontsize="large")
ax[0].legend(lgs, lgs_names, loc="upper right", fontsize="medium")
# ax_2.legend(fontsize="small")
ax[0].set_title("(a)", fontsize="large")


x_coord = []
y_coord = []
dx_coord = []
dy_coord = []
rect_set = []

for i in range(4):
    x = i * 10
    x_coord.append(x)

    y = profile_cloudy[CLOUDY_DAY][x]
    y_coord.append(y)

    dx = 10
    dx_coord.append(10)

    dy = profile_cloudy[CLOUDY_DAY][x + dx] - y
    dy_coord.append(dy)

    rect_set.append(patches.Rectangle((x, y), width=dx, height=dy, linewidth=1, edgecolor='b', facecolor='none', linestyle="--"))

for rect in rect_set:
    ax_2.add_patch(rect)

ax_2.vlines(x=x_coord[3], ymin=0, ymax=y_coord[2], linewidth=1, edgecolor='b', facecolor='none', linestyle="-")
ax_2.vlines(x=x_coord[3] + dx_coord[3], ymin=0, ymax=y_coord[3], linewidth=1, edgecolor='b', facecolor='none', linestyle="-")

ax_2.hlines(y=y_coord[3], xmin=0, xmax=x_coord[3], linewidth=1, edgecolor='b', facecolor='none', linestyle="-")
ax_2.hlines(y=y_coord[3] + dy_coord[3], xmin=0, xmax=x_coord[3] + dx_coord[3], linewidth=1, edgecolor='b', facecolor='none', linestyle="-")

ax_2.arrow(x=x_coord[3] - 10, y=82, dx=10, dy=0, length_includes_head=True, head_width=50, head_length=5, shape="full", color="b")
ax_2.arrow(x=x_coord[3] + dx_coord[3] + 10, y=82, dx=-10, dy=0, length_includes_head=True, head_width=50, head_length=5, shape="full", color="b")

ax_2.text(x=x_coord[3] + dx_coord[3]/2 - 0.5, y=82, s=r"$\Delta \tau$", ha="center", va="center", fontsize=10, color="b")

ax_2.text(x=x_coord[1], y=631, s=r"$q_t$", ha="left", va="center", fontsize=10, color="b")
ax_2.text(x=x_coord[1], y=823, s=r"$q_{t+\Delta \tau}$", ha="left", va="center", fontsize=10, color="b")


ax[1].scatter(x_t_clear, y_t_clear, color="red", marker="x", s=15)
ax[1].plot(x_t_clear, rg_fd_clear.intercept + rg_fd_clear.slope * x_t_clear, color='red', linewidth=0.4, label='Fitted line sunny')

ax[1].text(x=-0.56, y=6.00, s=f"FD={round(fd_clear, 2)}", ha="center", va="center", fontsize=10, color="r")
ax[1].text(x=-1.50, y=7.31, s=f"FD={round(fd_cloudy, 2)}", ha="center", va="center", fontsize=10, color="b")

ax[1].scatter(x_t_cloudy, y_t_cloudy, color="blue", marker="o", s=15)
ax[1].plot(x_t_cloudy, rg_fd_cloudy.intercept + rg_fd_cloudy.slope*x_t_cloudy, color='blue', linewidth=0.4, label='Fitted line cloudy')
ax[1].legend(fontsize="medium")

ax[1].set_ylim((3.5, 9))
ax[1].set_xlim((-3.3, 0.1))

ax[1].set_xlabel(r"$\ln(1/\Delta \tau)$", fontsize="large")
ax[1].set_ylabel(r"$\ln($S$(\Delta \tau) / {\Delta \tau}^2)$", fontsize="large")
ax[1].set_title("(b)", fontsize="large")

plt.savefig('figures/solar_model_proposal/fractal_dimension.pdf', dpi=700, bbox_inches='tight')

#%%

# #%% Clustering using Clear index and Fractal Index.
# csi_daily_values = day_mapping_frame["CSI_day"].dropna()
# fd_dict= {}
# for day_, data_dict_ in day_mapping_frame["metrics"].dropna().iteritems():
#     fd_dict[day_] = data_dict_["FD"]["FD_value"].item()
# fd_values = pd.DataFrame.from_dict(fd_dict, orient="index", columns=["FD"])
# csi_fd = pd.concat([csi_daily_values, fd_values], ignore_index=False, axis=1)
#
# # Hard Rules
# csi_fd_clustered = csi_fd.copy()
# csi_fd_clustered["cluster"] = 0
#
# idx_clear_sky = (csi_fd["CSI_day"] >= 0.6) & ((1 < csi_fd["FD"]) & (csi_fd["FD"]<=1.4))
# idx_partialy_cloudy = (csi_fd["CSI_day"] >= 0.6) & ((1.4 < csi_fd["FD"]) & (csi_fd["FD"]<=2.0))
# # idx_cloudy = (1.25 < csi_fd["FD"]) | ((csi_fd["FD"] <= 1.25) & (csi_fd["CSI_day"]<=0.5))
# idx_cloudy = csi_fd["CSI_day"]<=0.6
#
# csi_fd_clustered.loc[idx_clear_sky, ["cluster"]] = 0
# csi_fd_clustered.loc[idx_partialy_cloudy, ["cluster"]] = 2
# csi_fd_clustered.loc[idx_cloudy, ["cluster"]] = 1
#
# csi_fd_clustered_copula = csi_fd_clustered.copy()
#
# #%%
# np.random.seed(5)
# samples_per_cluster_v2 = sample_models_per_cluster(models_per_cluster)
# profiles_per_cluster = process_profiles_from_samples(samples_per_cluster_v2, day_mapping, label_frame)
#
# N_CLUSTERS_v2 = 3
# ALGORITHM_v2 = "KMeans"
# label_frame_v2 = pd.DataFrame(ap_cluster_solutions.loc[ALGORITHM_v2, N_CLUSTERS_v2], columns=["cluster"],
#                               index=clear_index_day_frame.index)
#
# csi_fd_clustered_copula["cluster"] = label_frame_v2["cluster"].values
#
# # Generated profiles by MVG and MVT models
# frame_generated_by_model = {}
# for model_type_ in ["MVT", "MVG"]:
#     frame_generated = []
#     n_clusters = len(profiles_per_cluster)
#     for cluster_i in range(n_clusters):
#         FD_0, _, _, _ = fractal_dimension(profiles_per_cluster[cluster_i][model_type_]["y_"], N_MAX=6)
#         CSI_day_0 = profiles_per_cluster[cluster_i][model_type_]["CSI_day"]
#         cluster_0 = (np.ones(len(CSI_day_0)) * cluster_i).astype(int)
#
#         generated_frame = pd.concat([pd.DataFrame(FD_0, columns=["FD"]),
#                                      pd.DataFrame(CSI_day_0, columns=["CSI_day"]),
#                                      pd.DataFrame(cluster_0, columns=["cluster"])],
#                                      axis=1)
#         frame_generated.append(generated_frame)
#     frame_generated_model = pd.concat(frame_generated, axis=0, ignore_index=True)
#     frame_generated_by_model[model_type_] = frame_generated_model
#
# fig, ax = plt.subplots(1, 4, figsize=(16, 4))
# plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, wspace=0.3)
# sns.scatterplot(data=csi_fd_clustered, x="CSI_day", y="FD", hue="cluster", ax=ax[0], palette="coolwarm")
# sns.scatterplot(data=csi_fd_clustered_copula, x="CSI_day", y="FD", hue="cluster", ax=ax[1], palette="coolwarm")
# sns.scatterplot(data=frame_generated_by_model["MVT"], x="CSI_day", y="FD", hue="cluster", ax=ax[2], palette="coolwarm")
# sns.scatterplot(data=frame_generated_by_model["MVG"], x="CSI_day", y="FD", hue="cluster", ax=ax[3], palette="coolwarm")
# ax[0].set_title("Fractal Dimension clustering")
# ax[1].set_title("Time series clustering")
# ax[2].set_title("Generated profiles MVT")
# ax[3].set_title("Generated profiles MVG")
#
# ax_flat = ax.flatten()
# for ax_ in ax_flat:
#     ax_.set_xlim([0, 1.14])
#     ax_.set_ylim([0.95, 2.01])
# # label_frame = csi_fd_clustered[["cluster"]]  # To plot the big figure with the clusters
#
# # #%% Distance metrics to merge the clusters
# # wasserstein_distance_d(profiles_per_cluster[3]["Original"]["y_prime"],
# #                        profiles_per_cluster[5]["Original"]["y_prime"], factor=1000)
# #
# # n_clusters = len(profiles_per_cluster)
# # # dist_matrix_emd = np.zeros((n_clusters, n_clusters))
# # dist_matrix_emd = np.diag([np.inf] * n_clusters)
# # dist_matrix_emd = np.ones((n_clusters, n_clusters)) * np.inf
# #
# # for ii in range(0, n_clusters):
# #     for jj in range(ii + 1, n_clusters):
# #         dist_matrix_emd[ii, jj] = wasserstein_distance_d(profiles_per_cluster[ii]["Original"]["y_prime"],
# #                                                          profiles_per_cluster[jj]["Original"]["y_prime"], factor=1000)
# #         # dist_matrix_emd[jj, ii] = dist_matrix_emd[ii, jj]
# #
# # print(dist_matrix_emd.round(3))
# #
# # sorted_distances = np.sort(dist_matrix_emd.flatten())
# # np.argwhere(dist_matrix_emd == sorted_distances[0])
#
# #%% Now that I have the models, I want to create the profiles
# START_DATE = "2021-03-01 00:00:00"
# END_DATE = "2021-09-30 23:50:00"
# RESAMPLE = "10T"
#
# data_aligned_fig = pd.concat([knmi_10min_w_m2[["qg"]], cs_haurwitz, cs_simplified_solis[["ghi_solis"]], cs_ineichen[["ghi_ineichen"]]], axis=1)
# data_aligned_fig.resample(RESAMPLE).mean()[START_DATE:END_DATE].plot()
# data_aligned_sliced_fig = data_aligned_fig.resample(RESAMPLE).mean()[START_DATE:END_DATE].copy()
# day_mapping_fig = day_processing_mapper(data_aligned_sliced_fig, IRR_THRESHOLD=20)
# day_mapping_frame_fig = pd.DataFrame.from_dict(day_mapping_fig).transpose()
# clear_index_day_frame_fig = day_mapping_frame_fig[["CSI_day"]]
#
# np.random.seed(1234)
# mvt_models = []
# pi_mixture = []
# for cluster_ in range(n_clusters):
#     mvt_models.append(models_per_cluster[cluster_]["models"]["MVT"])
#     pi_mixture.append(models_per_cluster[cluster_]["n_profiles_original"])
#
# pi_mixture_original = np.array(pi_mixture) / np.array(pi_mixture).sum()
# pi_mixture_dark = np.array([0.0, 1.0, 0.0])  # Dark days
# pi_mixture_sunny = np.array([0.0, 0.0, 1.0])  # Sunny days
#
# irr_measured = pivot_dataframe(data_aligned_sliced_fig[["qg"]])  # It has nan values
# irr_measured_original = pd.concat([irr_measured, clear_index_day_frame_fig["CSI_day"]], axis=1)
#
# pi_mixtures = [pi_mixture_original, pi_mixture_sunny, pi_mixture_dark]
# profiles_sampled = [data_aligned_sliced_fig]  # Original time series
# pivoted_series_sampled = [irr_measured_original]  # Original time series pivoted
#
# for pi_mixture_ in tqdm(pi_mixtures):
#     profile_series, pivoted_series = generate_profiles(pi_mixture_, mvt_models, day_mapping_fig, data_aligned_sliced_fig, START_DATE, END_DATE)
#     profile_series_concat = pd.concat([profile_series, data_aligned_sliced_fig[["ghi_haurwitz", "qg"]]], axis=1)
#
#     profiles_sampled.append(profile_series_concat)
#     pivoted_series_sampled.append(pivoted_series)
#
# n_steps = pivoted_series.shape[1]-1
# x_axis = pd.date_range(start="2021-11-01", periods=n_steps, freq=RESAMPLE)
#
# #%%
# # fig = plt.figure(figsize=(7.1, 3))
# fig = plt.figure(figsize=(14.2, 6))
# gs = gridspec.GridSpec(4, 4, wspace=0.35, hspace=0.35, left=0.08, bottom=0.07, right=0.95, top=0.95, width_ratios=[2, 2, 1, 1.5], height_ratios=[1, 1, 1, 1])
# ax = np.empty((4, 4), dtype=object)
#
# kk = 0
# for ii in range(4):
#     for jj in range(4):
#         ax[ii, jj] = fig.add_subplot(gs[kk])
#         kk += 1
#
# titles_list = [["(a)", "(b)", "(c)", "(d)"],
#                ["(e)", "(f)", "(g)", "(h)"],
#                ["(i)", "(j)", "(k)", "(l)"],
#                ["(m)", "(n)", "(o)", "(p)"]]
#
# y_labels_list = ["Original profiles\n$\pi=(0.44, 0.26, 0.3)$\nW/m${}^2$",
#                  "Synthetic profiles\n$\pi=(0.44, 0.26, 0.3)$\nW/m${}^2$",
#                  "Synthetic profiles\n$\pi=(0.0, 0.0, 1.0)$\nW/m${}^2$",
#                  "Synthetic profiles\n$\pi=(0.0, 1.0, 0.0)$\nW/m${}^2$"]
#
# for ii in range(4):
#     if ii == 0:
#         column_label = "qg"
#     else:
#         column_label = "qg_hat"
#
#     # locator = mdates.AutoDateLocator()
#     locator = mdates.AutoDateLocator(minticks=3, maxticks=15)
#     formatter = mdates.ConciseDateFormatter(locator)
#     formatter.formats = ['%y',  # ticks are mostly years
#                          '%b',  # ticks are mostly months
#                          '%d',  # ticks are mostly days
#                          '%H:%M',  # hrs
#                          '%H:%M',  # min
#                          '%S.%f', ]  # secs
#     formatter.offset_formats = [''] * 6
#
#     ax[ii, 0].plot(profiles_sampled[ii][[column_label,"ghi_haurwitz"]]["2021-06-01":"2021-06-15"].resample("30T").mean(), label=["1", "2"])
#     ax[ii, 0].set_ylim((-10, 1100))
#     ax[ii, 0].set_ylabel(y_labels_list[ii], fontsize="large")
#     ax[ii, 0].set_title(titles_list[ii][0])
#     ax[ii, 0].xaxis.set_major_locator(locator)
#     ax[ii, 0].xaxis.set_major_formatter(formatter)
#
#     locator = mdates.AutoDateLocator(minticks=3, maxticks=15)
#     formatter = mdates.ConciseDateFormatter(locator)
#     formatter.formats = ['%y',  # ticks are mostly years
#                          '%b',  # ticks are mostly months
#                          '%d',  # ticks are mostly days
#                          '%H:%M',  # hrs
#                          '%H:%M',  # min
#                          '%S.%f', ]  # secs
#     formatter.offset_formats = [''] * 6
#
#     ax[ii, 1].plot(profiles_sampled[ii][[column_label, "ghi_haurwitz"]]["2021-03-01":"2021-03-15"].resample("30T").mean(), label=["$\hat{q}_t$", "GHI Model"])
#     ax[ii, 1].set_ylim((-10, 1100))
#     ax[ii, 1].set_ylabel("W/m${}^2$")
#     ax[ii, 1].set_title(titles_list[ii][1])
#     ax[ii, 1].xaxis.set_major_locator(locator)
#     ax[ii, 1].xaxis.set_major_formatter(formatter)
#
#     norm_individual = mpl.colors.Normalize(vmin=0, vmax=1)
#     for _,(_, data_plot_) in enumerate(pivoted_series_sampled[ii].iterrows()):
#         ci_daily_ = data_plot_["CSI_day"]
#         ax[ii, 2].plot(x_axis, data_plot_.drop(["CSI_day"]).ravel(), linewidth=0.2, marker='.', markersize=1,
#                      markerfacecolor=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)),
#                      color=plt.cm.get_cmap('plasma')(norm_individual(ci_daily_)))
#     # ax[0,1].set_title("Synthetic load profiles")
#     ax[ii, 2].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax[ii, 2].set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=9)))
#     ax[ii, 2].set_ylabel("W/m${}^2$")
#     ax[ii, 2].set_ylim((-10, 1100))
#     ax[ii, 2].set_title(titles_list[ii][2])
#     cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('plasma')), ax=ax[ii,2])
#     cbar_2.ax.set_ylabel('Daily clear index [$K_d$]')
#
#     data_time_series = pivoted_series_sampled[ii].drop(columns=["CSI_day"])
#
#     x_lims = mdates.date2num([pd.to_datetime(2021 * 1000 + data_time_series.index[0], format='%Y%j'),
#                               pd.to_datetime(2021 * 1000 + data_time_series.index[-1], format='%Y%j')])
#     y_lims = mdates.date2num([x_axis[0], x_axis[-1] + pd.Timedelta(minutes=9)])
#
#     Z = data_time_series.values
#     z_min, z_max = 0, 1000
#     b = ax[ii, 3].imshow(Z.T,  cmap=plt.cm.get_cmap('plasma'), vmin=z_min, vmax=z_max, aspect='auto',
#                          extent=[x_lims[0], x_lims[1],  y_lims[0], y_lims[1]])
#     cbar_1 = plt.colorbar(b, ax=ax[ii, 3])
#     cbar_1.ax.set_ylabel('Global Irradiance\nW/m${}^2$')
#     ax[ii, 3].set_title(titles_list[ii][3])
#     ax[ii, 3].set_ylabel("Time of day")
#     ax[ii, 3].xaxis_date()
#     ax[ii, 3].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
#     ax[ii, 3].yaxis_date()
#     ax[ii, 3].yaxis.set_major_formatter(mdates.DateFormatter('%H'))
#
#     if ii == 3:
#         ax[ii, 1].legend(loc="upper right", labelspacing=0.3)
#         ax[ii, 2].set_xlabel("Time of day")
#         ax[ii, 3].set_xlabel("Month of year")
#
# plt.savefig('figures/solar_model_proposal/generative_profiles.png', dpi=700, bbox_inches='tight')
#
# #%% Surface plotPlot original data with the clear index
# irr_solar_model = pivot_dataframe(data_aligned_sliced[["ghi_haurwitz"]])
# irr_measured = pivot_dataframe(data_aligned_sliced[["qg"]])
# data_time_series = pivoted_series.drop(columns=["CSI_day"])
# data_time_series_slice = data_time_series.iloc[:]
# x_lims = mdates.date2num([pd.to_datetime(2021 * 1000 + data_time_series_slice.index[0], format='%Y%j'),
#                           pd.to_datetime(2021 * 1000 + data_time_series_slice.index[-1], format='%Y%j')])
# y_lims = mdates.date2num([x_axis[0], x_axis[-1] + pd.Timedelta(minutes=9)])
# Z = data_time_series_slice.values
# z_min, z_max = 0, 1000
# fig, ax = plt.subplots(1,1, figsize=(10, 10))
# ax.imshow(Z.T,  cmap=plt.cm.get_cmap('plasma'), vmin=z_min, vmax=z_max, aspect="auto",
#           extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]])
# ax.xaxis_date()
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b') )
# ax.yaxis_date()
# ax.yaxis.set_major_formatter(mdates.DateFormatter('%H') )
#
