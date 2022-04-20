from core.copula import EllipticalCopula
from core.clustering import (rlp_irradiance,
                             rlp_transformer,
                             load_time_series)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm
import seaborn as sns

def pivot_merge(data_cluster, *, annual_energy=None, irradiance=None, _regex="_ap", add_category=False):
    """
    Creates pandas data frame aligning each time step of the day for each transformer, adding extra information
    provided by the arguments.

    Arguments:
    ---------
        annual_energy: (pd.DataFrame): If provided, the annual energy will be concatenated to the final pivoted
            dataframe. This dataframe must have the annual energy consumption in GWh.
        irradiance: (pd.DataFrame): If provided, the irradiance time steps will be concatenated to the final
            pivoted dataframe. This frame must have the irradiance units in W/m^2.

            WARNING: The irradiance has an automatic threshold that will detect the number of time steps to be included.
                This threshold must be consistent between all scripts to have the same number of time steps.

        _regex: (str): Keyword to filter the columns of the load consumption dataframe.
            This must be "_ap" for active power and "_rp" for reactive power.
        add_category: (bool): Flag to add a column called "category", the category is a integer number that identifies
            the transformer. This is useful to do a stratified sampling over the final dataset. Using this row, you
            can sample the days proportionally to the number of days per transformer.

    """

    data_kw_resampled = data_cluster.filter(regex=_regex + "$").copy()
    transformer_name = data_kw_resampled.columns.to_list()

    data_kw_resampled['Day'] = data_kw_resampled.index.dayofyear
    data_kw_resampled['hour'] = data_kw_resampled.index.hour
    data_kw_resampled['dayIndex'] = data_kw_resampled.index.weekday
    data_kw_resampled['month'] = data_kw_resampled.index.month
    data_kw_resampled['year'] = data_kw_resampled.index.year
    data_kw_resampled['quarterIndex'] = data_kw_resampled.index.hour * 100 + data_kw_resampled.index.minute

    delta_times = np.sort(data_kw_resampled['quarterIndex'].unique())
    mapper = dict(zip(delta_times, ['q_' + str(ii) + _regex for ii in range(1, len(delta_times) + 1)]))

    transformers_samples = list()
    for category, transformer in tqdm(enumerate(transformer_name)):
        transfo_kw = data_kw_resampled.pivot(index='quarterIndex', columns='Day', values=transformer)

        if irradiance is not None:
            transfo_kw = pd.concat([transfo_kw, irradiance], axis=0)

        if annual_energy is not None:
            n_days = transfo_kw.shape[1]

            if _regex == "_ap":
                key_transformer = transformer
            else:
                key_transformer = transformer.replace("_rp", "_ap")

            energy_gwh = np.full((1, n_days), annual_energy.loc[key_transformer].item())
            energy_gwh_frame = pd.DataFrame(energy_gwh, index=['avg_gwh'], columns=transfo_kw.columns)
            transfo_kw = pd.concat([transfo_kw, energy_gwh_frame], sort=False)

        if add_category:
            n_days = transfo_kw.shape[1]
            category_number = np.full((1, n_days), category + 1, dtype=int)
            category_frame = pd.DataFrame(category_number, index=['category'], columns=transfo_kw.columns)
            transfo_kw = pd.concat([transfo_kw, category_frame], sort=False)

        transfo_kw.columns = [f"{transformer[:-3]}_" + str(day_idx) for day_idx in transfo_kw.columns]
        transformers_samples.append(transfo_kw)

    transformers_samples = pd.concat(transformers_samples, axis=1, sort=False)
    transformers_samples.rename(index=mapper, inplace=True)
    transformers_samples.index.name = ""

    return transformers_samples

def pivot_irradiance(data_cluster):
    cc, idx = rlp_irradiance(data_cluster)
    irr_pivoted = cc.transpose()
    delta_times_irr = np.sort(irr_pivoted.index.unique())
    mapper = dict(zip(delta_times_irr, [ii + "_qg" for ii in delta_times_irr]))
    irr_pivoted.rename(index=mapper, inplace=True)
    irr_pivoted.index.name = ""
    irradiance = irr_pivoted.copy()

    return irradiance, idx

def filter_transformer_per_cluster(data, clusters, k_cluster=None):

    cluster_transformer_count = clusters['cluster'].value_counts().sort_index()

    print(f"Cluster information for load profiles:\n{cluster_transformer_count}")
    idx = (clusters["cluster"] == k_cluster).values
    transformers_cluster = clusters["DALIBOX_ID"][idx].apply(lambda x: x.replace("_ap", "")).to_list()
    print(f"CLUSTER: {k_cluster}. Total transformers: {len(transformers_cluster)}")

    idx = np.zeros(len(data.columns), dtype=bool)
    for transformer_ in transformers_cluster:
        idx = idx | np.array(data.columns.str.contains(f"^{transformer_}_ap"))  # Active power columns
        idx = idx | np.array(data.columns.str.contains(f"^{transformer_}_rp"))  # Reactive power columns
    data_cluster = data.iloc[:, idx].copy()
    assert cluster_transformer_count[k_cluster] == int(data_cluster.shape[1] / 2), "Check the filter of clusters"
    data_cluster["qg"] = data["qg"].copy()

    return data_cluster

def filter_annual_energy_per_cluster(annual_energy, clusters, *, k_cluster: int =None, plot_annual_energy: bool = True):
    # Filter annual energy by cluster
    idx = (clusters["cluster"] == k_cluster).values
    transformers_cluster = clusters["DALIBOX_ID"][idx].apply(lambda x: x.replace("_ap", "")).to_list()

    idx = np.zeros(len(annual_energy), dtype=bool)
    transformers_annual_energy = annual_energy.reset_index()["index"]
    for transformer_ in transformers_cluster:
        idx = idx | (transformers_annual_energy == f"{transformer_}_ap").values  # Active power columns

    annual_energy_cluster = annual_energy[idx].copy()

    if plot_annual_energy:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.histplot(data=annual_energy_cluster.values.flatten(), kde=True, bins='fd', element='step', ax=ax)
        sns.rugplot(x=annual_energy_cluster.values.flatten(), ax=ax)
        ax.set_xlabel("Yearly Energy consumption GWh")
        ax.set_title(f"Cluster {k_cluster}")

    return annual_energy_cluster

def filter_per_cluster(data,
                       clusters,
                       annual_energy,
                       *,
                       add_reactive_power=True,
                       k_cluster: int = 0,
                       sub_sample_dataset: bool = True,
                       fraction_sub_sample: float = 0.1):

    data_cluster = filter_transformer_per_cluster(data, clusters, k_cluster=k_cluster)
    annual_energy = filter_annual_energy_per_cluster(annual_energy, clusters, k_cluster=k_cluster)

    irradiance, idx_irradiance = pivot_irradiance(data_cluster)

    if add_reactive_power:
        # TODO: The order of concatenation matters in the conditional sampling of the copula (Unknown behaviour)
        section_1 = pivot_merge(data_cluster,
                                add_category=sub_sample_dataset,
                                _regex="_ap")
        section_2 = pivot_merge(data_cluster, irradiance=irradiance, annual_energy=annual_energy,
                                _regex="_rp")
        data_cluster_copula = pd.concat([section_1, section_2], axis=0)
    else:
        data_cluster_copula = pivot_merge(data_cluster, irradiance=irradiance, annual_energy=annual_energy,
                                          add_category=sub_sample_dataset,
                                          _regex="_ap")

    data_cluster_copula.dropna(axis=1, inplace=True)
    data_cluster_copula = data_cluster_copula.transpose()

    if sub_sample_dataset:
        # Stratified stratify sampling to reduce the sample size
        subsample_dataset = data_cluster_copula.groupby("category").apply(lambda x: x.sample(frac=fraction_sub_sample)).copy()
        subsample_dataset = subsample_dataset.droplevel(level=0)
        subsample_dataset.drop(['category'], axis=1, inplace=True)

    else:
        subsample_dataset = data_cluster_copula.copy()

    return subsample_dataset, idx_irradiance

def create_rlp_and_annual_energy_per_cluster(data: pd.DataFrame,
                                             clusters: pd.DataFrame,
                                             annual_energy: pd.DataFrame,
                                             k_cluster: int = 0,
                                             plot_annual_energy: bool = False):
    """
    Creates and ACTIVE power representative load profile (RLP) of the selected cluster.
    Filters the annual energy consumption of the transformers in the selected cluster

    Arguments:
    ---------
        data: (pd.DataFrame): Timeseries dataframe with active, reactive power, irradiance.
        clusters: (pd.DataFrame): Frame with the name of the transformers and the cluster it belongs to
        annual_energy: (pd.DataFrame): Frame with the name of the transformers and the annual energy consumption of each
            one of them
        k_cluster (int): Cluster number which you want to filter the data and create the RLP.

    """

    data_filtered = filter_transformer_per_cluster(data, clusters, k_cluster=k_cluster)
    annual_energy_filtered = filter_annual_energy_per_cluster(annual_energy, clusters, k_cluster=k_cluster,
                                                              plot_annual_energy=plot_annual_energy)
    rlp_cluster = rlp_transformer(data_filtered, kwargs_filter={"regex": "_ap$"})

    assert len(rlp_cluster) == len(annual_energy_filtered), "Transformers quantities in the cluster does not agree."

    return rlp_cluster, annual_energy_filtered

def data_loader(file_time_series,
                file_transformer_outliers,
                file_load_clusters,
                file_annual_energy,
                *,
                freq: str = "30T"):
    data = load_time_series(file_time_series, file_path_outliers=file_transformer_outliers, freq=freq)
    clusters = pd.read_csv(file_load_clusters)
    annual_energy = pd.read_csv(file_annual_energy, index_col=0)

    return data, clusters, annual_energy

def create_rlp_and_annual_energy_for_all_clusters(file_time_series: str,
                                                  file_transformer_outliers: str,
                                                  file_load_clusters: str,
                                                  file_annual_energy: str) -> dict:

    data, clusters, annual_energy = data_loader(file_time_series,
                                                file_transformer_outliers,
                                                file_load_clusters,
                                                file_annual_energy)

    cluster_labels = clusters["cluster"].unique()
    cluster_labels.sort()

    rlp_clusters = {}
    for cluster_label in cluster_labels:
        rlp_cluster, annual_energy_filtered = create_rlp_and_annual_energy_per_cluster(data,
                                                                                       clusters,
                                                                                       annual_energy,
                                                                                       k_cluster=cluster_label)
        rlp_clusters[cluster_label] = {"rlp": rlp_cluster,
                                       "annual_energy": annual_energy_filtered}

    return rlp_clusters

def create_copula_models_for_all_clusters(file_time_series: str,
                                          file_transformer_outliers: str,
                                          file_load_clusters: str,
                                          file_annual_energy: str,
                                          add_reactive_power=True):
    """
    Builds copula models for each cluster. Each cluster is a group of time series of load consumption.

    Parameters:
    ----------
        file_time_series: str: file path of the time series dataset:
            The file contains: A column with a date stamp. Columns with the active and reactive power consumption
            for each transformer. i.e., 2 columns per transformer (one for active, one for reactive).
            One column with the solar global solar irradiance.
        file_transformer_outliers: str: file path of the file with the names of transformers that are considered outliers
        file_load_clusters: str: file path of the file with the names of the transformer and its cluster label
        file_annual_energy: str: file path with a file with transformer names and its annual energy consumption
        add_reactive_power: Bool: True if the final copula model includes the reactive power variables.

    Returns:
    -------
        copula_models: dict: dict: Nested dictionary with cluster number as the key. The second key could be:
            "copula" for the copula model, "original_data" for the original dataset to build the copula, and
            "idx_irradiance" with a boolean mask used to filter the sunlight times of the daily irradiance profile.

            e.g., copula_models[1]["copula"] -> retrieves the copula model object of the cluster 1.

    """

    data, clusters, annual_energy = data_loader(file_time_series,
                                                file_transformer_outliers,
                                                file_load_clusters,
                                                file_annual_energy)

    cluster_labels = clusters["cluster"].unique()
    cluster_labels.sort()

    copula_models = {}
    for cluster_label in tqdm(cluster_labels):
        data_copula, idx_irradiance = filter_per_cluster(data,
                                                         clusters,
                                                         annual_energy,
                                                         k_cluster=cluster_label,
                                                         add_reactive_power=add_reactive_power,
                                                         sub_sample_dataset=True,
                                                         fraction_sub_sample=0.1)

        copula_cluster = EllipticalCopula(data_copula.transpose().values)
        copula_cluster.fit()

        copula_models[cluster_label] = {"copula": copula_cluster,
                                        "original_data": data_copula,
                                        "idx_irradiance": idx_irradiance}

    return copula_models

def check_copula_model(copula_models: dict, k_cluster: int):
    copula_cluster = copula_models[k_cluster]["copula"]
    data_copula = copula_models[k_cluster]["original_data"]

    samples_cluster = copula_cluster.sample(1000, drop_inf=True)
    samples_cluster_frame = pd.DataFrame(samples_cluster.T, columns=data_copula.columns)

    filter_dict = [{"regex": "_ap$"}, {"regex": "_rp$"}, {"regex": "_qg$"}, {"items": ["avg_gwh"]}]

    fig, ax = plt.subplots(2, 4, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.5)
    y_label = {0: "Active Power [kW]",
               1: "Reactive Power [kVA]",
               2: "Global irradiance [W/m^2]",
               3: "Density"}

    for ii, regex_dict in enumerate(filter_dict):
        # ii = 0
        # regex_dict = {"regex": "_ap$"}
        if ii != 3:
            ax[0, ii].plot(data_copula.filter(**regex_dict).transpose().values, linewidth=0.2, color="gray")
            q_05 = np.nanquantile(data_copula.filter(**regex_dict).transpose().values, q=0.05, axis=1)
            q_50 = np.nanquantile(data_copula.filter(**regex_dict).transpose().values, q=0.50, axis=1)
            q_95 = np.nanquantile(data_copula.filter(**regex_dict).transpose().values, q=0.95, axis=1)
            ax[0, ii].plot(q_05, linewidth=0.5, color="red", linestyle="--")
            ax[0, ii].plot(q_50, linewidth=0.5, color="red", linestyle="-")
            ax[0, ii].plot(q_95, linewidth=0.5, color="red", linestyle="-.")
            ax[0, ii].xaxis.set_major_locator(ticker.MultipleLocator(5.0))

        else:
            # ax[0, ii].hist(data_copula.filter(items=["avg_gwh"]).transpose().values.flatten())
            sns.histplot(x=data_copula.filter(items=["avg_gwh"]).transpose().values.flatten(), element="step",
                         kde=True, ax=ax[0, ii])

            ax[0, ii].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax[0, ii].yaxis.set_major_locator(ticker.NullLocator())

        ax[0, ii].set_ylabel(y_label[ii])

    for ii, regex_dict in enumerate(filter_dict):
        if ii != 3:
            ax[1, ii].plot(samples_cluster_frame.filter(**regex_dict).transpose().values, linewidth=0.2, color="gray")
            q_05 = np.nanquantile(samples_cluster_frame.filter(**regex_dict).transpose().values, q=0.05, axis=1)
            q_50 = np.nanquantile(samples_cluster_frame.filter(**regex_dict).transpose().values, q=0.50, axis=1)
            q_95 = np.nanquantile(samples_cluster_frame.filter(**regex_dict).transpose().values, q=0.95, axis=1)
            ax[1, ii].plot(q_05, linewidth=0.5, color="red", linestyle="--")
            ax[1, ii].plot(q_50, linewidth=0.5, color="red", linestyle="-")
            ax[1, ii].plot(q_95, linewidth=0.5, color="red", linestyle="-.")
            ax[1, ii].xaxis.set_major_locator(ticker.MultipleLocator(5.0))
            ax[1, ii].set_xlabel("Time step [30 min]")

        else:
            # ax[1, ii].hist(samples_cluster_frame.filter(items=["avg_gwh"]).transpose().values.flatten())
            sns.histplot(x=samples_cluster_frame.filter(items=["avg_gwh"]).transpose().values.flatten(), element="step", kde=True, ax=ax[1, ii])
            # ax[1, ii].hist(samples_cluster_frame.filter(items=["avg_gwh"]).transpose().values.flatten())

            ax[1, ii].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax[1, ii].yaxis.set_major_locator(ticker.NullLocator())
            ax[1, ii].set_xlabel("Annual energy [GWh]")

        ax[1, ii].set_ylabel(y_label[ii])

    fig.suptitle(f"Copula modelling - Cluster: {k_cluster}\nUpper row original data - Lower row sampled from Copula")

def check_energy_levels(copula_models: dict, k_cluster: int, drop_inf: bool = True):

    def filter_power(data, index, regex="_ap", ):
        try:
            x_low = data.loc[index, ["q_32" + regex]].values.flatten()
            x_high = data.loc[index, ["q_33" + regex]].values.flatten()
        except:
            x_low = []
            x_high = []

        return x_low, x_high

    def seaborn_plot_power(x_low, y_low, x_high, y_high, ax=None, lims=None):
        sns.kdeplot(x=x_high, y=y_high, ax=ax, fill=False, linewidths=0.4, zorder=-1, colors="red")
        sns.scatterplot(x=x_high, y=y_high, ax=ax, color="red", s=10, marker="o", zorder=1)

        sns.kdeplot(x=x_low, y=y_low, ax=ax, fill=False, linewidths=0.4, zorder=-1, colors="grey")
        sns.scatterplot(x=x_low, y=y_low, ax=ax, color="k", s=10, marker="o", zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    data_copula = copula_models[k_cluster]["original_data"]
    copula_cluster = copula_models[k_cluster]["copula"]

    idx_low = data_copula["avg_gwh"].values < 0.25
    idx_high = data_copula["avg_gwh"].values > 0.9

    x_low_ap, y_low_ap = filter_power(data_copula, idx_low, regex="_ap")
    x_low_rp, y_low_rp = filter_power(data_copula, idx_low, regex="_rp")

    x_high_ap, y_high_ap = filter_power(data_copula, idx_high, regex="_ap")
    x_high_rp, y_high_rp = filter_power(data_copula, idx_high, regex="_rp")

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    seaborn_plot_power(x_low_ap, y_low_ap, x_high_ap, y_high_ap, ax=ax[0, 0], lims=(-30, 300))
    seaborn_plot_power(x_low_rp, y_low_rp, x_high_rp, y_high_rp, ax=ax[0, 1], lims=(-50, 40))

    mapper = dict(zip(data_copula.columns.to_list(),
                      ["x" + str(variable) for variable in range(1, len(data_copula.columns.to_list()) + 1)]))
    new_column = data_copula.columns.to_list()
    new_column.remove("avg_gwh")

    low_energy_samples = data_copula.loc[idx_low, ["avg_gwh"]].value_counts()
    samples_low_energy = []
    for _, (energy_value, n_samples) in tqdm(low_energy_samples.reset_index().iterrows()):
        samples_low = copula_cluster.sample(n_samples=int(n_samples),
                                            conditional=True,
                                            variables={mapper["avg_gwh"]: energy_value}, drop_inf=drop_inf)
        samples_low_frame = pd.DataFrame(samples_low.T, columns=new_column)
        samples_low_energy.append(samples_low_frame)
    samples_low_energy = pd.concat(samples_low_energy, axis=0).reset_index(drop=True)

    high_energy_samples = data_copula.loc[idx_high, ["avg_gwh"]].value_counts()
    samples_high_energy = []
    for _, (energy_value, n_samples) in tqdm(high_energy_samples.reset_index().iterrows()):
        samples_high = copula_cluster.sample(n_samples=int(n_samples),
                                             conditional=True,
                                             variables={mapper["avg_gwh"]: energy_value}, drop_inf=drop_inf)
        samples_high_frame = pd.DataFrame(samples_high.T, columns=new_column)
        samples_high_energy.append(samples_high_frame)
    samples_high_energy = pd.concat(samples_high_energy, axis=0).reset_index(drop=False)

    x_low_ap_sampled, y_low_ap_sampled = filter_power(samples_low_energy,
                                                      index=np.ones(len(samples_low_energy), dtype=bool), regex="_ap")
    x_low_rp_sampled, y_low_rp_sampled = filter_power(samples_low_energy,
                                                      index=np.ones(len(samples_low_energy), dtype=bool), regex="_rp")
    x_high_ap_sampled, y_high_ap_sampled = filter_power(samples_high_energy,
                                                        index=np.ones(len(samples_high_energy), dtype=bool),
                                                        regex="_ap")
    x_high_rp_sampled, y_high_rp_sampled = filter_power(samples_high_energy,
                                                        index=np.ones(len(samples_high_energy), dtype=bool),
                                                        regex="_rp")

    seaborn_plot_power(x_low_ap_sampled, y_low_ap_sampled, x_high_ap_sampled, y_high_ap_sampled, ax=ax[1, 0],
                       lims=(-30, 300))
    seaborn_plot_power(x_low_rp_sampled, y_low_rp_sampled, x_high_rp_sampled, y_high_rp_sampled, ax=ax[1, 1],
                       lims=(-50, 40))

def data_loader_irradiance(file_time_series_irradiance,
                           file_irradiance_clusters,
                           freq: str = "30T"):
    data_irradiance = load_time_series(file_time_series_irradiance, freq=freq)
    clusters_irradiance = pd.read_csv(file_irradiance_clusters)

    return data_irradiance, clusters_irradiance

def create_copula_model_irradiance_all_clusters(file_time_series_irradiance,
                                                file_irradiance_clusters,
                                                freq: str = "30T"):

    data_irradiance, clusters_irradiance = data_loader_irradiance(file_time_series_irradiance,
                                                                   file_irradiance_clusters,
                                                                   freq=freq)

    cluster_labels = clusters_irradiance["cluster"].unique()
    cluster_labels.sort()

    mixture_counts = clusters_irradiance["cluster"].value_counts().sort_index()
    mixture_prob = mixture_counts / mixture_counts.sum()

    copula_models_irradiance = {}

    for cluster_label in tqdm(cluster_labels):
        irradiance, idx_irradiance = pivot_irradiance(data_irradiance)
        cluste_irradiance_days_count = clusters_irradiance['cluster'].value_counts().sort_index()
        print(f"Cluster information for load profiles:\n{cluste_irradiance_days_count}")
        idx = (clusters_irradiance["cluster"] == cluster_label).values
        days_cluster = clusters_irradiance["Day"][idx].values
        data_irradiance_copula = irradiance[days_cluster]

        copula_cluster = EllipticalCopula(data_irradiance_copula.values)
        copula_cluster.fit()

        copula_models_irradiance[cluster_label] = {"copula": copula_cluster,
                                                   "original_data": data_irradiance_copula,
                                                   "idx_irradiance": idx_irradiance,
                                                   "pi": mixture_prob[cluster_label]}

    return copula_models_irradiance, mixture_prob