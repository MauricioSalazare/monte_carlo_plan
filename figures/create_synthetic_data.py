from core.copula import EllipticalCopula
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import pickle
import warnings
import multiprocessing as mp
import uuid
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def sampling_by_annual_energy(copula_model: EllipticalCopula,
                              data_original_copula: pd.DataFrame,
                              *,
                              energy_values: pd.Series = None,
                              add_uuid: bool = True,
                              add_day_id: bool = True) -> pd.DataFrame:
    """
    Conditional copula sampling by annual energy.

    Parameter:
    ----------
        copula_model: EllipticalCopula: Copula object to be sampled.
        data_original_copula: pd.DataFrame: Original pivoted data set. Dimensions (n x p): n: days p: variables.
        energy_values: pd.Series: The index is the annual energy level and the value is the number of days to be sampled
            by the copula model.
            If there the value is None, then the number of days to be sampled comes from the original data.
            E.g. if the original data set has 10 days of annual energ 0.134 [GWh], then the final frame will also have
            10 days.
        add_uid: bool: True if add a column of unique number identifier for the meter
        add_day_id": bool: True if add a column with a day number identifier.

    Return:
    -------
        samples_cluster_frame: pd.DataFrame: DataFrame with the generated profiles. Same dimensions, variable names,
            and index names than the original data frame. i.e., (n x p): n: days p: variables.

    """

    var_mapping = dict(zip(data_original_copula.columns,
                           [f"x{zz + 1}" for zz in range(len(data_original_copula.columns))]))
    cond_variable = var_mapping["avg_gwh"]

    if energy_values is None:
        energy_values_sample = data_original_copula.loc[:, "avg_gwh"].value_counts()
    else:
        energy_values_sample = energy_values.copy()

    avg_energy_list = []
    meter_uid = []
    day_id = []

    samples_list = []
    for energy_, count_ in tqdm(energy_values_sample.iteritems(),
                                total=energy_values_sample.shape[0],
                                desc="Sampling copula"):
        try:
            samples_list.append(copula_model.sample(count_,
                                                    conditional=True,
                                                    variables={cond_variable: energy_},
                                                    drop_inf=True))
        except RuntimeError as err:
            print(err)
            warnings.warn("Could not sample copula. Trying sampling without dropping nan values.", UserWarning)
            samples_list.append(copula_model.sample(count_,
                                                    conditional=True,
                                                    variables={cond_variable: energy_},
                                                    drop_inf=False))

        avg_energy_list += [energy_] * count_
        meter_uid += [uuid.uuid4().__str__()] * count_
        day_id += list(range(1, count_ + 1))

    avg_energy_frame = pd.DataFrame({"avg_gwh": avg_energy_list})
    meter_uid_frame = pd.DataFrame({"meter_uid": meter_uid})
    day_id_frame = pd.DataFrame({"day_id": day_id})

    if energy_values is None:
        samples_frame = pd.DataFrame(np.concatenate(samples_list, axis=1).T,
                                     index=data_original_copula.index,
                                     columns=data_original_copula.columns[:-1])
        samples_cluster_frame__ = pd.concat([samples_frame, data_original_copula.loc[:, "avg_gwh"]], axis=1)

    else:
        samples_frame = pd.DataFrame(np.concatenate(samples_list, axis=1).T,
                                     columns=data_original_copula.columns[:-1])
        samples_cluster_frame__ = pd.concat([samples_frame, avg_energy_frame], axis=1)

    if add_day_id:
        samples_cluster_frame__ = pd.concat([samples_cluster_frame__, day_id_frame], axis=1)
    if add_uuid:
        samples_cluster_frame__ = pd.concat([samples_cluster_frame__, meter_uid_frame], axis=1)

    return samples_cluster_frame__

def create_syntethic_profiles(copula_models, cluster, n_samples, **kwargs):
    annual_energy_values = copula_models[cluster]['original_data']['avg_gwh'].value_counts().sort_index()
    annual_energy_values_gwh = annual_energy_values.index.values.round(4)
    n_energy_values = len(annual_energy_values_gwh)

    annual_energy_to_sample_dict = dict(zip(annual_energy_values_gwh, np.repeat(n_samples, n_energy_values)))
    annual_energy_to_sample = pd.Series(annual_energy_to_sample_dict, name="avg_gwh")

    samples_copula_cluster_energy = sampling_by_annual_energy(copula_model=copula_models[cluster]["copula"],
                                                              data_original_copula=copula_models[cluster][
                                                                  "original_data"],
                                                              energy_values=annual_energy_to_sample,
                                                              **kwargs)
    samples_copula_cluster_energy["cluster"] = cluster


    return samples_copula_cluster_energy

if __name__ == "__main__":
    file_name_load_models = "../models/copula_model_load_with_reactive_power.pkl"
    with open(file_name_load_models, "rb") as pickle_file:
        copula_models = pickle.load(pickle_file)

    # The original data has the transformer count:
    # Cluster 0:  158 Transformers
    # Cluster 1:  170 Transformers
    # Cluster 2:  115 Transformers
    #            ----
    #     Total:  443 unique transformers ids.

    # create_syntethic_profiles(copula_models, 0, 10, add_uuid=True, add_day_id=True)

    n_days_per_transformer = 730  # 2 years of data (365 * 2)
    np.random.seed(12)
    n_cores = mp.cpu_count() - 1
    star_input = [(copula_models, 0, n_days_per_transformer),
                  (copula_models, 1, n_days_per_transformer),
                  (copula_models, 2, n_days_per_transformer)]

    with mp.Pool(processes=n_cores) as pool:
        solutions = pool.starmap(create_syntethic_profiles, star_input)

    #%%
    samples = pd.concat(solutions, axis=0)
    samples.reset_index(inplace=True, drop=True)

    samples_by_cluster = samples[samples["cluster"] == 0]
    samples_ = samples_by_cluster.drop(columns="cluster").copy()

    #%%
    # Process to pivot the data in melted presentation.
    ID = 150
    N_METERS = 33
    meters_data = []

    filter_dict = [{"regex": "_ap$"}, {"regex": "_rp$"}, "", ""]
    suffix = ["ap", "rp"]
    j = 0  # Only active power

    add_date_time_column = True

    x_axis = pd.date_range(start="2021-01-01", periods=48, freq="30T").strftime("%H:%M:%S")  # Remove year
    time_stamp_format = [f"q_{i}_{suffix[j]}" for i in range(1, 48 + 1)]
    dict_time_stamp_format = dict(zip(time_stamp_format, x_axis))


    meter_ids = samples_["meter_uid"].unique()
    for ID in trange(157-N_METERS, 157):
        idx_meter = samples_["meter_uid"] == meter_ids[ID]
        meter_data = samples_[idx_meter]
        ap_data_meter = meter_data.filter(**filter_dict[j])
        ap_column_names = ap_data_meter.columns.to_list()
        meter_data_ap = pd.concat([ap_data_meter, meter_data["day_id"].to_frame()], axis=1)
        meter_data_ap_melted = meter_data_ap.melt(id_vars=['day_id'], value_vars=ap_column_names)
        meter_data_ap_melted["time_stamp"] = meter_data_ap_melted["variable"].apply(lambda x: dict_time_stamp_format[x])
        meter_data_ap_melted.drop(columns="variable", inplace=True)

        if add_date_time_column:
            # % Convert a long range of ints into datetime
            start_date = pd.to_datetime("2020-12-31")
            meter_data_ap_melted["date"] = meter_data_ap_melted["day_id"].apply(lambda x: start_date +
                                                                                          datetime.timedelta(days=x))
            meter_data_ap_melted["date_time"] = meter_data_ap_melted["date"].astype(str) + " " + meter_data_ap_melted["time_stamp"]
            meter_data_ap_melted["date_time"] = pd.to_datetime(meter_data_ap_melted["date_time"])  # Todo: This is extremely slow improve with dictionary mapping
            meter_data_ap_melted = meter_data_ap_melted[["date_time", "value"]].copy()

        meter_data_ap_melted["meter_uid"] = meter_ids[ID]
        meters_data.append(meter_data_ap_melted)
    meters_data_all_melted = pd.concat(meters_data, axis=0, ignore_index=True)
    active_power = meters_data_all_melted.copy()

    #%% Unmelt by day_id
    if "day_id" in active_power.columns:
        unique_days = active_power["day_id"].unique()
        meter_data_pivoted = []
        for day_ in tqdm(unique_days, desc="Pivoting time series"):
            idx = meters_data_all_melted["day_id"] == day_
            filtered_meters = meters_data_all_melted[idx]
            meters_pivoted = filtered_meters.pivot(index='time_stamp', columns='meter_uid', values="value")
            meter_data_pivoted.append(meters_pivoted)

        meter_data_pivoted_frame = pd.concat(meter_data_pivoted, axis=0)
        meter_data_pivoted_frame.reset_index(drop=True, inplace=True)
    else:
        meter_data_pivoted_frame = active_power.pivot(index='date_time', columns='meter_uid', values="value")
    meter_data_pivoted_frame.round(3).to_csv("active_power_example.csv")
    #%%
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    meter_data_pivoted_frame.plot(ax=ax, legend=False)



