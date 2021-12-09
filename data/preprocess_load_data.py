import pandas as pd
from typing import List
import numpy as np
import calendar
import matplotlib.pyplot as plt

def load_consumption_data(file_name: str) -> pd.DataFrame:
    power = pd.read_csv(file_name,
                        parse_dates=True,
                        index_col='date',
                        date_parser=lambda col: pd.to_datetime(col, utc=True))
    power = power.resample('15T').mean()

    return power

def filter_power(data_frame: pd.DataFrame,
                 years: List[int] = None,
                 months: List[int] = None,
                 percentage=0.8) -> pd.DataFrame:
    """
    Filter transformers which contains at least a minimum amount of "percentage" of non-nan values in the dataset.
    Also, it is filtered by years and moths.

    """

    power_filtered = data_frame.copy()

    if years is not None:
        idx_ = np.zeros(len(power_filtered), dtype=bool)
        for year in years:
            idx_ = idx_ ^ (power_filtered.index.year == year)

        power_filtered = power_filtered[idx_]

        if months is not None:
            idx_ = np.zeros(len(power_filtered), dtype=bool)
            for month in months:
                idx_ = idx_ ^ (power_filtered.index.month == month)

            power_filtered = power_filtered[idx_]

    n_samples = len(power_filtered)
    idx = power_filtered.isna().sum(axis=0) >= n_samples * percentage
    transformers = power_filtered.loc[:, idx[~idx.values].index.to_list()]

    transformers_backward_fill = transformers.bfill()
    transformers_forward_fill = transformers_backward_fill.ffill()

    return transformers_forward_fill.copy()

def annual_energy_consumption(data_frame: pd.DataFrame, year: int) -> pd.DataFrame:
    idx = data_frame.index.year == year
    filter_data = data_frame[idx].copy()
    days = 366 if calendar.isleap(year) else 365

    assert filter_data.index.inferred_freq == '15T', "Frequency is not 15 min."
    assert len(filter_data) ==  days * 96, "Data incomplete"

    return (filter_data * 0.25).sum(axis=0, skipna=True).divide(1e6)

if __name__ == "__main__":
    file_name_active_power = r"raw_data\dali_data\active_power_enschede.csv"
    file_name_reactive_power = r"raw_data\dali_data\reactive_power_enschede.csv"

    active_power = load_consumption_data(file_name_active_power)
    reactive_power = load_consumption_data(file_name_reactive_power)

    ap_filtered = filter_power(active_power, years=[2020])
    rp_filtered = filter_power(reactive_power, years=[2020])

    set_ap_transformers = set(ap_filtered.columns)
    set_rp_transformers = set(rp_filtered.columns)

    # Transformers with sufficient data on both datasets:
    selected_transformers = set_ap_transformers.intersection(set_rp_transformers)
    selected_transformers -= {'ESD.001121-1'}

    ap_filtered = ap_filtered.loc[:, selected_transformers]
    rp_filtered = rp_filtered.loc[:, selected_transformers]

    # Anonymize transformers
    mapper_ap = dict(zip(selected_transformers, [f"tr_{ii}_ap" for ii in range(len(selected_transformers))]))
    mapper_rp = dict(zip(selected_transformers, [f"tr_{ii}_rp" for ii in range(len(selected_transformers))]))

    # Merge and export
    ap_filtered_tr = ap_filtered.rename(columns=mapper_ap)
    rp_filtered_tr = rp_filtered.rename(columns=mapper_rp)

    consumption_data = pd.concat([ap_filtered_tr, rp_filtered_tr], axis=1)
    year_consumption_Gwh = annual_energy_consumption(ap_filtered_tr, year=2020)

    year_consumption_Gwh.to_frame(name="energy_gwh").to_csv(r"processed_data/consumption_weather/year_consumption_gwh.csv")
    consumption_data.to_csv(r"processed_data/consumption_weather/consumption_active_reactive.csv")

    #%%
    start_date = "2020-07-17 00:00:00"
    end_date = "2020-07-22 00:00:00"

    ap_filtered_period = ap_filtered[(ap_filtered.index > start_date) & (ap_filtered.index < end_date)]
    rp_filtered_period = rp_filtered[(rp_filtered.index > start_date) & (rp_filtered.index < end_date)]

    #%%
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    ax[0].plot(ap_filtered_period, linewidth=0.3)
    ax[1].plot(rp_filtered_period, linewidth=0.3)
    for ax_ in ax:
        ax_.grid()
    ax[0].set_title("MV Distribution transformer load")
    ax[0].set_ylabel("Active Power [kW]")
    ax[1].set_ylabel("Reactive Power [kVA]")






