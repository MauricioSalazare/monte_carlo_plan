import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_name_power = r"processed_data/consumption_weather/consumption_active_reactive.csv"
power = pd.read_csv(file_name_power,
                    parse_dates=True,
                    index_col='date',
                    date_parser=lambda col: pd.to_datetime(col, utc=True))
power = power.resample('15T').mean()

file_name_irradiance = r"processed_data/consumption_weather/knmi_15min_qg.csv"
irradiance =  pd.read_csv(file_name_irradiance,
                          parse_dates=True,
                          index_col='date',
                          date_parser=lambda col: pd.to_datetime(col, utc=True))
irradiance = irradiance.resample('15T').mean()
merged_dataset = pd.concat([power, irradiance], axis=1, join="inner")

#%% Filter by months
months = [7, 8, 9]
idx = np.zeros(len(merged_dataset), dtype=bool)
for month in months:
    idx = idx ^ (merged_dataset.index.month == month)

merged_dataset_filtered = merged_dataset[idx]
active_power = merged_dataset_filtered.filter(regex="_ap$", axis=1)
reactive_power = merged_dataset_filtered.filter(regex="_rp$", axis=1)
irradiance = merged_dataset_filtered.filter(items=["qg"], axis=1)

#%% Plot the final time series
fig, ax = plt.subplots(3, 1, figsize=(15, 10))
ax[0].plot(active_power, linewidth=0.5)
ax[1].plot(reactive_power, linewidth=0.5)
ax[2].plot(irradiance, linewidth=0.9)
for ax_ in ax:
    ax_.grid()

merged_dataset_filtered.to_csv(r"processed_data/consumption_weather/time_series_data.csv")