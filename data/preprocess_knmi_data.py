"""
Compare KNMI datasets downloaded from 2 different sources:
1. https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens (hourly resolution)
2. https://dataplatform.knmi.nl/ (10 min resolution)
At hourly and 10 minutes resolution.

Save he 15min resolution KNMI irradiance data in a processed file.
The final file only contains irradiance ("qg") in the units of W/m^2

"""

import pandas as pd
import matplotlib.pyplot as plt


# 10 min resolution:
file_path = r"raw_data/knmi_data/10_min_resolution/knmi_data_enschede_nov_2021_10_min.csv"
batch_10_min = pd.read_csv(file_path, parse_dates=True, index_col='time')
batch_10_min.index = batch_10_min.index.tz_localize("UTC")
knmi_10min = batch_10_min["qg"].to_frame()

# Hourly resolution
file_path_1 = r"raw_data/knmi_data/hourly/uurgeg_290_2011-2020/uurgeg_290_2011-2020.txt"
file_path_2 = r"raw_data/knmi_data/hourly/uurgeg_290_2021-2030/uurgeg_290_2021-2030.txt"

file_path_3 = r"raw_data/knmi_data/hourly/uurgeg_290_1981-1990/uurgeg_290_1981-1990.txt"
file_path_4 = r"raw_data/knmi_data/hourly/uurgeg_290_1991-2000/uurgeg_290_1991-2000.txt"
file_path_5 = r"raw_data/knmi_data/hourly/uurgeg_290_2001-2010/uurgeg_290_2001-2010.txt"

batch_1 = pd.read_csv(file_path_1, skiprows=31)
batch_2 = pd.read_csv(file_path_2, skiprows=31)

# batch_2 = batch_2.apply(pd.to_numeric, errors='coerce')
# batch_2.apply(lambda x: x.astype("int64"))

batch_3 = pd.read_csv(file_path_3, skiprows=31)
batch_4 = pd.read_csv(file_path_4, skiprows=31)
batch_5 = pd.read_csv(file_path_5, skiprows=31)

batch_3 = batch_3.apply(pd.to_numeric, errors='coerce')
batch_4 = batch_4.apply(pd.to_numeric, errors='coerce')
batch_5 = batch_5.apply(pd.to_numeric, errors='coerce')



old_column_names = batch_1.columns.to_list()
new_column_names = [column_name.strip() for column_name in old_column_names]
mapper = dict(zip(old_column_names, new_column_names))

batch_1.rename(columns=mapper, inplace=True)
batch_2.rename(columns=mapper, inplace=True)
batch_3.rename(columns=mapper, inplace=True)
batch_4.rename(columns=mapper, inplace=True)
batch_5.rename(columns=mapper, inplace=True)

batch = pd.concat([batch_1, batch_2], ignore_index=True)
batch = pd.concat([batch_3, batch_4, batch_5, batch_1, batch_2], ignore_index=True)

batch.HH -= 1
batch['MINUTES'] = 0
batch['SECONDS'] = 0
batch['TIMESTAMP'] = batch[['YYYYMMDD', 'HH', 'MINUTES', 'SECONDS']].astype(str).agg('-'.join, axis=1)
batch["TIMESTAMP"] = pd.to_datetime(batch["TIMESTAMP"], format="%Y%m%d-%H-%M-%S")
batch.set_index("TIMESTAMP", drop=True, inplace=True)
batch.index = batch.index.tz_localize('UTC')
knmi_hourly = batch["Q"].to_frame(name="qg")

# knmi_hourly = knmi_hourly.dropna(axis=0)
knmi_hourly = knmi_hourly.resample('H').mean()
knmi_10min = knmi_10min.resample('10T').mean()

#%% Test if the data from two sources are similar
start_date = "2020-07-17 00:00:00"
end_date = "2020-07-22 00:00:00"

hourly_filtered = knmi_hourly[(knmi_hourly.index >  start_date) & (knmi_hourly.index < end_date)]
minutely_filtered = knmi_10min[(knmi_10min.index >  start_date) & (knmi_10min.index < end_date)]

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(hourly_filtered, color='r', label="Hourly in J/cm^2")
ax.plot(minutely_filtered, color="orange", label="10min in W/m^2")
ax.plot(minutely_filtered.resample('H').mean(), color="purple", label="10min resampled in W/m^2")
ax.plot(minutely_filtered.resample('H').mean() * 0.36, color="green", label="10min resampled in J/cm^2")
ax.plot(hourly_filtered * 2.77, color="blue", label="Hourly in W/m^2")
ax.set_title("Global irradiance in different units (2 different data sources)")
ax.set_ylabel("J/cm^2  or  W/m^w")
ax.grid()
ax.legend()

knmi_hourly_w_m2 = knmi_hourly * 2.77  # 2.77 is the factor fom J/cm^2 to W/m^2

knmi_15min = knmi_10min.resample('15T').mean()
idx = knmi_15min.index.year == 2020
knmi_15min_filtered = knmi_15min[idx]
knmi_15min_filtered.index.rename("date", inplace=True)
# knmi_15min_filtered.to_csv("processed_data/consumption_weather/knmi_15min_qg.csv")
# knmi_hourly_w_m2.to_csv("processed_data/consumption_weather/knmi_hourly_all_w_m2_qg.csv")