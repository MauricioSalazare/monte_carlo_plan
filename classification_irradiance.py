"""
After irradiance clustering, this script creates a k-means classifier to label the data of the irradiance
for days from 2011 - 2021

"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.clustering import (ClusteringRLP,
                             plot_scores,
                             rlp_irradiance,
                             load_time_series,
                             unique_filename,
                             plotly_plot_solutions)
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
# titles = {0: "Cloudy day",
#           1: "Sunny day",
#           2: "Dark/Rainy day"}


#%%
# Data for the Twente climatological station.
latitude = 52.27314817052
longitude = 6.8908745111116 + 8  # Offset to align sunrise and sunset (equivalent of shifting the time)
elevation = 33.2232  # Meters

twente = Location(latitude=latitude, longitude=longitude, tz='UTC', altitude=elevation, name='Twente')
# times = pd.date_range(start='20-07-01', end='2016-07-04', freq='1min', tz=twente.tz)
times = pd.date_range(start='1987-01-01', end='2021-11-21', freq='H', tz=twente.tz)
cs_simplified_solis = twente.get_clearsky(times, model="simplified_solis")
cs_haurwitz = twente.get_clearsky(times, model="haurwitz")
cs_ineichen = twente.get_clearsky(times, model="ineichen")

cs_simplified_solis.rename(columns={"ghi": "ghi_solis"}, inplace=True)
cs_haurwitz.rename(columns={"ghi": "ghi_haurwitz"}, inplace=True)
cs_ineichen.rename(columns={"ghi": "ghi_ineichen"}, inplace=True)

# cs.index = cs.index - pd.Timedelta(hours=0.5)

#%%
file_name=r"data/processed_data/consumption_weather/time_series_data.csv"
file_path_outliers = r"data/processed_data/consumption_weather/outlier_transformers.csv"
file_irradiance_full_data = r"data/processed_data/consumption_weather/knmi_hourly_all_w_m2_qg.csv"
file_irradiance_clusters = r"data/processed_data/consumption_weather/clustering_irradiance.csv"

data = load_time_series(file_name, file_path_outliers=file_path_outliers)
clusters_irradiance = pd.read_csv(file_irradiance_clusters, index_col=0)
knmi_hourly_w_m2 = pd.read_csv(file_irradiance_full_data,
                               parse_dates=True,
                               index_col='TIMESTAMP',
                               date_parser=lambda col: pd.to_datetime(col, utc=True))

#%% Merge CSI and Irradiance and check which model gives the best fit
knmi_csi = pd.concat([knmi_hourly_w_m2,
                      cs_simplified_solis["ghi_solis"],
                      cs_haurwitz["ghi_haurwitz"],  # This is the one used in sustainable energy paper ("Stochastic model for generation of high-resolution irradiance)
                      cs_ineichen["ghi_ineichen"]], ignore_index=False, axis=1)
# knmi_csi.loc['2021-05-27':'2021-06-08'].plot()
work_range = knmi_csi.loc['2020-01-01':'2020-12-30'].copy()
work_range.plot()
work_range.rename(columns={"ghi_solis": "ghi"}, inplace=True)



#%% Saturate the true readings to avoid values bigger than 1
knmi_test = work_range[["qg", "ghi"]].copy(deep=True)
# idx = (knmi_test["qg"] > knmi_test["ghi"]).values
# knmi_test.loc[idx, "qg"] = knmi_test.loc[idx, "ghi"]

# idx = (knmi_test["ghi"] == 0.0).values
# knmi_test.loc[idx, "qg"] = 0

#%%
fig, ax = plt.subplots(1,1, figsize=(20,5))
knmi_test.plot(ax=ax)
#%%


fig, ax = plt.subplots(1,1, figsize=(20,5))
knmi_test.plot(ax=ax)
knmi_test["csi"] = knmi_test["qg"] / knmi_test["ghi"]


knmi_test["csi"][knmi_test["csi"] > 2.0] = np.nan
knmi_test["csi"].replace(np.inf, 0, inplace=True)
# knmi_test["csi"].fillna(0, inplace=True)
# work_range["ghi"] / work_range["qg"]
ax_2= ax.twinx()
knmi_test["csi"].resample("D").mean().plot(drawstyle="steps-post", linewidth=2, ax=ax_2)


#%%
fig, ax = plt.subplots(1,1, figsize=(5,5))
knmi_test["csi"].plot.hist(bins=50, ax=ax)


#%%
csi_daily = knmi_test["csi"].resample("D").mean()  # If you resample per hour MUST USE ffill

#%% Load data and labels
irr_data_low_res, idx_mask_low_res = rlp_irradiance(data)
energy_irradiance = irr_data_low_res.multiply(0.25/1000).sum(axis=1)

#%%
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.histplot(x=energy_irradiance, ax= ax)

min_irr = energy_irradiance.min()
max_irr = energy_irradiance.max()

bin_irr = (max_irr - min_irr) / 3

idx_dark = (energy_irradiance < min_irr + bin_irr).values
idx_cloudy = ((energy_irradiance > min_irr + bin_irr)  & (energy_irradiance < min_irr + 2 * bin_irr)).values
idx_sunny = ((energy_irradiance > min_irr + 2 * bin_irr)  & (energy_irradiance < min_irr + 3 * bin_irr)).values


new_clustering = irr_data_low_res.copy()
new_clustering["cluster"] = 0
new_clustering["cluster"][idx_dark] = 2
new_clustering["cluster"][idx_cloudy] = 0
new_clustering["cluster"][idx_sunny] = 1


#%%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax_ = ax.flatten()

for ax_i, cluster_ in zip(ax_, [0,1,2]):
    idx_cluster = new_clustering["cluster"] == cluster_
    ax_i.plot(new_clustering[idx_cluster].values.T)


#%%





clean_day = irr_data_low_res.loc[213].values
clean_day_2 = irr_data_low_res.loc[266].values

dark_day = irr_data_low_res.loc[207].values
dark_day_2 = irr_data_low_res.loc[209].values

cloudy_day = irr_data_low_res.loc[217].values

#%%
fig, ax = plt.subplots(1,1)
ax.plot(clean_day, '-o', color='b')
ax.plot(clean_day_2, '-o', color='r')
ax_2 = ax.twinx()
# ax_2.plot(np.diff(clean_day), '--o', color='b')
# ax_2.plot(np.diff(clean_day_2), '--o', color='r')
# ax_2.plot(np.diff(np.diff(clean_day)), '--x', color='b')
ax.grid()






#%%
fig, ax = plt.subplots(1,1)
ax.plot(dark_day, '-o', color='k')
ax.plot(dark_day_2, '-o', color='g')
ax_2 = ax.twinx()
# ax_2.plot(np.diff(dark_day), '--o', color='k')
# ax_2.plot(np.diff(dark_day_2), '--o', color='g')
# ax_2.plot(np.diff(np.diff(dark_day)), '--x')
ax.grid()

#%%
fig, ax = plt.subplots(1,1)
# ax.plot(cloudy_day, '-o', color='orange')
# ax.plot(dark_day_2, '-o', color='o')
ax_2 = ax.twinx()
ax_2.plot(np.diff(cloudy_day), '--o', color='orange')
# ax_2.plot(np.diff(dark_day_2), '--o', color='g')
# ax_2.plot(np.diff(np.diff(dark_day)), '--x')
ax.grid()

#%%
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


kmeans = KMeans(n_clusters=3, random_state=1234, n_init=1000)
data_set = scale(irr_data_low_res, axis=0)
data_set = irr_data_low_res
kmeans.fit(data_set)
labels = kmeans.predict(data_set)


labeled_data_year = pd.concat([irr_data_low_res,
                               pd.DataFrame(labels, index=irr_data_low_res.index, columns=["cluster"])],
                               axis=1)
#%%
fig, ax = plt.subplots(1,1)
ax.plot(data_set.T)

#%%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax_ = ax.flatten()

for ax_i, cluster_ in zip(ax_, [0,1,2]):
    idx_cluster = labeled_data_year["cluster"] == cluster_
    ax_i.plot(labeled_data_year[idx_cluster].values.T)





#%%


irr_data_all, idx_mask = rlp_irradiance(data.resample("H").mean())
labeled_data = pd.concat([irr_data_all, clusters_irradiance], axis=1)
idx_nan = labeled_data.cluster.isna()
labeled_data = labeled_data[~idx_nan.values]
labeled_data.cluster = labeled_data.cluster.astype("int64")


#%% Train classifier
# Test scalers
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

data_X = labeled_data.iloc[:, :-1].values.copy()
y = labeled_data.iloc[:, -1].values.copy()

X = scale(data_X, axis=1)
mu_ = data_X.mean(axis=1)
std_ = data_X.std(axis=1)
scaled_X = (data_X - mu_[:,np.newaxis]) / std_[:, np.newaxis]
f_x = lambda x: (x - mu_[:,np.newaxis]) / std_[:, np.newaxis]


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)  # X == (n_samples, n_features)
neigh.predict(X)

#%%
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(X.T)

#%% Filter data and classify
year_range = np.arange(1988, 2021, 1)
month_range = np.arange(7, 10, 1)
mixture_years = {}
labeled_data_years = {}
for year_ in year_range:
    print(f"Year: {year_}")
    idx_1 = knmi_hourly_w_m2.index.year == year_
    # idx_2 = (knmi_hourly_w_m2.index.month == 7) | (knmi_hourly_w_m2.index.month == 8) | (knmi_hourly_w_m2.index.month == 9)
    # idx_2 = (knmi_hourly_w_m2.index.month == 3) | (knmi_hourly_w_m2.index.month == 4)
    for month_ in month_range:
        idx_2 = (knmi_hourly_w_m2.index.month == month_)

        idx_net = idx_1 & idx_2
        rlp_year, _ = rlp_irradiance(knmi_hourly_w_m2[idx_net], idx_mask=idx_mask)
        rlp_year_scaled = scale(rlp_year, axis=1)
        label_year  = neigh.predict(rlp_year_scaled)
        class_label, count_label = np.unique(label_year, return_counts=True)
        mixture_years[str(year_) + "-" + str(month_)] = dict(zip( class_label, count_label / len(label_year) ))

        labeled_data_year = pd.concat([rlp_year,
                                       pd.DataFrame(label_year, index=rlp_year.index, columns=["cluster"])],
                                       axis=1)
        labeled_data_years[str(year_) + "-" + str(month_)] = labeled_data_year

data_frame = pd.DataFrame.from_dict(mixture_years, orient="index")
data_frame.columns = ["Cloudy", "Sunny", "Dark"]

#%%
YEAR = "1990-7"
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax_ = ax.flatten()

for ax_i, cluster_ in zip(ax_, [0,1,2]):
    idx_cluster = labeled_data_years[YEAR]["cluster"] == cluster_
    ax_i.plot(labeled_data_years[YEAR][idx_cluster].values.T)


#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.scatter_ternary(data_frame, a="Cloudy", b="Dark", c="Sunny")
fig.show()
