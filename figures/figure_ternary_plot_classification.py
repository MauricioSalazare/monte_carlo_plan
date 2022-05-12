import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from core.clustering import rlp_irradiance, load_time_series


file_name=r"../data/processed_data/consumption_weather/time_series_data.csv"
file_path_outliers = r"../data/processed_data/consumption_weather/outlier_transformers.csv"
file_irradiance_full_data = r"../data/processed_data/consumption_weather/knmi_hourly_all_w_m2_qg.csv"
file_irradiance_clusters = r"../data/processed_data/consumption_weather/clustering_irradiance.csv"

data = load_time_series(file_name, file_path_outliers=file_path_outliers)
clusters_irradiance = pd.read_csv(file_irradiance_clusters, index_col=0)  # Days labeled by clustering algorithm
knmi_hourly_w_m2 = pd.read_csv(file_irradiance_full_data,
                               parse_dates=True,
                               index_col='TIMESTAMP',
                               date_parser=lambda col: pd.to_datetime(col, utc=True))


irr_data_all, idx_mask = rlp_irradiance(data.resample("H").mean())
labeled_data = pd.concat([irr_data_all, clusters_irradiance], axis=1)
idx_nan = labeled_data.cluster.isna()
labeled_data = labeled_data[~idx_nan.values]
labeled_data.cluster = labeled_data.cluster.astype("int64")


data_X = labeled_data.iloc[:, :-1].values.copy()
y = labeled_data.iloc[:, -1].values.copy()

X = scale(data_X, axis=1)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)  # X == (n_samples, n_features)


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
data_frame.columns = ["Cloudy", "Sunny", "Overcast"]
data_frame.to_csv("/fig")

#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
fig = px.scatter_ternary(data_frame, a="Cloudy", b="Overcast", c="Sunny", width=600, height=510)
fig.write_image(f"ternary_plot/ternary_classified_irradiance.pdf")
fig.show()