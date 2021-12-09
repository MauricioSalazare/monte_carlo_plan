from core.clustering import (ClusteringRLP,
                             plot_scores,
                             rlp_irradiance,
                             rlp_transformer,
                             rlp_active_reactive,
                             load_time_series,
                             unique_filename,
                             plotly_plot_solutions)
from sklearn.preprocessing import scale
import pandas as pd

file_name=r"data/processed_data/consumption_weather/time_series_data.csv"
file_path_outliers = r"data/processed_data/consumption_weather/outlier_transformers.csv"

data = load_time_series(file_name, file_path_outliers=file_path_outliers)
ap = rlp_transformer(data, kwargs_filter= {"regex": "_ap$"})

#%% Clustering
cluster_ap = ClusteringRLP()
cluster_ap.cluster_dataset(ap, plot_solutions=False)
score_values_ap = cluster_ap.get_clustering_scores()
plot_scores(score_values_ap["CDI"],
            score_values_ap["MDI"],
            score_values_ap["DBI"],
            score_values_ap["CHI"])


#%% Plot the profiles for active power
file_name = 'data/processed_data/plots/clustering/active_power_clusters.html'
algorithm_name = "GMM"
n_cluster = 3

ap_cluster_solutions = cluster_ap.get_cluster_labels()
ap_frame = ap.copy()
ap_scaled_frame = pd.DataFrame(scale(ap.values, axis=1), index= ap.index, columns=ap.columns)
ap_frame['cluster'] = ap_cluster_solutions.loc[algorithm_name, n_cluster]
ap_scaled_frame['cluster'] = ap_cluster_solutions.loc[algorithm_name, n_cluster]
plotly_plot_solutions(ap_scaled_frame, algorithm_name=algorithm_name, n_cluster=n_cluster, file_name=file_name)

#%% 1. create clustering results .csv file for transformers
file_path_load_clustering = "data/processed_data/consumption_weather/clustering_load.csv"
fixed_path = unique_filename(file_path_load_clustering)
ap_frame['cluster'].to_csv(fixed_path, index=True)


