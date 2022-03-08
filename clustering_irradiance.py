from core.clustering import (ClusteringRLP,
                             plot_scores,
                             rlp_irradiance,
                             load_time_series,
                             unique_filename,
                             plotly_plot_solutions)
from sklearn.preprocessing import scale
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

# titles = {0: "Cloudy day",
#           1: "Sunny day",
#           2: "Dark/Rainy day"}


file_name=r"data/processed_data/consumption_weather/time_series_data.csv"
file_path_outliers = r"data/processed_data/consumption_weather/outlier_transformers.csv"

data = load_time_series(file_name, file_path_outliers=file_path_outliers)

#%%
irr_data_all, _ = rlp_irradiance(data)

#%% Drop super low irradiance days and add them at the end
energy_irradiance = irr_data_all.multiply(0.25/1000).sum(axis=1)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.histplot(x=energy_irradiance, ax= ax)
idx = (energy_irradiance < 2.5).values

data_set_low_irradiance = irr_data_all.iloc[idx].values
_, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(data_set_low_irradiance.T, linewidth=0.1, color='#808080', alpha=0.8)
ax.plot(np.nanmean(data_set_low_irradiance.T, axis=1), linewidth=0.5, color='k', label='mean')
ax.plot(np.nanquantile(data_set_low_irradiance.T, q=0.05, axis=1),
        color='r', linewidth=0.5, linestyle='--', label='q0.05')
ax.plot(np.nanquantile(data_set_low_irradiance.T, q=0.95, axis=1),
        color='r', linewidth=0.5, linestyle='-', label='q0.95')
ax.grid(False)
ax.set_title(f"Low irradiance days. Count: ({data_set_low_irradiance.T.shape[1]}) -- Energy limit: {2.5} kWh/m^2",
                 fontsize=10, y=1.03, transform=ax.transAxes)
ax.set_ylim((0, 1000))

# low_irr_days = irr_data_all[idx].copy()
# irr_data = irr_data_all[~idx].copy()
irr_data = irr_data_all.copy()

#%% Clustering
cluster_irr = ClusteringRLP()
cluster_irr.cluster_dataset(irr_data, plot_solutions=False, end=20, axis=0, std_scale=True)
score_values_irr = cluster_irr.get_clustering_scores()
fig_scores_irradiance = plot_scores(score_values_irr["CDI"],
                                    score_values_irr["MDI"],
                                    score_values_irr["DBI"],
                                    score_values_irr["CHI"])
fig_scores_irradiance.suptitle("Scores irradiance")

#%%
file_name = 'data/processed_data/plots/clustering/irradiance_clusters.html'
algorithm_name = "GMM"
n_cluster = 3

irr_cluster_solutions = cluster_irr.get_cluster_labels()
irr_frame = irr_data.copy()
irr_scaled_frame = pd.DataFrame(scale(irr_data.values, axis=1), index= irr_data.index, columns=irr_data.columns)
irr_frame['cluster'] = irr_cluster_solutions.loc[algorithm_name, n_cluster]
irr_scaled_frame['cluster'] = irr_cluster_solutions.loc[algorithm_name, n_cluster]
plotly_plot_solutions(irr_frame, algorithm_name=algorithm_name, n_cluster=n_cluster, file_name=file_name)
print(f"Cluster count:\n{irr_scaled_frame['cluster'].value_counts()}")


#%% Use PCA to check improvement:
# n_components = 5
scaled = scale(irr_data.values, axis=1)
explain_ratio = 0.95
pca = PCA(n_components=explain_ratio)
pca_model = pca.fit(scaled)
X_r = pca_model.transform(scaled)
print(f'Number of components: {X_r.shape[1]}  --  Explained ratio: {pca_model.explained_variance_ratio_.sum():.2f}')
pca_data = pd.DataFrame(X_r, index=irr_data.index, columns=['PC' + str(ii) for ii in range(X_r.shape[1])])

cluster_pca = ClusteringRLP()
cluster_pca.cluster_dataset(pca_data, end=20, axis=0, std_scale=True)
score_values_pca = cluster_pca.get_clustering_scores()
fig_scores_pca = plot_scores(score_values_pca["CDI"],
                         score_values_pca["MDI"],
                         score_values_pca["DBI"],
                         score_values_pca["CHI"])
fig_scores_pca.suptitle("PCA Scores")

file_name_pca = 'data/processed_data/plots/clustering/irradiance_pca_clusters.html'
algorithm_name = "Birch"
n_cluster = 3

irr_pca_cluster_solutions = cluster_pca.get_cluster_labels()
irr_frame = irr_data.copy()
irr_scaled_frame = pd.DataFrame(scale(irr_data.values, axis=1), index= irr_data.index, columns=irr_data.columns)
irr_frame['cluster'] = irr_pca_cluster_solutions.loc[algorithm_name, n_cluster]
irr_scaled_frame['cluster'] = irr_pca_cluster_solutions.loc[algorithm_name, n_cluster]
plotly_plot_solutions(irr_frame, algorithm_name=algorithm_name, n_cluster=n_cluster, file_name=file_name_pca)
print(f"Cluster count:\n{irr_scaled_frame['cluster'].value_counts()}")


#%% MERGE CLUSTER WITH LOW IRRADIANCE DAYS
# CLUSTER_TO_MERGE = input("Select the CLUSTER_TO_MERGE (The one with lowest irradiance)!!!")
# low_irr_days["cluster"] = 2
# irr_data_final = pd.concat([irr_frame, low_irr_days], axis=0).sort_index()

irr_data_final = irr_frame.copy()

#%%
_, ax_ = plt.subplots(1, n_cluster, figsize=(20, 4))
ax = ax_.flatten()
for cluster_number, ax in enumerate(ax_):
    data_set = irr_data_final[irr_data_final["cluster"] == cluster_number].iloc[:,:-1].values

    ax.plot(data_set.T, linewidth=0.1, color='#808080', alpha=0.8)
    ax.plot(np.nanmean(data_set.T, axis=1), linewidth=0.5, color='k', label='mean')
    ax.plot(np.nanquantile(data_set.T, q=0.05, axis=1),
            color='r', linewidth=0.5, linestyle='--', label='q0.05')
    ax.plot(np.nanquantile(data_set.T, q=0.95, axis=1),
            color='r', linewidth=0.5, linestyle='-', label='q0.95')
    ax.grid(False)
    ax.set_title(f"({data_set.T.shape[1]})",
                 fontsize=10, y=1.03, transform=ax.transAxes)
    ax.set_ylim((0, 1000))

#%% Save solutions
# file_path_irradiance_clustering = "data/processed_data/consumption_weather/clustering_irradiance.csv"
# fixed_path = unique_filename(file_path_irradiance_clustering)
# irr_data_final["cluster"].to_csv(fixed_path, index=True)

