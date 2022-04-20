"""
Script is a copy of the clustering_load_profiles.py.
This script has the purpose of create the plots for the journal
"""

from core.clustering import (ClusteringRLP,
                             rlp_transformer,
                             load_time_series)
import pandas as pd
from sklearn import  metrics
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt

file_name=r"../data/processed_data/consumption_weather/time_series_data.csv"
file_path_outliers = r"../data/processed_data/consumption_weather/outlier_transformers.csv"

data = load_time_series(file_name, file_path_outliers=file_path_outliers)
ap = rlp_transformer(data, kwargs_filter= {"regex": "_ap$"})

#%% Clustering
cluster_ap = ClusteringRLP()
cluster_ap.cluster_dataset(ap, plot_solutions=True, init=2, end=15)
score_values_ap = cluster_ap.get_clustering_scores()
ap_cluster_solutions = cluster_ap.get_cluster_labels()
print(pd.DataFrame(ap_cluster_solutions[3]["GMM"]).value_counts())

#%% Print clustering solutions
N_CLUSTER = 3
algorithms = set(ap_cluster_solutions[N_CLUSTER].index)
for algo in algorithms:
    y_labels = ap_cluster_solutions[N_CLUSTER][algo]
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    plt.subplots_adjust(bottom=0.2, hspace = 0.3)
    for cluster, ax in zip(range(N_CLUSTER), axes.flatten()):
        ax.plot(ap.iloc[(y_labels == cluster), :].values.T, linewidth=0.3)
        ax.set_title(f'Cluster: {cluster} - Count: {np.sum((y_labels == cluster))}')
    plt.suptitle(f"Algorithm: {algo}")

#%% Reconfigure the mapping of cluster labels according to Spectral algorithm
algo_mapping = {}
algo_mapping["Spectral"] = {0: 0, 1: 1, 2: 2}
algo_mapping["Birch"] = {0: 1, 1: 0, 2: 2}
algo_mapping["GMM"] = {0: 2, 1: 0, 2: 1}
algo_mapping["HC(Ward)"] = {0: 1, 1: 2, 2: 0}
algo_mapping["KMeans"] = {0: 0, 1: 1, 2: 2}
algo_mapping["Kmedoids"] = {0: 1, 1: 2, 2: 0}

#%%
score_values_list = ["CDI", "MDI", "DBI", "CHI", "SI"]
title_list = {"CDI": "Clustering dispersion indicator  (Low-Good)",
              "MDI": "Modified Dunn Index  (Low-Good)",
              "DBI": "Davies-Bouldin Index (Low-Good)",
              "CHI": "Calinski-Habarasz  (High-Good)",
              "SI": "Silohuette Index" }

clustering_algorithms = ap_cluster_solutions.shape[0]
markers_cycle = list(islice(cycle(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '8', 's', 'p', 'P']),
                            clustering_algorithms))

#%%
fig, axs = plt.subplots(3, 2, figsize=(4, 8))
plt.subplots_adjust(bottom=0.05, hspace = 0.4, top=0.95, wspace=0.4)

ax = axs.flatten()
for ii, score_ in enumerate(score_values_list):
    for (index, value), marker in zip(score_values_ap[score_].iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[ii])
    ax[ii].set_title(title_list[score_])
    ax[ii].set_xticks(np.array(ap_cluster_solutions.columns))
    if ii ==4:
        ax[ii].legend()

ranked_scores_cluster_three = pd.concat([score_values_ap[key_][[3]].rename(columns={3: key_}) for key_ in score_values_ap], axis=1)
ranked_scores_cluster_three_filtered = ranked_scores_cluster_three[["DBI", "MDI", "CHI", "SI"]]
print("Scores for cluster 3")
print(ranked_scores_cluster_three_filtered.sort_values(by=["DBI", "MDI"]))

#%% Adjusted rand score to check the similarity between the clustering results. Reference is the Spectral clustering:
y_real = ap_cluster_solutions[N_CLUSTER]["Spectral"]
rand_scores = {}
for algo in algorithms:
    y_pred = ap_cluster_solutions[N_CLUSTER][algo]
    y_pred = [algo_mapping[algo][y_value] for y_value in y_pred]
    rand_scores[algo] = np.round(metrics.adjusted_rand_score(y_real, y_pred), 3)
rand_indexes = pd.DataFrame.from_dict(rand_scores, orient="index", columns=["RI"]).sort_values(by="RI", ascending=False)
print(rand_indexes)
#%% Count the number of transformers
name_mapping_ = {0: "C1", 1: "C2", 2: "C3"}

cluster_count = []
for algo in algorithms:
    aa = pd.DataFrame(ap_cluster_solutions[N_CLUSTER][algo], columns=[algo])
    aa = aa.value_counts().to_frame(name="cluster_count").sort_index()
    aa = aa.reset_index().rename(columns={algo: "cluster"})

    # aa["cluster"] = aa["cluster"].map(algo_mapping[algo])

    aa["cluster"] = aa["cluster"].map(name_mapping_)
    aa = aa.set_index("cluster")
    aa = aa.transpose().rename({"cluster_count": algo})
    cluster_count.append(aa)

cluster_count = pd.concat(cluster_count, axis=0)
cluster_count = cluster_count[["C1", "C2", "C3"]]

#%% Merge all the table
table = pd.concat([ranked_scores_cluster_three_filtered.round(3), rand_indexes.round(3), cluster_count], axis=1)
table = table.sort_values(by=["DBI", "MDI"])
table.to_csv("load_model/clustering_metrics.csv")
print(table)

