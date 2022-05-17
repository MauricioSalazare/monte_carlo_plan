"""
Script is a copy of the clustering_load_profiles.py.
This script has the purpose of create the plots for the journal
"""
from core.clustering import (ClusteringRLP,
                             rlp_transformer,
                             load_time_series)
import pandas as pd
from sklearn import metrics
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from core.figure_utils import set_figure_art
set_figure_art()
# mpl.rc('text', usetex=False)

file_name = r"../data/processed_data/consumption_weather/time_series_data.csv"
file_path_outliers = r"../data/processed_data/consumption_weather/outlier_transformers.csv"

data = load_time_series(file_name, file_path_outliers=file_path_outliers)
ap = rlp_transformer(data, kwargs_filter={"regex": "_ap$"})

#%% Clustering
np.random.seed(12345)
cluster_ap = ClusteringRLP()
cluster_ap.cluster_dataset(ap, plot_solutions=False, init=2, end=15)
score_values_ap = cluster_ap.get_clustering_scores()
ap_cluster_solutions = cluster_ap.get_cluster_labels()

#%% Print clustering solutions
N_CLUSTER = 3
algorithms = set(ap_cluster_solutions[N_CLUSTER].index)

for algo in algorithms:
    y_labels = ap_cluster_solutions[N_CLUSTER][algo]
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    plt.subplots_adjust(bottom=0.2, hspace=0.3)
    for cluster, ax in zip(range(N_CLUSTER), axes.flatten()):
        ax.plot(ap.iloc[(y_labels == cluster), :].values.T, linewidth=0.3)
        ax.set_title(f'Cluster: {cluster} - Count: {np.sum((y_labels == cluster))}')
    plt.suptitle(f"Algorithm: {algo}")

#%% Reconfigure the mapping of cluster labels according to Spectral algorithm
# YOU MUST DO THIS BY EYE!
algo_mapping = dict()
algo_mapping["Spectral"] = {0: 0, 1: 1, 2: 2}
algo_mapping["Birch"] = {0: 1, 1: 0, 2: 2}
algo_mapping["GMM"] = {0: 2, 1: 0, 2: 1}
algo_mapping["HC(Ward)"] = {0: 1, 1: 2, 2: 0}
algo_mapping["KMeans"] = {0: 0, 1: 1, 2: 2}
algo_mapping["Kmedoids"] = {0: 1, 1: 2, 2: 0}

#%%
score_values_list = ["CDI", "MDI", "DBI", "CHI", "SI"]
title_list = {"CDI": "Clustering dispersion indicator (Low-Good)",
              "MDI": "Modified Dunn Index (Low-Good)",
              "DBI": "Davies-Bouldin Index (Low-Good)",
              "CHI": "Calinski-Habarasz (High-Good)",
              "SI": "Silohuette Index (High-Good)"}

clustering_algorithms = ap_cluster_solutions.shape[0]
markers_cycle = list(islice(cycle(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '8', 's', 'p', 'P']),
                            clustering_algorithms))

#%%
fig, axs = plt.subplots(2, 2, figsize=(5, 4))
plt.subplots_adjust(left=0.09, right=0.95, bottom=0.20, hspace=0.5, top=0.95, wspace=0.4)
score_values_list_to_plot = ["MDI", "DBI", "CHI", "SI"]
title_subplot = ["(a)", "(b)", "(c)", "(d)"]
y_lims = [(0.5, 2.5), (0.5, 1.8), (200, 650), (0.12, 0.5)]

axin_axes = [[0.5, 0.1, 0.45, 0.30],
             [0.5, 0.1, 0.45, 0.30],
             [0.5, 0.5, 0.45, 0.30],
             [0.5, 0.5, 0.45, 0.30]]
axins_limits = [(2.5, 3.5, 0.75, 0.87),
                (2.5, 3.5, 0.80, 0.94),
                (2.5, 3.5, 604, 615),
                (2.5, 3.5, 0.37, 0.43)]

ax = axs.flatten()
for ii, score_ in enumerate(score_values_list_to_plot):
    for (index, value), marker in zip(score_values_ap[score_].iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[ii], markersize=4)
    ax[ii].set_title(title_subplot[ii], fontsize="large")
    ax[ii].set_ylabel(score_, fontsize="large")
    ax[ii].set_xlabel("Number of clusters", fontsize="large")
    ax[ii].set_xticks(np.array(ap_cluster_solutions.columns))
    ax[ii].set_ylim(y_lims[ii])

    axins = ax[ii].inset_axes(axin_axes[ii])
    for (index, value), marker in zip(score_values_ap[score_].iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=axins, markersize=4)

    x1, x2, y1, y2 = axins_limits[ii]
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax[ii].indicate_inset_zoom(axins, edgecolor="grey")

    if ii == 2:
        ax[ii].legend(fontsize="large",
                      bbox_to_anchor=(0.15, -0.20),
                      loc="upper left",
                      ncol=3,
                      title="Algorithms",
                      title_fontsize="large")
plt.savefig('load_model/cluster_metrics.pdf', dpi=700, bbox_inches='tight')

ranked_scores_cluster_three = pd.concat([score_values_ap[key_][[N_CLUSTER]].rename(columns={N_CLUSTER: key_}) for key_ in score_values_ap], axis=1)
ranked_scores_cluster_three_filtered = ranked_scores_cluster_three[["DBI", "MDI", "CHI", "SI"]]
print("Scores for cluster 3")
print(ranked_scores_cluster_three_filtered.sort_values(by=["CHI", "SI"]))

#%% Adjusted rand score to check the similarity between the clustering results. Reference is the Spectral clustering:
y_real = ap_cluster_solutions[N_CLUSTER]["KMeans"]
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
    temp_dataframe = pd.DataFrame(ap_cluster_solutions[N_CLUSTER][algo], columns=[algo])
    temp_dataframe = temp_dataframe.value_counts().to_frame(name="cluster_count").sort_index()
    temp_dataframe = temp_dataframe.reset_index().rename(columns={algo: "cluster"})

    temp_dataframe["cluster"] = temp_dataframe["cluster"].map(algo_mapping[algo])

    temp_dataframe["cluster"] = temp_dataframe["cluster"].map(name_mapping_)
    temp_dataframe = temp_dataframe.set_index("cluster")
    temp_dataframe = temp_dataframe.transpose().rename({"cluster_count": algo})
    cluster_count.append(temp_dataframe)

cluster_count = pd.concat(cluster_count, axis=0)
cluster_count = cluster_count[["C1", "C2", "C3"]]

#%% Merge all the table
table = pd.concat([ranked_scores_cluster_three_filtered.round(3), rand_indexes.round(3), cluster_count], axis=1)
table = table.sort_values(by=["CHI", "SI"], ascending=False)
table = table[["CHI","SI","DBI","MDI", "RI", "C1","C2","C3"]]
# table.to_csv("load_model/clustering_metrics.csv")
print(table)
