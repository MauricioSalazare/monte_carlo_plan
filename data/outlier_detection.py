from core.clustering import (ClusteringRLP,
                             plot_scores,
                             rlp_active_reactive,
                             rlp_transformer,
                             rlp_irradiance,
                             load_time_series,
                             unique_filename,
                             plot_solutions,
                             pca_3d,
                             plotly_pca_3d,
                             plotly_plot_solutions,
                             plotly_pca_all_algorithms_3d)
from core.outlier import (fit_sphere,
                          create_sphere_surface,
                          plotly_sphere,
                          polar_coordinates,
                          reject_outliers,
                          reject_outliers_mean,
                          distributions_and_outliers,
                          plotly_outlier_visualization,
                          rotate_angle_degrees)
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from statistics import median
import pandas as pd
import seaborn as sns
from pathlib import Path


file_path_outliers = r"processed_data/consumption_weather/outlier_transformers.csv"
file_name=r"processed_data/consumption_weather/time_series_data.csv"

data = load_time_series(file_name, file_path_outliers=file_path_outliers)
ap = rlp_transformer(data, kwargs_filter= {"regex": "_ap$"})

cluster_ap = ClusteringRLP()
cluster_ap.cluster_dataset(ap)
score_values_ap = cluster_ap.get_clustering_scores()
fig_scores = plot_scores(score_values_ap["CDI"],
            score_values_ap["MDI"],
            score_values_ap["DBI"],
            score_values_ap["CHI"])
fig_scores.suptitle("Full set Scores")

cluster_ap_cluster_solutions = cluster_ap.get_cluster_labels()

algorithm_name = "Birch"
n_cluster = 4

cluster_ap_labels = cluster_ap.get_cluster_labels(algorithm=algorithm_name, n_clusters=n_cluster)

plot_solutions(scale(ap.values, axis=1), cluster_ap_cluster_solutions, algorithm=algorithm_name, n_cluster=n_cluster)
plot_solutions(ap.values, cluster_ap_cluster_solutions, algorithm=algorithm_name, n_cluster=n_cluster)


#%% PCA over the data_set
# n_components = 5
scaled = scale(ap.values, axis=1)
explain_ratio = 0.95
pca = PCA(n_components=explain_ratio)
pca_model = pca.fit(scaled)
X_r = pca_model.transform(scaled)
print(f'Number of components: {X_r.shape[1]}  --  Explained ratio: {pca_model.explained_variance_ratio_.sum():.2f}')
pca_data = pd.DataFrame(X_r, index=ap.index, columns=['PC' + str(ii) for ii in range(X_r.shape[1])])

cluster_pca = ClusteringRLP()
cluster_pca.cluster_dataset(pca_data, std_scale=False)
score_values_pca = cluster_pca.get_clustering_scores()
fig_scores_pca = plot_scores(score_values_pca["CDI"],
                         score_values_pca["MDI"],
                         score_values_pca["DBI"],
                         score_values_pca["CHI"])
fig_scores_pca.suptitle("PCA Scores")

# Plot solutions
cluster_pca_labels = cluster_pca.get_cluster_labels(algorithm=algorithm_name, n_clusters=n_cluster)
cluster_pca_cluster_solutions = cluster_pca.get_cluster_labels()
pca_3d(pca_data, cluster_pca_cluster_solutions.loc[algorithm_name, n_cluster], algorithm_name)

pca_data_cluster = pca_data.copy(deep=True)
pca_data_cluster['cluster'] = cluster_pca_cluster_solutions.loc[algorithm_name, n_cluster]

sns.pairplot(pca_data_cluster.iloc[:,[0,1,2,3,-1]], hue="cluster", height=1.5, aspect=1,
             plot_kws={'s': 10, 'linewidth': 0.0, 'edgecolors': 'none'})

plotly_pca_3d(pca_data, cluster_pca_cluster_solutions.loc[algorithm_name, n_cluster], algorithm_name)
ap_frame = ap.copy()
scaled_frame = pd.DataFrame(scaled, index=ap.index, columns=ap.columns)
scaled_frame['cluster'] = cluster_pca_cluster_solutions.loc[algorithm_name, n_cluster]
ap_frame['cluster'] = cluster_pca_cluster_solutions.loc[algorithm_name, n_cluster]
plotly_plot_solutions(scaled_frame, algorithm_name=algorithm_name, n_cluster=n_cluster)
plotly_pca_all_algorithms_3d(pca_data, cluster_pca_cluster_solutions, n_cluster, rows=3, columns=3)



#%%#####################################################################################################################
#########################   SPHERE OUTLIER DETECTION   #################################################################
########################################################################################################################

coords_pca = pca_data.iloc[:, :3].values.copy()
centroid, r = fit_sphere(coords_pca)
x, y, z = create_sphere_surface(center=centroid, radius=r)
plotly_sphere(coords_pca, surface_sphere=(x, y, z), file_name="surface_overlaid")

# Re-center the data so it can be transformed to spherical coordinates
coords_pca[:,0] -= centroid[0]
coords_pca[:,1] -= centroid[1]
coords_pca[:,2] -= centroid[2]

x, y, z = create_sphere_surface(center=[0, 0, 0], radius=r)
plotly_sphere(coords_pca, surface_sphere=(x, y, z), file_name="surface_overlaid_centered")

r_pca, phi_pca_degrees, theta_pca_degrees = polar_coordinates(coords_pca)

idx_mean, _ = reject_outliers_mean(r_pca, m=4)
print(f'Outliers (radius heuristic) by MEAN statistic: {sum(~idx_mean)}')
print(cluster_pca_labels[~idx_mean].sort_values(by=['labels']))

idx_median, _ = reject_outliers(r_pca,m=4)
print(f'Outliers (radius heuristic) by MEDIAN statistic: {sum(~idx_median)}')
print(cluster_pca_labels[~idx_median].sort_values(by=['labels']))

#%%
fig, ax = plt.subplots(2, 4, figsize=(13, 7))
plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.4, left=0.05, right=0.95)
ax = ax.flatten()

lb={'radius': 0.03, 'phi': 0.02, 'theta': 0.05}
ub={'radius': 0.97, 'phi': 0.98, 'theta': 0.95}

r_pca, phi_pca_degrees_, theta_pca_degrees_ = polar_coordinates(coords_pca, ax=ax[0:4])
# _, idx_outliers = distributions_and_outliers(r_pca,
#                                              phi_pca_degrees_,
#                                              theta_pca_degrees_,
#                                              lb=lb,
#                                              ub=ub,
#                                              ax=ax[4:8])
phi_delta = 180 - median(phi_pca_degrees_)  # Keep one mode on the distribution
phi_pca_deg_rotated_wrapped = rotate_angle_degrees(phi_pca_degrees_, delta_degrees=phi_delta)
_, idx_outliers = distributions_and_outliers(r_pca,
                                             phi_pca_deg_rotated_wrapped,
                                             theta_pca_degrees_,
                                             lb=lb,
                                             ub=ub,
                                             ax=ax[4:8])

#%% Visualization using TWO techniques
pca_data_dali = pca_data.reset_index()
pca_data_clean = pca_data_dali[~idx_outliers]
pca_data_outliers = pca_data_dali[idx_outliers]

plotly_outlier_visualization(pca_data_clean, pca_data_outliers, file_name="outliers_my_model")

# LOF
outlier_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = outlier_model.fit_predict(scaled)
print(f'Outliers: {sum(y_pred==-1)}')
idx_lof_outliers = y_pred==-1

pca_data_clean = pca_data_dali[~idx_lof_outliers]
pca_data_outliers = pca_data_dali[idx_lof_outliers]

plotly_outlier_visualization(pca_data_clean, pca_data_outliers, file_name="outliers_LOF")

#%% Visualize the bad data
plotly_plot_solutions(scaled_frame.iloc[idx_outliers, :], algorithm_name=algorithm_name, n_cluster=n_cluster)
plotly_plot_solutions(ap_frame.iloc[idx_outliers, :], algorithm_name=algorithm_name, n_cluster=n_cluster)

#%% Visualize the good data
plotly_plot_solutions(scaled_frame.iloc[~idx_outliers, :], algorithm_name=algorithm_name, n_cluster=n_cluster)
plotly_plot_solutions(ap_frame.iloc[~idx_outliers, :], algorithm_name=algorithm_name, n_cluster=n_cluster)

#%%
drop_cluster = 20  # WATCH OUT: This number changes due to the random seed. Pick the high PV penetration cluster.
idx_cluster_to_drop = (ap_frame["cluster"] == drop_cluster).values
idx_outliers_final = idx_outliers | idx_cluster_to_drop

plotly_plot_solutions(ap_frame.iloc[~idx_outliers_final, :], algorithm_name=algorithm_name, n_cluster=n_cluster)


#%% Save outlier transformers (avoid deleting previous outlier files)
transformers_to_drop = pd.DataFrame(ap_frame.iloc[idx_outliers_final, :].index)
transformers_to_drop["DALIBOX_ID"] = transformers_to_drop["DALIBOX_ID"].apply(lambda x: x.replace("_ap", "")).to_frame()

fixed_path = unique_filename(file_path_outliers)
transformers_to_drop.to_csv(fixed_path, index=False)



