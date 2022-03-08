import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import mixture
from sklearn.preprocessing import scale
from sklearn.cluster import (KMeans,
                             MeanShift,
                             MiniBatchKMeans,
                             AgglomerativeClustering,
                             SpectralClustering,
                             DBSCAN,
                             OPTICS,
                             Birch)
from sklearn.neighbors import (kneighbors_graph,
                               LocalOutlierFactor)
from sklearn.metrics import pairwise_distances, calinski_harabasz_score, silhouette_score
from sklearn.metrics import davies_bouldin_score as dbi
from sklearn_extra.cluster import KMedoids
from .clustering_utils import Algorithms, mia, mdi, cdi
import matplotlib.pyplot as plt


class ClusteringRLP:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()

        self.trained_models = False

        # Identifier of each sample
        self.samples_names = None

        # Save all the clustering models
        self.cluster_models = None

        # Label for each cluster
        self.cluster_solutions_dataframe = None

        # Scores for clustering algorithms
        self.dbi_scores_dataframe = None
        self.mia_scores_dataframe = None
        self.mdi_scores_dataframe = None
        self.cdi_scores_dataframe = None
        self.calinski_scores_dataframe = None
        self.silhouette_scores_dataframe = None

        # All scores in one dictionary
        self.scores_solutions = None

        self._plot_dict()

    def cluster_dataset(self, data_set_, std_scale=True, sample_names=None, init=2, end=7, plot_solutions=False, axis=1):
        """
        Cluster data_set with different clustering algorithms

        Arguments:
        ----------
            data_set (pd.DataFrame):  Columns are variables and rows are samples  i.e., (n_samples, n_features)

        """

        if isinstance(data_set_, pd.DataFrame):
            data_set = data_set_.values.copy()
            self.samples_names = data_set_.index.to_list()

        elif isinstance(data_set_, np.ndarray):
            data_set = data_set_.copy()

            if sample_names is not None :
                assert len(sample_names) == data_set_.shape[0], \
                    "Length of list of sample names must be equal to number of samples."
                self.samples_names = sample_names
        else:
            print("Data set must be a pandas dataframe or a numpy array. Nothing was done")
            return

        if std_scale:
            data_set = scale(data_set, axis=axis)

        cluster_dbi_scores = dict()
        cluster_mia_scores = dict()
        cluster_mdi_scores = dict()
        cluster_cdi_scores = dict()
        cluster_calinski_scores = dict()
        cluster_silhouette_scores = dict()

        cluster_labels_solutions = dict()
        cluster_model = dict()

        for cluster_number in range(init, end):
            np.random.seed()
            print('-' * 70)
            print(f'Cluster number: {cluster_number}')
            params = {'n_cluster': cluster_number,
                      'n_neighbors': 10,
                      'eps': 0.8}

            # ------------------------------------------------------------------------------------------------------------------
            # Create cluster algorithms objects
            kmeans = KMeans(n_clusters=params['n_cluster'], random_state=1234, n_init=100)
            mini_batch_kmeans = MiniBatchKMeans(n_clusters=params['n_cluster'], n_init=5)

            # bandwidth = estimate_bandwidth(data_set, quantile=0.5, n_samples=50)
            # print(bandwidth)
            # mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

            connectivity = kneighbors_graph(data_set, n_neighbors=params['n_neighbors'], include_self=False)
            connectivity = 0.5 * (connectivity + connectivity.T)  # Make connectivity symmetric
            agg_ward = AgglomerativeClustering(n_clusters=params['n_cluster'],
                                               linkage='ward',
                                               connectivity=connectivity)
            average_linkage = AgglomerativeClustering(n_clusters=params['n_cluster'],
                                                      linkage='average',
                                                      # connectivity=connectivity,
                                                      affinity="euclidean")
            complete_linkage = AgglomerativeClustering(n_clusters=params['n_cluster'],
                                                       linkage='complete',
                                                       connectivity=connectivity,
                                                       affinity="euclidean")

            gmm = mixture.GaussianMixture(n_components=params['n_cluster'], n_init=50)
            spectral = SpectralClustering(n_clusters=params['n_cluster'],
                                          eigen_solver='arpack',
                                          affinity='nearest_neighbors',
                                          n_neighbors=params['n_neighbors'],
                                          n_init=50)
            # dbscan = DBSCAN(eps=params['eps'])
            birch = Birch(n_clusters=params['n_cluster'])

            kmedoids = KMedoids(metric="euclidean", n_clusters=params['n_cluster'])
            # ------------------------------------------------------------------------------------------------------------------

            clustering_algorithms = ((Algorithms.K_MEANS, kmeans),
                                     # (Algorithms.MINI_K_MEANS, mini_batch_kmeans),
                                     # ('MeanShift', mean_shift),
                                     (Algorithms.AGG_WARD, agg_ward),
                                     # (Algorithms.AGG_AVG, average_linkage),  # Removed because bad results
                                     # (Algorithms.AGG_COMP, complete_linkage),  # Removed because bad results
                                     (Algorithms.GMM, gmm),
                                     (Algorithms.SPECTRAL, spectral),
                                     # ('DBSCAN', dbscan),
                                     (Algorithms.BIRCH, birch),
                                     (Algorithms.K_MEDOIDS, kmedoids))

            dbi_scores = dict()
            mia_scores = dict()
            mdi_scores = dict()
            cdi_scores = dict()
            calinski_scores = dict()
            silhouette_scores = dict()

            labels_solutions = dict()

            for name, algorithm in tqdm(clustering_algorithms):
                print(f'Processing algorithm: {name}')
                algorithm.fit(data_set)
                cluster_model.update({cluster_number: {name: algorithm}})

                if hasattr(algorithm, 'labels_'):
                    y_labels = algorithm.labels_.astype(np.int)
                    labels_solutions.update({name: y_labels})
                else:
                    y_labels = algorithm.predict(data_set)
                    labels_solutions.update({name: y_labels})

                if plot_solutions:
                    fig, axes = plt.subplots(self._plot_grid[params['n_cluster']]["n_rows"],
                                             self._plot_grid[params['n_cluster']]["n_cols"],
                                             figsize=(14, 8))

                    for cluster, ax in zip(range(params['n_cluster']), axes.flatten()):
                        ax.plot(data_set[(y_labels == cluster), :].T, linewidth=0.3)
                        ax.set_title(f'Cluster: {cluster}')
                    fig.suptitle(name + " Clusters: " + str(params['n_cluster']))

                dbi_scores.update({name: dbi(data_set, y_labels)})
                mia_scores.update({name: mia(data_set, y_labels)})
                mdi_scores.update({name: mdi(data_set, y_labels)})
                cdi_scores.update({name: cdi(data_set, y_labels)})
                calinski_scores.update({name: calinski_harabasz_score(data_set, y_labels)})
                silhouette_scores.update({name: silhouette_score(data_set, y_labels)})

            cluster_labels_solutions.update({cluster_number: labels_solutions})
            cluster_dbi_scores.update({cluster_number: dbi_scores})
            cluster_mia_scores.update({cluster_number: mia_scores})
            cluster_mdi_scores.update({cluster_number: mdi_scores})
            cluster_cdi_scores.update({cluster_number: cdi_scores})
            cluster_calinski_scores.update({cluster_number: calinski_scores})
            cluster_silhouette_scores.update({cluster_number: silhouette_scores})

        # Save all the clustering models
        self.cluster_models = cluster_model

        # Label for each cluster
        self.cluster_solutions_dataframe = pd.DataFrame(cluster_labels_solutions)

        # Scores for clustering algorithms
        self.dbi_scores_dataframe = pd.DataFrame(cluster_dbi_scores)
        self.mia_scores_dataframe = pd.DataFrame(cluster_mia_scores)
        self.mdi_scores_dataframe = pd.DataFrame(cluster_mdi_scores)
        self.cdi_scores_dataframe = pd.DataFrame(cluster_cdi_scores)
        self.calinski_scores_dataframe = pd.DataFrame(cluster_calinski_scores)
        self.silhouette_scores_dataframe = pd.DataFrame(cluster_silhouette_scores)

        # Group scores in one dictionary for easy extraction:
        scores_names = ["DBI", "MIA", "MDI", "CDI", "CHI", "SI"]
        scores_values = [self.dbi_scores_dataframe,
                         self.mia_scores_dataframe,
                         self.mdi_scores_dataframe,
                         self.cdi_scores_dataframe,
                         self.calinski_scores_dataframe,
                         self.silhouette_scores_dataframe]

        self.scores_solutions = dict(zip(scores_names, scores_values))

        self.trained_models = True

    def get_clustering_scores(self, score_name: str = None, cluster: int = None):
        """
        Returns the clustering scoring results from the different clustering algorithms.
        Scoring techniques used:
            - Davies-Bouldin index (DBI)
            - Mean index adequacy (MIA)
            - Modified Dunn index (MDI)
            - Clustering dispersion indicator (CDI)
            - Silhouette Index (SI)

        Arguments:
        ----------
            score_name (str): Score acronym to be returned. Options ["DBI", "MIA", "MDI", "CDI", "CHI", "SI"]
            cluster (str): If score_name and cluster are provided it will narrow the results to the selected cluster.
                If only the cluster number is provided, then the return is all the scores for the specified cluster.

        Return:
        -------
            pd.Dataframe: Score solutions.

        """

        assert self.trained_models, "Fit a clustering algorithms first"

        if score_name is None and cluster is None:
            return self.scores_solutions

        elif score_name is None and cluster is not None:
            scores_cluster = pd.concat([scores_.loc[:, [cluster]]
                                        for _, scores_ in self.scores_solutions.items()], axis=1)
            scores_cluster.columns = list(self.scores_solutions.keys())

            return scores_cluster

        else:
            assert score_name in set(self.scores_solutions), \
                f"Invalid name, it should be any of: {set(self.scores_solutions)}"

            if cluster is not None:
                assert cluster in self.scores_solutions[score_name].columns,\
                    "Cluster requested is not in the scores dataset"
                return self.scores_solutions[score_name].loc[:, cluster]

            return self.scores_solutions[score_name]

    def get_clustering_models(self) -> dict:
        """
        Return all the models of each clustering algorithm for each number of clusters.
        The format of the key of the dictionary os: {"cluster_number": ""}
        """
        # return self.cluster_models
        raise NotImplementedError

    def get_cluster_labels(self, algorithm: str = None, n_clusters: int = None) -> pd.DataFrame:
        """
        Return the labels of the dataset for the specific algorithm and number of clusters

        """

        assert self.trained_models, "Train model first"

        if algorithm is None and n_clusters is None:
            return self.cluster_solutions_dataframe

        algorithms = self.get_available_algorithms()
        clusters = self.cluster_solutions_dataframe.columns

        assert algorithm in algorithms and n_clusters in clusters, f"Algorithm should be: {algorithms} and" \
                                                                    f"available clusters: {clusters}"

        y_labels = self.cluster_solutions_dataframe.loc[algorithm, n_clusters]

        if self.samples_names is not None:
            assert len(self.samples_names) == len(y_labels), "Length of labels and samples names does not agree."

            return pd.DataFrame({"labels": y_labels}, index=self.samples_names)

        return pd.DataFrame({"labels": y_labels})


    def get_available_algorithms(self):
        assert self.trained_models, "Train model first"

        return self.cluster_solutions_dataframe.index.to_list()

    def _plot_dict(self):
        """Helper dictionary for the subplots of matplotlib"""
        self._plot_grid = {1: {"n_rows": 1, "n_cols": 1},
                           2: {"n_rows": 1, "n_cols": 2},
                           3: {"n_rows": 2, "n_cols": 2},
                           4: {"n_rows": 2, "n_cols": 2},
                           5: {"n_rows": 2, "n_cols": 3},
                           6: {"n_rows": 2, "n_cols": 3},
                           7: {"n_rows": 3, "n_cols": 3},
                           8: {"n_rows": 3, "n_cols": 3},
                           9: {"n_rows": 3, "n_cols": 3},
                           10: {"n_rows": 3, "n_cols": 4},
                           11: {"n_rows": 3, "n_cols": 4},
                           12: {"n_rows": 3, "n_cols": 4},
                           13: {"n_rows": 4, "n_cols": 4},
                           14: {"n_rows": 4, "n_cols": 4},
                           15: {"n_rows": 4, "n_cols": 4},
                           16: {"n_rows": 4, "n_cols": 4},
                           17: {"n_rows": 5, "n_cols": 4},
                           18: {"n_rows": 5, "n_cols": 4},
                           19: {"n_rows": 5, "n_cols": 4},
                           20: {"n_rows": 5, "n_cols": 4}}



