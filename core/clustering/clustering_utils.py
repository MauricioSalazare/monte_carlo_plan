import pandas as pd
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances, calinski_harabasz_score, silhouette_score
from itertools import cycle, islice, product
from scipy.cluster.hierarchy import dendrogram
from matplotlib import rcdefaults
from core.utils import set_figure_art
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# rcdefaults()

class Algorithms:
    K_MEANS = 'KMeans'
    MINI_K_MEANS = 'MiniBatchKMeans'
    AGG_WARD = 'HC(Ward)'
    AGG_AVG = 'HC(Average)'
    AGG_COMP = 'HC(Complete)'
    GMM = 'GMM'
    SPECTRAL = 'Spectral'
    BIRCH = 'Birch'
    ENSEMBLE = 'Ensemble'
    K_MEDOIDS = 'Kmedoids'

class COLORS:
    LIST = ['#4E79A7',#'#2E91E5',
            '#F28E2B',#'#E15F99',
            '#59A14F',#'#1CA71C',
            '#E15759',#'#FB0D0D',
            '#DA16FF',
            '#222A2A',
            '#B68100',
            '#750D86',
            '#EB663B',
            '#511CFB',
            '#00A08B',
            '#FB00D1',
            '#FC0080',
            '#B2828D',
            '#6C7C32',
            '#778AAE',
            '#862A16',
            '#A777F1',
            '#620042',
            '#1616A7',
            '#DA60CA',
            '#6C4516',
            '#0D2A63',
            '#AF0038']

def unique_filename(file_path: str, *, extension: str = "csv"):
    """ Create a new file name if the file name exists """
    if Path(file_path).is_file():
        counter = 0
        ext = "_{}." + extension
        while Path(file_path[:-(len(extension) + 1)] + ext.format(counter)).is_file():
            counter += 1
        file_path = (file_path[:-(len(extension) + 1)] + ext.format(counter))

    return file_path

def load_time_series(file_name: str, * , freq: str ="15T", file_path_outliers: str = None):
    data = pd.read_csv(file_name,
                       parse_dates=True,
                       index_col='date',
                       date_parser=lambda col: pd.to_datetime(col, utc=True))
    data = data.resample(freq).mean()

    if file_path_outliers is not None:
        outlier_path = Path(file_path_outliers)
        if Path.is_file(outlier_path):
            outliers = pd.read_csv(outlier_path)
            outliers_list = outliers["DALIBOX_ID"].to_list()
            idx = np.zeros(len(data.columns), dtype=bool)

            print(f"Total transformers in the time series: {data.shape[1]}")
            print(f"Total transformer outliers list: {len(outliers_list)}")
            for outlier_column in outliers_list:
                idx = idx | np.array(data.columns.str.contains(f"^{outlier_column}_ap"))
                idx = idx | np.array(data.columns.str.contains(f"^{outlier_column}_rp"))
            print(f"Total columns removed: {sum(idx)}  (2 x total outlier list)")
            data = data.iloc[:, ~idx].copy()
            print(f"Total columns in data frame after outlier removal: {data.shape[1]}")

    return data

def rlp_transformer(data_set, plot_rlp=False, month=None, year=None, set_index=True, kwargs_filter=None):
    """
    Compute a daily representative load profile (RLP) of a transformer

    Arguments:
    ----------
        kwargs_filter: dict: Dictionary with the inputs for the filter methods on pandas
            e.g.,   option={"regex": "_ap$"} --> Filters all columns that ends with _ap ("Active power")
                    option={"regex": "_rp$"} --> Filters all columns that ends with _rp ("Reactive power")
                    option={"items": "gq"} --> Filters columns named "qg" ("Irradiance")
    """

    if kwargs_filter is not None:
        data_kw_resampled = data_set.filter(**kwargs_filter).copy()
    else:
        data_kw_resampled = data_set.copy()

    transformer_name = data_kw_resampled.columns.to_list()

    data_kw_resampled['Day'] = data_kw_resampled.index.dayofyear
    data_kw_resampled['hour'] = data_kw_resampled.index.hour
    data_kw_resampled['dayIndex'] = data_kw_resampled.index.weekday
    data_kw_resampled['month'] = data_kw_resampled.index.month
    data_kw_resampled['year'] = data_kw_resampled.index.year
    data_kw_resampled['quarterIndex'] = data_kw_resampled.index.hour * 100 + data_kw_resampled.index.minute

    delta_times = len(data_kw_resampled['quarterIndex'].unique())

    if month is not None:
        assert year is not None, "Specify the year as well."
        idx = (data_kw_resampled['year'] == year) & (data_kw_resampled['month'] == month)
        data_kw_resampled = data_kw_resampled[idx]

    rep_load_profile = list()
    for transformer in tqdm(transformer_name):
        transfo_kw = data_kw_resampled.pivot(index='quarterIndex', columns='Day', values=transformer)
        rep_load_profile.append(np.nanmean(transfo_kw, axis=1))

    rlp = pd.DataFrame(np.vstack(rep_load_profile), index=transformer_name,
                       columns=['q_' + str(ii) for ii in range(1, delta_times + 1)])

    rlp_export = rlp.reset_index()
    rlp_export.rename(columns={'index': 'DALIBOX_ID'}, inplace=True)

    if plot_rlp:
        plt.figure()
        plt.plot(rlp.transpose())
        plt.show()

    if set_index:
        rlp_export.set_index('DALIBOX_ID', inplace=True)

    return rlp_export

def rlp_active_reactive(data_set, plot_rlp=False, month=None, year=None):
    """
    Compute a daily representative load profile (RLP) of a transformer

    The RLP joins the profiles for active and reactive power.
    """

    rlp_active = rlp_transformer(data_set=data_set, plot_rlp=plot_rlp, month=month,
                                 year=year, kwargs_filter= {"regex": "_ap$"}, set_index=False)
    rlp_reactive = rlp_transformer(data_set=data_set, plot_rlp=plot_rlp, month=month,
                                   year=year, kwargs_filter={"regex": "_rp$"}, set_index=False)

    # Process index before concat
    rlp_active["DALIBOX_ID"] = rlp_active["DALIBOX_ID"].apply(lambda x: x.replace("_ap", ""))
    rlp_active.set_index("DALIBOX_ID", inplace=True)

    rlp_reactive["DALIBOX_ID"] = rlp_reactive["DALIBOX_ID"].apply(lambda x: x.replace("_rp", ""))
    rlp_reactive.set_index("DALIBOX_ID", inplace=True)

    # Change of columns before concat
    rlp_active.columns = [column + "_ap" for column in rlp_active.columns]
    rlp_reactive.columns = [column + "_rp" for column in rlp_reactive.columns]

    return pd.concat([rlp_active, rlp_reactive], axis=1)

def rlp_irradiance(data_set, plot_rlp=False, threshold: float = 50):
    data_irradiance = data_set.filter(items=["qg"]).copy()
    data_irradiance['Day'] = data_irradiance.index.dayofyear
    data_irradiance['quarterIndex'] = data_irradiance.index.hour * 100 + data_irradiance.index.minute

    delta_times = len(data_irradiance['quarterIndex'].unique())

    pivoted_irradiance = data_irradiance.pivot(index="Day", columns="quarterIndex", values="qg")
    pivoted_irradiance_clean = pivoted_irradiance.dropna(how="any")
    mapper = dict(zip(pivoted_irradiance_clean.columns, ['q_' + str(ii) for ii in range(1, delta_times + 1)]))
    pivoted_irradiance_clean = pivoted_irradiance_clean.rename(columns=mapper)

    idx = (pivoted_irradiance_clean > threshold).any(axis=0)
    pivoted_irradiance_filtered = pivoted_irradiance_clean.iloc[:, idx.values]

    if plot_rlp:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(pivoted_irradiance_filtered.transpose())

    return pivoted_irradiance_filtered, idx

def representative_load_profile(data_set, plot_rlp=False, month=None, year=None):
    data_kw_resampled = data_set.copy()
    transformer_name = data_set.columns.to_list()

    data_kw_resampled['Day'] = data_kw_resampled.index.dayofyear
    data_kw_resampled['hour'] = data_kw_resampled.index.hour
    data_kw_resampled['dayIndex'] = data_kw_resampled.index.weekday
    data_kw_resampled['month'] = data_kw_resampled.index.month
    data_kw_resampled['year'] = data_kw_resampled.index.year
    data_kw_resampled['quarterIndex'] = data_kw_resampled.index.hour * 100 + data_kw_resampled.index.minute

    if month is not None:
        assert year is not None, "Specify the year as well."
        idx = (data_kw_resampled['year'] == year) & (data_kw_resampled['month'] == month)
        data_kw_resampled = data_kw_resampled[idx]
    # TODO: Add filtering by month??

    rep_load_profile = list()
    for transformer in transformer_name:
        transfo_kw = data_kw_resampled.pivot(index='quarterIndex', columns='Day', values=transformer)
        rep_load_profile.append(np.nanmean(transfo_kw, axis=1))

    rlp = pd.DataFrame(np.vstack(rep_load_profile), index=transformer_name,
                       columns=['q_' + str(ii) for ii in range(1, 97)])

    rlp_export = rlp.reset_index()
    rlp_export.rename(columns={'index': 'DALIBOX_ID'}, inplace=True)
    # rlp_export.to_csv('data_HENGELO_dali_box.csv', index=False)

    if plot_rlp:
        plt.figure()
        plt.plot(rlp.transpose())
        plt.show()

    return rlp_export

def loading_per_cluster(cluster_solutions, data_kw, n_cluster=4, algorithm=Algorithms.AGG_WARD):
    loadings_clusters = list()
    for cluster_label in range(n_cluster):
        idx = cluster_solutions[n_cluster][algorithm] == cluster_label
        loadings_clusters.append(data_kw.loc[:, idx].sum(axis=1))
    loadings_clusters = pd.concat(loadings_clusters, axis=1)
    loadings_clusters.columns = ['cluster_' + str(ii) for ii in range(n_cluster)]
    loadings_clusters = pd.concat([loadings_clusters,
                                   pd.DataFrame(data_kw.sum(axis=1), columns=['sum_total'])], axis=1)

    return loadings_clusters

def plot_clusters_rlp(cluster_solutions_dataframe,
                      data_set,
                      n_cluster=4,
                      algorithm=Algorithms.AGG_WARD,
                      params={'rows': 2, 'columns': 2, 'x_in': 8, 'y_in': 8, 'top': 0.8}):

    assert (params['rows'] *  params['columns']) >= n_cluster, "Rows and columns of the plot wrongly defined."

    y_labels = cluster_solutions_dataframe.loc[algorithm, n_cluster]

    # %%
    fig, axes = plt.subplots(params['rows'], params['columns'], figsize=(params['x_in'], params['y_in']))
    plt.subplots_adjust(top=params['top'], bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.2)

    for cluster, ax in zip(range(n_cluster), axes.flatten()):
        ax.plot(data_set[(y_labels == cluster), :].T, linewidth=0.3, color='#808080')
        ax.plot(np.nanmean(data_set[(y_labels == cluster), :].T, axis=1), linewidth=1.5, color='k',
                label='mean')
        ax.plot(np.nanquantile(data_set[(y_labels == cluster), :].T, q=0.05, axis=1),
                color='r', linewidth=1.5, linestyle='--', label='q0.05')
        ax.plot(np.nanquantile(data_set[(y_labels == cluster), :].T, q=0.95, axis=1),
                color='r', linewidth=1.5, linestyle='-', label='q0.95')
        ax.set_title(f'Cluster: {cluster} - Total transf: {(y_labels == cluster).sum()}', fontsize='small')
    ax.legend(fontsize='small')
    fig.suptitle('Algorithm: ' + algorithm)

def plot_solutions(data_set, cluster_solutions, algorithm, n_cluster):
    if algorithm == 'ensemble':
        y_labels = cluster_solutions
    else:
        y_labels = cluster_solutions.loc[algorithm, n_cluster]

    m = int(np.floor(np.sqrt(n_cluster)))

    if m ** 2 != n_cluster:  # Perfect square
        m = m + 1

    fig, axes = plt.subplots(m, m, figsize=(14, 8))

    for cluster, ax in zip(range(n_cluster), axes.flatten()):
        ax.plot(data_set[(y_labels == cluster), :].T, linewidth=0.3)
        ax.set_title(f'Cluster: {cluster}')
    fig.suptitle(algorithm)

def pca_3d(pca_data, y_labels, algorithm_name):
    if isinstance(pca_data, pd.DataFrame):
        data_values = pca_data.values
    elif isinstance(pca_data, np.ndarray):
        data_values = pca_data
    else:
        print('Wrong data set input')
        return

    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers e.g. [0,0,3,3,10,10,10] => [0,0,1,1,2,2,2]
    n_cluster = len(le.classes_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # markers_cycle = list(islice(cycle(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '8', 's', 'p', 'P']),
    markers_cycle = list(islice(cycle(['.', 'o', 'v', 's', '*', '>', '1', '2', '8', 's', 'p', 'P']),
                                n_cluster))
    # colors = list(islice(cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'thistle', 'peru']),
    #                      n_cluster))

    colors = list(islice(cycle(COLORS.LIST),
                         n_cluster))

    for k in range(n_cluster):
        # ax.scatter(pca_data.loc[y_labels == k, ['PC0']].values,
        #            pca_data.loc[y_labels == k, ['PC1']].values,
        #            pca_data.loc[y_labels == k, ['PC2']].values, marker=markers_cycle[k], label=f'{k}', c=colors[k])
        ax.scatter(data_values[y_labels == k, 0],
                   data_values[y_labels == k, 1],
                   data_values[y_labels == k, 2], marker=markers_cycle[k], label=f'{k}', c=colors[k])
    ax.legend()
    ax.set_title(algorithm_name)

def plotly_pca_3d(pca_data, y_labels, algorithm_name):
    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers e.g. [0,0,3,3,10,10,10] => [0,0,1,1,2,2,2]
    n_cluster = len(le.classes_)

    colors = list(islice(cycle(COLORS.LIST),
                         n_cluster))

    df = pca_data.copy(deep=True)
    df = df.reset_index()
    df['Cluster'] = labels

    trace_list_3d = list()
    for cluster_number, color in zip(range(n_cluster), colors):
        df_filtered = df[df['Cluster'] == cluster_number]
        trace = go.Scatter3d(x=df_filtered['PC0'],
                             y=df_filtered['PC1'],
                             z=df_filtered['PC2'],
                             mode='markers',
                             text=df_filtered['DALIBOX_ID'],
                             opacity=0.7,
                             legendgroup="group" + str(cluster_number),
                             name="Cluster " + str(cluster_number),
                             showlegend=True,
                             marker=dict(size=5,
                                         color=color)
                             # scene=dict(xaxis=dict(title_text="xxxxxxxxx"))
                             )
        trace_list_3d.append(trace)

    fig = go.Figure(data=trace_list_3d)
    fig.update_layout({'scene1': dict(xaxis=dict(title_text="PCA0"),
                                      yaxis=dict(title_text="PCA1"),
                                      zaxis=dict(title_text="PCA2"))},
                      title_text='Algorithm: ' + algorithm_name,
                      title_x=0.5
                      )

    file_name_html = unique_filename('processed_data/plots/clustering/cluster_results.html', extension="html")
    fig.write_html(file_name_html, auto_open=True)

def plotly_plot_solutions(data_set_frame, *, algorithm_name, n_cluster, file_name: str = None):
    """
    Plot all the load profiles per cluster group

    """
    data_set_frame_values = data_set_frame.iloc[:, :-1].copy()  # Last column is the cluster

    m = int(np.floor(np.sqrt(n_cluster)))

    if m ** 2 != n_cluster:  # Perfect square
        m = m + 1

    fig = make_subplots(
        rows=m, cols=m,
        subplot_titles=[f'Cluster {ii}' for ii in range(n_cluster)],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    cluster_number = 0
    for row in range(1, m + 1):
        for column in range(1, m + 1):
            data_set_filtered = data_set_frame_values[data_set_frame['cluster'] == cluster_number]

            for dali_name, values in data_set_filtered.iterrows():
                fig.add_trace(go.Scatter(y=values,
                                         name=f'C{cluster_number}: {dali_name}',
                                         # legendgroup="group" + str(cluster_number),
                                         line=dict(width=1)),
                              row=row, col=column)
            cluster_number = cluster_number + 1
    fig.update_layout(
        title_text=algorithm_name,
        title_x=0.5
    )

    if file_name is None:
        file_name_html = unique_filename('processed_data/plots/clustering/profiles_plot.html', extension="html")
        fig.write_html(file_name_html, auto_open=True)
    else:
        file_name_html = unique_filename(file_name, extension="html")
        fig.write_html(file_name_html, auto_open=True)

def plotly_pca_all_algorithms_3d(pca_data, cluster_solutions_dataframe, n_cluster, rows=3, columns=3):
    # N_ALGORITHMS = len(cluster_solutions_dataframe.index.to_list())

    colors = list(islice(cycle(COLORS.LIST), n_cluster))

    subplots_indexing = list(product(range(1, rows + 1), range(1, columns + 1)))

    fig = make_subplots(
        rows=rows, cols=columns,
        specs=np.full((rows, columns), dict(type='scene')).tolist(),
        subplot_titles=cluster_solutions_dataframe.index.to_list(),
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    df = pca_data.reset_index()
    first_legend = True

    for algorithm_name, subplot_index in zip(cluster_solutions_dataframe.index.to_list(), subplots_indexing):
        print(algorithm_name)
        print(subplot_index)

        algorithm_labels = cluster_solutions_dataframe.loc[algorithm_name].loc[n_cluster]
        df['Cluster'] = algorithm_labels

        for cluster_number, color in zip(range(n_cluster), colors):
            df_filtered = df[df['Cluster'] == cluster_number]
            trace = go.Scatter3d(x=df_filtered['PC0'],
                                 y=df_filtered['PC1'],
                                 z=df_filtered['PC2'],
                                 mode='markers',
                                 text=df_filtered['DALIBOX_ID'],
                                 opacity=0.7,
                                 legendgroup="group" + str(cluster_number),
                                 name="Cluster " + str(cluster_number),
                                 showlegend=first_legend,
                                 marker=dict(size=3,
                                             color=color)
                                 )
            fig.add_trace(trace, row=subplot_index[0], col=subplot_index[1])

        first_legend = False  # Only show the legend of the first subplot plot (Avoid overcrowd of legends).

    # Update axis labels
    for ii in range(1, rows * columns + 1):
        kargs = {'scene' + str(ii): dict(xaxis=dict(title_text="PCA0"),
                                         yaxis=dict(title_text="PCA1"),
                                         zaxis=dict(title_text="PCA2"))}
        fig.update_layout(**kargs)

    fig.update_layout(
        title_text='Clustering results different algorithms',
        title_x=0.5
        # height=400,
        # width=800
    )

    file_name_html = unique_filename('processed_data/plots/clustering/pca_3d_algorithms.html', extension="html")
    fig.write_html(file_name_html, auto_open=True)

def plot_all_data(rlp_scaled, save_plot=True):
    fig, ax = plt.subplots(1, 1, figsize=(2/2.54, 1.16/2.54))
    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.95)
    ax.plot(rlp_scaled.T, linewidth=0.3, color='#808080')
    ax.plot(np.nanmean(rlp_scaled.T, axis=1), linewidth=1.5, color='k', label='mean')
    ax.plot(np.nanquantile(rlp_scaled.T, q=0.05, axis=1),
            color='r', linewidth=1.5, linestyle='--', label='q0.05')
    ax.plot(np.nanquantile(rlp_scaled.T, q=0.95, axis=1),
            color='r', linewidth=1.5, linestyle='-', label='q0.95')
    # ax.set_title(f'Cluster: 1 - Total transf: {rlp_scaled.shape[0]}', fontsize='small')
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if save_plot:
        plt.savefig('Figures/rlp.pdf', format='pdf')

def plot_scores(cdi_scores_dataframe,
                mdi_scores_dataframe,
                dbi_scores_dataframe,
                chi_scores_dataframe) -> plt.figure:

    assert ((cdi_scores_dataframe.shape[0] == mdi_scores_dataframe.shape[0]) &
            (dbi_scores_dataframe.shape[0] == chi_scores_dataframe.shape[0])), 'All scores must compare same algorithms'

    clustering_algorithms = cdi_scores_dataframe.shape[0]
    markers_cycle = list(islice(cycle(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '8', 's', 'p', 'P']),
                                clustering_algorithms))

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2, hspace = 0.3)

    ax = axs.flatten()
    for (index, value), marker in zip(cdi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[0])
    # ax[0].legend()
    ax[0].set_title('CDI')
    # ax[0].set_ylim((0.2, 0.8))
    ax[0].set_xticks(np.array(cdi_scores_dataframe.columns))

    for (index, value), marker in zip(mdi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[1])
    # ax[1].legend()
    ax[1].set_title('MDI')
    # ax[1].set_ylim((0, 3.0))
    ax[1].set_xticks(np.array(cdi_scores_dataframe.columns))

    for (index, value), marker in zip(dbi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[2])
    # ax[2].legend()
    ax[2].set_title('DBI')
    # ax[2].set_ylim((0.5, 1.75))
    ax[2].set_xticks(np.array(cdi_scores_dataframe.columns))
    ax[2].set_xlabel('Cluster')

    for (index, value), marker in zip(chi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[3])
    ax[3].set_title('CHI')
    # ax[3].set_ylim((250, 600))
    ax[3].set_xticks(np.array(cdi_scores_dataframe.columns))
    ax[3].set_xlabel('Cluster')
    ax[3].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2),
                 fancybox=True, shadow=False, ncol=3)

    return fig

def plot_scores_journal(cdi_scores_dataframe,
                        mdi_scores_dataframe,
                        dbi_scores_dataframe,
                        chi_scores_dataframe):
    set_figure_art()

    clustering_algorithms = cdi_scores_dataframe.shape[0]
    markers_cycle = list(islice(cycle(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '8', 's', 'p', 'P']),
                                clustering_algorithms))

    fig, axs = plt.subplots(2, 2, figsize=(3.35, 3))
    plt.subplots_adjust(top=0.98, bottom=0.3, hspace=0.4, wspace=0.4)

    ax = axs.flatten()
    for (index, value), marker in zip(cdi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[0])
    # ax[0].legend()
    ax[0].set_ylabel('CDI')
    ax[0].set_xticks(np.array(cdi_scores_dataframe.columns))
    ax[0].set_xlabel('Number of clusters')
    # ax[0].set_ylim(0.2, 0.8)

    for (index, value), marker in zip(mdi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[1])
    # ax[1].legend()
    ax[1].set_ylabel('MDI')
    ax[1].set_xticks(np.array(cdi_scores_dataframe.columns))
    ax[1].set_xlabel('Number of clusters')
    # ax[1].set_ylim(0, 3)

    for (index, value), marker in zip(dbi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[2])
    # ax[2].legend()
    ax[2].set_ylabel('DBI')
    # ax[2].set_ylim((0.5, 1.75))
    ax[2].set_xticks(np.array(cdi_scores_dataframe.columns))
    ax[2].set_xlabel('Number of clusters')

    for (index, value), marker in zip(chi_scores_dataframe.iterrows(), markers_cycle):
        value.plot.line(linewidth=0.4, label=index, marker=marker, ax=ax[3])
    ax[3].set_ylabel('CHI')
    # ax[3].set_ylim((250, 600))
    ax[3].set_xticks(np.array(cdi_scores_dataframe.columns))
    ax[3].set_xlabel('Number of clusters')
    ax[3].legend(loc='upper center',
                 bbox_to_anchor=(-0.25, -0.4),
                 fancybox=True,
                 shadow=False,
                 ncol=3,
                 fontsize=6)

    rcdefaults()  # Reset the config of matplotlib for the next plots

def corr_time_series(loading_series,
                     weather_data,
                     method='spearman',
                     plot_solution=False):
    data_merged = pd.merge(loading_series, weather_data,
                           how='inner',
                           right_index=True,
                           left_index=True)
    corr_matrix = data_merged.corr(method=method)

    if plot_solution:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.subplots(1, 1)
        plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)
        sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.get_cmap('PuOr'), ax=ax)
        ax.tick_params(axis='y', rotation=0)

    return corr_matrix

def mia(data_set, y_labels):  # mean_index_adequacy
    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers e.g. [0,0,3,3,10,10,10] => [0,0,1,1,2,2,2]
    n_labels = len(le.classes_)
    intra_dists = np.zeros(n_labels)
    for k in range(n_labels):
        cluster_k = data_set[labels == k]
        centroid = cluster_k.mean(axis=0)
        # Between members and its centroid
        # # Version 1
        # intra_dists[k] = np.sqrt(np.average(np.square(pairwise_distances(cluster_k, [centroid]))))

        # Version 2
        intra_dists[k] = np.average(np.square(pairwise_distances(cluster_k, [centroid])))

    # return np.sqrt(np.average(np.square(intra_dists)))  # Some papers has the square other not
    # return np.average(np.square(intra_dists))

    # For version 2
    return np.sqrt(np.average(intra_dists))

def mdi(data_set, y_labels):
    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers
    n_labels = len(le.classes_)
    # intra_dists = np.zeros(n_labels)
    intra_dists_cluster = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(data_set[0])), dtype=np.float)

    for k in range(n_labels):
        cluster_k = data_set[labels == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        # Between members and its centroid
        # intra_dists[k] = np.sqrt(np.average(pairwise_distances(cluster_k, [centroid])))
        # intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
        # intra_dists[k] = np.sqrt(np.average(np.square(pairwise_distances(cluster_k, [centroid]))))

        # Between all the members in the group
        pairwise_distances_matrix = pairwise_distances(cluster_k)

        if np.allclose(pairwise_distances_matrix, 0):  # 1 member only in the class
            intra_dists_cluster[k] = 0

        else:  # \hat{d}(D_k)
            intra_dists_cluster[k] = np.sqrt(
                np.average(
                    np.square(
                        pairwise_distances_matrix[np.triu_indices_from(pairwise_distances_matrix,
                                                                       k=1)])))

    centroid_distances = pairwise_distances(centroids)
    centroid_distances[centroid_distances == 0] = np.inf  # To avoid to pick up values where i == j

    return np.max(intra_dists_cluster) / np.min(centroid_distances, axis=1).min()

def cdi(data_set, y_labels):
    '''
    Clustering dispersion indicator (CDI)
        CDI =  \hat{d}(C)^{-1}  \sqrt{ K^{-1}  \sum^{K}_{k=1}  \hat{d}^2 (D_k)  }

        \hat{D_k} = \sqrt{1/(2M)  \sum^{M}_{m=1}  d^2(x^(m), X)}   where X is all the points in the same cluster k.


        This is calculated summing the upper triangular matrix of the matrix distances between points in the group.
        The sum of the upper triangular makes that the coefficient 1/(2M) became 1/M, because you are not summing
        repetitive distances e.g. (x_1 - x_2)^2 == (x_2 - x_1)^2.

    '''

    le = LabelEncoder()
    labels = le.fit_transform(y_labels)  # Normalize the label numbers
    n_labels = len(le.classes_)
    # intra_dists = np.zeros(n_labels)
    intra_dists_cluster = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(data_set[0])), dtype=np.float)

    for k in range(n_labels):
        cluster_k = data_set[labels == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        # Between members and its centroid
        # intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))

        # Between all the members in the group
        pairwise_distances_matrix = pairwise_distances(cluster_k)

        if np.allclose(pairwise_distances_matrix, 0):  # 1 member only in the class
            intra_dists_cluster[k] = 0

        else: # \hat{d}(D_k)
            intra_dists_cluster[k] =np.sqrt(
                                        np.average(
                                            np.square(
                                                pairwise_distances_matrix[np.triu_indices_from(pairwise_distances_matrix,
                                                                                               k=1)])))
    centroid_distances = pairwise_distances(centroids)
    intra_centroid_distances = np.sqrt(np.average(np.square(centroid_distances[np.triu_indices_from(centroid_distances,
                                                                                       k=1)] )))

    return (1 / intra_centroid_distances) * np.sqrt(np.average(np.square(intra_dists_cluster)))

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

