from core.copula import EllipticalCopula
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter
import matplotlib.dates as mdates
from core.figure_utils import set_figure_art
from tqdm import tqdm, trange
import pickle
import warnings
set_figure_art()
# mpl.rc('text', usetex=False)

def compute_quantiles(dataset: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """Compute quantiles of a numpy matrix of dimension (n x p) where n are variables and p are samples"""

    quant_05 = np.nanquantile(dataset, q=0.05, axis=1)
    quant_50 = np.nanquantile(dataset, q=0.50, axis=1)
    quant_95 = np.nanquantile(dataset, q=0.95, axis=1)

    return quant_05, quant_50, quant_95


def sampling_by_annual_energy(copula_model: EllipticalCopula,
                              data_original_copula: pd.DataFrame,
                              *,
                              energy_values: pd.Series = None) -> pd.DataFrame:
    """
    Conditional copula sampling by annual energy.

    Parameter:
    ----------
        copula_model: EllipticalCopula: Copula object to be sampled.
        data_original_copula: pd.DataFrame: Original pivoted data set. Dimensions (n x p): n: days p: variables.

    Return:
    -------
        samples_cluster_frame: pd.DataFrame: DataFrame with the generated profiles. Same dimensions, variable names,
            and index names than the original data frame. i.e., (n x p): n: days p: variables.

    """

    var_mapping = dict(zip(data_original_copula.columns,
                           [f"x{zz + 1}" for zz in range(len(data_original_copula.columns))]))
    cond_variable = var_mapping["avg_gwh"]

    if energy_values is None:
        energy_values_sample = data_original_copula.loc[:, "avg_gwh"].value_counts()
    else:
        energy_values_sample = energy_values.copy()

    avg_energy_list = []
    samples_list = []
    for energy_, count_ in tqdm(energy_values_sample.iteritems(),
                                total=energy_values_sample.shape[0],
                                desc="Sampling copula"):
        try:
            samples_list.append(copula_model.sample(count_,
                                                    conditional=True,
                                                    variables={cond_variable: energy_},
                                                    drop_inf=True))
        except RuntimeError as err:
            print(err)
            warnings.warn("Could not sample copula. Trying sampling without dropping nan values.", UserWarning)
            samples_list.append(copula_model.sample(count_,
                                                    conditional=True,
                                                    variables={cond_variable: energy_},
                                                    drop_inf=False))

        avg_energy_list += [energy_] * count_
    avg_energy_frame = pd.DataFrame({"avg_gwh": avg_energy_list})

    if energy_values is None:
        samples_frame = pd.DataFrame(np.concatenate(samples_list, axis=1).T,
                                     index=data_original_copula.index,
                                     columns=data_original_copula.columns[:-1])
        samples_cluster_frame__ = pd.concat([samples_frame, data_original_copula.loc[:, "avg_gwh"]], axis=1)

    else:
        samples_frame = pd.DataFrame(np.concatenate(samples_list, axis=1).T,
                                     columns=data_original_copula.columns[:-1])
        samples_cluster_frame__ = pd.concat([samples_frame, avg_energy_frame], axis=1)

    return samples_cluster_frame__

if __name__ == "__main__":
    file_name_load_models = "../models/copula_model_load_with_reactive_power.pkl"
    with open(file_name_load_models, "rb") as pickle_file:
        copula_models = pickle.load(pickle_file)

    #%%
    np.random.seed(12)
    samples_copula_all_cluster_energy = []
    annual_energy = {}
    for CLUSTER in trange(3, desc="Sampling copula models."):
        annual_energy_values = copula_models[CLUSTER]['original_data']['avg_gwh'].value_counts().sort_index()
        min_energy = annual_energy_values.index.min()
        max_energy = annual_energy_values.index.max()
        norm_individual = mpl.colors.Normalize(vmin=min_energy, vmax=max_energy * 1.1)
        annual_energy_values_gwh = annual_energy_values.index.values.round(4)
        n_energy_values = len(annual_energy_values_gwh)

        annual_energy[CLUSTER] = {"values_energy": annual_energy_values_gwh,
                                  "energy_counts": n_energy_values,
                                  "min": min_energy,
                                  "max": max_energy,
                                  "norm_object": norm_individual}

        annual_energy_to_sample_dict = dict(zip(annual_energy_values_gwh, np.repeat(100, n_energy_values)))
        annual_energy_to_sample = pd.Series(annual_energy_to_sample_dict, name="avg_gwh")

        samples_copula_cluster_energy = sampling_by_annual_energy(copula_model=copula_models[CLUSTER]["copula"],
                                                                  data_original_copula=copula_models[CLUSTER]["original_data"],
                                                                  energy_values=annual_energy_to_sample)
        samples_copula_cluster_energy["cluster"] = CLUSTER
        samples_copula_all_cluster_energy.append(samples_copula_cluster_energy)

    samples = pd.concat(samples_copula_all_cluster_energy, axis=0)
    samples.reset_index(inplace=True, drop=True)
    min_count_energy_values_to_plot = min([item["energy_counts"] for item in annual_energy.values()])

    #%%  Test the function to generate frames
    filter_dict = [{"regex": "_ap$"}, {"regex": "_rp$"}, "", ""]
    title_list = ["Active power", "Reactive power"]
    ylim_by_cluster = [(-30, 200), (-50, 100), (-30, 350)]

    x_axis = pd.date_range(start="2021-11-01", periods=48, freq="30T")

    fig, ax = plt.subplots(3, 2, figsize=(8, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.6, bottom=0.1, left=0.1, right=0.9, top=0.9)
    axes = [ax[0, :], ax[1, :], ax[2, :]]

    cbar_2 = [None, None, None]
    l4 = [None, None, None]

    for kk in trange(min_count_energy_values_to_plot):
        for cluster_number, ax in enumerate(axes):
            ax[0].clear()
            ax[1].clear()

            annual_energy_gwh = annual_energy[cluster_number]["values_energy"][-min_count_energy_values_to_plot:][kk]

            idx_cluster = samples["cluster"] == cluster_number
            idx_energy = samples["avg_gwh"] == annual_energy_gwh
            assert (idx_cluster & idx_energy).sum() != 0, "Empty set"

            samples_copula_cluster_energy_copy = samples.iloc[(idx_cluster & idx_energy).values, :].copy()
            norm_individual_ = annual_energy[cluster_number]["norm_object"]

            for ii, regex_dict in zip(range(2), filter_dict):
                for _, (_, data_plot_) in enumerate(samples_copula_cluster_energy_copy.iterrows()):
                    annual_energy_value_ = data_plot_["avg_gwh"]
                    ax[ii].plot(x_axis,
                                data_plot_.filter(**regex_dict).values,
                                linewidth=0.3,
                                marker='.',
                                markersize=1,
                                markerfacecolor=plt.cm.get_cmap('inferno')(norm_individual_(annual_energy_value_)),
                                color=plt.cm.get_cmap('inferno')(norm_individual_(annual_energy_value_)))

                ax[ii].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                ax[ii].set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=20)))

                q_05_high, q_50_high, q_95_high = compute_quantiles(samples_copula_cluster_energy_copy.filter(**regex_dict).values.transpose())
                ax[ii].plot(x_axis, q_05_high, linewidth=0.8, color="red", linestyle="-")
                l2 = ax[ii].plot(x_axis, q_50_high, linewidth=1.5, color="red", linestyle="--", label="High Energy - Perc. 50\%")
                ax[ii].plot(x_axis, q_95_high, linewidth=0.8, color="red", linestyle="-")

                if ii == 0:
                    ax[ii].set_ylim((-51, 430))
                    ax[ii].set_ylabel("[kW]", fontsize="x-large")

                if ii == 1:
                    if cbar_2[cluster_number] is None:
                        cbar_2[cluster_number] = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual_, cmap=plt.cm.get_cmap('inferno')),
                                              aspect=9, ax=ax[ii])
                        cbar_2[cluster_number].ax.set_ylabel('[GWh/year]', fontsize="xx-large")
                        l4[cluster_number] = cbar_2[cluster_number].ax.hlines(y=annual_energy_gwh, xmin=0, xmax=10, linewidth=2, color="cyan", label="High Energy - GWh/year")
                    else:
                        l4[cluster_number].remove()
                        l4[cluster_number] = cbar_2[cluster_number].ax.hlines(y=annual_energy_gwh, xmin=0, xmax=10, linewidth=2, color="cyan",
                                              label="High Energy - GWh/year")

                    ax[ii].set_ylabel("[kVAr]", fontsize="x-large")
                    ax[ii].set_ylim((-28, 350))
                    ax[ii].set_ylim(ylim_by_cluster[cluster_number])

                ax[ii].set_title(title_list[ii], fontsize="xx-large")
                ax[ii].set_xlabel("Time of day", fontsize="x-large")

                ax[0].text(1.2, 1.2, f"Cluster: {cluster_number}. Annual energy: {annual_energy_gwh} [GWh]", transform=ax[0].transAxes,
                        ha="center", va="center", color="k",
                        family="sans-serif", fontweight="light", fontsize=12)
        plt.savefig(rf'animations\frames\test{kk}.png')


    # %% Animation saved as a video (.mp4) or (.gif) files.
    fig, ax = plt.subplots(3, 2, figsize=(8, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.6, bottom=0.1, left=0.1, right=0.9, top=0.9)
    axes = [ax[0,:], ax[1,:], ax[2,:]]

    mpl.rcParams['animation.ffmpeg_path'] = r"C:\\ProgramData\\Miniconda3\\envs\\energym\\Library\\bin\\ffmpeg.exe"
    writer = FFMpegWriter(fps=5, extra_args=['-vcodec', 'libx264'])  # To generate .mp4
    # writer = animation.PillowWriter(fps=5)  # To generate .gif files

    with writer.saving(fig, r"animations\writer_test_profiles.mp4", 200):
        cbar_2 = [None, None, None]
        l4 = [None, None, None]

        for kk in trange(min_count_energy_values_to_plot):
            for cluster_number, ax in enumerate(axes):
                ax[0].clear()
                ax[1].clear()

                annual_energy_gwh = annual_energy[cluster_number]["values_energy"][-min_count_energy_values_to_plot:][kk]

                idx_cluster = samples["cluster"] == cluster_number
                idx_energy = samples["avg_gwh"] == annual_energy_gwh
                assert (idx_cluster & idx_energy).sum() != 0, "Empty set"

                samples_copula_cluster_energy_copy = samples.iloc[(idx_cluster & idx_energy).values, :].copy()
                norm_individual_ = annual_energy[cluster_number]["norm_object"]

                for ii, regex_dict in zip(range(2), filter_dict):
                    for _, (_, data_plot_) in enumerate(samples_copula_cluster_energy_copy.iterrows()):
                        annual_energy_value_ = data_plot_["avg_gwh"]
                        ax[ii].plot(x_axis,
                                    data_plot_.filter(**regex_dict).values,
                                    linewidth=0.3,
                                    marker='.',
                                    markersize=1,
                                    markerfacecolor=plt.cm.get_cmap('inferno')(norm_individual_(annual_energy_value_)),
                                    color=plt.cm.get_cmap('inferno')(norm_individual_(annual_energy_value_)))

                    ax[ii].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                    ax[ii].set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=20)))

                    q_05_high, q_50_high, q_95_high = compute_quantiles(
                        samples_copula_cluster_energy_copy.filter(**regex_dict).values.transpose())
                    ax[ii].plot(x_axis, q_05_high, linewidth=0.8, color="red", linestyle="-")
                    l2 = ax[ii].plot(x_axis, q_50_high, linewidth=1.5, color="red", linestyle="--",
                                     label="High Energy - Perc. 50\%")
                    ax[ii].plot(x_axis, q_95_high, linewidth=0.8, color="red", linestyle="-")

                    if ii == 0:
                        ax[ii].set_ylim((-51, 430))
                        ax[ii].set_ylabel("[kW]", fontsize="x-large")

                    if ii == 1:
                        if cbar_2[cluster_number] is None:
                            cbar_2[cluster_number] = plt.colorbar(
                                plt.cm.ScalarMappable(norm=norm_individual_, cmap=plt.cm.get_cmap('inferno')),
                                aspect=9, ax=ax[ii])
                            cbar_2[cluster_number].ax.set_ylabel('[GWh/year]', fontsize="xx-large")
                            l4[cluster_number] = cbar_2[cluster_number].ax.hlines(y=annual_energy_gwh, xmin=0, xmax=10,
                                                                                  linewidth=2, color="cyan",
                                                                                  label="High Energy - GWh/year")
                        else:
                            l4[cluster_number].remove()
                            l4[cluster_number] = cbar_2[cluster_number].ax.hlines(y=annual_energy_gwh, xmin=0, xmax=10,
                                                                                  linewidth=2, color="cyan",
                                                                                  label="High Energy - GWh/year")

                        ax[ii].set_ylabel("[kVAr]", fontsize="x-large")
                        # ax[ii].set_ylim((-28, 350))
                        ax[ii].set_ylim(ylim_by_cluster[cluster_number])

                    ax[ii].set_title(title_list[ii], fontsize="xx-large")
                    ax[ii].set_xlabel("Time of day", fontsize="x-large")

                    ax[0].text(1.2, 1.2, f"Cluster: {cluster_number}. Annual energy: {annual_energy_gwh} [GWh]",
                               transform=ax[0].transAxes,
                               ha="center", va="center", color="k",
                               family="sans-serif", fontweight="light", fontsize=12)
            writer.grab_frame()