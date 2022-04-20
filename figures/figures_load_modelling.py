from core.copula import EllipticalCopula
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import math
import re
from core.figure_utils import set_figure_art
from tqdm import tqdm
import seaborn as sns
import pickle
import warnings
set_figure_art()
# mpl.rc('text', usetex=False)


def cos_phi(p, q):
    """Computes power factor"""

    return math.cos(math.atan(q / p)) if p != 0.0 else 1.0


def power_factor_data(data: pd.DataFrame) -> pd.DataFrame:
    """Computes power factor for each time step of the daily time series"""

    p = re.compile(".*_rp*")
    time_step_number = [int(s.split("_")[1]) for s in data.columns if p.match(s)]

    assert time_step_number, "Reactive power variables are empty"

    pf_columns = []
    pf_values = []
    for tt in time_step_number:
        pf_columns.append(f"pf_{tt}")
        pf_values.append(data[[f"q_{tt}_ap",
                               f"q_{tt}_rp"]].apply(lambda x: cos_phi(x[f"q_{tt}_ap"], x[f"q_{tt}_rp"]), axis=1))

    pf_values_df = pd.concat(pf_values, axis=1)
    pf_values_df.columns = pf_columns

    return pf_values_df


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


file_name_load_models = "../models/copula_model_load_with_reactive_power.pkl"
with open(file_name_load_models, "rb") as pickle_file:
    copula_models = pickle.load(pickle_file)

# %% Sampling the copula conditioned with energy consumption. This takes sometime
np.random.seed(123456)
samples_copula_cluster = {}
min_energy = np.inf
max_energy = -np.inf
for cluster_ in range(3):
    copula_cluster = copula_models[cluster_]["copula"]
    data_copula_original = copula_models[cluster_]["original_data"]
    samples_copula_cluster[cluster_] = sampling_by_annual_energy(copula_cluster, data_copula_original)
    (min_energy_cluster, max_energy_cluster) = (data_copula_original["avg_gwh"].min(),
                                                data_copula_original["avg_gwh"].max())

    if min_energy_cluster < min_energy:
        min_energy = min_energy_cluster
    if max_energy_cluster > max_energy:
        max_energy = max_energy_cluster

# %%
norm_individual = mpl.colors.Normalize(vmin=min_energy, vmax=max_energy)
x_axis = pd.date_range(start="2021-11-01", periods=48, freq="30T")
filter_dict = [{"regex": "_ap$"}, {"regex": "_rp$"}, "", ""]

# %%
ax = np.empty((3, 4), dtype=object)
fig = plt.figure(figsize=(7, 6))
gs0 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2, hspace=None, left=0.08, bottom=0.16, right=0.98, top=0.95,
                        height_ratios=[1], width_ratios=[2, 1])  # Two columns to hold the different plots
gs00 = gs0[0].subgridspec(3, 2, wspace=0.35, hspace=0.35)  # Load profiles
gs01 = gs0[1].subgridspec(3, 2, hspace=0.35)  # Power factor and Annual energy

kk = 0
for ii in range(3):
    for jj in range(2):
        ax[ii, jj] = fig.add_subplot(gs00[kk])
        ax[ii, jj + 2] = fig.add_subplot(gs01[kk])
        kk += 1

titles_list = [["(a)", "(b)", "(c)", "(d)"],
               ["(e)", "(f)", "(g)", "(h)"],
               ["(i)", "(j)", "(k)", "(l)"]]

for cluster_ in range(3):
    copula_cluster = copula_models[cluster_]["copula"]
    data_copula = copula_models[cluster_]["original_data"]
    samples_cluster_frame = samples_copula_cluster[cluster_]

    # Sampling without considering yearly energy consumption
    samples_cluster = copula_cluster.sample(data_copula.shape[0], drop_inf=True)
    samples_cluster_frame_energy = pd.DataFrame(samples_cluster.T, columns=data_copula.columns)

    pf_values_original = power_factor_data(data=data_copula)
    pf_values_samples = power_factor_data(data=samples_cluster_frame)

    if cluster_ == 0:
        # Drop some outliers that looks ugly in the graph
        data_copula_clean = data_copula.drop(["tr_140_212", "tr_140_215"]).copy()
    else:
        data_copula_clean = data_copula.copy()

    for ii, regex_dict in enumerate(filter_dict):
        if (ii != 2) and (ii != 3):
            q_05, q_50, q_95 = compute_quantiles(data_copula_clean.filter(**regex_dict).values.transpose())
            q_05_s, q_50_s, q_95_s = compute_quantiles(samples_cluster_frame.filter(**regex_dict).values.transpose())

            for _, (_, data_plot_) in enumerate(data_copula_clean.iterrows()):
                annual_energy = data_plot_["avg_gwh"]
                ax[cluster_, ii].plot(x_axis,
                                      data_plot_.filter(**regex_dict).values,
                                      linewidth=0.3,
                                      marker='.',
                                      markersize=1,
                                      markerfacecolor=plt.cm.get_cmap('inferno')(norm_individual(annual_energy)),
                                      color=plt.cm.get_cmap('inferno')(norm_individual(annual_energy)))

            ax[cluster_, ii].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            ax[cluster_, ii].set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=29)))

            l1 = ax[cluster_, ii].plot(x_axis, q_05,
                                       linewidth=1.5, color="red", linestyle="--", label="Original 5\%")
            l2 = ax[cluster_, ii].plot(x_axis, q_50,
                                       linewidth=1.5, color="red", linestyle="-", label="Original 50\%")
            l3 = ax[cluster_, ii].plot(x_axis, q_95,
                                       linewidth=1.5, color="red", linestyle="-.", label="Original 95\%")

            l4 = ax[cluster_, ii].plot(x_axis, q_05_s,
                                       linewidth=1.5, color="blue", linestyle="--", label="Simulated 5\%")
            l5 = ax[cluster_, ii].plot(x_axis, q_50_s,
                                       linewidth=1.5, color="blue", linestyle="-", label="Simulated 50\%")
            l6 = ax[cluster_, ii].plot(x_axis, q_95_s,
                                       linewidth=1.5, color="blue", linestyle="-.", label="Simulated 95\%")

            lgs = l1 + l2 + l3 + l4 + l5 + l6
            lgs_names = [label_.get_label() for label_ in lgs]

            if ii == 1:
                cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('inferno')),
                                      ax=ax[cluster_, ii])
                cbar_2.ax.set_ylabel('[GWh/year]')

            if ii == 0:
                ax[cluster_, ii].set_ylim((-51, 430))
                ax[cluster_, ii].set_ylabel(r"$\bf{Cluster\; " + f"{cluster_ + 1}" + r"}$" + "\n Active power [kW]",
                                            fontsize="large")
            if ii == 1:
                ax[cluster_, ii].set_ylabel(f"Reactive power [kVAr]")

            if (cluster_ == 2) and (ii == 0):
                ax[cluster_, ii].legend(lgs,
                                        lgs_names,
                                        fontsize="medium",
                                        bbox_to_anchor=(0.4, -0.2),
                                        loc="upper left",
                                        ncol=2,
                                        title="Percentiles")

            if cluster_ == 2:
                ax[cluster_, ii].set_xlabel("Time of day")

        if ii == 2:  # Power factor
            ax[cluster_, ii].hist(pf_values_samples.values.flatten(),
                                  bins=150, alpha=0.5, density=True,
                                  histtype="stepfilled", linewidth=1.0,
                                  color=sns.color_palette()[0],
                                  edgecolor=sns.color_palette()[0],
                                  label="Sample MVT Copula")
            ax[cluster_, ii].hist(pf_values_original.values.flatten(),
                                  bins=150, alpha=0.5, density=True,
                                  histtype="stepfilled", linewidth=1.0,
                                  color=sns.color_palette()[1],
                                  edgecolor=sns.color_palette()[1],
                                  label="Original dataset")
            ax[cluster_, ii].set_xlim((0.9, 1.01))
            ax[cluster_, ii].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax[cluster_, ii].yaxis.set_major_locator(ticker.NullLocator())
            ax[cluster_, ii].set_ylabel("Density")

            if cluster_ == 2:
                ax[cluster_, ii].set_xlabel(r"cos$(\phi)$")
                ax[cluster_, ii].legend(fontsize="medium",
                                        bbox_to_anchor=(0.1, -0.3),
                                        loc="upper left",
                                        ncol=1,
                                        title="Density plots")

        if ii == 3:  # Annual energy
            sns.histplot(x=samples_cluster_frame_energy.filter(items=["avg_gwh"]).transpose().values.flatten(),
                         stat="density",
                         element="step",
                         alpha=0.5,
                         kde=True,
                         edgecolor=sns.color_palette()[0],
                         color=sns.color_palette()[0],
                         ax=ax[cluster_, ii],
                         line_kws={'color': sns.color_palette()[0]},
                         label="Sample MVT Copula")
            sns.histplot(x=data_copula_clean.filter(items=["avg_gwh"]).transpose().values.flatten(),
                         stat="density",
                         element="step",
                         alpha=0.5,
                         kde=True,
                         edgecolor=sns.color_palette()[1],
                         color=sns.color_palette()[1],
                         ax=ax[cluster_, ii],
                         line_kws={'color': sns.color_palette()[1]},
                         label="Original dataset")
            ax[cluster_, ii].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax[cluster_, ii].yaxis.set_major_locator(ticker.NullLocator())
            ax[cluster_, ii].set_ylabel("")
            ax[cluster_, ii].set_xlim((-0.01, 2.01))

            if cluster_ == 2:
                ax[cluster_, ii].set_xlabel("Annual Energy\n[GWh/year]")

        ax[cluster_, ii].set_title(titles_list[cluster_][ii], fontsize="large")

plt.savefig('load_model/sampling_load_models.png', dpi=700, bbox_inches='tight')

# %% Create samples conditioned by yearly energy consumption.
np.random.seed(12)
CLUSTER = 2
LOW_ANNUAL_ENERGY = 0.429440
HIGH_ANNUAL_ENERGY = 1.767857
annual_energy_to_sample = pd.Series({HIGH_ANNUAL_ENERGY: 100, LOW_ANNUAL_ENERGY: 100},
                                    name="avg_gwh")  # Dict[energy_level: samples]
samples_copula_cluster_energy = sampling_by_annual_energy(copula_model=copula_models[CLUSTER]["copula"],
                                                          data_original_copula=copula_models[CLUSTER]["original_data"],
                                                          energy_values=annual_energy_to_sample)
pf_values_samples_energy = power_factor_data(samples_copula_cluster_energy)

#%%
filter_dict = [{"regex": "_ap$"}, {"regex": "_rp$"}, "", ""]
title_list = ["(a)", "(b)"]
fig, ax = plt.subplots(1, 2, figsize=(7, 3))
plt.subplots_adjust(wspace=0.4, bottom=0.2, left=0.1, right=0.95, top=0.9)

for ii, regex_dict in zip(range(2), filter_dict):
    for _, (_, data_plot_) in enumerate(samples_copula_cluster_energy.iterrows()):
        annual_energy = data_plot_["avg_gwh"]
        if (annual_energy == LOW_ANNUAL_ENERGY) and (ii == 1) and (data_plot_.filter(**regex_dict).values > 75).any():
            # Cut ugly outliers
            pass
        else:
            ax[ii].plot(x_axis,
                        data_plot_.filter(**regex_dict).values,
                        linewidth=0.3,
                        marker='.',
                        markersize=1,
                        markerfacecolor=plt.cm.get_cmap('inferno')(norm_individual(annual_energy)),
                        color=plt.cm.get_cmap('inferno')(norm_individual(annual_energy)))

    ax[ii].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax[ii].set_xlim((x_axis[0], x_axis[-1] + pd.Timedelta(minutes=20)))

    idx_low = (samples_copula_cluster_energy["avg_gwh"] == LOW_ANNUAL_ENERGY).values
    idx_high = (samples_copula_cluster_energy["avg_gwh"] == HIGH_ANNUAL_ENERGY).values

    _, q_50_low, _ = compute_quantiles(
        samples_copula_cluster_energy.iloc[idx_low, :].filter(**regex_dict).values.transpose())
    _, q_50_high, _ = compute_quantiles(
        samples_copula_cluster_energy.iloc[idx_high, :].filter(**regex_dict).values.transpose())

    l1 = ax[ii].plot(x_axis, q_50_low, linewidth=1.5, color="blue", linestyle="--", label="Low Energy - Perc. 50\%")
    l2 = ax[ii].plot(x_axis, q_50_high, linewidth=1.5, color="red", linestyle="--", label="High Energy - Perc. 50\%")

    if ii == 0:
        ax[ii].set_ylim((-51, 430))
        ax[ii].set_ylabel("Active power [kW]")

    if ii == 1:
        cbar_2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_individual, cmap=plt.cm.get_cmap('inferno')),
                              ax=ax[ii])
        cbar_2.ax.set_ylabel('[GWh/year]')
        l3 = cbar_2.ax.hlines(y=0.429440, xmin=0, xmax=10, linewidth=2, color="blue", label="Low Energy - GWh/year")
        l4 = cbar_2.ax.hlines(y=1.767857, xmin=0, xmax=10, linewidth=2, color="red", label="High Energy - GWh/year")

        ax[ii].set_ylabel("Reactive power [kVAr]")
        ax[ii].set_ylim((-28, 180))

        lgs = l1 + l2 + [l3] + [l4]
        lgs_names = [label_.get_label() for label_ in lgs]
        ax[0].legend(lgs,
                     lgs_names,
                     fontsize="medium",
                     bbox_to_anchor=(-0.2, -0.15),
                     loc="upper left",
                     ncol=4,
                     handlelength=1.5)

    ax[ii].set_title(title_list[ii], fontsize="large")
    ax[ii].set_xlabel("Time of day")

plt.savefig('load_model/sampling_conditioned.png', dpi=700, bbox_inches='tight')
