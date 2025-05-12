# %%  Importing the necessary modules
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib as mat

sns.set_theme(style="ticks")
mat.rcParams["figure.dpi"] = 300

# %% import custom functions
from src import utils


# %% Random number
rng = np.random.RandomState(0)


# %% settings
metric = "I_8RI_AJ"
base_year = 2006
sample_buffer = 50
extrapolate = True
clip_samples = True
nLOC_samples = 60


# %% Load GCM means and sigmas
gcm_means = pd.read_csv("./data/gcm_mean_loca2_varavg_lm_cv-flow-weighted.csv")
gcm_sigs = pd.read_csv("./data/gcm_sigs_loca2_varavg_lm_cv-flow-weighted.csv").T


# %% dt dp arrays
dp = np.array([-25, -12.5, 0, 12.5, 25])
dt = np.array([0, 1, 2, 3, 4, 5])
dt_dict = {
    'T2P075': 2,
    'T3P075': 3,
    'T4P075': 4,
    'T5P075': 5,
    'T1P088': 1,
    'T2P088': 2,
    'T3P088': 3,
    'T4P088': 4,
    'T5P088': 5,
    'T0P100': 0,
    'T1P100': 1,
    'T2P100': 2,
    'T3P100': 3,
    'T4P100': 4,
    'T5P100': 5,
    'T1P113': 1,
    'T2P113': 2,
    'T3P113': 3,
    'T4P113': 4,
    'T5P113': 5,
    'T2P125': 2,
    'T3P125': 3,
    'T4P125': 4,
    'T5P125': 5,
}
dp_dict = {
    'T2P075': -25.0,
    'T3P075': -25.0,
    'T4P075': -25.0,
    'T5P075': -25.0,
    'T1P088': -12.5,
    'T2P088': -12.5,
    'T3P088': -12.5,
    'T4P088': -12.5,
    'T5P088': -12.5,
    'T0P100': 0,
    'T1P100': 0,
    'T2P100': 0,
    'T3P100': 0,
    'T4P100': 0,
    'T5P100': 0,
    'T1P113': 12.5,
    'T2P113': 12.5,
    'T3P113': 12.5,
    'T4P113': 12.5,
    'T5P113': 12.5,
    'T2P125': 25.0,
    'T3P125': 25.0,
    'T4P125': 25.0,
    'T5P125': 25.0
}
dp_extrap = np.array([-25.0, -12.5, 0, 12.5, 25.0])
dt_extrap = np.array([0, 1, 2, 3, 4, 5, 6])
dt_dict_extrap = {
    'T2P075': 2,
    'T3P075': 3,
    'T4P075': 4,
    'T5P075': 5,
    'T6P075': 6,
    'T1P088': 1,
    'T2P088': 2,
    'T3P088': 3,
    'T4P088': 4,
    'T5P088': 5,
    'T6P088': 6,
    'T0P100': 0,
    'T1P100': 1,
    'T2P100': 2,
    'T3P100': 3,
    'T4P100': 4,
    'T5P100': 5,
    'T6P100': 6,
    'T1P113': 1,
    'T2P113': 2,
    'T3P113': 3,
    'T4P113': 4,
    'T5P113': 5,
    'T6P113': 6,
    'T2P125': 2,
    'T3P125': 3,
    'T4P125': 4,
    'T5P125': 5,
    'T6P125': 6
}
dp_dict_extrap = {
    'T2P075': -25.0,
    'T3P075': -25.0,
    'T4P075': -25.0,
    'T5P075': -25.0,
    'T6P075': -25.0,
    'T1P088': -12.5,
    'T2P088': -12.5,
    'T3P088': -12.5,
    'T4P088': -12.5,
    'T5P088': -12.5,
    'T6P088': -12.5,
    'T0P100': 0,
    'T1P100': 0,
    'T2P100': 0,
    'T3P100': 0,
    'T4P100': 0,
    'T5P100': 0,
    'T6P100': 0,
    'T1P113': 12.5,
    'T2P113': 12.5,
    'T3P113': 12.5,
    'T4P113': 12.5,
    'T5P113': 12.5,
    'T6P113': 12.5,
    'T2P125': 25.0,
    'T3P125': 25.0,
    'T4P125': 25.0,
    'T5P125': 25.0,
    'T6P125': 25.0
}

# %% metrics data
metrics_thresholds = pd.read_csv(
    "./data/metrics_thresholds.csv", header=None, index_col=0
).to_dict()[1]

metrics_rs = pd.read_csv("./data/metrics_rs.csv")
metrics = metrics_rs.columns[2:]
metrics_data = {}
for m in metrics:
    df = metrics_rs[["dt", "dp", m]]
    df.columns = ["dt", "dp", "Baseline"]
    metrics_data[m] = df.copy()

if extrapolate == True:
    metrics_rs_extrap = pd.read_csv("./data/metrics_rs_extrapolated.csv")
    metrics_data_extrap = {}
    for m in metrics:
        df = metrics_rs_extrap[["dt", "dp", m]]
        df.columns = ["dt", "dp", "Baseline"]
        metrics_data_extrap[m] = df.copy()


# %% plotting
f, ax = plt.subplots(figsize=(7, 6), facecolor=None)

metric = "I_8RI_AJ"
metric_levels = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
ph = 2085

sample_new = True

for n in range(nLOC_samples + 1):
    if sample_new:
        # - build cc distributions sample at the planning horizon
        x_sample, y_sample, samples, rv = utils.get_gcm_dist(
            ph - base_year, gcm_means, gcm_sigs, rng, 100000, dp, dt
        )
        
        # - clip samples
        if clip_samples:
            if extrapolate == True:
                x_sample_clip, y_sample_clip, samples = utils.clip_samples(
                    samples, dp_dict_extrap, dt_dict_extrap
                )
            else:
                x_sample_clip, y_sample_clip, samples = utils.clip_samples(
                    samples, dp_dict, dt_dict
                )

        # -  interpolate samples
        df_future_hist = utils.build_future_hist(
            [str(ph)],
            [samples],
            metrics_data if extrapolate == False else metrics_data_extrap,
            metrics,
            metrics_thresholds,
            ["Baseline"],
        )

    data = df_future_hist.loc[
        (df_future_hist["Strategy"] == "Baseline")
        & (df_future_hist["Metric"] == metric)
        & (df_future_hist["PH"] == str(ph))
    ]
    #- scatter sampledvalues
    # plt.scatter(x=data["dp"], y=data["dt"], c='data["value_diff"]', cmap="RdBu", norm=utils.MidpointNormalize(midpoint=0), alpha=0.5, zorder=2, s=1,)
    # plt.scatter(x=data["dp"], y=data["dt"], color='grey', s=2, edgecolor='w', lw= 0.1, alpha=0.2, zorder=2)

    # data_na = data.loc[data["value_diff"].isna()]
    # plt.scatter(x=data_na["dp"], y=data_na["dt"], color="k", s=2, alpha=0.5, zorder=9)


    #- response surface interior
    metric_values = utils.get_metric_values(metrics_data, metric, "Baseline")
    metric_values = metric_values - metrics_thresholds[metric]
    dp_course_grid, dt_course_grid = np.mgrid[
        dp.min() : dp.max() + 12.5 : 12.5, dt.min() : dt.max() + 1 : 1
    ]
    rs = ax.contourf(dp_course_grid, dt_course_grid, metric_values, zorder=0, cmap="RdBu", norm=utils.MidpointNormalize(midpoint=0), alpha=1.0, levels=metric_levels)

    #- response surface extrapolated
    if extrapolate == True:
        metric_values_extrap = utils.get_metric_values(metrics_data_extrap, metric, "Baseline")
        metric_values_extrap = metric_values_extrap - metrics_thresholds[metric]
        dp_course_grid_extrap, dt_course_grid_extrap = np.mgrid[
            dp_extrap.min() : dp_extrap.max() + 12.5 : 12.5, dt_extrap.min() : dt_extrap.max() + 1 : 1
        ]
        rs_extrap = ax.contourf(dp_course_grid_extrap, dt_course_grid_extrap, metric_values_extrap, zorder=0, cmap="RdBu", norm=utils.MidpointNormalize(midpoint=0), alpha=0.8, levels=metric_levels)

    #- GCM cdf
    contuour_step_dp, contuour_step_dt = 0.05, 0.05
    dp_fine_grid_extrap, dt_fine_grid_extrap, cdf_extrap = utils.cumulative_cdf(
        dp_extrap, contuour_step_dp, dt_extrap, contuour_step_dt, rv
    )
    gcm_cdf = ax.contour(dp_fine_grid_extrap, dt_fine_grid_extrap, cdf_extrap, levels=[0.65, 0.95], colors="k", alpha=0.8, zorder=11)
    # plt.clabel(CS=gcm_cdf, inline=True, fmt="%.2f", fontsize="small", rightside_up=True, colors="k")

    data_sorted = data.sort_values("value", ascending=True)
    data_sorted["dt_norm"] = (data_sorted["dt"] - data_sorted["dt"].min()) / (data_sorted["dt"].max() - data_sorted["dt"].min())
    data_sorted["dp_norm"] = (data_sorted["dp"] - data_sorted["dp"].min()) / (data_sorted["dp"].max() - data_sorted["dp"].min())
    tp_center = data_sorted[["dt", "dp"]].mean().to_numpy()
    tp_center_norm = data_sorted[["dt_norm", "dp_norm"]].mean().to_numpy()

    # expected = np.searchsorted(data_sorted['value'],np.mean(data_sorted['value']))
    # idx_low, idx_hi = int(expected - 25), int(expected + 25)
    # points = data_sorted.iloc[idx_low:idx_hi][['dt_norm','dp_norm']].to_numpy()
    # points_diff = points - tp_center_norm
    # point_select = data_sorted.iloc[idx_low:idx_hi].iloc[np.argmin(np.sqrt(np.sum(points_diff*points_diff, axis=1)))]
    # plt.scatter(tp_center[1], tp_center[0],color='b',edgecolor='k',s=20,zorder=12)
    # plt.annotate('EV',xy=[tp_center[1], tp_center[0]+0.1],color='blue',fontsize=7,zorder=12)

    p_95percent = data_sorted.iloc[4050:5050]
    sns.scatterplot(data=p_95percent,x='dp',y='dt',color='purple',s=2, edgecolor='w',linewidth=0.1, alpha=0.1,zorder=9)

    p_75percent = data_sorted.iloc[24950:25050]
    sns.scatterplot(data=p_75percent,x='dp',y='dt',color='orange',s=2, edgecolor='w',linewidth=0.1, alpha=0.1,zorder=9)

    p_50percent = data_sorted.iloc[49950:50050]
    sns.scatterplot(data=p_50percent,x='dp',y='dt',color='blue',s=2, edgecolor='w',linewidth=0.1, alpha=0.1,zorder=9)

    for exceedance, color in zip([0.5, 0.25, 0.05], ['blue', 'orange', 'purple']):
        idx_low, idx_hi = int(exceedance*len(data_sorted) - 50), int(exceedance*len(data_sorted) + 50)
        points = data_sorted.iloc[idx_low:idx_hi][['dt_norm','dp_norm']].to_numpy()
        points_diff = points - tp_center_norm
        point_select = data_sorted.iloc[idx_low:idx_hi].reset_index(drop=True).iloc[np.argmin(np.sqrt(np.sum(points_diff*points_diff, axis=1)))]
        plt.scatter(point_select['dp'], point_select['dt'], facecolor=color,  edgecolor='k', linewidth=0.2, s=5, zorder=12)
 
plt.annotate("Change in Temperature (C)", xy=[-0.15, 1.03], xycoords="axes fraction", color="k", fontsize=12, zorder=12, horizontalalignment="left")
plt.annotate("Change from\ncurrent\n(maf)", xy=[1.03, 1.03], xycoords="axes fraction", color="k", fontsize=12, zorder=12, horizontalalignment="center",)
plt.annotate("Eight River April-July Flow Response Surface", xy=[0.5, 1.1], xycoords="axes fraction", color="k", fontsize=12, zorder=12, horizontalalignment="center",)
plt.annotate(f'Level of Concern Values TH = {ph}', xy=[0.5, 1.06], xycoords="axes fraction", color='k', fontsize=7, horizontalalignment='center')
plt.ylim(-0.25, 6.25)
plt.xlim(-27, 27)
plt.yticks([0, 1, 2, 3, 4, 5, 6])
plt.xticks([-25, -12.5, 0, 12.5, 25])
plt.grid(linestyle="dashed", color="k", lw=0.5)
plt.xlabel("Change in Precipitation (%)")
plt.ylabel("")  # Change in Temperature ($\degree$C)
# plt.title('95th Percentile Level-of-concern Selection on\nEight River April-July Runoff Response')
# plt.title('10k Samples of Likely Climate Changes at 2043')
# plt.title('10k Samples of Likely Climate Changes at 2043\nwith Eight River April-July Runoff Response')
# plt.title('Worst 5% of 10k Samples of Likely Climate Changes at 2043\nwith Eight River April-July Runoff Response')
# plt.title('95th Percentile Level-of-concern Selection\nwith Eight River April-July Runoff Response')

# ax.get_legend().remove()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = f.colorbar(rs, cax=cax)
# cbar.set_label('Change from current (million acre-feet)')
# cax.set_xticks([])
# cax.set_yticks([])

sns.despine(trim=True, offset=10)

# plt.savefig('./ouput/figures/rs.svg')

# %%
f, ax = plt.subplots(figsize=(6, 5), facecolor=None)
sns.ecdfplot(data, complementary=True, y=data["value_diff"], ax=ax)
ax.set_xlabel(
    "Proportion of 10k Samples of Likely Climate Changes at 2043\n(non-exceedance probability)"
)
ax.set_xlabel("")
ax.set_ylabel("Change from current\n(thousand acre-feet)")
ax.set_ylabel("")
ax.set_title("Eight River April-July Runoff Response")
ax.set_title("")
ax.set_xticks(np.arange(0, 1 + 0.1, 0.1))
ax.set_xticklabels([str(int(i * 100)) + "%" for i in np.arange(0, 1 + 0.1, 0.1)])
ax.set_xlim(-0.02, 1.02)
ax.grid(which="both")
plt.savefig("./output/figures/cdf.svg")
# sns.despine(trim=True,offset=10)



# %%
