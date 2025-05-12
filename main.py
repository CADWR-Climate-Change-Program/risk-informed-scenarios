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
if extrapolate==False:
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
else:
    dp = np.array([-25.0, -12.5, 0, 12.5, 25.0])
    dt = np.array([0, 1, 2, 3, 4, 5, 6])
    dt_dict = {
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
    dp_dict = {
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
if extrapolate == False:
    metrics_rs = pd.read_csv("./data/metrics_rs.csv")
else:
    metrics_rs = pd.read_csv("./data/metrics_rs_extrapolated.csv")
metrics = metrics_rs.columns[2:]
metrics_data = {}
for m in metrics:
    df = metrics_rs[["dt", "dp", m]]
    df.columns = ["dt", "dp", "Baseline"]
    metrics_data[m] = df.copy()



# %% get levels of concern
metric_LOC = pd.DataFrame()

# - loop through planning horizons
for ph in np.arange(2045, 2050 + 1, 1):
    print(ph)

    ph_idx = ph - base_year

    # - loop through multiple samples
    for n in range(nLOC_samples):

        # - build cc sample at the planning horizon
        x_sample, y_sample, samples, rv = utils.get_gcm_dist(
            ph_idx, gcm_means, gcm_sigs, rng, 100000, dp, dt
        )

        # - clip samples
        if clip_samples:
            x_sample_clip, y_sample_clip, samples = utils.clip_samples(samples, dp_dict, dt_dict)

        # -  interpolate samples
        df_future_hist = utils.build_future_hist([str(ph)], [samples], metrics_data, metrics, metrics_thresholds, ["Baseline"])

        # - sort and normalize dt dp
        data = df_future_hist.loc[(df_future_hist.PH == str(ph)) & (df_future_hist.Metric == metric)].copy()
        data_sorted = data.sort_values("value", ascending=True)
        data_sorted.insert(0,"dt_norm",(data_sorted["dt"] - data_sorted["dt"].min()) / (data_sorted["dt"].max() - data_sorted["dt"].min()))
        data_sorted.insert(0,"dp_norm",(data_sorted["dp"] - data_sorted["dp"].min()) / (data_sorted["dp"].max() - data_sorted["dp"].min()))
        tp_center_norm = data_sorted[["dt_norm", "dp_norm"]].mean().to_numpy()

        # - get levels of concern (each exceedance)
        for exceedance in [0.5, 0.25, 0.05]:
            # - get the exceedance indices
            idx_low, idx_hi = int(exceedance * len(data_sorted) - sample_buffer), int(
                exceedance * len(data_sorted) + sample_buffer
            )
            points = data_sorted.iloc[idx_low:idx_hi][["dt_norm", "dp_norm"]].to_numpy()
            # - get the closest point to the center
            points_diff = points - tp_center_norm
            point_select = (
                data_sorted.iloc[idx_low:idx_hi]
                .reset_index(drop=True)
                .iloc[np.argmin(np.sqrt(np.sum(points_diff * points_diff, axis=1)))]
            )
            # - append the results to the dataframe
            metric_LOC = pd.concat(
                [
                    metric_LOC,
                    pd.DataFrame(
                        {
                            "n": n + 1,
                            "Metric": metric,
                            "PH": ph,
                            "LOC": 100 - 100 * exceedance,
                            "dp": point_select["dp"],
                            "dt": point_select["dt"],
                        },
                        index=[0],
                    ),
                ],
                axis=0,
            )

# %% save levels of concern
metric_LOC.to_csv(
    f"./output/{metric}_LOC.csv", index=False
)

# %%
