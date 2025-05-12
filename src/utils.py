import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

from scipy.stats import multivariate_normal
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import percentileofscore

import matplotlib.colors as colors


def get_gcm_dist(ph, means, sigs, rng, n_sample, x, y):
    ph_means = means.iloc[ph][["DP", "DT"]].to_list()
    ph_sigs = np.flip(sigs.iloc[ph * 2 - 1 : ph * 2 + 1].to_numpy(), (0, 1))
    x_sample, y_sample = rng.multivariate_normal(ph_means, ph_sigs, n_sample).T
    samples = pd.DataFrame({"x": x_sample, "y": y_sample})
    rv = multivariate_normal(ph_means, ph_sigs)
    return x_sample, y_sample, samples, rv


def construct_interpolated_data(metrics_data, metric, strategy, num_divisions):
    z = metrics_data[metric][strategy].to_numpy()
    x = metrics_data[metric]["dp"]
    y = metrics_data[metric]["dt"]
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    x_grid, y_grid = np.linspace(x.min(), x.max(), num_divisions), np.linspace(
        y.min(), y.max(), num_divisions
    )
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = interp(X, Y)
    return x_grid, y_grid, Z


def interpolate_samples(samples, x_grid, y_grid, Z):
    interpolated_samples = []
    for i in range(len(samples[0])):
        x_idx = np.searchsorted(x_grid, (100 + samples[0][i]) / 100) - 1
        y_idx = np.searchsorted(y_grid, samples[1][i]) - 1
        z = Z[y_idx, x_idx]
        interpolated_samples.append(z)
    return interpolated_samples


def get_metric_values(metric_dict, metric, strategy):
    df = metric_dict[metric][[strategy, "dt", "dp"]]
    return df.pivot_table(index="dp", columns="dt", values=strategy).to_numpy()


def clip_samples(samples, dp_dict, dt_dict):
    dt_array = np.array(list(dt_dict.values()))
    dp_array = np.array(list(dp_dict.values()))
    df = pd.DataFrame([dp_array, dt_array]).T
    df["surface"] = "bounds"
    geometry = [Point(xy) for xy in zip(dp_array, dt_array)]
    point_gdf = gpd.GeoDataFrame(df, geometry=geometry)
    poly_gdf = point_gdf.dissolve("surface").convex_hull
    geometry = [Point(xy) for xy in zip(samples["x"], samples["y"])]
    sample_gdf = gpd.GeoDataFrame(samples, geometry=geometry)
    sample_clip = gpd.clip(sample_gdf, poly_gdf)
    x_sample_clip = sample_clip["x"].values
    y_sample_clip = sample_clip["y"].values
    return x_sample_clip, y_sample_clip, sample_clip[["x", "y"]]


def build_future_hist(
    samples_ph_labels, samples_ph, metrics_data, metrics, metric_thresholds, strategies
):
    df_future_hist = pd.DataFrame()
    for m in metrics:
        for ph, samples in zip(samples_ph_labels, samples_ph):
            samples_x = samples["x"].values
            samples_y = samples["y"].values

            for s in strategies:
                x_grid, y_grid, z = construct_interpolated_data(
                    metrics_data, m, s, 1000
                )
                interpolated_samples = interpolate_samples(
                    [samples_x, samples_y], x_grid, y_grid, z
                )
                df = pd.DataFrame(
                    {
                        "Metric": m,
                        "PH": ph,
                        "Strategy": s,
                        "value": interpolated_samples,
                        "value_diff": [
                            i - metric_thresholds[m] for i in interpolated_samples
                        ],
                        "dp": samples_x,
                        "dt": samples_y,
                    }
                )
                df = df.loc[df.value != -999]
                df_future_hist = pd.concat([df_future_hist, df], axis=0)
    return df_future_hist


def cumulative_cdf(dp, dp_step, dt, dt_step, rv):
    x, y = np.mgrid[
        dp.min() : dp.max() + dp_step : dp_step, dt.min() : dt.max() + dt_step : dt_step
    ]
    z = pd.DataFrame(rv.pdf(np.dstack((x, y))))
    z = z / np.sum(z,axis=0).sum()
    z.columns = [i for i in y[0]]
    z["dp"] = [i for i in x.T[0]]
    z_cum = z.melt(id_vars=["dp"]).sort_values("value")
    z_cum["value"] = 1 - z_cum["value"].cumsum()
    cdf = z_cum.pivot_table(index="dp", columns="variable").to_numpy()
    return x, y, cdf


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
