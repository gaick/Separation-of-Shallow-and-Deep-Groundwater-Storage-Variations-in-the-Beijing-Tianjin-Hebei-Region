"""
深层 GWS 分离与可视化。

ΔGWS_deep = ΔGWS_total − ΔGWS_shallow
  - ΔGWS_total: jingjinji_groundwater_storage_final.nc (delta_gws)
  - ΔGWS_shallow: 3DCNN 预测（测试期）或观测格网（全时段验证）

输出:
  results/figures/deep_gws/deep_spatial_maps_8months.png
  results/figures/deep_gws/deep_regional_timeseries.png
  results/figures/deep_gws/pp_vs_ecp_comparison.png
  results/json/deep_gws/deep_gws_summary.json
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from gws_io import DATA_DIR, RESULTS_PROPOSED, TIME_STEPS, get_test_target_indices, load_coords, regional_mean, squeeze_grid
from gws_paths import DEEP_GWS_FIG_DIR, DEEP_GWS_JSON_DIR, ensure_parent

NC_DIR = "./data/nc"
FIG_DIR = DEEP_GWS_FIG_DIR
SHALLOW_CNN_DIR = RESULTS_PROPOSED

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Arial Unicode MS", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
})


def _monthly_aggregate(ds: xr.Dataset, var: str, ref_lat: np.ndarray, ref_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将不规则时间 nc 聚合为月均值，并重采样到参考格网（最近邻）。"""
    da = ds[var]
    monthly = da.resample(time="MS").mean()
    monthly = monthly.sel(
        lat=slice(ref_lat.min(), ref_lat.max()),
        lon=slice(ref_lon.min(), ref_lon.max()),
    )
    src_lat = monthly.lat.values
    src_lon = monthly.lon.values
    src_data = monthly.values  # (T, Hs, Ws)

    lat_idx = np.array([np.argmin(np.abs(src_lat - la)) for la in ref_lat])
    lon_idx = np.array([np.argmin(np.abs(src_lon - lo)) for lo in ref_lon])
    out = src_data[:, lat_idx[:, None], lon_idx[None, :]]

    times = pd.to_datetime(monthly.time.values)
    return times, out.astype(np.float32)


def _huang2015_1deg_is_pp(lon_val: np.ndarray, lat_val: np.ndarray) -> np.ndarray:
    """判断 1° GRACE 网格是否归属山前平原 PP（Huang et al., 2015, GRL Fig.1）。"""
    lon_c = np.floor(lon_val) + 0.5
    lat_c = np.floor(lat_val) + 0.5
    pp = lon_c <= 114.5
    pp |= (lon_c == 115.5) & (lat_c >= 38.5)
    pp |= (lon_c == 116.5) & (lat_c >= 40.5)
    return pp


def build_pp_ecp_masks(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """山前平原 PP vs 中东部平原 ECP（Huang et al., 2015, doi:10.1002/2014GL062498）。

    PP：太行山山前冲洪积平原 + 冲积扇，以浅层潜水开采为主；
    ECP：冲积平原 + 滨海平原，以深层承压水开采为主。
    按原文 §2.1，对每个 0.25° 格点按其所在 1° GRACE 网格的水文地质归属划分
    （分区依据 Wu et al., 1996; Wang et al., 2010，与 Huang Fig.1 一致）。
    """
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    pp = _huang2015_1deg_is_pp(lon2d, lat2d)
    ecp = ~pp
    return {"pp": pp, "ecp": ecp}


def masked_regional_mean(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    means = []
    for t in range(grid.shape[0]):
        vals = grid[t][mask]
        means.append(float(np.nanmean(vals)) if vals.size else np.nan)
    return np.array(means)


def plot_spatial_maps(deep: np.ndarray, lat: np.ndarray, lon: np.ndarray, labels: list[str], out_path: str, spatial_mask: np.ndarray | None = None) -> None:
    n = deep.shape[0]
    if spatial_mask is None:
        spatial_mask = np.isfinite(deep).any(axis=0)
    rows, cols = np.where(spatial_mask)
    pad = 0.08
    extent = (
        float(lon[cols.min()] - pad), float(lon[cols.max()] + pad),
        float(lat[rows.min()] - pad), float(lat[rows.max()] + pad),
    ) if rows.size else (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    vmax = np.nanpercentile(np.abs(deep), 95)

    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.45 * n + 0.4), gridspec_kw={"hspace": 0.08})
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.set_facecolor("#d9d9d9")
        im = ax.pcolormesh(lon, lat, deep[i], cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.contour(lon, lat, spatial_mask.astype(float), levels=[0.5], colors="#333333", linewidths=0.55)
        ax.set_title(f"深层 GWS {labels[i]} (cm)", fontsize=9, pad=3)
        if i == n - 1:
            ax.set_xlabel("经度 °E", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylabel("纬度 °N", fontsize=8)
        ax.tick_params(labelsize=7, length=2)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    fig.suptitle("深层地下水储量变化空间分布（测试期）", fontsize=11, y=0.995)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def plot_regional_ts(times: np.ndarray, deep_total: np.ndarray, deep_ecp: np.ndarray, deep_pp: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    t = pd.to_datetime(times)
    ax.plot(t, deep_total, "k-", lw=2, label="全区平均")
    ax.plot(t, deep_ecp, "r--", lw=1.8, label="中东部平原 (ECP)")
    ax.plot(t, deep_pp, "g--", lw=1.8, label="山前平原 (PP)")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("深层 GWS 异常 (cm)")
    ax.set_xlabel("时间")
    ax.set_title("深层地下水储量变化 — PP/ECP 对比（2018–2024）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_subregion_bar(ecp_mean: float, pp_mean: float, ecp_trend: float, pp_trend: float, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["中东部平原\n(ECP)", "山前平原\n(PP)"]
    axes[0].bar(labels, [ecp_mean, pp_mean], color=["#c0392b", "#27ae60"])
    axes[0].set_ylabel("测试期平均深层 GWS (cm)")
    axes[0].set_title("测试期空间均值对比")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, [ecp_trend, pp_trend], color=["#c0392b", "#27ae60"])
    axes[1].set_ylabel("线性趋势 (cm/月)")
    axes[1].set_title("测试期深层 GWS 变化趋势")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("深层 GWS：ECP vs PP（Huang et al., 2015 分区）", fontsize=12)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def linear_trend(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return 0.0
    coef = np.polyfit(x[valid], y[valid], 1)
    return float(coef[0])


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    lat, lon, all_time = load_coords()
    Y_obs = np.load(os.path.join(DATA_DIR, "Y_shallow_gws.npy"))
    spatial_mask = np.any(~np.isnan(Y_obs) & (np.abs(Y_obs) > 1e-6), axis=0)

    ds = xr.open_dataset(os.path.join(NC_DIR, "jingjinji_groundwater_storage_final.nc"))
    nc_times, gws_total = _monthly_aggregate(ds, "delta_gws", lat, lon)
    ds.close()

    # 对齐 2018-01 ~ 2024-12
    if len(all_time):
        ref_times = pd.to_datetime(all_time)
    else:
        ref_times = pd.date_range("2018-01-01", periods=len(Y_obs), freq="MS")

    shallow_obs = Y_obs.copy()
    shallow_obs[:, ~spatial_mask] = np.nan

    # 全时段深层（观测浅层）
    deep_full = np.full_like(shallow_obs, np.nan)
    for i, t in enumerate(ref_times):
        idx = np.where(nc_times == t)[0]
        if len(idx) == 0:
            # 最近月匹配
            j = int(np.argmin(np.abs(nc_times - t)))
        else:
            j = idx[0]
        if j < gws_total.shape[0]:
            deep_full[i] = gws_total[j] - shallow_obs[i]

    masks = build_pp_ecp_masks(lat, lon)
    pp_m = masks["pp"] & spatial_mask
    ecp_m = masks["ecp"] & spatial_mask

    deep_regional = masked_regional_mean(deep_full, spatial_mask)
    deep_pp = masked_regional_mean(deep_full, pp_m)
    deep_ecp = masked_regional_mean(deep_full, ecp_m)

    # 测试期：用 3DCNN 预测浅层
    n_seq = len(Y_obs) - TIME_STEPS
    test_idx = get_test_target_indices(len(Y_obs), n_seq)
    test_labels = [str(ref_times[i])[:7] for i in test_idx]

    pred_path = os.path.join(SHALLOW_CNN_DIR, "Y_pred_real.npy")
    if os.path.exists(pred_path):
        shallow_pred = squeeze_grid(np.load(pred_path))
    else:
        print("⚠️ 未找到 3DCNN 预测，测试期使用 Persistence 浅层代替")
        shallow_pred = np.stack([Y_obs[i - 1] for i in test_idx])

    deep_test = []
    for k, t_idx in enumerate(test_idx):
        t = ref_times[t_idx]
        j_arr = np.where(nc_times == t)[0]
        j = int(j_arr[0]) if len(j_arr) else int(np.argmin(np.abs(nc_times - t)))
        total_t = gws_total[j]
        shallow_t = shallow_pred[k]
        deep_t = total_t - shallow_t
        deep_t[~spatial_mask] = np.nan
        deep_test.append(deep_t)
    deep_test = np.stack(deep_test)

    # 图 1：测试期空间图
    plot_spatial_maps(
        deep_test, lat, lon, test_labels,
        os.path.join(FIG_DIR, "deep_spatial_maps_8months.png"),
        spatial_mask=spatial_mask,
    )

    # 图 2：全时段区域序列
    plot_regional_ts(
        ref_times, deep_regional, deep_ecp, deep_pp,
        os.path.join(FIG_DIR, "deep_regional_timeseries.png"),
    )

    # 图 3：ECP vs PP（测试期）
    test_ecp = masked_regional_mean(deep_test, ecp_m)
    test_pp = masked_regional_mean(deep_test, pp_m)
    plot_subregion_bar(
        float(np.nanmean(test_ecp)),
        float(np.nanmean(test_pp)),
        linear_trend(test_ecp),
        linear_trend(test_pp),
        os.path.join(FIG_DIR, "pp_vs_ecp_comparison.png"),
    )

    # 与 CSV 文献结果交叉验证
    csv_path = os.path.join(NC_DIR, "gws_literature_method_results_correct_sy.csv")
    csv_deep_mean = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        csv_deep_mean = float(df["GWS_Deep"].dropna().mean())

    fp_ecp = float(np.nanmean(deep_ecp))
    fp_pp = float(np.nanmean(deep_pp))
    tp_ecp = float(np.nanmean(test_ecp))
    tp_pp = float(np.nanmean(test_pp))
    if fp_ecp < fp_pp:
        interpretation = (
            "ECP（深层承压水主采区）深层 GWS 负异常幅度大于 PP（山前平原），"
            "与 Huang et al. (2015) 水文地质认识一致"
        )
    else:
        interpretation = (
            f"Huang et al. (2015) PP/ECP 分区下，全时段深层 GWS 均值 PP={fp_pp:.1f} cm、"
            f"ECP={fp_ecp:.1f} cm；测试期 ECP 趋势 {linear_trend(test_ecp):.2f} cm/月、"
            f"PP 趋势 {linear_trend(test_pp):.2f} cm/月。"
            "注：Huang 原文对比的是 PP 浅层与 ECP 深层井观测，"
            "与本文格网化深层分离残差在同一指标上对比时需审慎解读"
        )

    summary = {
        "formula": "GWS_deep = GWS_total - GWS_shallow",
        "subregion_method": "Huang et al. (2015, GRL) PP/ECP via 1° hydrogeologic grids (Wu et al., 1996; Wang et al., 2010)",
        "total_gws_source": "jingjinji_groundwater_storage_final.nc (delta_gws)",
        "shallow_test_source": SHALLOW_CNN_DIR,
        "test_months": test_labels,
        "full_period": {
            "regional_mean_deep_cm": float(np.nanmean(deep_regional)),
            "pp_mean_deep_cm": float(np.nanmean(deep_pp)),
            "ecp_mean_deep_cm": float(np.nanmean(deep_ecp)),
            "pp_trend_cm_per_month": linear_trend(deep_pp),
            "ecp_trend_cm_per_month": linear_trend(deep_ecp),
            "funnel_mean_deep_cm": float(np.nanmean(deep_ecp)),
            "non_funnel_mean_deep_cm": float(np.nanmean(deep_pp)),
            "funnel_trend_cm_per_month": linear_trend(deep_ecp),
            "non_funnel_trend_cm_per_month": linear_trend(deep_pp),
        },
        "test_period": {
            "pp_mean_deep_cm": float(np.nanmean(test_pp)),
            "ecp_mean_deep_cm": float(np.nanmean(test_ecp)),
            "pp_trend_cm_per_month": linear_trend(test_pp),
            "ecp_trend_cm_per_month": linear_trend(test_ecp),
            "funnel_mean_deep_cm": float(np.nanmean(test_ecp)),
            "non_funnel_mean_deep_cm": float(np.nanmean(test_pp)),
            "funnel_trend_cm_per_month": linear_trend(test_ecp),
            "non_funnel_trend_cm_per_month": linear_trend(test_pp),
            "regional_mean_deep_cm": float(np.nanmean(masked_regional_mean(deep_test, spatial_mask))),
        },
        "literature_csv_deep_mean_cm": csv_deep_mean,
        "interpretation": interpretation,
    }
    summary_path = os.path.join(DEEP_GWS_JSON_DIR, "deep_gws_summary.json")
    ensure_parent(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 深层 GWS 图已保存至 {FIG_DIR}/")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
