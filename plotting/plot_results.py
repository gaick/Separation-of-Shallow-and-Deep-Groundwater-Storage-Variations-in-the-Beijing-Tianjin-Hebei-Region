"""
浅层 GWS 3DCNN 预测结果可视化（含过拟合诊断）。

生成图：
  1. 时空填色对比（8个月 × 观测/预测/误差）
  2. 区域平均时间序列
  3. 散点图（每个点 = 一个格点在某月的值）
  4. 空间 RMSE 分布
  5. 训练/验证/测试泛化对比（过拟合诊断）
  6. 学习曲线（Train vs Val 的 R²/MSE/MAE）
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from gws_io import PROPOSED_MODEL_LABEL, RESULTS_DIR, calculate_metrics, load_prediction_results, regional_mean, squeeze_grid
from gws_paths import json_file
from plotting.plot_locale import PlotLocale, get_locale, setup_matplotlib

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
_LOC: PlotLocale = get_locale("zh")
LEARNING_CURVES_PATH = json_file(RESULTS_DIR, "learning_curves.json")
GENERALIZATION_PATH = json_file(RESULTS_DIR, "generalization_metrics.json")

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Arial Unicode MS", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
})

FIGURE_GUIDE = """\
图1 时空填色对比图 (spatial_maps_8months.png / spatial_maps_8months_p1.png / _p2.png)
  - 每一行 = 测试集的一个月（2024-05 ~ 2024-12，共8个月）
  - 左列「观测值」= 真实浅层地下水储量变化（cm）
  - 中列「预测值」= 本文模型输出
  - 右列「误差」= 预测值 − 观测值（越接近白色越好）
  - p1=5~8月, p2=9~12月（论文排版推荐用分页版）
  - 颜色：红=正值（储量增加），蓝=负值（储量减少）

图2 区域平均时间序列 (regional_mean_timeseries.png)
  - 把 27×25 格网每月做空间平均，得到一条时间曲线
  - 蓝线=观测，橙虚线=预测
  - 用于快速看 8 个月整体趋势是否跟得上

图3 散点图 (scatter_pred_vs_true.png)
  - 每个绿点 = 一个「格点 × 月份」的观测-预测配对（共约 675×8=5400 点）
  - 越贴近黑色虚线（y=x）说明预测越准
  - 左上角 R²/RMSE 为全格网全月份统计

图4 空间 RMSE 分布 (mean_spatial_rmse.png)
  - 每个格点在 8 个月上的平均预测误差（RMSE，cm）
  - 黄色=误差小，红色=误差大
  - 边缘格点误差通常更大（卷积边界效应）

图5 泛化能力对比 (generalization_comparison.png) 【过拟合诊断】
  - 对比训练集 / 验证集 / 测试集的 R²、MAE、MSE
  - 对应论文表 3-1 数值见 generalization_metrics_table.csv
  - 若训练集远高于测试集 → 可能过拟合
  - 若三者接近 → 模型泛化良好

图6 学习曲线 (learning_curves.png)
  - 横轴=训练轮次，纵轴=指标
  - 蓝线=训练集，橙线=验证集
  - 验证集曲线持续上升后平稳，且与训练集差距不大 → 训练健康
"""


def _month_label(t) -> str:
    s = str(t)
    return s[:7] if len(s) >= 7 else s


def _masked_for_plot(grid: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[:, ~spatial_mask] = np.nan
    return out


NODATA_COLOR = "#d9d9d9"

# 京津冀省界 shapefile（复制 Windows 上 D:/data/2022年省界/sheng2022.shp 到此处）
JJJ_SHAPEFILE_CANDIDATES = (
    os.environ.get("JJJ_SHAPEFILE", ""),
    "./data/boundary/sheng2022.shp",
    "./data/boundary/jjj_provinces.shp",
)
JJJ_PROVINCE_NAMES = ("北京市", "天津市", "河北省")
_boundary_cache: object = None  # None=未加载, False=无数据, GeoDataFrame=成功


def _load_jjj_boundary():
    """加载京津冀三省界；需本地 shapefile + geopandas。"""
    global _boundary_cache
    if _boundary_cache is not None:
        return _boundary_cache if _boundary_cache is not False else None

    shape_path = next((p for p in JJJ_SHAPEFILE_CANDIDATES if p and os.path.exists(p)), None)
    if not shape_path:
        print("⚠️  未找到京津冀省界 shapefile，子图仅显示有效格网轮廓。"
              "请将 sheng2022.shp 放到 data/boundary/ 或设置环境变量 JJJ_SHAPEFILE")
        _boundary_cache = False
        return None

    try:
        import geopandas as gpd
    except ImportError:
        print("⚠️  未安装 geopandas，无法绘制省界。运行: pip install geopandas")
        _boundary_cache = False
        return None

    gdf = gpd.read_file(shape_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    name_col = next((c for c in ("省", "NAME", "name", "Province", "province") if c in gdf.columns), None)
    if name_col:
        gdf = gdf[gdf[name_col].isin(JJJ_PROVINCE_NAMES)]
    if gdf.empty:
        print(f"⚠️  shapefile 中未筛到 {JJJ_PROVINCE_NAMES}，将绘制全部要素")
        gdf = gpd.read_file(shape_path)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

    _boundary_cache = gdf
    print(f"✅ 已加载京津冀省界: {shape_path} ({len(gdf)} 个要素)")
    return gdf


def _draw_admin_boundary(ax: plt.Axes, gdf) -> None:
    """在子图上叠加省界（黑色轮廓，类似示例图）。"""
    if gdf is None:
        return
    gdf.boundary.plot(ax=ax, color="#111111", linewidth=0.75, zorder=4)


def _mask_extent(
    lat: np.ndarray, lon: np.ndarray, spatial_mask: np.ndarray, pad: float = 0.08
) -> tuple[float, float, float, float]:
    """裁剪到有效格网范围，去掉四周无数据留白。"""
    rows, cols = np.where(spatial_mask)
    if rows.size == 0:
        return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    return (
        float(lon[cols.min()] - pad),
        float(lon[cols.max()] + pad),
        float(lat[rows.min()] - pad),
        float(lat[rows.max()] + pad),
    )


def _cmap_with_nodata(name: str) -> mcolors.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=NODATA_COLOR)
    return cmap


def _symmetric_limits(*arrays: np.ndarray, pct: float = 98) -> tuple[float, float]:
    stacked = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if stacked.size == 0:
        return -1.0, 1.0
    lim = np.percentile(np.abs(stacked), pct)
    return -max(lim, 0.1), max(lim, 0.1)


def write_figure_guide(out_dir: str) -> str | None:
    if _LOC.lang != "zh":
        return None
    path = os.path.join(out_dir, _LOC.figure_guide_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(FIGURE_GUIDE)
    return path


def _figure_size_for_extent(
    n_rows: int,
    n_cols: int,
    extent: tuple[float, float, float, float],
    panel_width: float = 1.85,
    *,
    wspace: float = 0.14,
    hspace: float = 0.16,
) -> tuple[float, float]:
    """按经纬度跨度计算 figsize；为子图间距和外侧坐标轴标题留空。"""
    lon_min, lon_max, lat_min, lat_max = extent
    geo_h_over_w = (lat_max - lat_min) / max(lon_max - lon_min, 1e-6)
    panel_h = panel_width * geo_h_over_w
    gap_w = max(0, n_cols - 1) * wspace * panel_width
    gap_h = max(0, n_rows - 1) * hspace * panel_h
    fig_w = panel_width * n_cols + gap_w + 1.35   # 右侧双 colorbar 留宽
    fig_h = panel_h * n_rows + gap_h + 0.55       # 总标题 + 底部轴标
    return fig_w, fig_h


def _draw_spatial_panel(
    ax: plt.Axes,
    arr: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cmap: mcolors.Colormap,
    vmin: float,
    vmax: float,
    extent: tuple[float, float, float, float],
    spatial_mask: np.ndarray,
    boundary_gdf=None,
    *,
    month: str = "",
    col_title: str = "",
) -> plt.cm.ScalarMappable:
    ax.set_facecolor(NODATA_COLOR)
    lon_min, lon_max, lat_min, lat_max = extent
    im = ax.pcolormesh(
        lon, lat, arr, cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", rasterized=True,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.contour(
        lon, lat, spatial_mask.astype(float),
        levels=[0.5], colors="#666666", linewidths=0.35, linestyles="--", zorder=3,
    )
    _draw_admin_boundary(ax, boundary_gdf)
    if month:
        ax.text(
            0.04, 0.96, month, transform=ax.transAxes,
            fontsize=7, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, linewidth=0),
        )
    if col_title:
        ax.set_title(col_title, fontsize=8, fontweight="bold", pad=3)
    # 参考 Fig.2：每个子图各自带经纬度刻度，轴标题统一挂在外侧
    ax.tick_params(labelsize=6, length=2, pad=1, width=0.45, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    return im


def _save_spatial_figure(
    y_true_m: np.ndarray,
    y_pred_m: np.ndarray,
    error: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    times,
    row_indices: list[int],
    out_path: str,
    suptitle: str,
    spatial_mask: np.ndarray,
    pred_label: str | None = None,
) -> str:
    if pred_label is None:
        model = _model_title(os.path.dirname(out_path) if "figures" in out_path else RESULTS_DIR)
        pred_label = _LOC.predicted.format(model=model)
    n = len(row_indices)
    vmin, vmax = _symmetric_limits(y_true_m[row_indices], y_pred_m[row_indices])
    err = error[row_indices]
    err_lim = np.nanpercentile(np.abs(err[np.isfinite(err)]), 98) if np.isfinite(err).any() else 1.0
    err_lim = max(float(err_lim), 0.1)
    extent = _mask_extent(lat, lon, spatial_mask)
    boundary_gdf = _load_jjj_boundary()
    cmap_val = _cmap_with_nodata("RdBu_r")
    cmap_err = _cmap_with_nodata("coolwarm")
    col_titles = (_LOC.observed, pred_label, _LOC.error.format(lim=err_lim))

    wspace, hspace = 0.14, 0.16
    fig_w, fig_h = _figure_size_for_extent(n, 3, extent, wspace=wspace, hspace=hspace)
    fig, axes = plt.subplots(n, 3, figsize=(fig_w, fig_h))
    if n == 1:
        axes = np.array([axes])

    im_val = im_err = None
    for ri, i in enumerate(row_indices):
        month = _month_label(times[i])
        row_axes = axes[ri]
        im_val = _draw_spatial_panel(
            row_axes[0], y_true_m[i], lat, lon, cmap_val, vmin, vmax, extent, spatial_mask,
            boundary_gdf, month=month, col_title=col_titles[0] if ri == 0 else "",
        )
        _draw_spatial_panel(
            row_axes[1], y_pred_m[i], lat, lon, cmap_val, vmin, vmax, extent, spatial_mask,
            boundary_gdf, col_title=col_titles[1] if ri == 0 else "",
        )
        im_err = _draw_spatial_panel(
            row_axes[2], error[i], lat, lon, cmap_err, -err_lim, err_lim, extent, spatial_mask,
            boundary_gdf, col_title=col_titles[2] if ri == 0 else "",
        )

    fig.subplots_adjust(left=0.08, right=0.80, top=0.91, bottom=0.08, wspace=wspace, hspace=hspace)
    fig.supylabel(_LOC.latitude, fontsize=8, x=0.02)
    fig.supxlabel(_LOC.longitude, fontsize=8, y=0.015)

    cax_val = fig.add_axes([0.835, 0.12, 0.012, 0.70])
    cb1 = fig.colorbar(im_val, cax=cax_val)
    cb1.set_label("GWS (cm)", fontsize=7, labelpad=6)
    cb1.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb1.ax.yaxis.set_label_position("left")

    cax_err = fig.add_axes([0.908, 0.12, 0.012, 0.70])
    cb2 = fig.colorbar(im_err, cax=cax_err)
    cb2.set_label(_LOC.error_cm, fontsize=7, labelpad=5)
    cb2.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb2.ax.yaxis.set_ticks_position("right")
    cb2.ax.yaxis.set_label_position("right")

    fig.suptitle(suptitle, fontsize=9.5, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_path


def plot_spatial_maps(data: dict, out_dir: str) -> list[str]:
    y_pred = squeeze_grid(data["y_pred"])
    y_true = squeeze_grid(data["y_true"])
    lat, lon = data["lat"], data["lon"]
    mask = data["spatial_mask"]
    times = data["test_time"]
    n = y_pred.shape[0]

    y_true_m = _masked_for_plot(y_true, mask)
    y_pred_m = _masked_for_plot(y_pred, mask)
    error = y_pred_m - y_true_m

    paths = []
    root_for_title = os.path.dirname(out_dir) if os.path.basename(out_dir) == "figures" else out_dir
    model_title = _model_title(root_for_title)
    pred_label = _LOC.predicted.format(model=model_title)
    part_upper = "Part 1" if _LOC.lang == "en" else "上"
    part_lower = "Part 2" if _LOC.lang == "en" else "下"

    paths.append(_save_spatial_figure(
        y_true_m, y_pred_m, error, lat, lon, times, list(range(n)),
        os.path.join(out_dir, "spatial_maps_8months.png"),
        _LOC.spatial_suptitle_full.format(model=model_title),
        mask,
        pred_label=pred_label,
    ))

    mid = n // 2
    paths.append(_save_spatial_figure(
        y_true_m, y_pred_m, error, lat, lon, times, list(range(0, mid)),
        os.path.join(out_dir, "spatial_maps_8months_p1.png"),
        _LOC.spatial_suptitle_part.format(
            part=part_upper, start=_month_label(times[0]), end=_month_label(times[mid - 1]),
        ),
        mask,
        pred_label=pred_label,
    ))
    paths.append(_save_spatial_figure(
        y_true_m, y_pred_m, error, lat, lon, times, list(range(mid, n)),
        os.path.join(out_dir, "spatial_maps_8months_p2.png"),
        _LOC.spatial_suptitle_part.format(
            part=part_lower, start=_month_label(times[mid]), end=_month_label(times[-1]),
        ),
        mask,
        pred_label=pred_label,
    ))
    return paths


def plot_timeseries(data: dict, out_dir: str) -> str:
    mask = data["spatial_mask"]
    y_true = squeeze_grid(data["y_true"])
    y_pred = squeeze_grid(data["y_pred"])
    times = data["test_time"]
    labels = [_month_label(t) for t in times]

    true_mean = regional_mean(y_true, mask)
    pred_mean = regional_mean(y_pred, mask)
    m = calculate_metrics(y_true, y_pred)
    mse, rmse, mae, r2 = m["MSE"], m["RMSE"], m["MAE"], m["R2"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.plot(x, true_mean, "o-", color="#1f77b4", lw=2, ms=7, label=_LOC.observed_regional)
    ax.plot(x, pred_mean, "s--", color="#ff7f0e", lw=2, ms=7, label=_LOC.predicted_regional)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel(_LOC.month)
    ax.set_ylabel(_LOC.shallow_gws_change)
    ax.set_title(_LOC.timeseries_title, fontweight="bold")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(loc="best")
    ax.text(
        0.02, 0.98,
        f"{_LOC.grid_metrics}\nR²={r2:.3f}\nMSE={mse:.3f} cm²\nMAE={mae:.3f} cm\nRMSE={rmse:.3f} cm",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    path = os.path.join(out_dir, "regional_mean_timeseries.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter(data: dict, out_dir: str) -> str:
    mask = data["spatial_mask"]
    y_true = squeeze_grid(data["y_true"])
    y_pred = squeeze_grid(data["y_pred"])

    t_flat = y_true[:, mask].ravel()
    p_flat = y_pred[:, mask].ravel()
    valid = np.isfinite(t_flat) & np.isfinite(p_flat) & (np.abs(t_flat) > 1e-5)
    t_flat, p_flat = t_flat[valid], p_flat[valid]

    m = calculate_metrics(t_flat, p_flat)
    mse, rmse, mae, r2 = m["MSE"], m["RMSE"], m["MAE"], m["R2"]
    lim = _symmetric_limits(t_flat, p_flat)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(t_flat, p_flat, alpha=0.35, s=12, c="#2ca02c", edgecolors="none")
    ax.plot(lim, lim, "k--", lw=1, label=_LOC.ideal_line)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed (cm)" if _LOC.lang == "en" else "观测值 (cm)")
    ax.set_ylabel("Predicted (cm)" if _LOC.lang == "en" else "预测值 (cm)")
    ax.set_title(_LOC.scatter_title, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.text(
        0.05, 0.95,
        f"R²={r2:.3f}\nMSE={mse:.3f} cm²\nMAE={mae:.3f} cm\nRMSE={rmse:.3f} cm\nN={len(t_flat)} {_LOC.points_suffix}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    path = os.path.join(out_dir, "scatter_pred_vs_true.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mean_spatial_rmse(data: dict, out_dir: str) -> str:
    mask = data["spatial_mask"]
    y_true = squeeze_grid(data["y_true"])
    y_pred = squeeze_grid(data["y_pred"])
    lat, lon = data["lat"], data["lon"]

    sq_err = (y_pred - y_true) ** 2
    rmse_map = np.sqrt(np.nanmean(sq_err, axis=0)).astype(float)
    rmse_map[~mask] = np.nan

    extent = _mask_extent(lat, lon, mask)
    boundary_gdf = _load_jjj_boundary()
    cmap = _cmap_with_nodata("YlOrRd")

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.set_facecolor(NODATA_COLOR)
    im = ax.pcolormesh(lon, lat, rmse_map, cmap=cmap, shading="auto", rasterized=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.contour(lon, lat, mask.astype(float), levels=[0.5], colors="#666666", linewidths=0.35, linestyles="--")
    _draw_admin_boundary(ax, boundary_gdf)
    ax.set_xlabel(_LOC.longitude)
    ax.set_ylabel(_LOC.latitude)
    ax.set_title(_LOC.rmse_title, fontweight="bold")
    plt.colorbar(im, ax=ax, label="RMSE (cm)", fraction=0.046, pad=0.02)
    path = os.path.join(out_dir, "mean_spatial_rmse.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _model_title(results_dir: str) -> str:
    cfg_path = json_file(results_dir, "model_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            label = json.load(f).get("model_label", PROPOSED_MODEL_LABEL)
    else:
        label = PROPOSED_MODEL_LABEL
    if _LOC.lang == "en":
        return _LOC.proposed_model_label
    return label


def write_generalization_table(results_dir: str, out_dir: str) -> str | None:
    path = json_file(results_dir, "generalization_metrics.json")
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        gen = json.load(f)

    fieldnames = [_LOC.dataset_col, "R²", "MAE (cm)", "RMSE (cm)", "MSE (cm²)"]
    rows = []
    for split, label in [("train", _LOC.train), ("val", _LOC.val), ("test", _LOC.test)]:
        m = gen[split]
        rows.append({
            _LOC.dataset_col: label,
            "R²": f"{m['R2']:.3f}",
            "MAE (cm)": f"{m['MAE']:.3f}",
            "RMSE (cm)": f"{m['RMSE']:.3f}",
            "MSE (cm²)": f"{m['MSE']:.3f}",
        })

    out = os.path.join(out_dir, "generalization_metrics_table.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot_generalization(results_dir: str, out_dir: str) -> str | None:
    path = json_file(results_dir, "generalization_metrics.json")
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        gen = json.load(f)

    splits = ["train", "val", "test"]
    labels = [_LOC.train, _LOC.val, _LOC.test]
    r2 = [gen[s]["R2"] for s in splits]
    mae = [gen[s]["MAE"] for s in splits]
    mse = [gen[s]["MSE"] for s in splits]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.16, wspace=0.32)

    bar_kw = dict(color=colors, edgecolor="white", linewidth=0.8, width=0.52)

    axes[0].bar(labels, r2, **bar_kw)
    axes[0].set_ylabel("R²", fontsize=10)
    axes[0].set_ylim(0, 1.14)
    axes[0].set_title("(a) R²", fontweight="bold", fontsize=10, pad=8)
    axes[0].tick_params(axis="both", labelsize=9)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for i, v in enumerate(r2):
        axes[0].text(i, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, mae, **bar_kw)
    axes[1].set_ylabel("MAE (cm)", fontsize=10)
    axes[1].set_ylim(0, max(mae) * 1.28)
    axes[1].set_title("(b) MAE", fontweight="bold", fontsize=10, pad=8)
    axes[1].tick_params(axis="both", labelsize=9)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for i, v in enumerate(mae):
        axes[1].text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(labels, mse, **bar_kw)
    axes[2].set_ylabel("MSE (cm²)", fontsize=10)
    axes[2].set_ylim(0, max(mse) * 1.28)
    axes[2].set_title("(c) MSE", fontweight="bold", fontsize=10, pad=8)
    axes[2].tick_params(axis="both", labelsize=9)
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    for i, v in enumerate(mse):
        axes[2].text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    gap = gen["train"]["R2"] - gen["test"]["R2"]
    model_title = _model_title(results_dir)
    fig.suptitle(
        _LOC.generalization_title.format(model=model_title),
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.04,
        f"Train R² − Test R² = {gap:+.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555",
    )

    out = os.path.join(out_dir, "generalization_comparison.png")
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def plot_learning_curves(results_dir: str, out_dir: str) -> str | None:
    path = json_file(results_dir, "learning_curves.json")
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_r2 = [h["train"]["R2"] for h in history]
    val_r2 = [h["val"]["R2"] for h in history]
    train_mse = [h["train"]["MSE"] for h in history]
    val_mse = [h["val"]["MSE"] for h in history]
    train_mae = [h["train"]["MAE"] for h in history]
    val_mae = [h["val"]["MAE"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [
        (train_r2, val_r2, "R²", _LOC.higher_better),
        (train_mse, val_mse, "MSE (cm²)", _LOC.lower_better),
        (train_mae, val_mae, "MAE (cm)", _LOC.lower_better),
    ]
    for ax, (tr, va, ylab, hint) in zip(axes, pairs):
        ax.plot(epochs, tr, "-", color="#1f77b4", lw=1.5, label=_LOC.train)
        ax.plot(epochs, va, "-", color="#ff7f0e", lw=1.5, label=_LOC.val)
        ax.set_xlabel(_LOC.epoch)
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} ({hint})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(_LOC.learning_curves_title, fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "learning_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main(results_dir: str | None = None, out_dir: str | None = None, lang: str = "zh") -> None:
    global _LOC
    _LOC = get_locale(lang)
    setup_matplotlib(lang)

    root = results_dir or RESULTS_DIR
    if not os.path.exists(os.path.join(root, "Y_pred_real.npy")):
        raise FileNotFoundError(
            f"未找到预测结果。请先运行 AutoTrain.py 或 torch_test.py\n期望目录: {root}"
        )

    data = load_prediction_results(root)
    out_dir = out_dir or os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    paths = [
        write_figure_guide(out_dir),
        *plot_spatial_maps(data, out_dir),
        plot_timeseries(data, out_dir),
        plot_scatter(data, out_dir),
        plot_mean_spatial_rmse(data, out_dir),
    ]

    gen_path = plot_generalization(root, out_dir)
    if gen_path:
        paths.append(gen_path)

    table_path = write_generalization_table(root, out_dir)
    if table_path:
        paths.append(table_path)

    lc_path = plot_learning_curves(root, out_dir)
    if lc_path:
        paths.append(lc_path)

    print("\n📊 可视化完成：" if lang == "zh" else "\n📊 Figures saved:")
    for p in paths:
        if p:
            print(f"   {p}")
    if lang == "zh":
        print("\n💡 请先阅读 figures/00_图例说明.txt 了解每张图的含义")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot shallow GWS prediction results")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Results directory")
    parser.add_argument("--out-dir", default=None, help="Output figure directory")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh", help="Figure language")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir, args.lang)
