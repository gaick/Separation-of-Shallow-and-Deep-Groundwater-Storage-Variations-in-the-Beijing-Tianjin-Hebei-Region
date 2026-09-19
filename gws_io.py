"""浅层 GWS：数据 I/O、指标与结果读写（无 PyTorch 依赖，供绘图使用）。"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

import numpy as np

from gws_paths import (
    LEGACY_AUTOTRAIN_DIR,
    RESULTS_DIR,
    RESULTS_PROPOSED,
    ensure_parent,
    json_file,
)

TIME_STEPS = 12
PROPOSED_MODEL_ID = "baseline_2enc_res"
PROPOSED_MODEL_LABEL = "3DCNN(M2·2层·残差+末时刻·本文)"

_DATA_DIR_CANDIDATES = ("./data/npy", "./data/data/npy")


def resolve_data_dir(required: bool = True) -> str:
    """自动定位 npy 数据目录（兼容 data/npy 与 data/data/npy）。"""
    for path in _DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(path, "Y_shallow_gws.npy")):
            return path
    if not required:
        return "./data/npy"
    raise FileNotFoundError(
        "未找到浅层 GWS 数据，请确认存在以下之一:\n"
        "  ./data/npy/Y_shallow_gws.npy\n"
        "  ./data/data/npy/Y_shallow_gws.npy"
    )


# Soft resolve so importing metrics/helpers works without proprietary data
# (examples/quick_test.py does not need real BTH arrays).
DATA_DIR = resolve_data_dir(required=False)
METRICS_PATH = json_file(RESULTS_DIR, "metrics.json")
BEST_PARAMS_PATH = json_file(RESULTS_DIR, "best_params.json")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, min_samples: int = 10) -> dict[str, float]:
    """格点–月份 pooled 评价指标（水利/水文机器学习常用集合）。"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    valid_mask = (np.abs(y_true) > 1e-5) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
    n_valid = int(np.sum(valid_mask))
    empty = {
        "MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "R2": -1.0, "NSE": -1.0,
        "R": 0.0, "Bias": 0.0, "PBIAS": 0.0, "KGE": -1.0,
        "NRMSE": 0.0, "NRMSE_std": 0.0, "IOA": 0.0, "RMAE": 0.0, "n_samples": 0,
    }
    if n_valid < min_samples:
        return empty

    obs = y_true[valid_mask]
    sim = y_pred[valid_mask]

    mse = float(np.mean((obs - sim) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(obs - sim)))
    bias = float(np.mean(sim - obs))

    obs_mean = float(np.mean(obs))
    sim_mean = float(np.mean(sim))
    obs_std = float(np.std(obs))
    sim_std = float(np.std(sim))

    ss_res = float(np.sum((obs - sim) ** 2))
    ss_tot = float(np.sum((obs - obs_mean) ** 2))
    nse = float(1 - ss_res / (ss_tot + 1e-8))

    if obs_std > 1e-8 and sim_std > 1e-8:
        r = float(np.corrcoef(obs, sim)[0, 1])
    else:
        r = 0.0

    alpha = sim_std / (obs_std + 1e-8)
    beta = sim_mean / (obs_mean + 1e-8) if abs(obs_mean) > 1e-8 else 1.0
    kge = float(1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    obs_range = float(np.max(obs) - np.min(obs))
    nrmse = rmse / (obs_range + 1e-8)
    nrmse_std = rmse / (obs_std + 1e-8)

    denom_ioa = float(np.sum((np.abs(sim - obs_mean) + np.abs(obs - obs_mean)) ** 2))
    ioa = float(1 - ss_res / (denom_ioa + 1e-8))

    pbias = float(100.0 * np.sum(sim - obs) / (np.sum(np.abs(obs)) + 1e-8))
    rmae = float(mae / (np.mean(np.abs(obs)) + 1e-8))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": nse,
        "NSE": nse,
        "R": r,
        "Bias": bias,
        "PBIAS": pbias,
        "KGE": kge,
        "NRMSE": nrmse,
        "NRMSE_std": nrmse_std,
        "IOA": ioa,
        "RMAE": rmae,
        "n_samples": n_valid,
    }


def inspect_data_summary(
    data_dir: str | None = None,
    y_filename: str = "Y_shallow_gws.npy",
) -> dict[str, Any]:
    """加载前检查数据形状、单位与数值范围。"""
    root = data_dir or DATA_DIR
    X = np.load(os.path.join(root, "X_drivers.npy"))
    Y = np.load(os.path.join(root, y_filename))
    valid_y = Y[np.isfinite(Y) & (np.abs(Y) > 1e-6)]
    summary = {
        "data_dir": root,
        "y_filename": y_filename,
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "Y_unit": "cm",
        "Y_min": float(valid_y.min()),
        "Y_max": float(valid_y.max()),
        "Y_mean": float(valid_y.mean()),
        "Y_std": float(valid_y.std()),
        "X_standardized": True,
        "Y_standardized": abs(valid_y.mean()) < 0.5 and abs(valid_y.std() - 1.0) < 0.5,
        "X_pre_standardized": True,
        "note": "Y 为 cm 物理量；X 驱动因子 ch0-4 已 Z-score；训练时仅对 GWS 历史通道与 Y 做一次标准化",
    }
    return summary


def print_data_summary(
    data_dir: str | None = None,
    y_filename: str = "Y_shallow_gws.npy",
) -> dict[str, Any]:
    s = inspect_data_summary(data_dir, y_filename)
    print(f"📁 数据目录: {s['data_dir']}")
    print(f"   X {tuple(s['X_shape'])} (驱动因子已标准化)")
    print(f"   Y {tuple(s['Y_shape'])} 单位={s['Y_unit']}")
    print(
        f"   Y 范围: [{s['Y_min']:.2f}, {s['Y_max']:.2f}] cm | "
        f"均值={s['Y_mean']:.2f} 标准差={s['Y_std']:.2f}"
    )
    if s["Y_standardized"]:
        print("   ⚠️ Y 近似 Z-score 分布，请确认是否为物理量 cm")
    else:
        print("   ✓ Y 为物理量(cm)，训练时将仅用训练月份做标准化")
    return s


def load_coords(data_dir: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = data_dir or DATA_DIR
    lat_path = os.path.join(root, "lat_coords.npy")
    lon_path = os.path.join(root, "lon_coords.npy")
    time_path = os.path.join(root, "time_coords.npy")
    if all(os.path.exists(p) for p in (lat_path, lon_path, time_path)):
        return np.load(lat_path), np.load(lon_path), np.load(time_path)

    npz_path = os.path.join(root, "drivers_and_gws_data.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data["lat"], data["lon"], data["time"]

    lat = np.linspace(36.0, 42.5, 27)
    lon = np.linspace(113.5, 119.5, 25)
    return lat, lon, np.array([])


def get_test_target_indices(n_raw_months: int, n_sequences: int) -> np.ndarray:
    val_end = int(n_sequences * 0.9)
    seq_indices = np.arange(val_end, n_sequences)
    return seq_indices + TIME_STEPS


def get_test_times(all_time: np.ndarray, n_sequences: int) -> np.ndarray:
    target_idx = get_test_target_indices(len(all_time), n_sequences)
    return all_time[target_idx]


def squeeze_grid(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0, :, :]
    return arr


def regional_mean(grid: np.ndarray, spatial_mask: np.ndarray | None = None) -> np.ndarray:
    grid = squeeze_grid(grid)
    means = []
    for t in range(grid.shape[0]):
        slice_ = grid[t]
        if spatial_mask is not None:
            vals = slice_[spatial_mask]
        else:
            vals = slice_[np.abs(slice_) > 1e-5]
        means.append(float(np.nanmean(vals)) if vals.size else 0.0)
    return np.array(means)


def save_best_params(params: dict[str, Any], results_dir: str | None = None) -> None:
    root = results_dir or RESULTS_DIR
    os.makedirs(root, exist_ok=True)
    path = json_file(root, "best_params.json")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def load_best_params_from(results_dir: str) -> dict[str, Any]:
    path = json_file(results_dir, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return load_best_params()


def load_best_params() -> dict[str, Any]:
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, encoding="utf-8") as f:
            return json.load(f)
    legacy = json_file(LEGACY_AUTOTRAIN_DIR, "best_params.json")
    if os.path.exists(legacy):
        with open(legacy, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"未找到 best_params.json，请先运行 AutoTrain.py。\n期望路径: {BEST_PARAMS_PATH}"
    )


def save_prediction_results(
    y_pred_real: np.ndarray,
    y_true_real: np.ndarray,
    spatial_mask: np.ndarray,
    metrics: dict[str, float],
    source: str = "predict",
    results_dir: str | None = None,
    data_dir: str | None = None,
) -> str:
    root = results_dir or RESULTS_DIR
    os.makedirs(root, exist_ok=True)

    y_pred = squeeze_grid(y_pred_real)
    y_true = squeeze_grid(y_true_real)
    n_months = y_pred.shape[0]

    lat, lon, all_time = load_coords(data_dir)
    if len(all_time):
        n_sequences = len(all_time) - TIME_STEPS
        test_time = get_test_times(all_time, n_sequences)
    else:
        test_time = np.arange(n_months)

    np.save(os.path.join(root, "Y_pred_real.npy"), y_pred)
    np.save(os.path.join(root, "Y_true_real.npy"), y_true)
    np.save(os.path.join(root, "spatial_mask.npy"), spatial_mask)
    np.save(os.path.join(root, "lat.npy"), lat)
    np.save(os.path.join(root, "lon.npy"), lon)
    np.save(os.path.join(root, "test_time.npy"), test_time)

    true_mean = regional_mean(y_true, spatial_mask)
    pred_mean = regional_mean(y_pred, spatial_mask)
    time_labels = [str(t)[:7] for t in test_time] if len(test_time) == n_months else list(range(1, n_months + 1))

    csv_path = os.path.join(root, "regional_mean_summary.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Month", "True_Regional_Mean_(cm)", "Pred_Regional_Mean_(cm)", "Abs_Error_(cm)"])
        for i in range(n_months):
            writer.writerow([
                time_labels[i],
                f"{true_mean[i]:.6f}",
                f"{pred_mean[i]:.6f}",
                f"{abs(true_mean[i] - pred_mean[i]):.6f}",
            ])

    meta = {
        "source": source,
        "n_test_months": int(n_months),
        "grid_shape": list(y_pred.shape),
        "units": "cm",
        "description": "Full-grid shallow GWS predictions on test set (8 months)",
        **metrics,
    }
    metrics_path = json_file(root, "metrics.json")
    ensure_parent(metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 全格网预测结果已保存至: {root}")
    print(f"   Y_pred_real.npy  {y_pred.shape}  |  Y_true_real.npy  {y_true.shape}")
    return root


def load_prediction_results(results_dir: str | None = None) -> dict[str, Any]:
    root = results_dir or RESULTS_DIR
    y_pred = np.load(os.path.join(root, "Y_pred_real.npy"))
    y_true = np.load(os.path.join(root, "Y_true_real.npy"))
    spatial_mask = np.load(os.path.join(root, "spatial_mask.npy"))
    lat = np.load(os.path.join(root, "lat.npy"))
    lon = np.load(os.path.join(root, "lon.npy"))
    test_time = np.load(os.path.join(root, "test_time.npy"), allow_pickle=True)

    metrics = {}
    metrics_path = json_file(root, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "spatial_mask": spatial_mask,
        "lat": lat,
        "lon": lon,
        "test_time": test_time,
        "metrics": metrics,
        "results_dir": root,
    }
