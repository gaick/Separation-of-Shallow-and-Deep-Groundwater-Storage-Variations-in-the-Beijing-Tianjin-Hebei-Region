"""Persistence 基线：下一月浅层 GWS = 上一月观测值（Y(t)=Y(t-1)）。"""

from __future__ import annotations

import bootstrap  # noqa: F401
import json
import os

import numpy as np

from gws_io import (
    DATA_DIR,
    RESULTS_PROPOSED,
    TIME_STEPS,
    calculate_metrics,
    get_test_target_indices,
    load_coords,
    regional_mean,
    save_prediction_results,
)
from gws_paths import BASELINE_PERSISTENCE_DIR, COMPARE_JSON_DIR, ensure_parent, json_file

RESULTS_DIR = BASELINE_PERSISTENCE_DIR
HIGH_CHANGE_MONTH_IDX = [1, 2, 3, 4]  # 2024-06 ~ 09
HIGH_CHANGE_THRESHOLD_CM = 0.3


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return calculate_metrics(y_true, y_pred)


def _squeeze_grid_stack(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        return arr[:, 0]
    return arr


def compute_stratified_comparison(
    y_true: np.ndarray,
    y_pred_cnn: np.ndarray,
    y_pred_per: np.ndarray,
    Y_raw: np.ndarray,
    spatial_mask: np.ndarray,
    test_target_idx: np.ndarray,
) -> dict:
    """分层评价：全测试集 / 高变化月份 / 高变化格点。"""
    yt = _squeeze_grid_stack(y_true)
    yc = _squeeze_grid_stack(y_pred_cnn)
    yp = _squeeze_grid_stack(y_pred_per)

    delta = np.zeros_like(yt)
    for i, t_idx in enumerate(test_target_idx):
        delta[i] = np.abs(Y_raw[t_idx] - Y_raw[t_idx - 1])

    high_cell_mask = (delta > HIGH_CHANGE_THRESHOLD_CM) & spatial_mask
    high_months = HIGH_CHANGE_MONTH_IDX

    lag1_accs = []
    for i in range(Y_raw.shape[1]):
        for j in range(Y_raw.shape[2]):
            if not spatial_mask[i, j]:
                continue
            s = Y_raw[:, i, j]
            s = s[np.isfinite(s)]
            if len(s) > 13:
                lag1_accs.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))

    def subset_metrics(yt_sub, yc_sub, yp_sub):
        return {
            "cnn": _metrics_dict(yt_sub, yc_sub),
            "persistence": _metrics_dict(yt_sub, yp_sub),
        }

    full = subset_metrics(yt, yc, yp)
    months = subset_metrics(yt[high_months], yc[high_months], yp[high_months])
    cells = subset_metrics(yt[high_cell_mask], yc[high_cell_mask], yp[high_cell_mask])

    month_rows = []
    month_labels = ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"]
    for i, label in enumerate(month_labels):
        mc = _metrics_dict(yt[i : i + 1], yc[i : i + 1])
        mp = _metrics_dict(yt[i : i + 1], yp[i : i + 1])
        dY = float(np.nanmean(delta[i][spatial_mask]))
        month_rows.append({
            "month": label,
            "mean_abs_delta_Y_cm": dY,
            "cnn_MAE": mc["MAE"],
            "persistence_MAE": mp["MAE"],
            "cnn_wins_MAE": mc["MAE"] < mp["MAE"],
        })

    return {
        "lag1_autocorr_mean": float(np.mean(lag1_accs)) if lag1_accs else None,
        "high_change_threshold_cm": HIGH_CHANGE_THRESHOLD_CM,
        "high_change_months": [month_labels[i] for i in high_months],
        "full_test": full,
        "high_change_months_metrics": months,
        "high_change_cells_metrics": {
            **cells,
            "n_samples": int(high_cell_mask.sum()),
        },
        "monthly_breakdown": month_rows,
    }


def persistence_predict(Y_raw: np.ndarray, spatial_mask: np.ndarray, test_target_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """测试期每样本：预测值 = 目标月前一月（物理量 cm）。"""
    y_true, y_pred = [], []
    for t_idx in test_target_idx:
        prev_idx = t_idx - 1
        if prev_idx < 0:
            continue
        true_t = Y_raw[t_idx].copy()
        pred_t = Y_raw[prev_idx].copy()
        true_t[~spatial_mask] = 0.0
        pred_t[~spatial_mask] = 0.0
        y_true.append(true_t)
        y_pred.append(pred_t)
    return np.stack(y_true)[:, np.newaxis, :, :], np.stack(y_pred)[:, np.newaxis, :, :]


def main() -> None:
    Y_raw = np.load(os.path.join(DATA_DIR, "Y_shallow_gws.npy"))
    spatial_mask = np.any(~np.isnan(Y_raw) & (np.abs(Y_raw) > 1e-6), axis=0)

    n_seq = len(Y_raw) - TIME_STEPS
    test_target_idx = get_test_target_indices(len(Y_raw), n_seq)

    y_true, y_pred = persistence_predict(Y_raw, spatial_mask, test_target_idx)
    metrics = calculate_metrics(y_true, y_pred)

    print("=" * 60)
    print("Persistence 基线（Y(t)=Y(t-1)）— 测试集")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  NSE  = {metrics['NSE']:.4f}")
    print(f"  KGE  = {metrics['KGE']:.4f}")
    print(f"  MAE  = {metrics['MAE']:.4f} cm")
    print(f"  RMSE = {metrics['RMSE']:.4f} cm")
    print(f"  RMAE = {metrics['RMAE']:.4f}")
    print("=" * 60)

    save_prediction_results(
        y_pred, y_true, spatial_mask, metrics,
        source="persistence_baseline",
        results_dir=RESULTS_DIR,
    )

    # 气候态基线：训练期各格点逐月均值
    n_seq = len(Y_raw) - TIME_STEPS
    train_end = int(n_seq * 0.8)
    train_targets = np.arange(TIME_STEPS, train_end + TIME_STEPS)
    month_ids = np.array([t % 12 for t in train_targets])
    climatology = np.zeros_like(Y_raw)
    for m in range(12):
        climatology[m] = np.nanmean(Y_raw[train_targets[month_ids == m]], axis=0)

    clim_pred, clim_true = [], []
    for t_idx in test_target_idx:
        clim_pred.append(climatology[t_idx % 12])
        clim_true.append(Y_raw[t_idx])
    clim_pred = np.stack(clim_pred)[:, np.newaxis, :, :]
    clim_true = np.stack(clim_true)[:, np.newaxis, :, :]
    clim_metrics = calculate_metrics(clim_true, clim_pred)

    print("\n气候态基线（训练期逐月均值）— 测试集")
    print(f"  R²   = {clim_metrics['R2']:.4f}")
    print(f"  MAE  = {clim_metrics['MAE']:.4f} cm")

    # 空间结构：格点级相关系数（衡量空间分布保真度）
    def spatial_corr(y_t, y_p, m):
        corrs = []
        for k in range(y_t.shape[0]):
            a, b = y_t[k, 0][m].flatten(), y_p[k, 0][m].flatten()
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() > 10:
                corrs.append(float(np.corrcoef(a[valid], b[valid])[0, 1]))
        return float(np.mean(corrs)) if corrs else 0.0

    cnn_pred_path = os.path.join(RESULTS_PROPOSED, "Y_pred_real.npy")
    spatial_cnn, spatial_per = None, spatial_corr(y_true, y_pred, spatial_mask)
    if os.path.exists(cnn_pred_path):
        y_cnn = np.load(cnn_pred_path)[:, np.newaxis, :, :]
        spatial_cnn = spatial_corr(y_true, y_cnn, spatial_mask)

    _, _, all_time = load_coords()
    true_mean = regional_mean(y_true, spatial_mask)
    pred_mean = regional_mean(y_pred, spatial_mask)
    if len(all_time):
        labels = [str(t)[:7] for t in all_time[test_target_idx]]
    else:
        labels = [f"test_{i+1}" for i in range(len(test_target_idx))]

    summary = {
        "baseline": "persistence",
        "description": "Y(t) = Y(t-1), same test months as 3DCNN",
        "test_metrics": metrics,
        "climatology_metrics": clim_metrics,
        "spatial_corr": {"persistence": spatial_per, "cnn": spatial_cnn},
        "months": labels,
        "regional_mean_true": true_mean.tolist(),
        "regional_mean_pred": pred_mean.tolist(),
        "note": "浅层GWS月尺度lag-1自相关约0.95，Persistence为强基线；3DCNN价值在于融合驱动因子、支持无上月观测的业务场景及深层分离流程",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = json_file(RESULTS_DIR, "baseline_summary.json")
    ensure_parent(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 与 3DCNN 基线对比
    cnn_path = json_file(RESULTS_PROPOSED, "metrics.json")
    if os.path.exists(cnn_path):
        with open(cnn_path, encoding="utf-8") as f:
            cnn = json.load(f)
        cnn_r2 = cnn.get("R2", cnn.get("test_R2"))
        cnn_mae = cnn.get("MAE", cnn.get("test_MAE"))
        per_r2 = metrics["R2"]
        per_mae = metrics["MAE"]
        print("\n对比 3D-CNN 本文模型:")
        print(f"  ΔR²   = {cnn_r2 - per_r2:+.4f}")
        print(f"  ΔMAE  = {per_mae - cnn_mae:+.4f} cm")
        compare = {
            "persistence": metrics,
            "climatology": clim_metrics,
            "cnn_baseline": _metrics_dict(y_true, np.load(cnn_pred_path)[:, np.newaxis, :, :]),
            "delta_R2_cnn_vs_persistence": cnn_r2 - per_r2,
            "delta_MAE_cm_persistence_vs_cnn": per_mae - cnn_mae,
            "spatial_corr": {"persistence": spatial_per, "cnn": spatial_cnn},
            "stratified": compute_stratified_comparison(
                y_true,
                np.load(cnn_pred_path)[:, np.newaxis, :, :],
                y_pred,
                Y_raw,
                spatial_mask,
                test_target_idx,
            ),
        }
        compare_path = os.path.join(COMPARE_JSON_DIR, "baseline_comparison.json")
        os.makedirs(COMPARE_JSON_DIR, exist_ok=True)
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(compare, f, indent=2, ensure_ascii=False)

        st = compare["stratified"]
        hm = st["high_change_months_metrics"]
        hc = st["high_change_cells_metrics"]
        print("\n分层对比 — 高变化月份 (2024-06~09):")
        print(f"  3DCNN MAE={hm['cnn']['MAE']:.4f}  MSE={hm['cnn']['MSE']:.4f}  R²={hm['cnn']['R2']:.4f}")
        print(f"  Per  MAE={hm['persistence']['MAE']:.4f}  MSE={hm['persistence']['MSE']:.4f}  R²={hm['persistence']['R2']:.4f}")
        print(f"分层对比 — 高变化格点 (|ΔY|>{HIGH_CHANGE_THRESHOLD_CM}cm, n={hc['n_samples']}):")
        print(f"  3DCNN MAE={hc['cnn']['MAE']:.4f}  MSE={hc['cnn']['MSE']:.4f}")
        print(f"  Per  MAE={hc['persistence']['MAE']:.4f}  MSE={hc['persistence']['MSE']:.4f}")


if __name__ == "__main__":
    main()
