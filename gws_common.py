"""浅层 GWS 时空预测：数据预处理、评估与训练工具。"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gws_imputation import IMPUTATION_METHODS, apply_imputation, observed_mask_from_y
from gws_io import (
    DATA_DIR,
    RESULTS_DIR,
    TIME_STEPS,
    calculate_metrics,
    load_best_params,
    load_coords,
    load_prediction_results,
    print_data_summary,
    regional_mean,
    save_best_params,
    save_prediction_results,
    squeeze_grid,
)
from gws_model_variants import (  # noqa: F401 — re-export
    IN_CHANNELS,
    MODEL_REGISTRY,
    Simple_ST_Net,
    Simple_ST_Net_Plus1Enc,
    count_parameters,
)
from gws_paths import (
    RESULTS_BASELINE,
    RESULTS_COMPARE,
    RESULTS_DIR,
    RESULTS_PROPOSED,
    ensure_parent,
    json_file,
)

N_TRIALS = 30
EPOCHS_PER_TRIAL = 60
FINAL_TRAIN_EPOCHS = 300
PATIENCE = 30
WARMUP_EPOCHS = 5
N_DRIVER_CHANNELS = 5
GWS_CHANNEL_IDX = 5

MODEL_PATH = os.path.join(RESULTS_DIR, "shallow_best_model.pth")
NORM_STATS_PATH = json_file(RESULTS_DIR, "norm_stats.json")
LEARNING_CURVES_PATH = json_file(RESULTS_DIR, "learning_curves.json")
GENERALIZATION_PATH = json_file(RESULTS_DIR, "generalization_metrics.json")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.mps.manual_seed(seed)
    except AttributeError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _save_norm_stats(
    Y_mean: float,
    Y_std: float,
    train_month_end: int,
    results_dir: str = RESULTS_DIR,
    extra: dict | None = None,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    stats = {
        "normalization": "single_pass",
        "note": "驱动因子(前5通道)使用npy内建Z-score；GWS历史(第6通道)与Y目标仅用训练月份做一次性Z-score",
        "train_raw_months": f"0..{train_month_end - 1}",
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "driver_channels": "pre-standardized in npy (no second scaling)",
    }
    if extra:
        stats.update(extra)
    path = json_file(results_dir, "norm_stats.json")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def _rescale_driver_channels(
    X_drivers: np.ndarray,
    spatial_mask: np.ndarray,
    norm_month_end: int,
) -> np.ndarray:
    """仅用测试前月份重算驱动因子 Z-score，避免全时段标准化带来的分布漂移。"""
    X_rescaled = X_drivers.copy()
    for ch in range(X_drivers.shape[1]):
        ref = X_drivers[:norm_month_end, ch][:, spatial_mask]
        mu = float(np.nanmean(ref))
        sd = float(np.nanstd(ref) + 1e-8)
        X_rescaled[:, ch] = (X_drivers[:, ch] - mu) / sd
    return X_rescaled


def _monthly_climatology(Y_raw: np.ndarray, spatial_mask: np.ndarray, month_end: int) -> np.ndarray:
    clim = np.zeros((12, *Y_raw.shape[1:]))
    counts = np.zeros(12)
    for m in range(month_end):
        mo = m % 12
        clim[mo] += np.where(spatial_mask, Y_raw[m], 0.0)
        counts[mo] += 1
    for mo in range(12):
        if counts[mo] > 0:
            clim[mo] /= counts[mo]
    return clim


def load_and_preprocess_data(
    save_stats: bool = True,
    results_dir: str = RESULTS_DIR,
    data_dir: str | None = None,
    y_filename: str = "Y_shallow_gws.npy",
    split_mode: str = "default",
    test_seq_count: int = 8,
    inner_val_seq_count: int = 8,
    norm_mode: str = "train_months",
    rescale_drivers_pretest: bool = False,
    target_mode: str = "absolute",
    imputation_method: str = "baseline_zero",
) -> tuple[Any, ...]:
    """
    单次标准化：
      - 驱动因子 ch0-4：默认 npy 已 Z-score；可选仅用测试前月份重标准化
      - GWS 历史 ch5 + 目标 Y：按 norm_mode 选取月份估计 mean/std，做唯一一次 Z-score
      - target_mode=monthly_anomaly：先减月气候态再标准化，预测时加回气候态
      - imputation_method：无效格点填补策略（评价仍在原始观测掩膜上）

    split_mode:
      - default: 80/10/10 序列切分
      - holdout_last_n: 最后 n 个序列作测试，其余序列末尾切 inner_val 用于早停
    """
    root = data_dir or DATA_DIR
    print("⏳ 正在读取浅层地下水数据...")
    print_data_summary(root, y_filename)

    X_drivers = np.load(os.path.join(root, "X_drivers.npy"))
    Y_raw = np.load(os.path.join(root, y_filename))
    lat, lon, _ = load_coords(root)

    X_drivers = np.transpose(X_drivers, (0, 3, 1, 2))  # (T, 5, H, W)

    observed_mask = observed_mask_from_y(Y_raw)
    print(f"🌍 有效空间区域占比: {np.sum(observed_mask) / observed_mask.size:.2f}")

    n_seq = len(Y_raw) - TIME_STEPS
    if split_mode == "holdout_last_n":
        test_count = min(test_seq_count, max(1, n_seq // 5))
        train_end = n_seq - test_count
        val_end = n_seq
        inner_val = min(inner_val_seq_count, max(1, train_end // 5))
        inner_train_end = train_end - inner_val
        print(
            f"📌 切分模式 holdout_last_n：inner_train={inner_train_end} | "
            f"inner_val={inner_val} | test={test_count}"
        )
    else:
        train_end = int(n_seq * 0.8)
        val_end = int(n_seq * 0.9)
        inner_train_end = train_end
        inner_val = val_end - train_end

    if norm_mode == "pre_test_months":
        norm_month_end = val_end + TIME_STEPS
        norm_label = f"测试前月 0~{norm_month_end - 1}"
    else:
        norm_month_end = train_end + TIME_STEPS
        norm_label = f"训练月 0~{norm_month_end - 1}"

    if rescale_drivers_pretest:
        driver_note = f"已用 {norm_label} 重标准化（填补后）"
    else:
        driver_note = "保持 npy 内建 Z-score，不再二次标准化"

    if imputation_method not in IMPUTATION_METHODS:
        raise ValueError(f"未知 imputation_method: {imputation_method}")

    imp = apply_imputation(
        imputation_method, Y_raw, X_drivers, observed_mask, lat, lon, norm_month_end,
    )
    Y_raw = imp.Y
    X_drivers = imp.X
    print(f"🧩 缺失值填补: {imp.note}")

    if rescale_drivers_pretest:
        X_drivers = _rescale_driver_channels(X_drivers, observed_mask, norm_month_end)

    monthly_clim = None
    Y_work = Y_raw.copy()
    if target_mode == "monthly_anomaly":
        monthly_clim = _monthly_climatology(Y_raw, observed_mask, norm_month_end)
        Y_work = Y_raw - monthly_clim[np.arange(len(Y_raw)) % 12]
        target_note = "月距平（去月气候态）"
    else:
        target_note = "绝对量 (cm)"

    Y_train_slice = Y_work[:norm_month_end]
    Y_valid_data = Y_train_slice[:, observed_mask]
    Y_mean = float(np.nanmean(Y_valid_data))
    Y_std = float(np.nanstd(Y_valid_data) + 1e-8)
    print(f"📊 GWS 单次标准化（{norm_label}，{target_note}）：mean={Y_mean:.4f}，std={Y_std:.4f}")
    print(f"   驱动因子 ch0-4：{driver_note}")

    Y_norm = (Y_work - Y_mean) / Y_std
    gws_history_norm = Y_norm.copy()

    X_norm = np.concatenate([X_drivers, np.expand_dims(gws_history_norm, axis=1)], axis=1)

    # 输入通道保留填补值；目标 Y 仅在原始观测格点参与损失
    Y_norm = Y_norm * observed_mask.astype(Y_norm.dtype)
    X_norm = np.nan_to_num(X_norm, nan=0.0)
    Y_norm = np.nan_to_num(Y_norm, nan=0.0)

    spatial_mask = observed_mask

    if save_stats:
        extra = {
            "split_mode": split_mode,
            "norm_mode": norm_mode,
            "target_mode": target_mode,
            "imputation_method": imputation_method,
            "imputation_note": imp.note,
            "rescale_drivers_pretest": rescale_drivers_pretest,
            "test_seq_count": test_seq_count if split_mode == "holdout_last_n" else val_end - train_end,
        }
        if monthly_clim is not None:
            extra["monthly_climatology_cm"] = monthly_clim.tolist()
        _save_norm_stats(
            Y_mean, Y_std, norm_month_end, results_dir,
            extra=extra,
        )

    X_seq, Y_seq = [], []
    for i in range(n_seq):
        X_seq.append(X_norm[i: i + TIME_STEPS])
        Y_seq.append(Y_norm[i + TIME_STEPS])

    X_seq = np.array(X_seq)
    Y_seq = np.expand_dims(np.array(Y_seq), axis=1)
    print(f"序列数据形状: X={X_seq.shape}, Y={Y_seq.shape}")

    if split_mode == "holdout_last_n":
        print(
            f"训练集:{inner_train_end} | 验证集(早停):{inner_val} | "
            f"测试集:{n_seq - train_end}"
        )
        return (
            X_seq[:inner_train_end], Y_seq[:inner_train_end],
            X_seq[inner_train_end:train_end], Y_seq[inner_train_end:train_end],
            X_seq[train_end:], Y_seq[train_end:],
            Y_mean, Y_std, spatial_mask,
        )

    print(f"训练集:{train_end} | 验证集:{val_end - train_end} | 测试集:{n_seq - val_end}")
    return (
        X_seq[:train_end], Y_seq[:train_end],
        X_seq[train_end:val_end], Y_seq[train_end:val_end],
        X_seq[val_end:], Y_seq[val_end:],
        Y_mean, Y_std, spatial_mask,
    )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    Y_mean: float,
    Y_std: float,
    device: torch.device,
    spatial_mask: np.ndarray | None = None,
) -> dict[str, float]:
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in loader:
            preds.append(model(bx.to(device)).cpu().numpy())
            trues.append(by.numpy())
    preds_real = (np.concatenate(preds) * Y_std) + Y_mean
    trues_real = (np.concatenate(trues) * Y_std) + Y_mean
    if preds_real.ndim == 4 and preds_real.shape[1] == 1:
        preds_real = preds_real[:, 0]
        trues_real = trues_real[:, 0]
    if spatial_mask is not None:
        t_flat = trues_real[:, spatial_mask].ravel()
        p_flat = preds_real[:, spatial_mask].ravel()
        valid = np.isfinite(t_flat) & np.isfinite(p_flat) & (np.abs(t_flat) > 1e-5)
        return calculate_metrics(t_flat[valid], p_flat[valid])
    return calculate_metrics(trues_real, preds_real)


def metrics_to_str(metrics: dict[str, float]) -> str:
    return (
        f"R²={metrics['R2']:.4f} | NSE={metrics['NSE']:.4f} | KGE={metrics['KGE']:.4f} | "
        f"MAE={metrics['MAE']:.4f}cm | RMSE={metrics['RMSE']:.4f}cm | "
        f"R={metrics['R']:.4f} | IOA={metrics['IOA']:.4f}"
    )


def save_generalization_metrics(metrics: dict[str, dict[str, float]], results_dir: str = RESULTS_DIR) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = json_file(results_dir, "generalization_metrics.json")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_learning_curves(history: list[dict[str, Any]], results_dir: str = RESULTS_DIR) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = json_file(results_dir, "learning_curves.json")
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = (torch.abs(targets) > 1e-5).float()
    loss = (preds - targets) ** 2
    return (loss * mask).sum() / (mask.sum() + 1e-8)


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, base_lr: float, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.base_lr
        return [lr for _ in self.optimizer.param_groups]
