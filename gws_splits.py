"""按日历年份划分 train/val/test（rolling-origin 等补充实验）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from gws_io import DATA_DIR, TIME_STEPS, load_coords


@dataclass(frozen=True)
class CalendarSplit:
    name: str
    train_target_years: tuple[int, ...]
    val_year: int
    test_year: int
    train_label: str
    val_label: str
    test_label: str


RO_1 = CalendarSplit(
    name="RO-1",
    train_target_years=(2019, 2020, 2021),
    val_year=2022,
    test_year=2023,
    train_label="2018–2021",
    val_label="2022",
    test_label="2023",
)

RO_2 = CalendarSplit(
    name="RO-2",
    train_target_years=(2019, 2020, 2021, 2022),
    val_year=2023,
    test_year=2024,
    train_label="2018–2022",
    val_label="2023",
    test_label="2024",
)

DEFAULT_RO_SPLITS = (RO_1, RO_2)


def _target_year(time_coords: np.ndarray, target_month_idx: int) -> int:
    if len(time_coords):
        return int(str(time_coords[target_month_idx])[:4])
    return 2018 + target_month_idx // 12


def seq_indices_for_target_years(
    n_raw_months: int,
    time_coords: np.ndarray,
    years: set[int],
) -> np.ndarray:
    """返回预测目标月落在给定年份内的序列索引。"""
    indices = []
    for seq_idx in range(n_raw_months - TIME_STEPS):
        target_idx = seq_idx + TIME_STEPS
        if _target_year(time_coords, target_idx) in years:
            indices.append(seq_idx)
    return np.array(indices, dtype=int)


def load_and_preprocess_calendar_split(
    split: CalendarSplit,
    save_stats: bool = False,
    results_dir: str | None = None,
) -> tuple[Any, ...]:
    """
    按日历年份划分数据：
      - 训练：目标月年份 ∈ train_target_years（2018 起算时最早目标为 2019-01）
      - 验证：目标月年份 = val_year
      - 测试：目标月年份 = test_year
    GWS 标准化仅用训练期原始月份估计 mean/std。
    """
    from gws_common import _save_norm_stats

    X_drivers = np.load(os.path.join(DATA_DIR, "X_drivers.npy"))
    Y_raw = np.load(os.path.join(DATA_DIR, "Y_shallow_gws.npy"))
    X_drivers = np.transpose(X_drivers, (0, 3, 1, 2))

    _, _, time_coords = load_coords()
    spatial_mask = np.any(~np.isnan(Y_raw) & (np.abs(Y_raw) > 1e-6), axis=0)
    n_raw = len(Y_raw)
    n_seq = n_raw - TIME_STEPS

    train_seq = seq_indices_for_target_years(
        n_raw, time_coords, set(split.train_target_years),
    )
    val_seq = seq_indices_for_target_years(n_raw, time_coords, {split.val_year})
    test_seq = seq_indices_for_target_years(n_raw, time_coords, {split.test_year})

    if len(train_seq) == 0 or len(val_seq) == 0 or len(test_seq) == 0:
        raise ValueError(
            f"{split.name} 划分无效: train={len(train_seq)}, val={len(val_seq)}, test={len(test_seq)}"
        )

    train_month_end = int(train_seq.max() + TIME_STEPS)
    Y_train_slice = Y_raw[:train_month_end]
    Y_valid_data = Y_train_slice[:, spatial_mask]
    Y_mean = float(np.nanmean(Y_valid_data))
    Y_std = float(np.nanstd(Y_valid_data) + 1e-8)

    Y_norm = (Y_raw - Y_mean) / Y_std
    gws_history_norm = Y_norm.copy()
    X_norm = np.concatenate([X_drivers, np.expand_dims(gws_history_norm, axis=1)], axis=1)
    X_norm[:, :, ~spatial_mask] = 0.0
    Y_norm[:, ~spatial_mask] = 0.0
    X_norm = np.nan_to_num(X_norm, nan=0.0)
    Y_norm = np.nan_to_num(Y_norm, nan=0.0)

    if save_stats and results_dir:
        _save_norm_stats(Y_mean, Y_std, train_month_end, results_dir)

    X_seq, Y_seq = [], []
    for i in range(n_seq):
        X_seq.append(X_norm[i: i + TIME_STEPS])
        Y_seq.append(Y_norm[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    Y_seq = np.expand_dims(np.array(Y_seq), axis=1)

    print(
        f"📅 {split.name}: train={len(train_seq)} val={len(val_seq)} test={len(test_seq)} "
        f"| norm months 0..{train_month_end - 1}"
    )

    return (
        X_seq[train_seq], Y_seq[train_seq],
        X_seq[val_seq], Y_seq[val_seq],
        X_seq[test_seq], Y_seq[test_seq],
        Y_mean, Y_std, spatial_mask,
    )


def persistence_for_calendar_test(
    split: CalendarSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """日历测试划分上的 Persistence 预测（物理量 cm）。"""
    Y_raw = np.load(os.path.join(DATA_DIR, "Y_shallow_gws.npy"))
    _, _, time_coords = load_coords()
    spatial_mask = np.any(~np.isnan(Y_raw) & (np.abs(Y_raw) > 1e-6), axis=0)
    n_raw = len(Y_raw)

    test_seq = seq_indices_for_target_years(n_raw, time_coords, {split.test_year})
    target_idx = test_seq + TIME_STEPS

    y_true, y_pred = [], []
    for t_idx in target_idx:
        prev_idx = t_idx - 1
        true_t = Y_raw[t_idx].copy()
        pred_t = Y_raw[prev_idx].copy()
        true_t[~spatial_mask] = 0.0
        pred_t[~spatial_mask] = 0.0
        y_true.append(true_t)
        y_pred.append(pred_t)

    return (
        np.stack(y_true)[:, np.newaxis, :, :],
        np.stack(y_pred)[:, np.newaxis, :, :],
        spatial_mask,
    )
