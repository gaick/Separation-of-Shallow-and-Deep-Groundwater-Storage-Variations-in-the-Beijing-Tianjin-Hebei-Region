"""
横向对比实验的超参空间定义。

策略：
  - 共享超参（lr / batch_size / dropout / weight_decay / hidden_channels）
    全部模型使用相同值，默认取自已有 3DCNN Optuna 结果。
  - 模型专属超参
    各模型独立 Optuna（TPE 贝叶斯优化），在固定共享超参下搜索最优。
"""

from __future__ import annotations

from typing import Any, Callable

import optuna

from gws_st_models import ST_MODEL_REGISTRY

# 共享超参键（训练循环与 build_model 均使用）
SHARED_PARAM_KEYS = ("lr", "batch_size", "dropout_rate", "weight_decay", "hidden_channels")

# 各模型专属超参键
MODEL_SPECIFIC_KEYS: dict[str, tuple[str, ...]] = {
    "stnet_3dcnn_1enc": (),
    "stnet_3dcnn": (),
    "stnet_3dcnn_proposed": (),
    "convlstm": ("num_layers", "kernel_size"),
    "convgru": ("num_layers", "kernel_size"),
    "tcn": ("num_layers", "kernel_size"),
    "stconv_21d": ("num_blocks",),
    "unet3d": ("depth",),
    "convlstm_unet": ("convlstm_layers", "unet_depth", "kernel_size"),
}

SKIP_SPECIFIC_TUNING = frozenset({"stnet_3dcnn_1enc", "stnet_3dcnn", "stnet_3dcnn_proposed"})


def default_shared_params() -> dict[str, Any]:
    """无已有结果时的共享超参默认值。"""
    return {
        "lr": 2.04e-4,
        "batch_size": 8,
        "dropout_rate": 0.095,
        "weight_decay": 1.21e-6,
        "hidden_channels": 24,
    }


def load_shared_params() -> dict[str, Any]:
    """优先读取 shallow_gws/best_params.json，否则用默认值。"""
    import os
    from gws_io import load_best_params_from
    from gws_paths import RESULTS_BASELINE, json_file

    baseline = RESULTS_BASELINE
    if os.path.exists(json_file(baseline, "best_params.json")):
        bp = load_best_params_from(baseline)
    else:
        try:
            from gws_io import load_best_params
            bp = load_best_params()
        except FileNotFoundError:
            bp = default_shared_params()
    return {k: bp[k] for k in SHARED_PARAM_KEYS if k in bp}


def suggest_model_specific(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """在 Optuna trial 中采样模型专属超参。"""
    if model_name == "convlstm":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        }
    if model_name == "convgru":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        }
    if model_name == "tcn":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        }
    if model_name == "stconv_21d":
        return {
            "num_blocks": trial.suggest_int("num_blocks", 1, 2),
        }
    if model_name == "unet3d":
        return {
            "depth": trial.suggest_int("depth", 1, 2),
        }
    if model_name == "convlstm_unet":
        return {
            "convlstm_layers": trial.suggest_int("convlstm_layers", 1, 2),
            "unet_depth": trial.suggest_int("unet_depth", 1, 2),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        }
    return {}


def get_param_suggest_fn(model_name: str) -> Callable[[optuna.Trial], dict[str, Any]]:
    return lambda trial: suggest_model_specific(trial, model_name)


def merge_params(shared: dict[str, Any], specific: dict[str, Any]) -> dict[str, Any]:
    """合并为 train_model 所需的 best_params 字典。"""
    merged = dict(shared)
    merged.update(specific)
    return merged


def validate_model_name(name: str) -> type:
    if name not in ST_MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}，可选: {list(ST_MODEL_REGISTRY)}")
    return ST_MODEL_REGISTRY[name]
