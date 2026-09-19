"""固定超参训练流程（基线 / 对比实验共用）。"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gws_common import (
    FINAL_TRAIN_EPOCHS,
    PATIENCE,
    WARMUP_EPOCHS,
    WarmupLR,
    evaluate_loader,
    masked_mse_loss,
    metrics_to_str,
)
from gws_io import save_prediction_results
from gws_model_variants import count_parameters
from gws_paths import ensure_parent, json_file


def _paths(results_dir: str) -> dict[str, str]:
    return {
        "model": os.path.join(results_dir, "shallow_best_model.pth"),
        "norm": json_file(results_dir, "norm_stats.json"),
        "curves": json_file(results_dir, "learning_curves.json"),
        "generalization": json_file(results_dir, "generalization_metrics.json"),
        "config": json_file(results_dir, "model_config.json"),
    }


def train_model(
    model_class: Type[nn.Module],
    results_dir: str,
    best_params: dict[str, Any],
    data_pack: tuple,
    device: torch.device,
    model_name: str,
    model_label: str,
    model_kwargs: dict[str, Any] | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """用固定超参训练一个变体，结果写入 results_dir。"""
    X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_mean, Y_std, spatial_mask = data_pack
    os.makedirs(results_dir, exist_ok=True)
    paths = _paths(results_dir)

    extra = model_kwargs or {}
    # X: (N, T, C, H, W) — C = 驱动通道数 + GWS 历史
    in_channels = int(extra.pop("in_channels", X_train.shape[2]))
    model = model_class(
        in_channels=in_channels,
        dropout_rate=best_params["dropout_rate"],
        hidden_channels=best_params["hidden_channels"],
        **extra,
    ).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    scheduler_warmup = WarmupLR(optimizer, WARMUP_EPOCHS, best_params["lr"])
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=10, min_lr=1e-7,
    )

    bs = best_params["batch_size"]
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=bs, shuffle=True, num_workers=0,
    )
    train_eval_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=bs, shuffle=False, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
        batch_size=bs, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test)),
        batch_size=bs, shuffle=False, num_workers=0,
    )

    best_val_r2 = -float("inf")
    best_weights = None
    early_stop = 0
    history: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"▶ 训练模型: {model_label}")
    print(f"  输出目录: {results_dir}")
    print(f"  参数量: {n_params:,}")
    print(f"{'=' * 60}\n")

    for epoch in range(FINAL_TRAIN_EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = masked_mse_loss(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()
            train_loss += loss.item()

        if epoch < WARMUP_EPOCHS:
            scheduler_warmup.step()

        train_m = evaluate_loader(model, train_eval_loader, Y_mean, Y_std, device, spatial_mask)
        val_m = evaluate_loader(model, val_loader, Y_mean, Y_std, device, spatial_mask)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "train": train_m,
            "val": val_m,
        })

        if epoch >= WARMUP_EPOCHS:
            scheduler_plateau.step(val_m["R2"])

        if val_m["R2"] > best_val_r2:
            best_val_r2 = val_m["R2"]
            best_weights = copy.deepcopy(model.state_dict())
            early_stop = 0
            print(f"Epoch {epoch:3d} | 🔥 新最佳验证集")
            print(f"           Train  {metrics_to_str(train_m)}")
            print(f"           Val    {metrics_to_str(val_m)}")
        else:
            early_stop += 1
            if early_stop >= PATIENCE:
                print(f"Epoch {epoch:3d} | ⏹ 早停（{early_stop}轮无提升）")
                break

    model.load_state_dict(best_weights)

    gen = {
        "train": evaluate_loader(model, train_eval_loader, Y_mean, Y_std, device, spatial_mask),
        "val": evaluate_loader(model, val_loader, Y_mean, Y_std, device, spatial_mask),
        "test": evaluate_loader(model, test_loader, Y_mean, Y_std, device, spatial_mask),
    }

    torch.save(best_weights, paths["model"])
    for key in ("curves", "generalization", "config"):
        ensure_parent(paths[key])
    with open(paths["curves"], "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(paths["generalization"], "w", encoding="utf-8") as f:
        json.dump(gen, f, indent=2, ensure_ascii=False)
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "model_label": model_label,
            "params": n_params,
            "best_params": best_params,
            "model_specific": extra,
        }, f, indent=2, ensure_ascii=False)

    t_preds, t_trues = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in test_loader:
            t_preds.append(model(bx.to(device)).cpu().numpy())
            t_trues.append(by.numpy())
    t_preds_real = (np.concatenate(t_preds) * Y_std) + Y_mean
    t_trues_real = (np.concatenate(t_trues) * Y_std) + Y_mean

    test_m = gen["test"]
    save_prediction_results(
        t_preds_real, t_trues_real, spatial_mask,
        metrics={
            "model": model_name,
            **{f"test_{k}": v for k, v in test_m.items()},
            "train_R2": gen["train"]["R2"],
            "val_R2": gen["val"]["R2"],
        },
        source="compare_experiment",
        results_dir=results_dir,
        data_dir=data_dir,
    )

    print(f"\n📊 {model_label} — 三分集结果:")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}  {metrics_to_str(gen[split])}")

    return {"generalization": gen, "params": n_params, "results_dir": results_dir}
