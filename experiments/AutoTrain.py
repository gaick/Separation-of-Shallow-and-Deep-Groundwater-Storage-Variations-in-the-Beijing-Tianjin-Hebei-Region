"""自动调参训练浅层 GWS 3DCNN，保存最佳权重、泛化指标与学习曲线。"""

import bootstrap  # noqa: F401

import copy
import os
import warnings

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset

from gws_common import (
    EPOCHS_PER_TRIAL,
    FINAL_TRAIN_EPOCHS,
    MODEL_PATH,
    N_TRIALS,
    PATIENCE,
    RESULTS_DIR,
    WARMUP_EPOCHS,
    WarmupLR,
    evaluate_loader,
    get_device,
    load_and_preprocess_data,
    masked_mse_loss,
    metrics_to_str,
    save_best_params,
    save_generalization_metrics,
    save_learning_curves,
    save_prediction_results,
    set_seed,
)
from gws_model_variants import Simple_ST_Net

warnings.filterwarnings("ignore")
set_seed(42)


def objective(trial, X_train, Y_train, X_val, Y_val, Y_mean, Y_std, device):
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.25)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [8, 16, 24])

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = Simple_ST_Net(in_channels=6, dropout_rate=dropout_rate, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_warmup = WarmupLR(optimizer, WARMUP_EPOCHS, lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PER_TRIAL - WARMUP_EPOCHS,
    )

    best_val_r2 = -float("inf")
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = masked_mse_loss(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()

        if epoch < WARMUP_EPOCHS:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()

        val_metrics = evaluate_loader(model, val_loader, Y_mean, Y_std, device)
        best_val_r2 = max(best_val_r2, val_metrics["R2"])
        trial.report(best_val_r2, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_r2


def main():
    device = get_device()
    print(f"🚀 使用设备: {device}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_mean, Y_std, spatial_mask = load_and_preprocess_data()

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_warmup_steps=20)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda t: objective(t, X_train, Y_train, X_val, Y_val, Y_mean, Y_std, device),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\n🏆 浅层地下水最佳参数组合: {study.best_params}")
    bp = study.best_params
    save_best_params(bp)

    model = Simple_ST_Net(
        in_channels=6,
        dropout_rate=bp["dropout_rate"],
        hidden_channels=bp["hidden_channels"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    scheduler_warmup = WarmupLR(optimizer, WARMUP_EPOCHS, bp["lr"])
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=10, min_lr=1e-7,
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=bp["batch_size"], shuffle=True, num_workers=0,
    )
    train_eval_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=bp["batch_size"], shuffle=False, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
        batch_size=bp["batch_size"], shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test)),
        batch_size=bp["batch_size"], shuffle=False, num_workers=0,
    )

    best_val_r2 = -float("inf")
    best_model_weights = None
    early_stop_count = 0
    learning_history: list[dict] = []

    print("\n🔥 开始浅层地下水终极训练...")
    print("   （验证集指标用于选模；训练/验证/测试三分集对比用于判断过拟合）\n")

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

        train_metrics = evaluate_loader(model, train_eval_loader, Y_mean, Y_std, device)
        val_metrics = evaluate_loader(model, val_loader, Y_mean, Y_std, device)
        val_r2 = val_metrics["R2"]

        learning_history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "train": train_metrics,
            "val": val_metrics,
        })

        if epoch >= WARMUP_EPOCHS:
            scheduler_plateau.step(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stop_count = 0
            print(f"Epoch {epoch:3d} | 🔥 新最佳验证集")
            print(f"           Train  {metrics_to_str(train_metrics)}")
            print(f"           Val    {metrics_to_str(val_metrics)}")
            print(
                f"           LR={optimizer.param_groups[0]['lr']:.7f} | "
                f"Train Loss(norm)={train_loss / len(train_loader):.4f}"
            )
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE:
                print(f"Epoch {epoch:3d} | ⏹ 早停触发（{early_stop_count}轮验证集无提升）")
                break

    save_learning_curves(learning_history)

    print("\n✅ 加载最佳权重，评估训练/验证/测试集...")
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    gen_metrics = {
        "train": evaluate_loader(model, train_eval_loader, Y_mean, Y_std, device),
        "val": evaluate_loader(model, val_loader, Y_mean, Y_std, device),
        "test": evaluate_loader(model, test_loader, Y_mean, Y_std, device),
    }
    save_generalization_metrics(gen_metrics)

    print("\n" + "=" * 60)
    print("📊 【泛化能力对比 — 判断过拟合】")
    print("   若 Train R² 远高于 Test R² → 可能过拟合")
    print("   若 Train ≈ Val ≈ Test → 泛化良好")
    print("-" * 60)
    for split in ("train", "val", "test"):
        print(f"  {split:5s}  {metrics_to_str(gen_metrics[split])}")
    print("=" * 60)

    test_m = gen_metrics["test"]
    t_preds, t_trues = [], []
    model.eval()
    with torch.no_grad():
        for bx, by in test_loader:
            t_preds.append(model(bx.to(device)).cpu().numpy())
            t_trues.append(by.numpy())
    t_preds_real = (np.concatenate(t_preds) * Y_std) + Y_mean
    t_trues_real = (np.concatenate(t_trues) * Y_std) + Y_mean

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(best_model_weights, MODEL_PATH)
    print(f"\n✅ 模型权重: {MODEL_PATH}")
    print(f"✅ 最佳参数: {os.path.join(RESULTS_DIR, 'best_params.json')}")

    save_prediction_results(
        t_preds_real,
        t_trues_real,
        spatial_mask,
        metrics={
            **{f"test_{k}": v for k, v in test_m.items()},
            "train_R2": gen_metrics["train"]["R2"],
            "val_R2": gen_metrics["val"]["R2"],
        },
        source="AutoTrain",
    )
    print("💡 运行 python plot_results.py 查看结果图与过拟合诊断图")


if __name__ == "__main__":
    main()
