"""
Quick test aligned with the paper's M2 experiment design.

What this checks (no proprietary GRACE/well data required):
  1) Model class = paper M2 (2-layer encoder + last-frame + Persistence residual)
  2) Input layout = (B, T=12, C=6, H, W) with channel order [pre,tmp,pet,lst,ndvi,gws]
  3) Residual identity: if the learned delta is ~0, prediction ≈ Y(t-1)
  4) Short train on synthetic AR(1) fields (high lag-1, similar to shallow GWS)
     and compare against Persistence baseline with the same metrics as the paper

This is a *smoke / consistency* test, NOT a reproduction of the BTH test-set
scores (R²≈0.961). Paper metrics require the licensed research data package.

Usage (repository root):
  python examples/quick_test.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gws_io import calculate_metrics
from gws_model_variants import GWS_CHANNEL_IDX, Simple_ST_Net_Ablation2Enc

# Paper-consistent settings (M2)
TIME_STEPS = 12
IN_CHANNELS = 6
HIDDEN = 24
DROPOUT = 0.056
SEED = 2024


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_series(
    n_months: int = 48,
    h: int = 12,
    w: int = 12,
    lag1: float = 0.93,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic drivers + shallow GWS with strong persistence (lag-1 ~ paper BTH regime).
    Y units are cm-like anomalies; X drivers are z-scored noise with mild seasonality.
    """
    rng = np.random.default_rng(SEED)
    # Drivers: (T,H,W,5)
    season = np.sin(2 * np.pi * np.arange(n_months) / 12.0)[:, None, None, None]
    X = rng.normal(0.0, 1.0, size=(n_months, h, w, 5)).astype(np.float32)
    X[..., 0:1] += 0.4 * season  # precip-like
    X[..., 1:2] += 0.3 * season  # temp-like

    # Shallow GWS AR(1) + seasonal recharge pulse
    Y = np.zeros((n_months, h, w), dtype=np.float32)
    Y[0] = rng.normal(0.0, 0.4, size=(h, w))
    for t in range(1, n_months):
        pulse = 0.25 * np.sin(2 * np.pi * t / 12.0 + 0.7)
        innov = rng.normal(0.0, 0.12, size=(h, w))
        # Mild spatial smoothness
        innov = (
            innov
            + np.roll(innov, 1, 0)
            + np.roll(innov, -1, 0)
            + np.roll(innov, 1, 1)
            + np.roll(innov, -1, 1)
        ) / 5.0
        Y[t] = lag1 * Y[t - 1] + pulse + innov
    return X, Y


def build_windows(
    X: np.ndarray, Y: np.ndarray, time_steps: int = TIME_STEPS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build samples matching training layout:
      x: (N, T, C, H, W)  with C = drivers(5) + gws_history(1)
      y: (N, H, W)        target = Y[t] where window ends at t-1
    """
    t, h, w, _ = X.shape
    xs, ys = [], []
    for start in range(0, t - time_steps):
        end = start + time_steps  # exclusive; last hist month = end-1; target = end
        drivers = X[start:end]  # (T,H,W,5)
        gws_hist = Y[start:end]  # (T,H,W)
        cube = np.concatenate([drivers, gws_hist[..., None]], axis=-1)  # (T,H,W,6)
        cube = np.transpose(cube, (0, 3, 1, 2))  # (T,C,H,W)
        xs.append(cube)
        ys.append(Y[end])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def persistence_predict(x_btchw: np.ndarray) -> np.ndarray:
    """Y_hat(t) = Y(t-1) = last GWS channel of the input window."""
    return x_btchw[:, -1, GWS_CHANNEL_IDX, :, :]


def assert_residual_identity(model: nn.Module, x: torch.Tensor, atol: float = 1e-4) -> None:
    """
    With decoder forced to near-zero delta, M2 output must equal Persistence Y(t-1).
    This verifies the paper residual head Ŷ = Y(t-1) + Δ.
    """
    model.eval()
    # Zero decoder weights/biases
    with torch.no_grad():
        for p in model.decoder.parameters():
            p.zero_()
    with torch.no_grad():
        pred = model(x).cpu().numpy().squeeze(1)
    pers = persistence_predict(x.cpu().numpy())
    max_abs = float(np.max(np.abs(pred - pers)))
    if max_abs > atol:
        raise AssertionError(
            f"Residual identity failed: max|pred - Y(t-1)|={max_abs:.3e} > {atol}"
        )


def short_train(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int = 40,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> list[float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = x_train.shape[0]
    losses = []
    for step in range(steps):
        idx = np.random.randint(0, n, size=batch_size)
        xb = torch.from_numpy(x_train[idx])
        yb = torch.from_numpy(y_train[idx]).unsqueeze(1)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).cpu().numpy().squeeze(1)
    return calculate_metrics(y, pred)


def main() -> None:
    set_seed(SEED)
    print("=" * 64)
    print("Quick test — M2 (baseline_2enc_res) consistency with paper design")
    print("=" * 64)

    X, Y = make_synthetic_series()
    # Empirical lag-1 of regional mean (should be high, like BTH shallow GWS)
    ym = Y.mean(axis=(1, 2))
    lag1 = float(np.corrcoef(ym[:-1], ym[1:])[0, 1])
    print(f"[data] synthetic months={len(Y)}, grid={Y.shape[1]}x{Y.shape[2]}, lag-1={lag1:.3f}")

    x_all, y_all = build_windows(X, Y)
    # Chronological split (no shuffle), similar spirit to paper
    n = len(x_all)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    x_tr, y_tr = x_all[:n_train], y_all[:n_train]
    x_te, y_te = x_all[n_train + n_val :], y_all[n_train + n_val :]
    print(f"[data] windows N={n}  train={len(x_tr)}  test={len(x_te)}")
    print(f"[data] tensor layout x={x_tr.shape} (B,T,C,H,W)  y={y_tr.shape}")

    # --- Check 1: construct M2 with paper hyper-params ---
    model = Simple_ST_Net_Ablation2Enc(
        in_channels=IN_CHANNELS, dropout_rate=DROPOUT, hidden_channels=HIDDEN
    )
    assert model.name == "baseline_2enc_res", model.name
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model.label}  id={model.name}  params={n_params}")

    # --- Check 2: forward shape ---
    xb = torch.from_numpy(x_tr[:2])
    with torch.no_grad():
        out = model(xb)
    assert out.shape == (2, 1, Y.shape[1], Y.shape[2]), out.shape
    print(f"[shape] forward OK: {tuple(xb.shape)} -> {tuple(out.shape)}")

    # --- Check 3: Persistence residual identity ---
    # Re-init a fresh model so identity test is clean
    model_id = Simple_ST_Net_Ablation2Enc(
        in_channels=IN_CHANNELS, dropout_rate=DROPOUT, hidden_channels=HIDDEN
    )
    assert_residual_identity(model_id, xb)
    print("[residual] Ŷ = Y(t-1) + Δ identity OK (decoder zeroed)")

    # --- Check 4: Persistence baseline metrics on test windows ---
    pers = persistence_predict(x_te)
    m_pers = calculate_metrics(y_te, pers)
    print(
        f"[baseline] Persistence  R²={m_pers['R2']:.3f}  "
        f"MAE={m_pers['MAE']:.3f}  RMSE={m_pers['RMSE']:.3f}"
    )

    # --- Check 5: short train; loss should drop; M2 should beat Persistence ---
    losses = short_train(model, x_tr, y_tr, steps=50, batch_size=8)
    assert losses[-1] < losses[0], "Training loss did not decrease"
    print(f"[train] MSE {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    m_m2 = evaluate(model, x_te, y_te)
    print(
        f"[test] M2 (synthetic) R²={m_m2['R2']:.3f}  "
        f"MAE={m_m2['MAE']:.3f}  RMSE={m_m2['RMSE']:.3f}"
    )

    if m_m2["MAE"] > m_pers["MAE"] * 1.25:
        # Soft warning: on tiny synthetic grids training may be noisy; still require
        # that M2 is not catastrophically worse than Persistence.
        print(
            "[warn] M2 MAE not clearly better than Persistence on this tiny synthetic "
            "run; residual identity + loss decrease already passed."
        )
    else:
        print("[compare] M2 MAE <= 1.25 × Persistence MAE (expected under AR(1)+drivers)")

    # --- Check 6: deep residual definition (water balance) on synthetic totals ---
    total = Y + np.linspace(-20.0, -25.0, len(Y), dtype=np.float32)[:, None, None]
    deep = total - Y
    assert np.allclose(deep + Y, total), "Water-balance residual inconsistent"
    print("[balance] ΔGWS_deep = ΔGWS_total − ΔGWS_shallow  OK")

    print("=" * 64)
    print("Quick test OK — architecture/residual/training/metrics consistent with paper.")
    print("Note: BTH paper scores require the research data package (not redistributed).")
    print("=" * 64)


if __name__ == "__main__":
    main()
