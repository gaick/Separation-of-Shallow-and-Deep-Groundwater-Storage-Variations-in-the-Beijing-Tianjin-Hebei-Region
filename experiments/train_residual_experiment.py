"""
训练改进版 3DCNN：末时刻时间聚合 + Persistence 残差学习。

Ŷ(t) = Y(t-1) + Δ，其中 Δ 由 3 层 Conv3D Encoder + 2D Decoder 预测；
时间维取末帧（对应 t-1），不再对 12 个月做 mean。

用法:
  python train_residual_experiment.py
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import json
import os

from gws_common import RESULTS_BASELINE, get_device, load_and_preprocess_data, set_seed
from gws_io import load_best_params_from, save_best_params
from gws_model_variants import Simple_ST_Net_Plus1Enc, count_parameters
from gws_paths import COMPARE_JSON_DIR, PLUS1ENC_DIR, RESULTS_PROPOSED, ensure_parent, json_file
from gws_train import train_model

RESULTS_DIR = RESULTS_PROPOSED
COMPARE_JSON = os.path.join(COMPARE_JSON_DIR, "residual_vs_persistence.json")


def _load_persistence_metrics() -> dict:
    path = os.path.join(COMPARE_JSON_DIR, "baseline_comparison.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["persistence"]
    return {}


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"🚀 设备: {device}")

    if os.path.exists(json_file(PLUS1ENC_DIR, "best_params.json")):
        bp = load_best_params_from(PLUS1ENC_DIR)
    elif os.path.exists(json_file(RESULTS_BASELINE, "best_params.json")):
        bp = load_best_params_from(RESULTS_BASELINE)
    else:
        raise FileNotFoundError("未找到 best_params.json")

    data = load_and_preprocess_data(save_stats=True, results_dir=RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_best_params(bp, RESULTS_DIR)

    model_cls = Simple_ST_Net_Plus1Enc
    probe = model_cls(
        dropout_rate=bp["dropout_rate"],
        hidden_channels=bp["hidden_channels"],
    )
    print(f"📐 模型: {model_cls.label} | 参数量: {count_parameters(probe):,}")
    print("   改进: 时间维末帧聚合 + Ŷ(t)=Y(t-1)+Δ（Persistence 残差）")

    result = train_model(
        model_cls,
        RESULTS_DIR,
        bp,
        data,
        device,
        model_cls.name,
        model_cls.label,
    )

    test = result["generalization"]["test"]
    pers = _load_persistence_metrics()
    delta_r2 = test["R2"] - pers.get("R2", 0) if pers else None
    delta_mae = pers.get("MAE", 0) - test["MAE"] if pers else None

    summary = {
        "model": model_cls.name,
        "label": model_cls.label,
        "improvements": [
            "temporal_pool=last (末时刻，非 mean)",
            "residual: Y_hat = Y(t-1) + delta",
        ],
        "test": test,
        "persistence_baseline": pers,
        "delta_R2_vs_persistence": delta_r2,
        "delta_MAE_cm_persistence_minus_model": delta_mae,
        "beats_persistence_R2": delta_r2 > 0 if delta_r2 is not None else None,
        "beats_persistence_MAE": delta_mae > 0 if delta_mae is not None else None,
    }
    os.makedirs(os.path.dirname(COMPARE_JSON), exist_ok=True)
    ensure_parent(COMPARE_JSON)
    with open(COMPARE_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("📊 改进版 vs Persistence（测试集）")
    print("-" * 60)
    print(f"  改进版  R²={test['R2']:.4f}  MAE={test['MAE']:.4f} cm")
    if pers:
        print(f"  Pers.   R²={pers['R2']:.4f}  MAE={pers['MAE']:.4f} cm")
        print(f"  ΔR²={delta_r2:+.4f}  ΔMAE(Pers-Model)={delta_mae:+.4f} cm")
        if delta_r2 and delta_r2 > 0:
            print("  ✅ 测试集 R² 已超过 Persistence")
        elif delta_mae and delta_mae > 0:
            print("  ✅ 测试集 MAE 已低于 Persistence")
        else:
            print("  ⚠️  全测试集仍未超过 Persistence，见高变化子集分析")
    print("=" * 60)
    print(f"✅ 结果: {RESULTS_DIR}")
    print(f"✅ 对比: {COMPARE_JSON}")


if __name__ == "__main__":
    main()
