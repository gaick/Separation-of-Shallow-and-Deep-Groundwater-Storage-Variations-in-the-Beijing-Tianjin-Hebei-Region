"""
3DCNN Encoder 深度消融实验（5 个变体，固定超参）。

变体（均含末时刻聚合 + Persistence 残差 Ŷ=Y(t-1)+Δ）：
  enc1        — M1：1 层 Conv3D
  shallow_gws — M2：2 层
  enc3_flat   — M3：3 层恒宽
  plus1enc    — M4：3 层加宽（本文）
  enc4_flat   — M5：4 层封顶

用法:
  python compare_experiment.py
  python compare_experiment.py --skip-trained   # 跳过已有 generalization_metrics 的模型
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import argparse
import json
import os
import shutil

from gws_common import RESULTS_BASELINE, RESULTS_COMPARE, get_device, load_and_preprocess_data, set_seed
from gws_io import load_best_params, load_best_params_from, save_best_params
from gws_model_variants import ABLATION_VARIANTS, results_dir_for
from gws_paths import ensure_parent, json_file
from gws_train import train_model


def _copy_norm_stats(src: str, dst: str) -> None:
    src_f = json_file(src, "norm_stats.json")
    dst_f = json_file(dst, "norm_stats.json")
    if not os.path.exists(src_f):
        return
    if os.path.abspath(src_f) == os.path.abspath(dst_f):
        return
    os.makedirs(dst, exist_ok=True)
    ensure_parent(dst_f)
    shutil.copy2(src_f, dst_f)


def _is_trained(results_dir: str) -> bool:
    return os.path.exists(json_file(results_dir, "generalization_metrics.json"))


def run_compare_experiment(
    skip_baseline: bool = False,
    skip_trained: bool = False,
) -> str:
    set_seed(42)
    device = get_device()
    print(f"🚀 设备: {device}")

    baseline_dir = RESULTS_BASELINE
    if os.path.exists(json_file(baseline_dir, "best_params.json")):
        bp = load_best_params_from(baseline_dir)
    else:
        bp = load_best_params()
    print(f"✅ 固定超参: {bp}")

    data = load_and_preprocess_data(save_stats=True, results_dir=baseline_dir)
    save_best_params(bp, baseline_dir)

    summary = []
    norm_src = baseline_dir

    for model_cls in ABLATION_VARIANTS:
        out_dir = results_dir_for(model_cls)
        is_baseline = model_cls.name == "baseline_2enc_res"

        if skip_baseline and is_baseline:
            print(f"\n⏭ 跳过基线: {model_cls.label}（--skip-baseline）")
            if _is_trained(out_dir):
                with open(json_file(out_dir, "model_config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                with open(json_file(out_dir, "generalization_metrics.json"), encoding="utf-8") as f:
                    gen = json.load(f)
                summary.append({
                    "model": model_cls.name,
                    "label": model_cls.label,
                    "params": cfg.get("params", 0),
                    "results_dir": out_dir,
                    **{f"test_{k}": v for k, v in gen["test"].items()},
                })
            continue

        if skip_trained and _is_trained(out_dir):
            print(f"\n⏭ 已有结果，跳过训练: {model_cls.label} → {out_dir}")
            with open(json_file(out_dir, "model_config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            with open(json_file(out_dir, "generalization_metrics.json"), encoding="utf-8") as f:
                gen = json.load(f)
            summary.append({
                "model": model_cls.name,
                "label": model_cls.label,
                "params": cfg.get("params", 0),
                "results_dir": out_dir,
                **{f"test_{k}": v for k, v in gen["test"].items()},
            })
            continue

        os.makedirs(out_dir, exist_ok=True)
        _copy_norm_stats(norm_src, out_dir)
        save_best_params(bp, out_dir)

        result = train_model(
            model_cls, out_dir, bp, data, device,
            model_cls.name, model_cls.label,
        )
        summary.append({
            "model": model_cls.name,
            "label": model_cls.label,
            "params": result["params"],
            "results_dir": out_dir,
            **{f"test_{k}": v for k, v in result["generalization"]["test"].items()},
        })

    os.makedirs(RESULTS_COMPARE, exist_ok=True)
    compare_path = os.path.join(RESULTS_COMPARE, "experiment_summary.json")
    ensure_parent(compare_path)
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Encoder 深度消融（统一：末时刻聚合+Persistence残差）；固定超参；对比 Conv3D 层数/通道策略",
            "best_params": bp,
            "models": summary,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 消融实验完成: {compare_path}")
    return compare_path


def main(skip_baseline: bool = False, skip_trained: bool = False) -> None:
    run_compare_experiment(skip_baseline=skip_baseline, skip_trained=skip_trained)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true", help="不重新训练 2层基线")
    parser.add_argument("--skip-trained", action="store_true", help="跳过已有结果的变体")
    args = parser.parse_args()
    main(skip_baseline=args.skip_baseline, skip_trained=args.skip_trained)
