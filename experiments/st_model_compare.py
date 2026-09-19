"""
时空模型横向对比实验。

流程：
  1. 固定共享超参（来自 shallow_gws Optuna 结果）
  2. 各模型专属超参贝叶斯优化（st_model_autotune.py）
  3. 用最优超参完整训练并在测试集评估

用法:
  python st_model_compare.py                    # 调参 + 训练 + 汇总
  python st_model_compare.py --skip-tuning      # 跳过调参，直接训练
  python st_model_compare.py --skip-trained       # 跳过已有结果
  python st_model_compare.py --tuning-only       # 仅调参
"""

from __future__ import annotations

import bootstrap  # noqa: F401
import argparse
import json
import os
import shutil

from gws_common import RESULTS_BASELINE, get_device, load_and_preprocess_data, set_seed
from gws_hyperparams import load_shared_params, merge_params
from gws_io import save_best_params
from gws_st_models import ST_COMPARE_VARIANTS, ST_MODEL_REGISTRY, results_dir_for
from gws_paths import MODELS_ROOT, ST_COMPARE_JSON_DIR, ensure_parent, json_file
from gws_train import train_model
from st_model_autotune import tune_all

SUMMARY_PATH = os.path.join(ST_COMPARE_JSON_DIR, "st_experiment_summary.json")
RESULTS_ROOT = MODELS_ROOT


def _copy_norm_stats(src: str, dst: str) -> None:
    src_f = json_file(src, "norm_stats.json")
    if os.path.exists(src_f):
        os.makedirs(dst, exist_ok=True)
        dst_f = json_file(dst, "norm_stats.json")
        ensure_parent(dst_f)
        shutil.copy2(src_f, dst_f)


def _is_trained(results_dir: str) -> bool:
    return os.path.exists(json_file(results_dir, "generalization_metrics.json"))


def _load_best_params(model_name: str) -> dict:
    from st_model_autotune import results_dir_for_model
    bp_path = json_file(results_dir_for_model(model_name), "best_params.json")
    if os.path.exists(bp_path):
        with open(bp_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("best_params", data)
    return merge_params(load_shared_params(), {})


def _model_specific_from_params(model_name: str, params: dict) -> dict:
    from gws_hyperparams import MODEL_SPECIFIC_KEYS
    keys = MODEL_SPECIFIC_KEYS.get(model_name, ())
    return {k: params[k] for k in keys if k in params}


def run_st_compare(
    skip_tuning: bool = False,
    skip_trained: bool = False,
    tuning_only: bool = False,
    n_trials: int = 25,
    force_tuning: bool = False,
    models: list[str] | None = None,
) -> str:
    set_seed(42)
    device = get_device()
    print(f"🚀 设备: {device}")

    shared = load_shared_params()
    print(f"✅ 共享超参（全部模型相同）: {shared}")

    variant_names = models or [c.name for c in ST_COMPARE_VARIANTS]
    train_names = set(variant_names)

    if not skip_tuning:
        tune_all(model_names=variant_names, n_trials=n_trials, force=force_tuning)
    if tuning_only:
        print("ℹ️  --tuning-only：跳过完整训练")
        return SUMMARY_PATH

    baseline_dir = RESULTS_BASELINE
    data = load_and_preprocess_data(save_stats=True, results_dir=baseline_dir)
    norm_src = baseline_dir

    summary = []
    for model_cls in ST_COMPARE_VARIANTS:
        out_dir = results_dir_for(model_cls, RESULTS_ROOT)
        bp = _load_best_params(model_cls.name)
        specific = _model_specific_from_params(model_cls.name, bp)

        if model_cls.name not in train_names:
            if _is_trained(out_dir):
                print(f"\n📂 载入已有结果: {model_cls.label}")
                with open(json_file(out_dir, "model_config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                with open(json_file(out_dir, "generalization_metrics.json"), encoding="utf-8") as f:
                    gen = json.load(f)
                summary.append(_summary_row(model_cls, cfg, gen, out_dir, bp, specific))
            continue

        if skip_trained and _is_trained(out_dir):
            print(f"\n⏭ 已有结果，跳过: {model_cls.label} → {out_dir}")
            with open(json_file(out_dir, "model_config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            with open(json_file(out_dir, "generalization_metrics.json"), encoding="utf-8") as f:
                gen = json.load(f)
            summary.append(_summary_row(model_cls, cfg, gen, out_dir, bp, specific))
            continue

        os.makedirs(out_dir, exist_ok=True)
        _copy_norm_stats(norm_src, out_dir)
        save_best_params(bp, out_dir)

        result = train_model(
            model_cls,
            out_dir,
            bp,
            data,
            device,
            model_cls.name,
            model_cls.label,
            model_kwargs=specific,
        )
        summary.append(_summary_row(
            model_cls,
            {"params": result["params"]},
            result["generalization"],
            out_dir,
            bp,
            specific,
        ))

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    payload = {
        "note": "时空模型横向对比（5模型：3DCNN×2 + ConvLSTM + ConvGRU + STConv2+1D）；共享超参一致",
        "shared_params": shared,
        "models": summary,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 横向对比完成: {SUMMARY_PATH}")
    _print_ranking(summary)
    _run_output_plots(summary)
    return SUMMARY_PATH


def _run_output_plots(summary: list[dict]) -> None:
    """为每个模型生成预测结果图，并输出横向对比汇总图。"""
    from plot_results import main as plot_results
    from plot_st_compare import main as plot_st_compare

    for m in summary:
        rd = m["results_dir"]
        pred = os.path.join(rd, "Y_pred_real.npy")
        if os.path.exists(pred):
            print(f"\n📊 绘制预测结果图: {m['label']} → {rd}/figures/")
            plot_results(rd)
        else:
            print(f"⚠️  跳过绘图（无预测结果）: {m['label']}")

    print("\n📊 绘制横向对比汇总图...")
    if len(summary) >= 2:
        plot_st_compare()
    else:
        print("⚠️  模型数不足 2，跳过横向对比汇总图")


def _summary_row(model_cls, cfg, gen, out_dir, bp, specific) -> dict:
    test = gen["test"] if "test" in gen else gen
    return {
        "model": model_cls.name,
        "label": model_cls.label,
        "params": cfg.get("params", 0),
        "results_dir": out_dir,
        "shared_params": {k: bp[k] for k in ("lr", "batch_size", "dropout_rate", "weight_decay", "hidden_channels") if k in bp},
        "model_specific": specific,
        **{f"test_{k}": v for k, v in test.items()},
        **{f"train_{k}": v for k, v in gen.get("train", {}).items()},
        **{f"val_{k}": v for k, v in gen.get("val", {}).items()},
    }


def _print_ranking(summary: list[dict]) -> None:
    if not summary:
        return
    ranked = sorted(summary, key=lambda m: m.get("test_R2", -1), reverse=True)
    print("\n📊 测试集 R² 排名:")
    for i, m in enumerate(ranked, 1):
        print(f"  {i}. {m['label']:20s}  R²={m['test_R2']:.4f}  MAE={m['test_MAE']:.3f}cm  参数={m['params']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="时空模型横向对比实验")
    parser.add_argument("--skip-tuning", action="store_true", help="跳过专属超参搜索")
    parser.add_argument("--skip-trained", action="store_true", help="跳过已有 generalization_metrics 的模型")
    parser.add_argument("--tuning-only", action="store_true", help="仅运行贝叶斯优化")
    parser.add_argument("--force-tuning", action="store_true", help="强制重跑调参")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--models", nargs="*", default=None, help="指定模型名")
    args = parser.parse_args()
    run_st_compare(
        skip_tuning=args.skip_tuning,
        skip_trained=args.skip_trained,
        tuning_only=args.tuning_only,
        n_trials=args.n_trials,
        force_tuning=args.force_tuning,
        models=args.models,
    )


if __name__ == "__main__":
    main()
