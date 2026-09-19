"""
浅层 GWS 3DCNN 一键流程入口。

默认一次跑完 5 个 Encoder 消融变体训练 + 各模型结果图 + 对比汇总图。

用法:
  python run_pipeline.py
  python run_pipeline.py --skip-trained   # 跳过已有结果的变体
  python run_pipeline.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import bootstrap  # noqa: F401 — 确保子目录脚本可被 import
from gws_paths import COMPARE_JSON_DIR


def _step(title: str) -> None:
    print(f"\n{'=' * 60}\n▶ {title}\n{'=' * 60}")


def _all_result_dirs() -> list[str]:
    summary = os.path.join(COMPARE_JSON_DIR, "experiment_summary.json")
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            data = json.load(f)
        from gws_paths import resolve_model_dir
        return [resolve_model_dir(m["results_dir"]) for m in data["models"]]
    from gws_model_variants import ABLATION_VARIANTS, results_dir_for
    return [results_dir_for(c) for c in ABLATION_VARIANTS]


def run_post_analysis() -> None:
    """Persistence 基线、深层 GWS 图、论文 docx。"""
    _step("Persistence / 气候态基线评估")
    from baseline_persistence import main as baseline_main
    baseline_main()

    _step("深层 GWS 分离与漏斗区对比图")
    from plot_deep_gws import main as plot_deep_main
    plot_deep_main()

    _step("基线对比柱状图")
    from plot_baseline_comparison import main as plot_base_main
    plot_base_main()

    _step("生成论文 2.4–5 节 Word 文档")
    from generate_paper_sections import build_doc
    build_doc()


def run_plots() -> None:
    from plot_compare import main as plot_compare
    from plot_results import main as plot_results

    for d in _all_result_dirs():
        pred = os.path.join(d, "Y_pred_real.npy")
        if os.path.exists(pred):
            _step(f"绘制结果图: {d}")
            plot_results(d)

    _step("绘制消融对比图")
    plot_compare()

    print("\n" + "=" * 60)
    print("📂 输出目录")
    for d in _all_result_dirs():
        if os.path.exists(os.path.join(d, "figures")):
            print(f"   {d}/figures/")
    print("   ./results/figures/compare/")
    print("   ./results/json/compare/")
    print("=" * 60)


def run_training(autotrain: bool = False, skip_baseline: bool = False, skip_trained: bool = False) -> None:
    from compare_experiment import run_compare_experiment

    if autotrain:
        _step("Optuna 自动调参 + 基线模型训练")
        from AutoTrain import main as autotrain_main
        autotrain_main()
        skip_baseline = True
        print("\nℹ️  基线已由 AutoTrain 完成，消融实验将跳过基线重训")

    _step("Encoder 深度消融实验（5 变体）")
    run_compare_experiment(skip_baseline=skip_baseline, skip_trained=skip_trained)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="浅层 GWS 3DCNN 一键流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--autotrain", action="store_true",
        help="先运行 AutoTrain（Optuna 调参 + 基线训练），再训练加深模型",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="跳过训练，仅根据已有结果画图",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="跳过基线训练（需已有基线结果；常与 --autotrain 联用）",
    )
    parser.add_argument(
        "--skip-trained", action="store_true",
        help="跳过已有 generalization_metrics.json 的变体",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="仅运行基线对比、深层GWS图及论文docx生成",
    )
    args = parser.parse_args()

    try:
        if args.analysis_only:
            _step("仅后分析模式")
            run_post_analysis()
        elif args.plot_only:
            _step("仅画图模式")
            run_plots()
            run_post_analysis()
        else:
            run_training(
                autotrain=args.autotrain,
                skip_baseline=args.skip_baseline,
                skip_trained=args.skip_trained,
            )
            run_plots()
            run_post_analysis()
    except FileNotFoundError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹ 用户中断")
        sys.exit(130)

    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()
