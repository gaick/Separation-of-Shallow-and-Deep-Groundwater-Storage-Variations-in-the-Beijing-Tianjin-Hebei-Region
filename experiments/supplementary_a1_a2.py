"""
附录表 A1（rolling-origin）与表 A2（统计显著性）补充实验。

用法:
  PYTHONPATH=. .venv/bin/python experiments/supplementary_a1_a2.py           # 全部
  PYTHONPATH=. .venv/bin/python experiments/supplementary_a1_a2.py --a2-only   # 仅显著性（快）
  PYTHONPATH=. .venv/bin/python experiments/supplementary_a1_a2.py --a1-only   # 仅 rolling-origin（慢）
"""

from __future__ import annotations

import argparse
import bootstrap  # noqa: F401
import json
import os
from math import erfc, exp, lgamma, log, sqrt
from typing import Any

import numpy as np

from gws_io import (
    DATA_DIR,
    TIME_STEPS,
    calculate_metrics,
    get_test_target_indices,
    load_prediction_results,
)
from gws_paths import (
    BASELINE_PERSISTENCE_DIR,
    RESULTS_PROPOSED,
    ST_COMPARE_MODELS_ROOT,
    TABLES_ROOT,
)

from gws_model_variants import Simple_ST_Net_Ablation2Enc

PAPER_MODEL = Simple_ST_Net_Ablation2Enc
PAPER_MODEL_LABEL = "M2·2层Encoder"
HIGH_CHANGE_MONTH_IDX = [1, 2, 3, 4]  # 2024-06 ~ 2024-09（相对测试集起点）
HIGH_CHANGE_THRESHOLD_CM = 0.3
SUPP_JSON_DIR = os.path.join(TABLES_ROOT, "supplementary")
RO_MODELS_ROOT = os.path.join("./results/models/supplementary/rolling_origin")


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return calculate_metrics(y_true, y_pred)


def _squeeze(arr: np.ndarray) -> np.ndarray:
    return arr[:, 0] if arr.ndim == 4 else arr


def _paired_errors(y_true: np.ndarray, y_pred: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
    yt = _squeeze(y_true)
    yp = _squeeze(y_pred)
    abs_err = np.abs(yt - yp)
    mask = spatial_mask[np.newaxis, :, :] & (np.abs(yt) > 1e-5) & np.isfinite(yt) & np.isfinite(yp)
    return abs_err[mask]


def _t_cdf(x: float, df: int) -> float:
    """Student-t 累积分布（x >= 0）。"""
    if df <= 0:
        return 0.5
    t = df / (df + x * x)
    a, b = df / 2.0, 0.5
    return 1.0 - 0.5 * _regularized_incomplete_beta(t, a, b)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(log(x) * a + log(1 - x) * b - ln_beta) / a
    f, c, d = 1.0, 1.0, 1.0
    for m in range(1, 201):
        if m % 2 == 1:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(c * d - 1.0) < 1e-10:
            break
    return front * (f - 1.0)


def _paired_ttest_pvalue(diff: np.ndarray) -> tuple[float, float]:
    """单侧检验：H1 mean(diff) > 0（A 优于 B）。"""
    n = len(diff)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    if std_d < 1e-15:
        return 0.0, 1.0
    t_stat = mean_d / (std_d / sqrt(n))
    df = n - 1
    if t_stat >= 0:
        p_one = 1.0 - _t_cdf(t_stat, df)
    else:
        p_one = _t_cdf(-t_stat, df)
    return float(t_stat), float(min(p_one, 1.0))


def _wilcoxon_pvalue(diff: np.ndarray) -> tuple[float, float]:
    """Wilcoxon 符号秩检验（正态近似，单侧：A 优于 B）。"""
    diff = diff[np.abs(diff) > 1e-12]
    n = len(diff)
    if n < 10:
        return float("nan"), float("nan")
    ranks = _rankdata(np.abs(diff))
    w_plus = float(np.sum(ranks[diff > 0]))
    mu = n * (n + 1) / 4.0
    sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma < 1e-15:
        return w_plus, 1.0
    z = (w_plus - mu) / sigma
    p_one = 0.5 * erfc(z / sqrt(2.0))
    return w_plus, float(min(max(p_one, 0.0), 1.0))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=float)
    sorted_v = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i: j + 1]] = avg_rank
        i = j + 1
    return ranks


def _run_significance_test(
    err_a: np.ndarray,
    err_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    diff = err_b - err_a  # 正数表示 A 误差更小（A 更好）
    n = len(diff)
    t_stat, t_p = _paired_ttest_pvalue(diff)
    w_stat, w_p = _wilcoxon_pvalue(diff)

    sem = float(np.std(diff, ddof=1) / sqrt(n))
    ci_low = float(np.mean(diff) - 1.96 * sem)
    ci_high = float(np.mean(diff) + 1.96 * sem)
    alpha = 0.05
    significant = bool(t_p < alpha and np.mean(diff) > 0)

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_pairs": int(n),
        "mean_abs_error_a_cm": float(np.mean(err_a)),
        "mean_abs_error_b_cm": float(np.mean(err_b)),
        "mean_improvement_cm": float(np.mean(diff)),
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "ci95_improvement_cm": [ci_low, ci_high],
        "alpha": alpha,
        "significant_at_005": significant,
        "conclusion": f"{label_a} 显著优于 {label_b}" if significant else "差异未达 0.05 显著性",
    }


def _high_change_mask(
    y_true: np.ndarray,
    spatial_mask: np.ndarray,
    subset: str,
) -> np.ndarray:
    """返回与 flatten 后误差向量对齐的布尔掩膜。"""
    yt = _squeeze(y_true)
    n_months, h, w = yt.shape
    Y_raw = np.load(os.path.join(DATA_DIR, "Y_shallow_gws.npy"))
    n_seq = len(Y_raw) - TIME_STEPS
    test_target_idx = get_test_target_indices(len(Y_raw), n_seq)

    delta = np.zeros_like(yt)
    for i, t_idx in enumerate(test_target_idx):
        delta[i] = np.abs(Y_raw[t_idx] - Y_raw[t_idx - 1])

    base_mask = spatial_mask[np.newaxis, :, :] & (np.abs(yt) > 1e-5) & np.isfinite(yt)

    if subset == "high_change_months":
        month_mask = np.zeros(n_months, dtype=bool)
        month_mask[HIGH_CHANGE_MONTH_IDX] = True
        return base_mask & month_mask[:, np.newaxis, np.newaxis]

    if subset == "high_change_cells":
        return base_mask & (delta > HIGH_CHANGE_THRESHOLD_CM)

    return base_mask


def run_table_a2() -> dict[str, Any]:
    paper = load_prediction_results(RESULTS_PROPOSED)
    pers = load_prediction_results(BASELINE_PERSISTENCE_DIR)
    conv_dir = os.path.join(ST_COMPARE_MODELS_ROOT, "convlstm")
    conv = load_prediction_results(conv_dir)

    spatial_mask = paper["spatial_mask"]
    y_true = paper["y_true"]

    err_paper = _paired_errors(y_true, paper["y_pred"], spatial_mask)
    err_pers = _paired_errors(y_true, pers["y_pred"], spatial_mask)
    err_conv = _paired_errors(y_true, conv["y_pred"], spatial_mask)

    full_vs_pers = _run_significance_test(err_paper, err_pers, "M2", "Persistence")
    full_vs_conv = _run_significance_test(err_paper, err_conv, "M2", "ConvLSTM")

    hc_mask = _high_change_mask(y_true, spatial_mask, "high_change_months")
    yt = _squeeze(y_true)
    err_paper_hc = np.abs(yt - _squeeze(paper["y_pred"]))[hc_mask]
    err_pers_hc = np.abs(yt - _squeeze(pers["y_pred"]))[hc_mask]
    err_conv_hc = np.abs(yt - _squeeze(conv["y_pred"]))[hc_mask]
    hc_vs_pers = _run_significance_test(err_paper_hc, err_pers_hc, "M2", "Persistence")
    hc_vs_conv = _run_significance_test(err_paper_hc, err_conv_hc, "M2", "ConvLSTM")

    rows = [
        {
            "comparison_group": "M2 vs Persistence（全测试集）",
            "test_method": "配对 t 检验 / Wilcoxon",
            **full_vs_pers,
        },
        {
            "comparison_group": "M2 vs ConvLSTM（无残差，全测试集）",
            "test_method": "配对 t 检验 / Wilcoxon",
            **full_vs_conv,
        },
        {
            "comparison_group": "M2 vs Persistence（高变化子集，2024-06~09）",
            "test_method": "配对 t 检验 / Wilcoxon",
            **hc_vs_pers,
        },
        {
            "comparison_group": "M2 vs ConvLSTM（无残差，高变化子集，2024-06~09）",
            "test_method": "配对 t 检验 / Wilcoxon",
            **hc_vs_conv,
        },
    ]

    result = {
        "table": "A2",
        "description": "关键模型对比的统计显著性检验（测试集 2024-05~12）",
        "high_change_subset": "2024-06 ~ 2024-09 四个月",
        "rows": rows,
    }

    os.makedirs(SUPP_JSON_DIR, exist_ok=True)
    out_path = os.path.join(SUPP_JSON_DIR, "table_a2_significance.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print_table_a2(result)
    print(f"\n✅ 表 A2 结果: {out_path}")
    return result


def _train_ro_fold(split, device, best_params: dict, force: bool = False) -> dict[str, Any]:
    from gws_io import save_best_params
    from gws_paths import json_file
    from gws_train import train_model
    from gws_splits import load_and_preprocess_calendar_split

    results_dir = os.path.join(RO_MODELS_ROOT, split.name.lower())
    metrics_path = json_file(results_dir, "generalization_metrics.json")
    config_path = json_file(results_dir, "model_config.json")

    if not force and os.path.exists(metrics_path) and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("model_name") == PAPER_MODEL.name:
            with open(metrics_path, encoding="utf-8") as f:
                gen = json.load(f)
            print(f"⏭  {split.name} 已有 M2 结果，跳过训练")
            return {"split": split.name, "metrics": gen["test"], "results_dir": results_dir, "skipped": True}

    data = load_and_preprocess_calendar_split(split, save_stats=True, results_dir=results_dir)
    os.makedirs(results_dir, exist_ok=True)
    save_best_params(best_params, results_dir)

    train_model(
        PAPER_MODEL,
        results_dir,
        best_params,
        data,
        device,
        PAPER_MODEL.name,
        f"{PAPER_MODEL.label} ({split.name})",
    )

    with open(metrics_path, encoding="utf-8") as f:
        gen = json.load(f)
    return {"split": split.name, "metrics": gen["test"], "results_dir": results_dir, "skipped": False}


def run_table_a1(device, force: bool = False) -> dict[str, Any]:
    from gws_io import load_best_params_from
    from gws_splits import DEFAULT_RO_SPLITS

    bp = load_best_params_from(RESULTS_PROPOSED)
    rows = []

    for split in DEFAULT_RO_SPLITS:
        fold = _train_ro_fold(split, device, bp, force=force)
        m = fold["metrics"]
        rows.append({
            "scheme": split.name,
            "train_period": split.train_label,
            "val_period": split.val_label,
            "test_period": split.test_label,
            "R2": m["R2"],
            "MAE_cm": m["MAE"],
            "RMSE_cm": m["RMSE"],
            "note": "滚动起点验证",
            "results_dir": fold["results_dir"],
        })

    r2_vals = [r["R2"] for r in rows]
    mae_vals = [r["MAE_cm"] for r in rows]
    rmse_vals = [r["RMSE_cm"] for r in rows]

    avg_row = {
        "scheme": "RO-平均",
        "train_period": "—",
        "val_period": "—",
        "test_period": "—",
        "R2": f"{np.mean(r2_vals):.4f} ± {np.std(r2_vals, ddof=1):.4f}",
        "MAE_cm": f"{np.mean(mae_vals):.4f} ± {np.std(mae_vals, ddof=1):.4f}",
        "RMSE_cm": f"{np.mean(rmse_vals):.4f} ± {np.std(rmse_vals, ddof=1):.4f}",
        "note": "报告指标分布而非单次点估计",
    }

    result = {
        "table": "A1",
        "description": f"rolling-origin 验证结果（{PAPER_MODEL_LABEL}）",
        "rows": rows,
        "average": avg_row,
    }

    os.makedirs(SUPP_JSON_DIR, exist_ok=True)
    out_path = os.path.join(SUPP_JSON_DIR, "table_a1_rolling_origin.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print_table_a1(result)
    print(f"\n✅ 表 A1 结果: {out_path}")
    return result


def _print_table_a1(result: dict) -> None:
    print("\n" + "=" * 72)
    print("表 A1  rolling-origin 验证结果")
    print("=" * 72)
    print(f"{'方案':<8} {'训练期':<12} {'验证期':<6} {'测试期':<6} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 72)
    for r in result["rows"]:
        print(
            f"{r['scheme']:<8} {r['train_period']:<12} {r['val_period']:<6} {r['test_period']:<6} "
            f"{r['R2']:>8.4f} {r['MAE_cm']:>9.4f} {r['RMSE_cm']:>9.4f}"
        )
    a = result["average"]
    print("-" * 72)
    print(
        f"{a['scheme']:<8} {'—':<12} {'—':<6} {'—':<6} "
        f"{a['R2']:>8} {a['MAE_cm']:>10} {a['RMSE_cm']:>10}"
    )


def _print_table_a2(result: dict) -> None:
    print("\n" + "=" * 72)
    print("表 A2  统计显著性检验")
    print("=" * 72)
    for row in result["rows"]:
        print(f"\n{row['comparison_group']}")
        print(f"  配对 t: stat={row['paired_t_statistic']:.2f}, p={row['paired_t_pvalue']:.2e}")
        print(f"  Wilcoxon: stat={row['wilcoxon_statistic']:.0f}, p={row['wilcoxon_pvalue']:.2e}")
        print(f"  95% CI (误差改善): [{row['ci95_improvement_cm'][0]:.4f}, {row['ci95_improvement_cm'][1]:.4f}] cm")
        print(f"  结论: {row['conclusion']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="附录表 A1/A2 补充实验")
    parser.add_argument("--a1-only", action="store_true", help="仅运行 rolling-origin")
    parser.add_argument("--force-retrain", action="store_true", help="强制重训 rolling-origin（换模型后使用）")
    parser.add_argument("--a2-only", action="store_true", help="仅运行显著性检验")
    args = parser.parse_args()

    run_a1 = not args.a2_only
    run_a2 = not args.a1_only

    if run_a2:
        run_table_a2()

    if run_a1:
        from gws_common import get_device, set_seed

        set_seed(42)
        device = get_device()
        print(f"\n🚀 rolling-origin 训练设备: {device} | 模型: {PAPER_MODEL_LABEL}")
        run_table_a1(device, force=args.force_retrain)


if __name__ == "__main__":
    main()
