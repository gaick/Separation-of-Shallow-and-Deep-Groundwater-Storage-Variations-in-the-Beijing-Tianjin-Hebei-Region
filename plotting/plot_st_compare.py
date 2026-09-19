"""时空模型横向对比可视化（含 Persistence 统计基线与本文 3DCNN）。"""

from __future__ import annotations

import argparse

import bootstrap  # noqa: F401
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from gws_io import PROPOSED_MODEL_ID, PROPOSED_MODEL_LABEL, RESULTS_PROPOSED
from gws_paths import BASELINE_PERSISTENCE_DIR, COMPARE_JSON_DIR, ST_COMPARE_FIG_DIR, ST_COMPARE_JSON_DIR, json_file
from plotting.plot_locale import get_locale, setup_matplotlib

_LABEL_EN = {
    "3DCNN(1层-浅)": "3DCNN (1-layer)",
    "3DCNN(1层·浅)": "3DCNN (1-layer)",
    "3DCNN(2层-基线)": "3DCNN (2-layer baseline)",
    "3DCNN(2层·基线)": "3DCNN (2-layer baseline)",
    "3DCNN(残差+末时刻-本文)": "3DCNN (M2, proposed)",
    "3DCNN(M2·2层·残差+末时刻·本文)": "3DCNN (M2, proposed)",
    "Persistence(基线)": "Persistence (baseline)",
}


def _display_label(label: str, lang: str) -> str:
    if lang == "en":
        return _LABEL_EN.get(label, label.replace("层", "-layer").replace("基线", "baseline").replace("本文", "proposed"))
    return label

SUMMARY_PATH = os.path.join(ST_COMPARE_JSON_DIR, "st_experiment_summary.json")
BASELINE_JSON = os.path.join(COMPARE_JSON_DIR, "baseline_comparison.json")
EXTENDED_SUMMARY_PATH = os.path.join(ST_COMPARE_JSON_DIR, "st_experiment_summary_with_baseline.json")


def _load_dl_models() -> list[dict]:
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"未找到 {SUMMARY_PATH}，请先运行 st_model_compare.py")
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)["models"]


def _entry_from_test(
    model_id: str,
    label: str,
    test: dict,
    params: int = 0,
    kind: str = "dl",
    results_dir: str = "",
) -> dict:
    return {
        "model": model_id,
        "label": label,
        "params": params,
        "kind": kind,
        "results_dir": results_dir,
        "test_R2": test["R2"],
        "test_MAE": test["MAE"],
        "test_RMSE": test["RMSE"],
        "test_MSE": test["MSE"],
    }


def _load_extended_models(include_paper_model: bool = True, lang: str = "zh") -> list[dict]:
    """DL 横向对比 + Persistence 基线 + 本文最终模型。"""
    loc = get_locale(lang)
    models = list(_load_dl_models())
    for m in models:
        m.setdefault("kind", "dl")

    if include_paper_model:
        gen_path = json_file(RESULTS_PROPOSED, "generalization_metrics.json")
        cfg_path = json_file(RESULTS_PROPOSED, "model_config.json")
        if os.path.exists(gen_path):
            with open(gen_path, encoding="utf-8") as f:
                gen = json.load(f)
            params = 211849
            if os.path.exists(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    params = json.load(f).get("params", params)
            paper = _entry_from_test(
                PROPOSED_MODEL_ID,
                loc.proposed_model_label if lang == "en" else PROPOSED_MODEL_LABEL,
                gen["test"],
                params=params,
                kind="paper",
                results_dir=RESULTS_PROPOSED,
            )
            models = [m for m in models if m.get("kind") != "paper"]
            models.append(paper)

    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON, encoding="utf-8") as f:
            bl = json.load(f)
        pers = _entry_from_test(
            "persistence",
            loc.persistence_baseline,
            bl["persistence"],
            params=0,
            kind="baseline",
            results_dir=BASELINE_PERSISTENCE_DIR,
        )
        models.append(pers)

    return models


def _save_extended_summary(models: list[dict]) -> None:
    os.makedirs(os.path.dirname(EXTENDED_SUMMARY_PATH), exist_ok=True)
    with open(EXTENDED_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": "含 DL 横向对比、本文最终模型（残差+末时刻）、Persistence 统计基线",
            "models": models,
        }, f, indent=2, ensure_ascii=False)


def main(include_paper_model: bool = True, out_dir: str | None = None, lang: str = "zh") -> None:
    loc = get_locale(lang)
    setup_matplotlib(lang)
    models = _load_extended_models(include_paper_model=include_paper_model, lang=lang)
    if len(models) < 2:
        raise FileNotFoundError("对比模型数量不足")

    _save_extended_summary(models)

    labels = [_display_label(m["label"], lang) for m in models]
    test_r2 = [m["test_R2"] for m in models]
    test_mae = [m["test_MAE"] for m in models]
    test_rmse = [m["test_RMSE"] for m in models]
    params = [m["params"] for m in models]
    kinds = [m.get("kind", "dl") for m in models]

    out_dir = out_dir or ST_COMPARE_FIG_DIR
    os.makedirs(out_dir, exist_ok=True)
    n = len(labels)

    kind_color = {"dl": "#66c2a5", "paper": "#fc8d62", "baseline": "#8da0cb"}
    colors = [kind_color.get(k, "#66c2a5") for k in kinds]
    hatches = ["//" if k == "baseline" else "" for k in kinds]

    fig, axes = plt.subplots(1, 3, figsize=(max(16, 2.6 * n), 5.5))
    metrics = [
        (test_r2, "Test R²", loc.higher_better, (0, 1.05)),
        (test_mae, "Test MAE (cm)", loc.lower_better, None),
        (test_rmse, "Test RMSE (cm)", loc.lower_better, None),
    ]
    for ax, (vals, title, hint, ylim) in zip(axes, metrics):
        bars = ax.bar(range(n), vals, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("black")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        sep = ", " if loc.lang == "en" else "，"
        ax.set_title(f"{title}\n({hint})" if loc.lang == "en" else f"{title}\n（{hint}）", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=kind_color["dl"], label=loc.st_legend_dl),
        Patch(facecolor=kind_color["paper"], label=loc.st_legend_paper),
        Patch(facecolor=kind_color["baseline"], hatch="//", label=loc.st_legend_persistence),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(loc.st_suptitle, fontsize=12, fontweight="bold", y=1.06)
    out1 = os.path.join(out_dir, "st_metrics_comparison.png")
    fig.tight_layout()
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    dl_models = [m for m in models if m.get("kind") != "baseline"]
    ref_p = dl_models[0]["params"] if dl_models else 1
    for m in models:
        p, r2, lb, k = m["params"], m["test_R2"], _display_label(m["label"], lang), m.get("kind", "dl")
        if k == "baseline":
            ax2.scatter(0, r2, s=160, c=kind_color["baseline"], marker="D",
                        edgecolors="black", zorder=3, label="_nolegend_")
            ax2.annotate(lb, (0, r2), textcoords="offset points", xytext=(8, 4), fontsize=8)
            continue
        ratio = p / ref_p if ref_p else 1.0
        ax2.scatter(p, r2, s=120, c=[kind_color.get(k, kind_color["dl"])],
                    edgecolors="black", zorder=3)
        suffix = f"×{ratio:.1f} {loc.params_ratio_suffix}" if loc.lang == "en" else f"×{ratio:.1f}{loc.params_ratio_suffix}"
        ax2.annotate(f"{lb}\n{suffix}", (p, r2), textcoords="offset points",
                     xytext=(6, 4), fontsize=7)
    ax2.set_xlabel(loc.st_params_xlabel)
    ax2.set_ylabel("Test R²")
    ax2.set_title(loc.st_params_title, fontweight="bold")
    ax2.grid(alpha=0.3)
    out2 = os.path.join(out_dir, "st_params_vs_r2.png")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"✅ {out1}")
    print(f"✅ {out2}")
    print(f"✅ {EXTENDED_SUMMARY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    main(out_dir=args.out_dir, lang=args.lang)
