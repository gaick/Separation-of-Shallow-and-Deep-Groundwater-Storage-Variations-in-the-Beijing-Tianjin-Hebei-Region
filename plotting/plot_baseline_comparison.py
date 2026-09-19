"""绘制基线对比图：全测试集 + 高变化子集。"""

from __future__ import annotations

import bootstrap  # noqa: F401
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from gws_io import PROPOSED_MODEL_LABEL
from gws_paths import COMPARE_FIG_DIR, COMPARE_JSON_DIR

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Arial Unicode MS", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
})

OUT_FULL = os.path.join(COMPARE_FIG_DIR, "baseline_comparison.png")
OUT_STRAT = os.path.join(COMPARE_FIG_DIR, "baseline_stratified_comparison.png")


def _bar_panel(ax, names, m4_vals, per_vals, title, ylabel):
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, per_vals, w, label="Persistence", color="#3498db")
    ax.bar(x + w / 2, m4_vals, w, label=PROPOSED_MODEL_LABEL, color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, (pv, mv) in enumerate(zip(per_vals, m4_vals)):
        better = "本文" if mv < pv else "Per"
        ax.text(i, max(pv, mv) * 1.02, better, ha="center", fontsize=8, color="#27ae60")


def main():
    path = os.path.join(COMPARE_JSON_DIR, "baseline_comparison.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    clim = d["climatology"]
    per = d["persistence"]
    cnn = d["cnn_baseline"]

    # 图1：全测试集三方法
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = ["气候态", "Persistence", "本文模型"]
    r2 = [clim["R2"], per["R2"], cnn["R2"]]
    mae = [clim["MAE"], per["MAE"], cnn["MAE"]]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    axes[0].bar(names, r2, color=colors)
    axes[0].set_ylabel("R²")
    axes[0].set_title("全测试集 R²")
    axes[0].set_ylim(min(r2) - 0.2, 1.0)
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(r2):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(names, mae, color=colors)
    axes[1].set_ylabel("MAE (cm)")
    axes[1].set_title("全测试集 MAE")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(mae):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    fig.suptitle(f"浅层 GWS 预测：基线 vs {PROPOSED_MODEL_LABEL}（全测试集）", fontsize=12)
    os.makedirs(os.path.dirname(OUT_FULL), exist_ok=True)
    fig.savefig(OUT_FULL, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✅ {OUT_FULL}")

    # 图2：高变化子集 M4 vs Persistence
    st = d.get("stratified", {})
    hm = st.get("high_change_months_metrics", {})
    hc = st.get("high_change_cells_metrics", {})
    if not hm or not hc:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    subsets = ["高变化月份\n(06~09)", "高变化格点\n(|ΔY|>0.3cm)"]
    _bar_panel(
        axes[0], subsets,
        [hm["cnn"]["MAE"], hc["cnn"]["MAE"]],
        [hm["persistence"]["MAE"], hc["persistence"]["MAE"]],
        "MAE 对比（越低越好）", "MAE (cm)",
    )
    _bar_panel(
        axes[1], subsets,
        [hm["cnn"]["MSE"], hc["cnn"]["MSE"]],
        [hm["persistence"]["MSE"], hc["persistence"]["MSE"]],
        "MSE 对比（越低越好）", "MSE (cm²)",
    )
    fig.suptitle(f"高变化子集：{PROPOSED_MODEL_LABEL} vs Persistence", fontsize=12)
    fig.savefig(OUT_STRAT, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✅ {OUT_STRAT}")


if __name__ == "__main__":
    main()
