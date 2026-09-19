"""
顶刊风格架构图 — 基于 architecture_spec.py 精确绘制
运行: python plotting/draw_architecture.py
"""

import bootstrap  # noqa: F401
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
import architecture_spec as spec

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
})

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "input":  "#A8D8A8",
    "conv3d": "#6BAED6",
    "pool":   "#FD8D3C",
    "conv2d": "#9E9AC8",
    "output": "#FC9272",
    "op":     "#E0E0E0",
    "arrow":  "#333333",
    "group":  "#666666",
}


def darken(hex_color, amount=0.22):
    c = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return tuple(max(0, v * (1 - amount)) for v in c)


def draw_3d_block(ax, x, y, w, h, d, color, alpha=0.9, zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#333",
                           linewidth=1.0, alpha=alpha, zorder=zorder))
    ax.add_patch(Polygon([(x, y+h), (x+d*0.55, y+h+d*0.35), (x+w+d*0.55, y+h+d*0.35), (x+w, y+h)],
                         facecolor=darken(color), edgecolor="#333", linewidth=0.8, alpha=alpha, zorder=zorder-1))
    ax.add_patch(Polygon([(x+w, y), (x+w+d*0.55, y+d*0.35), (x+w+d*0.55, y+h+d*0.35), (x+w, y+h)],
                         facecolor=darken(color, 0.35), edgecolor="#333", linewidth=0.8, alpha=alpha, zorder=zorder-1))
    return x + w / 2, y + h / 2


def block_width(layer):
    """方块厚度 ∝ 通道数；输入层额外体现时间维"""
    if layer["id"] == "input":
        return 3.2
    if layer["id"] == "tmean":
        return 0.5
    ch = layer["out_ch"]
    return 0.6 + 0.06 * min(ch, 48)


def block_height(layer):
    if layer["id"] in ("input", "output"):
        return 2.4
    return 2.0


def draw_architecture():
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=300)
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(-1.2, 5.8)
    ax.axis("off")

    # ── 布局参数 ──
    x = 0.8
    y_base = 1.8
    positions = {}

    for layer in spec.DRAW_LAYERS:
        w = block_width(layer)
        h = block_height(layer)
        d = 0.35 + w * 0.08
        color = COLORS[layer["color_key"]]
        cx, cy = draw_3d_block(ax, x, y_base, w, h, d, color)
        positions[layer["id"]] = (x, x + w, cx, cy)

        # 主标签
        if layer["in_ch"] != layer["out_ch"] and layer["kernel"]:
            ch_text = f"{layer['in_ch']}$\\rightarrow${layer['out_ch']}"
        elif layer["id"] == "input":
            ch_text = f"{spec.IN_CHANNELS} ch"
        else:
            ch_text = f"{layer['out_ch']} ch"

        ax.text(cx, y_base - 0.35, layer["name"], ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="#111")
        ax.text(cx, y_base - 0.65, ch_text, ha="center", va="top", fontsize=8.5, color="#333")

        if layer["kernel"]:
            ax.text(cx, y_base - 0.88, f"k={layer['kernel']}", ha="center", va="top",
                    fontsize=7.5, color="#555", style="italic")
        if layer["extra"]:
            ax.text(cx, y_base - 1.08, layer["extra"], ha="center", va="top",
                    fontsize=7, color="#666")

        # 张量形状（方块上方）
        ax.text(cx, y_base + h + d * 0.35 + 0.15, layer["shape"],
                ha="center", va="bottom", fontsize=7, color="#444",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#CCC", alpha=0.9))

        x += w + 1.1

    # ── 箭头 ──
    ids = [l["id"] for l in spec.DRAW_LAYERS]
    for i in range(len(ids) - 1):
        _, x1, _, cy = positions[ids[i]]
        x0, _, _, _ = positions[ids[i + 1]]
        ax.annotate("", xy=(x0 - 0.08, cy), xytext=(x1 + 0.08, cy),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["arrow"], lw=1.6))

    # permute 标注（input → enc1 之间）
    _, x1, _, cy = positions["input"]
    x0, _, _, _ = positions["enc1"]
    mx = (x1 + x0) / 2
    ax.text(mx, cy + 0.55, "permute\n(B,T,C,H,W)$\\rightarrow$(B,C,T,H,W)",
            ha="center", va="bottom", fontsize=6.5, color="#888")

    # ── 分组虚线框 ──
    group_ranges = {}
    for lid, (x0, x1, _, _) in positions.items():
        layer = next(l for l in spec.DRAW_LAYERS if l["id"] == lid)
        g = layer["group"]
        if g not in group_ranges:
            group_ranges[g] = [x0, x1]
        else:
            group_ranges[g][0] = min(group_ranges[g][0], x0)
            group_ranges[g][1] = max(group_ranges[g][1], x1)

    group_labels = dict(spec.GROUPS)
    for g, (x0, x1) in group_ranges.items():
        pad = 0.25
        rect = FancyBboxPatch((x0 - pad, 0.55), x1 - x0 + 2 * pad, 4.5,
                              boxstyle="round,pad=0.02", facecolor="none",
                              edgecolor=COLORS["group"], linestyle="--", linewidth=1.3)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, 5.15, group_labels.get(g, g),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLORS["group"])

    # ── 输入通道说明 ──
    drivers = ", ".join(spec.DRIVERS)
    ax.text(positions["input"][2], 0.85,
            f"Channels: [{drivers}, {spec.GWS_HISTORY_CH}]\n"
            f"Grid: {spec.H}$\\times${spec.W}  |  Window: {spec.TIME_STEPS} months",
            ha="center", va="top", fontsize=7.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF0", edgecolor="#8BC78B", alpha=0.9))

    # ── 标题 ──
    ax.set_title(
        f"Architecture of Simple_ST_Net  ($C_{{hidden}}$={spec.HIDDEN_CHANNELS}, "
        f"Jing-Jin-Ji, {spec.RESOLUTION})",
        fontsize=13, fontweight="bold", pad=18
    )

    # ── 图例 ──
    legend_items = [
        ("input", "Input Data"), ("conv3d", "3D Convolution"),
        ("pool", "Temporal Aggregation"), ("conv2d", "2D Convolution"),
        ("output", "Prediction"),
    ]
    for i, (key, label) in enumerate(legend_items):
        lx = 12.5
        ly = -0.85 - i * 0.22
        ax.add_patch(Rectangle((lx, ly), 0.28, 0.14, facecolor=COLORS[key], edgecolor="#333", lw=0.8))
        ax.text(lx + 0.38, ly + 0.07, label, fontsize=7.5, va="center")

    path = os.path.join(OUT_DIR, "fig2_model_architecture.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {path}")
    return path


def draw_sliding_window():
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=300)
    n_months, window = 20, spec.TIME_STEPS
    months = range(n_months)
    ax.plot(months, [0]*n_months, "k-", lw=0.8)
    for m in months:
        ax.text(m, -0.4, f"M{m+1}", ha="center", fontsize=7, color="#666")

    colors = ["#6BAED6", "#FD8D3C", "#A8D8A8"]
    for i, start in enumerate([0, 4, 8]):
        y = 0.35 + i * 0.85
        rect = FancyBboxPatch((start, y), window, 0.5, boxstyle="round,pad=0.01",
                              facecolor=colors[i], edgecolor="#333", alpha=0.75, lw=0.9)
        ax.add_patch(rect)
        ax.text(start + window/2, y + 0.25,
                f"Input: {window} months $\\times$ {spec.IN_CHANNELS} ch $\\times$ "
                f"{spec.H}$\\times${spec.W}",
                ha="center", va="center", fontsize=7.5)
        ax.annotate(f"Predict\nM{start+window+1}",
                    xy=(start + window + 0.2, y + 0.25), xytext=(start + window + 0.05, y + 0.25),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0), fontsize=7, va="center")

    ax.set_xlim(-1, n_months + 2)
    ax.set_ylim(-0.9, 3.0)
    ax.axis("off")
    ax.set_title("Sliding-Window Input Construction", fontsize=12, fontweight="bold", pad=10)
    path = os.path.join(OUT_DIR, "fig1_sliding_window.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {path}")
    return path


if __name__ == "__main__":
    print("📐 架构规格:")
    print(f"   输入: {spec.TIME_STEPS}mo × {spec.IN_CHANNELS}ch × {spec.H}×{spec.W}")
    print(f"   hidden_channels = {spec.HIDDEN_CHANNELS}")
    for l in spec.LAYERS:
        print(f"   {l['name']:14s}  {l.get('op',''):12s}  shape={l['shape']}")
    print()
    draw_sliding_window()
    draw_architecture()
