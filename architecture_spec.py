"""
Simple_ST_Net 网络结构规格（与 AutoTrain.py / torch_test.py 一致）
所有绘图脚本应引用此文件，保证图与代码一致。
"""

# ── 数据 ──────────────────────────────────────────────
REGION = "京津冀"
RESOLUTION = "0.25°"
TIME_RANGE = "2018-01 ~ 2024-12"
N_MONTHS = 84
H, W = 27, 25

DRIVERS = ["pre", "tmp", "pet", "lst", "ndvi"]  # 降水/气温/蒸散发/地表温度/NDVI
N_DRIVERS = len(DRIVERS)
GWS_HISTORY_CH = "gws"  # 第6通道：历史浅层GWS
IN_CHANNELS = N_DRIVERS + 1  # 6

TIME_STEPS = 12  # 滑动窗口长度

# ── 最优超参（torch_test.py BEST_PARAMS）──────────────
HIDDEN_CHANNELS = 24
DROPOUT = 0.056
BATCH_SIZE = 8

# ── 各层定义（顺序即 forward 流程）────────────────────
# 每项: name, op, in_ch, out_ch, kernel, extra, out_shape (C,T,H,W) 或 (C,H,W)
LAYERS = [
    {
        "id": "input",
        "group": "Input",
        "name": "Input Tensor",
        "op": "Sliding Window",
        "in_ch": IN_CHANNELS,
        "out_ch": IN_CHANNELS,
        "kernel": None,
        "extra": f"{TIME_STEPS} months",
        "shape": f"({IN_CHANNELS}, {TIME_STEPS}, {H}, {W})",
        "note": "5 drivers + GWS history",
        "color_key": "input",
    },
    {
        "id": "permute",
        "group": "Input",
        "name": "Permute",
        "op": "permute(0,2,1,3,4)",
        "in_ch": IN_CHANNELS,
        "out_ch": IN_CHANNELS,
        "kernel": None,
        "extra": "",
        "shape": f"({IN_CHANNELS}, {TIME_STEPS}, {H}, {W})",
        "note": "(B,T,C,H,W)→(B,C,T,H,W)",
        "color_key": "op",
        "is_op": True,
    },
    {
        "id": "enc1",
        "group": "3D Encoder",
        "name": "Conv3D-1",
        "op": "Conv3d",
        "in_ch": IN_CHANNELS,
        "out_ch": HIDDEN_CHANNELS,
        "kernel": "3×3×3",
        "extra": "BN + GELU + Dropout3d",
        "shape": f"({HIDDEN_CHANNELS}, {TIME_STEPS}, {H}, {W})",
        "note": "padding=(1,1,1)",
        "color_key": "conv3d",
    },
    {
        "id": "enc2",
        "group": "3D Encoder",
        "name": "Conv3D-2",
        "op": "Conv3d",
        "in_ch": HIDDEN_CHANNELS,
        "out_ch": HIDDEN_CHANNELS * 2,
        "kernel": "3×3×3",
        "extra": "BN + GELU + Dropout3d",
        "shape": f"({HIDDEN_CHANNELS * 2}, {TIME_STEPS}, {H}, {W})",
        "note": "padding=(1,1,1)",
        "color_key": "conv3d",
    },
    {
        "id": "tmean",
        "group": "Temporal Agg.",
        "name": "Temporal Mean",
        "op": "mean(dim=2)",
        "in_ch": HIDDEN_CHANNELS * 2,
        "out_ch": HIDDEN_CHANNELS * 2,
        "kernel": None,
        "extra": "T: 12→1",
        "shape": f"({HIDDEN_CHANNELS * 2}, {H}, {W})",
        "note": "沿时间维聚合",
        "color_key": "pool",
    },
    {
        "id": "dec1",
        "group": "2D Decoder",
        "name": "Conv2D-1",
        "op": "Conv2d",
        "in_ch": HIDDEN_CHANNELS * 2,
        "out_ch": HIDDEN_CHANNELS,
        "kernel": "3×3",
        "extra": "GELU",
        "shape": f"({HIDDEN_CHANNELS}, {H}, {W})",
        "note": "padding=1",
        "color_key": "conv2d",
    },
    {
        "id": "dec2",
        "group": "2D Decoder",
        "name": "Conv2D-2",
        "op": "Conv2d",
        "in_ch": HIDDEN_CHANNELS,
        "out_ch": 1,
        "kernel": "1×1",
        "extra": "",
        "shape": f"(1, {H}, {W})",
        "note": "",
        "color_key": "conv2d",
    },
    {
        "id": "output",
        "group": "Output",
        "name": "Shallow GWS",
        "op": "Prediction Map",
        "in_ch": 1,
        "out_ch": 1,
        "kernel": None,
        "extra": "denormalize",
        "shape": f"(1, {H}, {W})",
        "note": "下一月浅层地下水储量变化",
        "color_key": "output",
    },
]

# 绘图用：跳过 permute（太小，用箭头标注即可）
DRAW_LAYERS = [l for l in LAYERS if l["id"] != "permute"]

GROUPS = [
    ("Input", "Input Data\n(5 drivers + GWS history)"),
    ("3D Encoder", "3D Spatiotemporal Encoder"),
    ("Temporal Agg.", "Temporal Aggregation"),
    ("2D Decoder", "2D Spatial Decoder"),
    ("Output", "Output"),
]
