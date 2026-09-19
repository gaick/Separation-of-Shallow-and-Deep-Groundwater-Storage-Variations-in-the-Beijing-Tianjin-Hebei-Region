"""
3DCNN 消融变体（对比实验用）

说明：
  - 对比维度是 **3D Encoder 中 Conv3D 的层数** 与 **通道扩展策略**
  - 全部消融变体统一：**末时刻时间聚合 + Persistence 残差**（Ŷ=Y(t-1)+Δ）
  - hidden_channels 是卷积宽度超参（如 24），不是额外再叠 MLP「隐藏层」
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gws_paths import MODELS_ROOT

IN_CHANNELS = 6
GWS_CHANNEL_IDX = 5  # 第 6 通道：GWS 历史（窗口末帧 = Y(t-1)）


def _enc3d(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.BatchNorm3d(out_ch),
        nn.GELU(),
        nn.Dropout3d(p=dropout),
    )


def _dec2d(in_ch: int, mid_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(mid_ch, 1, kernel_size=1),
    )


class _STNetBase(nn.Module):
    """默认：3D Encoder + 时间维 mean 池化 + 2D Decoder。"""

    temporal_pool: str = "mean"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.encoder(x)
        x = self._pool_time(x)
        out = self.decoder(x)
        return self._post_decode(x, out)

    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_pool == "last":
            return x[:, :, -1, :, :]
        return torch.mean(x, dim=2)

    def _post_decode(self, x_enc_last: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return out


class _STNetBaseResidual(_STNetBase):
    """残差头：输出 ΔGWS，加窗口末帧 GWS（即 Y(t-1)）。"""

    use_residual: bool = True

    def _post_decode(self, x_enc_last: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if not self.use_residual:
            return out
        # x 未传入 gws_last，子类 forward 需 override 或在 _post_decode 从 module 取
        return out


class _STNetBaseResidualLastT(_STNetBase):
    """末时刻时间聚合 + Persistence 残差：Ŷ(t)=Y(t-1)+Δ。"""

    temporal_pool: str = "last"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bcthw = x.permute(0, 2, 1, 3, 4).contiguous()
        gws_last = x_bcthw[:, GWS_CHANNEL_IDX : GWS_CHANNEL_IDX + 1, -1, :, :]
        x = self.encoder(x_bcthw)
        x = x[:, :, -1, :, :]
        delta = self.decoder(x)
        return delta + gws_last


class _STNetBaseLegacy(_STNetBase):
    """旧版：时间 mean 池化、无残差（仅 ST 横向对比参照用）。"""

    temporal_pool: str = "mean"


class Simple_ST_Net_Shallow(_STNetBaseResidualLastT):
    """消融 M1：1 层 Conv3D Encoder。"""

    name = "shallow_1enc"
    label = "M1·1层Encoder"
    results_subdir = "ablation/enc1"

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.2, hidden_channels: int = 16):
        super().__init__()
        h = hidden_channels
        self.encoder = _enc3d(in_channels, h, dropout_rate)
        self.decoder = _dec2d(h, h)


class Simple_ST_Net(_STNetBaseLegacy):
    """2 层 Conv3D（旧结构，AutoTrain / 横向对比参照）。"""

    name = "baseline_2enc"
    label = "基线 2层Encoder"
    results_subdir = "shallow_gws"

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.2, hidden_channels: int = 16):
        super().__init__()
        h, h2 = hidden_channels, hidden_channels * 2
        self.encoder = nn.Sequential(
            _enc3d(in_channels, h, dropout_rate),
            _enc3d(h, h2, dropout_rate),
        )
        self.decoder = _dec2d(h2, h)


class Simple_ST_Net_Ablation2Enc(_STNetBaseResidualLastT):
    """消融 M2：2 层 Conv3D Encoder。"""

    name = "baseline_2enc_res"
    label = "M2·2层Encoder"

    results_subdir = "shallow_gws"

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.2, hidden_channels: int = 16):
        super().__init__()
        h, h2 = hidden_channels, hidden_channels * 2
        self.encoder = nn.Sequential(
            _enc3d(in_channels, h, dropout_rate),
            _enc3d(h, h2, dropout_rate),
        )
        self.decoder = _dec2d(h2, h)


class Simple_ST_Net_Plus1Flat(_STNetBaseResidualLastT):
    """消融 M3：3 层 Encoder，通道恒宽。"""

    name = "plus1_flat_3enc"
    label = "M3·3层(恒宽)"
    results_subdir = "ablation/enc3_flat"

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.2, hidden_channels: int = 16):
        super().__init__()
        h = hidden_channels
        self.encoder = nn.Sequential(
            _enc3d(in_channels, h, dropout_rate),
            _enc3d(h, h, dropout_rate),
            _enc3d(h, h, dropout_rate),
        )
        self.decoder = _dec2d(h, h)


class Simple_ST_Net_Plus1Enc(_STNetBaseResidualLastT):
    """消融 M4 / 本文最终模型：3 层 Encoder 加宽 + 残差 + 末时刻。"""

    name = "plus1enc_3enc"
    label = "M4·3层(加宽)"
    results_subdir = "shallow_gws_plus1enc_reslastt"

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.2, hidden_channels: int = 16):
        super().__init__()
        h, h2, h4 = hidden_channels, hidden_channels * 2, hidden_channels * 4
        self.encoder = nn.Sequential(
            _enc3d(in_channels, h, dropout_rate),
            _enc3d(h, h2, dropout_rate),
            _enc3d(h2, h4, dropout_rate),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(h4, h2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h2, h, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h, 1, kernel_size=1),
        )


class Simple_ST_Net_Plus2Flat(_STNetBaseResidualLastT):
    """消融 M5：4 层 Encoder，通道封顶 2h。"""

    name = "plus2_flat_4enc"
    label = "M5·4层(封顶)"
    results_subdir = "ablation/enc4_flat"

    def __init__(self, in_channels: int = 6, dropout_rate: float = 0.2, hidden_channels: int = 16):
        super().__init__()
        h, h2 = hidden_channels, hidden_channels * 2
        self.encoder = nn.Sequential(
            _enc3d(in_channels, h, dropout_rate),
            _enc3d(h, h2, dropout_rate),
            _enc3d(h2, h2, dropout_rate),
            _enc3d(h2, h2, dropout_rate),
        )
        self.decoder = _dec2d(h2, h)


# 消融实验（统一残差+末时刻；M2 与旧 shallow_gws 目录共用）
ABLATION_VARIANTS: list[type[_STNetBase]] = [
    Simple_ST_Net_Shallow,
    Simple_ST_Net_Ablation2Enc,
    Simple_ST_Net_Plus1Flat,
    Simple_ST_Net_Plus1Enc,
    Simple_ST_Net_Plus2Flat,
]

# 向后兼容别名
Simple_ST_Net_Plus1Enc_ResLastT = Simple_ST_Net_Plus1Enc

MODEL_REGISTRY: dict[str, type[nn.Module]] = {cls.name: cls for cls in ABLATION_VARIANTS}


def results_dir_for(model_cls: type[_STNetBase], root: str = MODELS_ROOT) -> str:
    return f"{root.rstrip('/')}/{model_cls.results_subdir}"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_param_table(hidden_channels: int = 24) -> list[tuple[str, int]]:
    """不依赖 torch 运行时可手算；有 torch 时精确计数。"""
    rows = []
    for cls in ABLATION_VARIANTS:
        m = cls(hidden_channels=hidden_channels, dropout_rate=0.1)
        rows.append((cls.label, count_parameters(m)))
    return rows
