"""
时空模型横向对比：ConvLSTM、3D-UNet、ConvLSTM-UNet 及 3DCNN 参照。

所有模型统一接口：
  - 输入 (B, T, C, H, W)，T=12，C=6
  - 输出 (B, 1, H, W)
  - __init__(in_channels, dropout_rate, hidden_channels, **model_kwargs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gws_model_variants import Simple_ST_Net, Simple_ST_Net_Plus1Enc, Simple_ST_Net_Shallow
from gws_paths import MODELS_ROOT

IN_CHANNELS = 6


class _STModelBase(nn.Module):
    """统一 permute：输入 (B,T,C,H,W) → 各子类内部格式。"""

    name: str = "base"
    label: str = "Base"
    results_subdir: str = "st_compare/base"

    def _to_bcthw(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3, 4).contiguous()


# ── ConvLSTM ──────────────────────────────────────────


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state if state is not None else self._init_state(x)
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def _init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        z = x.new_zeros(b, self.hidden_ch, h, w)
        return z, z


class ConvLSTMModel(_STModelBase):
    """多层 ConvLSTM + 2D 解码头。"""

    name = "convlstm"
    label = "ConvLSTM"
    results_subdir = "st_compare/convlstm"

    def __init__(
        self,
        in_channels: int = 6,
        dropout_rate: float = 0.1,
        hidden_channels: int = 24,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout2d(p=dropout_rate)

        cells: list[ConvLSTMCell] = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, _, h, w = x.shape
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        for ti in range(t):
            inp = x[:, ti]
            for li, cell in enumerate(self.cells):
                h_out, c_out = cell(inp, states[li])
                states[li] = (h_out, c_out)
                inp = h_out
            inp = self.dropout(inp)
        return self.decoder(inp)


# ── ConvGRU ───────────────────────────────────────────


class ConvGRUCell(nn.Module):
    """ConvGRU 单元（比 ConvLSTM 门控更少，小样本常用）。"""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv_gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad)
        self.conv_cand = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = x.new_zeros(x.size(0), self.hidden_ch, x.size(2), x.size(3))
        combined = torch.cat([x, h], dim=1)
        z, r = torch.chunk(self.conv_gates(combined), 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        h_tilde = torch.tanh(self.conv_cand(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


class ConvGRUModel(_STModelBase):
    name = "convgru"
    label = "ConvGRU"
    results_subdir = "st_compare/convgru"

    def __init__(
        self,
        in_channels: int = 6,
        dropout_rate: float = 0.1,
        hidden_channels: int = 24,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout2d(p=dropout_rate)
        cells: list[ConvGRUCell] = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            cells.append(ConvGRUCell(in_ch, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = x.shape
        states: list[torch.Tensor | None] = [None] * self.num_layers
        for ti in range(t):
            inp = x[:, ti]
            for li, cell in enumerate(self.cells):
                h_out = cell(inp, states[li])
                states[li] = h_out
                inp = h_out
            inp = self.dropout(inp)
        return self.decoder(inp)


# ── (2+1)D 分解时空卷积 ───────────────────────────────


class STConvBlock(nn.Module):
    """先空间 1×3×3，再时间 3×1×1，参数量低于完整 3D 卷积。"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Dropout3d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STConv21DModel(_STModelBase):
    """(2+1)D 分解卷积：空间-时间分离，适合小格网。"""

    name = "stconv_21d"
    label = "STConv(2+1D)"
    results_subdir = "st_compare/stconv_21d"

    def __init__(
        self,
        in_channels: int = 6,
        dropout_rate: float = 0.1,
        hidden_channels: int = 24,
        num_blocks: int = 2,
    ):
        super().__init__()
        blocks: list[STConvBlock] = []
        ch = in_channels
        for i in range(num_blocks):
            out = hidden_channels if i == 0 else hidden_channels * min(i + 1, 2)
            blocks.append(STConvBlock(ch, out, dropout_rate))
            ch = out
        self.blocks = nn.ModuleList(blocks)
        self.decoder = nn.Sequential(
            nn.Conv2d(ch, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bcthw(x)
        for blk in self.blocks:
            x = blk(x)
        x = torch.mean(x, dim=2)
        return self.decoder(x)


# ── TCN（逐格点 1D 卷积，小格网上效果差，保留供参考）────────


class TemporalConvModel(_STModelBase):
    """沿时间维 1D 卷积提取时序特征，再 2D 卷积解码（纯卷积时序模型）。"""

    name = "tcn"
    label = "TCN"
    results_subdir = "st_compare/tcn"

    def __init__(
        self,
        in_channels: int = 6,
        dropout_rate: float = 0.1,
        hidden_channels: int = 24,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        blocks: list[nn.Module] = []
        ch = in_channels
        for _ in range(num_layers):
            blocks.extend([
                nn.Conv1d(ch, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
            ])
            ch = hidden_channels
        self.temporal = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → 逐格点沿 T 做 1D 卷积
        b, t, c, h, w = x.shape
        x = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, c, t)
        x = self.temporal(x)[:, :, -1]  # 取末时刻特征
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.decoder(x)


# ── 3D U-Net ──────────────────────────────────────────


def _conv3d_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.GELU(),
        nn.Dropout3d(p=dropout),
    )


class UNet3DModel(_STModelBase):
    """轻量 3D U-Net：仅空间下采样，时间维保持。"""

    name = "unet3d"
    label = "3D-UNet"
    results_subdir = "st_compare/unet3d"

    def __init__(
        self,
        in_channels: int = 6,
        dropout_rate: float = 0.1,
        hidden_channels: int = 24,
        depth: int = 2,
    ):
        super().__init__()
        self.depth = depth
        h = hidden_channels

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = h * (2 ** i)
            self.enc_blocks.append(_conv3d_block(ch, out_ch, dropout_rate))
            self.pools.append(nn.MaxPool3d(kernel_size=(1, 2, 2)))
            ch = out_ch

        self.bottleneck = _conv3d_block(ch, ch * 2, dropout_rate)
        dec_ch = ch * 2

        self.dec_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in reversed(range(depth)):
            enc_ch = h * (2 ** i)
            self.upconvs.append(
                nn.ConvTranspose3d(dec_ch, enc_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
            self.dec_blocks.append(_conv3d_block(enc_ch * 2, enc_ch, dropout_rate))
            dec_ch = enc_ch

        self.head = nn.Conv3d(h, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bcthw(x)  # (B, C, T, H, W)
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        x = self.head(x)
        return torch.mean(x, dim=2)  # 时间维聚合 → (B, 1, H, W)


# ── ConvLSTM + U-Net ──────────────────────────────────


class ConvLSTMUNetModel(_STModelBase):
    """ConvLSTM 时序编码 + 2D U-Net 空间解码。"""

    name = "convlstm_unet"
    label = "ConvLSTM-UNet"
    results_subdir = "st_compare/convlstm_unet"

    def __init__(
        self,
        in_channels: int = 6,
        dropout_rate: float = 0.1,
        hidden_channels: int = 24,
        convlstm_layers: int = 2,
        unet_depth: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.convlstm_layers = convlstm_layers
        self.unet_depth = unet_depth

        cells: list[ConvLSTMCell] = []
        for i in range(convlstm_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        h = hidden_channels
        self.enc2d = nn.ModuleList()
        self.pools2d = nn.ModuleList()
        ch = h
        for i in range(unet_depth):
            out_ch = h * (2 ** min(i, 1))  # h → 2h，封顶 2h
            self.enc2d.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Dropout2d(p=dropout_rate),
                )
            )
            self.pools2d.append(nn.MaxPool2d(2))
            ch = out_ch

        self.bottleneck2d = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),
        )

        self.dec2d = nn.ModuleList()
        self.up2d = nn.ModuleList()
        dec_ch = ch
        for i in reversed(range(unet_depth)):
            enc_ch = h * (2 ** min(i, 1))
            self.up2d.append(nn.ConvTranspose2d(dec_ch, enc_ch, 2, stride=2))
            self.dec2d.append(
                nn.Sequential(
                    nn.Conv2d(enc_ch * 2, enc_ch, 3, padding=1),
                    nn.GELU(),
                    nn.Dropout2d(p=dropout_rate),
                )
            )
            dec_ch = enc_ch

        self.head = nn.Conv2d(h, 1, kernel_size=1)

    def _run_convlstm(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = x.shape
        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.convlstm_layers
        for ti in range(t):
            inp = x[:, ti]
            for li, cell in enumerate(self.cells):
                h_out, c_out = cell(inp, states[li])
                states[li] = (h_out, c_out)
                inp = h_out
            inp = self.dropout(inp)
        return inp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._run_convlstm(x)
        skips = []
        for enc, pool in zip(self.enc2d, self.pools2d):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck2d(x)
        for up, dec, skip in zip(self.up2d, self.dec2d, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


# ── 3DCNN 参照（复用已有实现）──────────────────────────


class STNet3DCNNShallow(_STModelBase, Simple_ST_Net_Shallow):
    """3DCNN 浅层（1 层 Encoder），参数量最小。"""

    name = "stnet_3dcnn_1enc"
    label = "3DCNN(1层·浅)"
    results_subdir = "st_compare/3dcnn_1enc"


class STNet3DCNNRef(_STModelBase, Simple_ST_Net):
    """3DCNN 基线（2 层 Encoder），用于横向对比参照。"""

    name = "stnet_3dcnn"
    label = "3DCNN(2层·基线)"
    results_subdir = "st_compare/3dcnn_baseline"


class STNet3DCNNProposed(_STModelBase, Simple_ST_Net_Plus1Enc):
    """3DCNN 本文模型（3 层加宽），用于横向对比参照。"""

    name = "stnet_3dcnn_proposed"
    label = "3DCNN(3层·本文)"
    results_subdir = "st_compare/3dcnn_proposed"


# ── 注册表 ────────────────────────────────────────────

# 论文横向对比：5 种轻量架构（不含 TCN，小格网上不稳定）
ST_COMPARE_VARIANTS: list[type[_STModelBase]] = [
    STNet3DCNNShallow,
    STNet3DCNNRef,
    ConvLSTMModel,
    ConvGRUModel,
    STConv21DModel,
]

_ALL_MODEL_CLASSES = [
    *ST_COMPARE_VARIANTS,
    TemporalConvModel,
    UNet3DModel,
    ConvLSTMUNetModel,
    STNet3DCNNProposed,
]
ST_MODEL_REGISTRY: dict[str, type[_STModelBase]] = {cls.name: cls for cls in _ALL_MODEL_CLASSES}


def build_model(model_cls: type[_STModelBase], shared: dict, specific: dict | None = None) -> nn.Module:
    """用共享超参 + 模型专属超参实例化模型。"""
    specific = specific or {}
    kwargs = {
        "in_channels": IN_CHANNELS,
        "dropout_rate": shared["dropout_rate"],
        "hidden_channels": shared["hidden_channels"],
        **specific,
    }
    return model_cls(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def results_dir_for(model_cls: type[_STModelBase], root: str = MODELS_ROOT) -> str:
    return f"{root.rstrip('/')}/{model_cls.results_subdir}"
