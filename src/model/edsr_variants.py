"""
Unified Lightweight Refinement Network (ULRNet) for Efficient Image Super-Resolution:
- Gated Asymmetric Depthwise Conv (GADWConv)
- Unified Lightweight Attention (ULAttention)
- Learnable Residual Scaling (LRS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
import math


class GADWConv(nn.Module):
    """
    GADWConv: Asymmetric depthwise conv with grouped 1x1 gating conv.
    The gate conv uses groups=4 when possible to reduce parameter cost.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # asymmetric depthwise: (1x3) then (3x1)
        self.dw1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels, bias=True)
        self.dw2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=True)
        self.dw_dil3 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=True)
        # use grouped 1x1 gating to reduce params when channels divisible by 4
        gate_groups = 4 if (channels % 4 == 0) else 1
        self.gate_conv = nn.Conv2d(channels, channels, 1, groups=gate_groups, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 主分支
        out1 = self.dw1(x)
        out1 = self.dw2(out1)

        # 并联的扩张卷积分支
        out2 = self.dw_dil3(x)

        # 汇合两个分支
        out = out1 + out2

        # 轻量通道注意力（Attention-Guided Gate）
        gp = x.mean(dim=(2, 3), keepdim=True)  # Global avg pooling (B,C,1,1)
        ca_scale = torch.sigmoid(gp)           # Simple ECA-like modulation
        gate = self.sigmoid(self.gate_conv(x) * ca_scale)

        # 门控融合 + 残差连接
        return out * gate + x


# --- GADWConvLite: lightweight depthwise conv + ECA1D gating ---
class GADWConvLite(nn.Module):
    """
    GADWConvLite: simplified depthwise conv with lightweight ECA-style gating.
    - 仅使用单层depthwise + ECA1D全局通道注意力
    - 参数与计算量更低，适合与ULAttention共存
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=True)
        # adaptive ECA 1D kernel size based on channel count (keep small odd kernel)
        t = int(abs((math.log2(max(1, channels)) * 2 + 1)))
        k = t if (t % 2 == 1 and t >= 1) else max(3, t + 1)
        k = min(k, channels) if channels > 1 else 3
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dw(x)
        y = x.mean(dim=(2, 3), keepdim=True)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.eca_conv(y)
        y = self.sigmoid(y.permute(0, 2, 1).unsqueeze(-1))
        out = out * y + x
        return out


class ULAttention(nn.Module):
    """
    ULAttention (改良融合版)
    - Channel: ECA-style 1D conv on pooled descriptor (very low parameter cost).
    - Spatial: avg+max concat -> small conv (3x3) producing a single-channel map.
    - Fusion: learnable temperature-softmax weights to combine CA and SA maps.
    - Returns x + out (residual-style) so it can be used block-level or stage-level.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ECA-style 1D conv kernel selection (small odd kernel)
        gamma = 2
        b = 1
        t = int(abs((math.log2(max(1, channel)) * gamma + b)))
        k = t if (t % 2 == 1 and t >= 1) else max(3, t + 1)
        # ensure kernel not larger than channel
        k = min(k, channel) if channel > 1 else 3
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

        # Spatial attention: avg + max -> small conv (2 -> 1)
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

        # learnable fusion weights and temperature
        self.fuse = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        # log temperature: initial value for tau ~0.3
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.3), dtype=torch.float32))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel branch (ECA)
        y = self.global_pool(x).view(b, c)       # (B, C)
        y = y.unsqueeze(1)                       # (B, 1, C)
        y = self.eca_conv(y)                     # (B, 1, C)
        ca = torch.sigmoid(y).view(b, c, 1, 1)   # (B, C, 1, 1)
        ca_map = ca.expand_as(x)

        # Spatial branch
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa_in = torch.cat([avg, mx], dim=1)     # (B,2,H,W)
        sa = torch.sigmoid(self.sa_conv(sa_in)) # (B,1,H,W)
        sa_map = sa.expand(b, c, h, w)

        # Fusion with temperature
        tau = torch.exp(self.log_tau) + 1e-6
        w = torch.softmax(self.fuse / tau, dim=0)
        out = x * (w[0] * ca_map + w[1] * sa_map)

        return x + out


# NOTE: StageULAttention merged into GADWResidualBlock to simplify module hierarchy
class GADWResidualBlock(nn.Module):
    """
    GADWResidualBlock: Residual block using GADWConv and optional integrated
    ULAttention (previously StageULAttention). The stage-level attention wrapper
    has been merged into this block to simplify the codebase while preserving
    functionality. If use_attention is True, an internal 1x1 projection => ULAttention =>
    1x1 projection is applied and scaled by a learnable alpha before adding to the
    block input.
    """
    def __init__(self, channels, kernel_size=3, res_scale=0.1,
                 use_dwconv=True, idx=0, use_attention=True):
        super().__init__()
        self.use_dwconv = use_dwconv
        self.use_attention = use_attention
        padding = kernel_size // 2

        if use_dwconv:
            self.conv1 = GADWConvLite(channels)
            self.conv2 = GADWConvLite(channels)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)

        # lightweight activation
        self.act = nn.ReLU6(inplace=True)

        # learnable residual scaling
        self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))

        # Integrate StageULAttention functionality here: 1x1 proj -> ULAttention -> 1x1 proj
        if self.use_attention:
            # small bottleneck/projections to keep overhead minimal
            self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            self.att = ULAttention(channels, reduction=8)
            self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            # learnable scalar to control magnitude of the attention residual
            self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            # placeholders to avoid attribute errors when attention disabled
            self.proj_in = None
            self.att = None
            self.proj_out = None
            self.alpha = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        out = out * self.res_scale

        if self.use_attention and self.att is not None:
            y = self.proj_in(out)
            y = self.att(y)
            y = self.proj_out(y)
            return x + self.alpha * y
        else:
            return x + out


class ULRNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [
            nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=True),
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.ReLU(inplace=True),
        ]

        use_dw = getattr(args, 'use_dwconv', True)
        use_att = getattr(args, 'use_attention', True)

        m_body = []
        total_blocks = n_resblocks
        for i in range(n_resblocks):
            use_dw_block = use_dw
            block = GADWResidualBlock(n_feats, kernel_size=3, res_scale=0.1,
                                           use_dwconv=use_dw_block, idx=i, use_attention=use_att)
            block.total_blocks = total_blocks
            m_body.append(block)
        m_body.append(conv(n_feats, n_feats, 3))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        # Remove global_att usage, keep None
        self.global_att = None

        m_tail = [
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1)
        ]

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        body_layers = list(self.body)
        last_conv = body_layers[-1]
        blocks = body_layers[:-1]

        out = x
        for block in blocks:
            out = block(out)
        res = last_conv(out)

        # self.global_att is None, so skip attention here
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(f"Error copying parameter {name}, model shape {own_state[name].size()}, checkpoint shape {param.size()}")
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'Unexpected key {name} in state_dict')


def make_model(args, parent=False):
    net = ULRNet(args)
    check_modules(net)
    return net


def check_modules(net):
    has_gadw = any(isinstance(m, GADWConv) for m in net.modules())
    has_ulatt = any(isinstance(m, ULAttention) for m in net.modules())
    print(f"[INFO] GADWConv: {has_gadw}, ULAttention: {has_ulatt}")
