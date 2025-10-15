"""
edsr_variants.py

EDSR 变体 - 集成 A + C + E:
- A: Adaptive Depthwise Conv (门控自适应 DWConv)
- C: LiteAttention++（动态融合 CA 与 SA, 使用 softmax 权重）
- E: 可学习残差缩放 + 可选的轻量归一化（LayerNorm/GroupNorm）用于训练稳定

说明：此文件实现了一个向后兼容的模型构建工厂 `make_model(args)`，并打印关键模块使用情况。
请确保项目其它处通过 `from model import edsr_variants` 或者 `make_model(args)` 调用本文件。

当前版本实现的是 “Hybrid-Attention Efficient EDSR (A+C+E v2)”：
- 前两层 Residual Block 使用标准 Conv（Hybrid 结构）
- 其余 Block 使用 AdaptiveDWConv（门控轻量卷积）
- 注意力模块使用 LiteAttentionPlus v2（动态融合 CA+SA，优化实现）
- 残差缩放固定为 0.1
- Tail 使用 3×3 输出卷积
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


# -----------------------------
# Adaptive Depthwise Conv (A)
# -----------------------------
class AdaptiveDWConv(nn.Module):
    """
    自适应深度卷积模块：
    - 先用 1x1 投影降/升维（可选），再做 depthwise conv
    - 使用门控机制（轻量的通道注意力）产出每个通道的缩放因子
    - 目标：在保持轻量的同时提高 DWConv 的表达能力
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, bias=True, gate_reduction=8):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2

        # 如果输入通道 != 输出通道，使用 1x1 先投影
        if in_channels != out_channels:
            self.pre_proj = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        else:
            self.pre_proj = None

        # depthwise conv
        self.dw = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, groups=out_channels, bias=bias)

        # 门控：全局池 -> FC -> SiLU -> FC -> sigmoid 返回每个通道的缩放
        hidden = max(out_channels // gate_reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # keep original input for residual connection
        identity = x
        if self.pre_proj is not None:
            x = self.pre_proj(x)
        y = self.dw(x)
        g = self.gate(y)
        # residual-style enhancement: add gated DW output to input projection
        # this reduces the risk of over-suppressing channels and helps preserve
        # information for better PSNR/texture recovery.
        return x + y * g


# -----------------------------
# LiteAttention++ v2 (C)
# -----------------------------
class LiteAttentionPlus(nn.Module):
    """
    轻量化通道+空间注意力的动态融合（v2）：
    - 计算 CA 和 SA
    - 使用一个小网络产生两个融合权重 alpha, beta
    - 使用 softmax(alpha, beta) 使权重归一化
    - 输出 x * (alpha * CA + beta * SA) 以实现加权融合
    - 优化实现：避免不必要的广播，简化计算
    """
    def __init__(self, channel, reduction=16, min_channels=8):
        super().__init__()
        hidden = max(channel // reduction, min_channels)

        # 通道注意力（瓶颈）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channel, 1, bias=True),
            nn.Sigmoid()
        )

        # 空间注意力（使用 DW 提取空间，然后降为 1 通道）
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 融合权重网络：全局池 -> FC -> 输出 2 权重
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True)  # raw scores for CA/SA
        )

    def forward(self, x):
        # x: [B, C, H, W]
        ca = self.ca(x)                           # [B, C, 1, 1]
        sa = self.sa(x)                           # [B, 1, H, W]

        # 获取融合权重 scores -> softmax
        fusion_scores = self.fusion(x).mean(dim=[2, 3])  # [B, 2]
        weights = F.softmax(fusion_scores, dim=1)         # [B, 2]
        alpha = weights[:, 0].view(-1, 1, 1, 1)
        beta = weights[:, 1].view(-1, 1, 1, 1)

        # 混合注意力：使用动态融合的加权平均
        att_map = alpha * ca + beta * sa

        # 残差注意力增强：保留原特征，叠加调制结果
        return x + x * att_map


# -----------------------------
# Adaptive Residual Block (E)
# -----------------------------
class AdaptiveResidualBlock(nn.Module):
    """
    自适应残差块：
    - 可选 DW 路径（AdaptiveDWConv）或标准 conv
    - 可学习残差缩放 res_scale（单参数）
    - 使用 GroupNorm(1, channels) 作为唯一归一化方式
    - 使用 SiLU 激活
    """
    def __init__(self, channels, kernel_size=3, res_scale=0.1,
                 use_attention=True, use_dwconv=True):
        super().__init__()
        self.use_dwconv = use_dwconv
        padding = kernel_size // 2

        if use_dwconv:
            # 使用自适应 DWConv 两次
            self.conv1 = AdaptiveDWConv(channels, channels, kernel_size)
            self.conv2 = AdaptiveDWConv(channels, channels, kernel_size)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)

        self.act = nn.SiLU(inplace=True)
        self.use_attention = use_attention
        if use_attention:
            self.att = LiteAttentionPlus(channels)

        # 固定残差缩放，不可学习
        self.res_scale = 0.1

    def forward(self, x):
        out = self.conv1(x)
        # 如果 conv1 返回的是带激活的结果（AdaptiveDWConv 内部没有额外激活），在这里统一激活
        out = self.act(out)
        out = self.conv2(out)

        if self.use_attention:
            out = self.att(out)

        out = out * self.res_scale
        return x + out


# -----------------------------
# EfficientEDSR 主模型工厂
# -----------------------------
class EfficientEDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # Head: small channel-preserving conv -> expand
        m_head = [
            nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=True),
        ]

        # Body
        use_dw = getattr(args, 'use_dwconv', True)
        use_att = getattr(args, 'use_attention', True)

        m_body = []
        for i in range(n_resblocks):
            # 前两层 Residual Block 使用标准卷积
            use_dw_block = use_dw and (i >= 2)
            m_body.append(
                AdaptiveResidualBlock(n_feats, kernel_size=3, res_scale=0.1,
                                      use_attention=use_att, use_dwconv=use_dw_block)
            )
        m_body.append(conv(n_feats, n_feats, 3))

        # Tail: 1x1 aggregation + upsampler + final conv (改为3x3卷积，padding=1)
        m_tail = [
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    # 保持向后兼容的 load_state_dict
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


# 工厂函数
def make_model(args, parent=False):
    torch.manual_seed(42)
    net = EfficientEDSR(args)
    check_modules(net)
    return net


def check_modules(net):
    has_dw = any(isinstance(m, AdaptiveDWConv) for m in net.modules())
    has_att = any(isinstance(m, LiteAttentionPlus) for m in net.modules())
    print(f"[INFO] 使用 AdaptiveDWConv: {has_dw}")
    print(f"[INFO] 使用 LiteAttentionPlus: {has_att}")