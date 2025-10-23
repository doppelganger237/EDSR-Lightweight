"""
ULRNet（Unified Lightweight Refinement Network）
用于高效图像超分辨率的轻量卷积网络。

核心模块：
1. SDWConv：轻量深度可分卷积 + 通道注意力（ECA）
2. ULAttentionPlus：统一通道-空间注意力模块
3. SDWResidualBlock：残差块结构，整合注意力与高频增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


# --- 轻量深度卷积模块 ---
class SDWConv(nn.Module):
    """SDWConv：轻量深度卷积"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=True),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        out = self.conv(x)
        # Mean+Std通道增强，使用var_mean替换
        var_mean = torch.var_mean(out, dim=(2, 3), keepdim=True, unbiased=False)
        y = var_mean[1] + var_mean[0].sqrt()
        return out * (1 + 0.2 * y)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """官方论文 NAFBlock 实现"""
    def __init__(self, c, expansion=2, reduction=16, e_lambda=1e-4):
        super().__init__()
        hidden = c * expansion

        # 第一条路径: PW1 -> SG1 -> DW -> ECA -> PW2
        self.pw1 = nn.Conv2d(c, hidden * 2, 1, bias=True)
        self.sg1 = SimpleGate()
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True)
        self.eca = ECA(hidden, k_size=3)
        self.pw2 = nn.Conv2d(hidden, c, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

        # 第二条路径: PW3 -> SG2 -> PW4 (FFN2)
        self.pw3 = nn.Conv2d(c, hidden * 2, 1, bias=True)
        self.sg2 = SimpleGate()
        self.pw4 = nn.Conv2d(hidden, c, 1, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        shortcut = x

        # 第一条路径
        x1 = self.pw1(x)
        x1 = self.sg1(x1)
        x1 = self.dw(x1)
        x1 = self.eca(x1)
        x1 = self.pw2(x1)
        x = shortcut + self.beta * x1

        # 第二条路径 (FFN2)
        x2 = self.pw3(x)
        x2 = self.sg2(x2)
        x2 = self.pw4(x2)
        out = x + self.gamma * x2
        return out


# --- 统一通道与空间注意力模块 ---

class ULAttentionPlus(nn.Module):
    """
    串行注意力模块（CA→GroupNorm→SA→Residual Add）：
    - 通道注意力：ECA + 均值+方差
    - 中间归一化：GroupNormLayer 稳定特征分布
    - 空间注意力：avg+max pooling -> DWConv+PWConv -> h-sigmoid
    - 残差融合输出
    """
    def __init__(self, channel):
        super().__init__()
        # 通道注意力（ECA）2D实现
        self.eca_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        # GroupNorm 替代 LayerNorm2d
        self.norm = nn.GroupNorm(1, channel, eps=1e-6, affine=True)
        # 空间注意力（SimAM）
        self.simam = SimAM()

    def forward(self, x):
        b, c, h, w = x.size()
        # --- 通道注意力 ---
        var_mean = torch.var_mean(x, dim=(2,3), keepdim=True, unbiased=False)
        y = var_mean[1] + var_mean[0].sqrt()
        ca_map = torch.sigmoid(self.eca_conv(x))
        x_ca = x * ca_map

        # --- GroupNorm ---
        x_ca = self.norm(x_ca)

        # --- 空间注意力 ---
        x_sa = self.simam(x_ca)

        # --- 残差融合 ---
        out = x + x_sa
        return out

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(energy)

# 新增 Fusion1x1 类
class Fusion1x1(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fuse = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        out = x1 + x2
        out = self.fuse(out)
        out = self.act(out)
        return out

# --- 残差块结构 ---
class SDWResidualBlock(nn.Module):
    """SDWResidualBlock：残差块，包含两层轻量卷积及 SimAM 注意力"""
    def __init__(self, channels, kernel_size=3, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        padding = kernel_size // 2

        # 第一层保持3x3（局部特征）
        self.conv1 = SDWConv(channels, kernel_size=3)
        # 第二层使用普通 3x3 深度卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # 仅使用 SimAM 空间注意力
        self.att = SimAM()

    def forward(self, x):
        # 第一层轻量卷积
        out = self.conv1(x)
        # 第二层轻量卷积
        out = self.conv2(out)

        # SimAM 空间注意力
        out = out * self.res_scale
        y = self.att(out)
        return x + out + y


# --- 主干网络 ---
class ULRNet(nn.Module):
    """ULRNet：统一轻量超分网络（Head → Body → Tail）"""
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_resblocks = args.n_resblocks
        kernel_size = 3
        n_feats = args.n_feats
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # 网络头部：浅层特征提取
        m_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=1, bias=True)]

        # 网络主体：残差块堆叠 + 额外卷积
        m_body = []
        # 前半段: 使用 NAFBlock
        for _ in range(n_resblocks // 2):
            m_body.append(NAFBlock(n_feats))
        self.fusion = Fusion1x1(n_feats)
        # 后半段: 启用 DWConv 和 Attention
        for _ in range(n_resblocks // 2, n_resblocks):
            m_body.append(
                SDWResidualBlock(
                    n_feats, kernel_size=kernel_size, res_scale=args.res_scale
                )
            )
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.body_naf = nn.Sequential(*m_body[:n_resblocks // 2])
        self.body_sdw = nn.Sequential(*m_body[n_resblocks // 2:-1])
        self.body_tail = m_body[-1]

        self.head = nn.Sequential(*m_head)
        # 删除原 self.body 定义
        # self.body = nn.Sequential(*m_body)
        # 轻量 DW+PW+PixelShuffle 上采样
        class PixelShuffleUpsample(nn.Module):
            def __init__(self, c, scale):
                super().__init__()
                self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
                self.pw = nn.Conv2d(c, c * (scale ** 2), kernel_size=1, bias=True)
                self.ps = nn.PixelShuffle(scale)
            def forward(self, x):
                x = self.dw(x)
                x = self.pw(x)
                x = self.ps(x)
                return x

        self.tail = nn.Sequential(
            PixelShuffleUpsample(n_feats, scale),
            conv(n_feats, args.n_colors, kernel_size)
        )

    def forward(self, x):
        # 输入预处理：减均值
        x = self.sub_mean(x)
        # 浅层特征提取
        x = self.head(x)

        naf_out = self.body_naf(x)
        sdw_out = self.body_sdw(naf_out)
        assert naf_out.shape == sdw_out.shape, f"Shape mismatch: {naf_out.shape} vs {sdw_out.shape}"
        fused = self.fusion(naf_out, sdw_out)
        res = self.body_tail(fused)

        # 残差连接
        res += x

        # 上采样与输出恢复
        x = self.tail(res)
        # 加均值恢复原始图像分布
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
    has_gadw = any(isinstance(m, SDWConv) for m in net.modules())
    has_ulatt = any(isinstance(m, ULAttentionPlus) for m in net.modules())
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")
    print(f"[INFO] SDWConv: {has_gadw}, ULAttentionPlus: {has_ulatt}")

    print('NAF blocks:', sum(isinstance(m, NAFBlock) for m in net.modules()))
    print('SDW blocks:', sum(isinstance(m, SDWResidualBlock) for m in net.modules()))


# SCA 模块（官方论文实现）
class SCA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return x * y


# 新增 ECA 类 (2D版本)
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, height, width)
        y = self.avg_pool(x)  # (batch, channels, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (batch, 1, channels)
        y = self.conv(y)  # (batch, 1, channels)
        y = self.sigmoid(y)  # (batch, 1, channels)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (batch, channels, 1, 1)
        return x * y.expand_as(x)
