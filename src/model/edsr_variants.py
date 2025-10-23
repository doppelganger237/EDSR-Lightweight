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

class ResBlock(nn.Module):
    """ResBlock: 标准残差块，支持可选注意力模块"""
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.SiLU(True), res_scale=1, use_attention=False):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.use_attention = use_attention
        if self.use_attention:
            self.att = ULAttentionPlus(n_feats)
            self.proj_out = nn.Conv2d(n_feats, n_feats, kernel_size=1, bias=True)
            self.alpha = nn.Parameter(torch.ones(n_feats, 1, 1) * 0.1)
        else:
            self.att = None
            self.proj_out = None
            self.alpha = None

    def forward(self, x):
        res = self.body(x)
        res = res * self.res_scale
        if self.use_attention and self.att is not None:
            y_att = self.att(res)
            y = self.proj_out(y_att)
            scaled_alpha = self.alpha
            return x + res + scaled_alpha * y
        else:
            return x + res
        


# --- LayerNorm2d 替换 GroupNorm ---
class LayerNorm2d(nn.Module):
    """LayerNorm for Conv2d feature maps (B, C, H, W)"""
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


# --- 轻量深度卷积模块 ---
class SDWConv(nn.Module):
    """SDWConv：轻量深度卷积"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)


    def forward(self, x):
        out = self.dw(x)
        out = F.silu(out, inplace=True)
        out = self.pw(out)
        out = F.silu(out, inplace=True)
        # Mean+Std通道增强，使用var_mean替换
        var_mean = torch.var_mean(out, dim=(2, 3), keepdim=True, unbiased=False)
        y = var_mean[1] + var_mean[0].sqrt()
        return out * (1 + 0.2 * y)


class NAFBlock(nn.Module):
    def __init__(self, c, expansion=2):
        super().__init__()
        hidden = c * expansion
        self.norm = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, hidden * 2, 1, bias=True)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=5, padding=2, groups=hidden * 2, bias=True)
        self.sg = SimpleGate()
        self.pw2 = nn.Conv2d(hidden, c, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        res = self.norm(x)
        res = self.pw1(res)
        res = self.dw(res)
        res = self.sg(res)
        res = self.pw2(res)
        return x + self.beta * res


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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
        # 通道注意力（ECA）
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # LayerNorm2d 替代 GroupNorm
        self.norm = LayerNorm2d(channel)
        # 空间注意力（SimAM）
        self.simam = SimAM()

    def forward(self, x):
        b, c, h, w = x.size()
        # --- 通道注意力 ---
        var_mean = torch.var_mean(x, dim=(2,3), keepdim=True, unbiased=False)
        y = var_mean[1] + var_mean[0].sqrt()
        y = y.squeeze(-1).permute(0, 2, 1)  # B x 1 x C
        y = self.eca_conv(y)
        ca_map = torch.sigmoid(y.permute(0, 2, 1).unsqueeze(-1))  # B x C x 1 x 1
        ca_map = ca_map.expand_as(x)
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

# --- 轻量特征蒸馏模块（低参数版本） ---
class FeatureDistillationBlock(nn.Module):
    """融合版轻量蒸馏模块：带1×1通道融合"""
    def __init__(self, c, dist_ratio=0.25):
        super().__init__()
        d_c = int(c * dist_ratio)
        self.fuse = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(c, d_c, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(d_c, c, kernel_size=3, padding=1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_fuse = self.act(self.fuse(x))
        distilled = self.act(self.conv1(x_fuse))
        fused = self.act(self.conv2(distilled))
        return x + fused

# 新增 Fusion1x1 类
class Fusion1x1(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fuse = nn.Conv2d(c * 2, c, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.fuse(out)
        out = self.act(out)
        return out

# --- 残差块结构 ---
class SDWResidualBlock(nn.Module):
    """SDWResidualBlock：残差块，包含两层轻量卷积、高频增强及统一注意力"""
    def __init__(self, channels, kernel_size=3, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        padding = kernel_size // 2

        # 第一层保持3x3（局部特征）
        self.conv1 = SDWConv(channels, kernel_size=3)
        # 第二层使用 dilated DWConv (5×5, dilation=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2*2, dilation=2, groups=channels, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True)
        )

        self.att = ULAttentionPlus(channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.ones(channels, 1, 1) * 0.1)

        # 创建固定拉普拉斯高频增强卷积核
        lap_kernel = torch.tensor([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap_kernel.unsqueeze(0).unsqueeze(0))
        # 固定高频增强强度，稳定训练
        self.hf_scale = 0.05  # 固定高频增强强度，稳定训练

    def forward(self, x):
        # 第一层轻量卷积
        out = self.conv1(x)
        # 第二层轻量卷积
        out = self.conv2(out)

        # 高频增强
        freq_out = F.conv2d(x, self.lap_kernel.repeat(x.size(1), 1, 1, 1), padding=1, groups=x.size(1))
        out = out + self.hf_scale * freq_out

        out = out * self.res_scale

        y_att = self.att(out)
        y = self.proj_out(y_att)
        scaled_alpha = self.alpha
        return x + out + scaled_alpha * y


# --- 主干网络 ---
class ULRNet(nn.Module):
    """ULRNet：统一轻量超分网络（Head → Body → Tail）"""
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_resblocks = args.n_resblocks
        kernel_size = 3
        n_feats = args.n_feats
        scale = args.scale[0]

        use_dw = args.use_dwconv
        use_att = args.use_attention

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # 网络头部：浅层特征提取
        m_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=1, bias=True)]

        # 网络主体：残差块堆叠 + 额外卷积
        m_body = []
        if use_dw:
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
        else:
            for i in range(n_resblocks):
                m_body.append(
                    ResBlock(
                        conv, n_feats, kernel_size, res_scale=args.res_scale, use_attention=use_att
                    )
                )
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # 网络尾部：上采样与颜色恢复
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # 输入预处理：减均值
        x = self.sub_mean(x)
        # 浅层特征提取
        x = self.head(x)

        # 分割 body 为前半段和后半段
        n_resblocks = len([m for m in self.body if isinstance(m, (NAFBlock, SDWResidualBlock))])
        half = n_resblocks // 2
        naf_out = self.body[:half](x)
        sdw_out = self.body[half:-1](naf_out)
        fused = self.fusion(naf_out, sdw_out)
        res = self.body[-1](fused)

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
