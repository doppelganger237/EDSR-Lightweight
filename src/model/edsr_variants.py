
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
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, use_attention=False):
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
            self.att = ULAttentionPlus(n_feats, gamma_init=0.1)
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
        
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x


# --- 轻量深度卷积模块 ---
class SDWConv(nn.Module):
    """SDWConv：轻量深度卷积"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)


    def forward(self, x):
        out = self.pw(self.dw(x))
        # Mean+Std通道增强
        mean = out.mean(dim=(2, 3), keepdim=True)
        std = out.std(dim=(2, 3), keepdim=True)
        y = mean + std
        return out * (1 + 0.2 * y)


# --- 统一通道与空间注意力模块 ---

class ULAttentionPlus(nn.Module):
    """
    串行注意力模块（CA→LayerNorm→SA→Residual Add）：
    - 通道注意力：ECA + 均值+方差
    - 中间归一化：LayerNorm 稳定特征分布
    - 空间注意力：avg+max pooling -> DWConv+PWConv -> h-sigmoid
    - 残差融合输出
    """
    def __init__(self, channel, gamma_init=0.1):
        super().__init__()
        # 通道注意力（ECA）
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # LayerNorm
        self.norm = BiasFreeLayerNorm(channel)
        # 空间注意力（轻量结构）
        self.sa_dw = nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False)
        self.sa_pw = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        # --- 通道注意力 ---
        mean = x.mean(dim=(2,3), keepdim=True)
        std = x.std(dim=(2,3), keepdim=True)
        y = mean + std
        y = y.squeeze(-1).permute(0, 2, 1)  # B x 1 x C
        y = self.eca_conv(y)
        ca_map = torch.sigmoid(y.permute(0, 2, 1).unsqueeze(-1))  # B x C x 1 x 1
        ca_map = ca_map.expand_as(x)
        x_ca = x * ca_map

        # --- LayerNorm ---
        x_ca = self.norm(x_ca)

        # --- 空间注意力 ---
        avg = torch.mean(x_ca, dim=1, keepdim=True)
        mx, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = torch.cat([avg, mx], dim=1)
        sa = self.sa_pw(self.sa_dw(sa))
        sa = F.hardsigmoid(sa)
        x_sa = x_ca * sa

        # --- 残差融合 ---
        out = x + x_sa
        return out

# --- 残差块结构 ---
class SDWResidualBlock(nn.Module):
    """SDWResidualBlock：残差块，包含两层轻量卷积、高频增强及统一注意力"""
    def __init__(self, channels, kernel_size=3, res_scale=1,
                 use_dwconv=True, use_attention=True):
        super().__init__()
        self.res_scale = res_scale
        self.use_dwconv = use_dwconv
        self.use_attention = use_attention
        padding = kernel_size // 2

        if use_dwconv:
            self.conv1 = SDWConv(channels)
            self.conv2 = SDWConv(channels)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)

        if self.use_attention:
            self.att = ULAttentionPlus(channels, gamma_init=0.1)
            self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            self.alpha = nn.Parameter(torch.ones(channels, 1, 1) * 0.1)
        else:
            self.att = None
            self.proj_out = None
            self.alpha = None

        # 固定拉普拉斯高频增强卷积核
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

        # 高频增强：固定拉普拉斯高频增强
        freq_out = F.conv2d(x, self.lap_kernel.repeat(x.size(1), 1, 1, 1), padding=1, groups=x.size(1))
        out = out + self.hf_scale * freq_out

        out = out * self.res_scale

        if self.use_attention and self.att is not None:
            y_att = self.att(out)
            y = self.proj_out(y_att)
            scaled_alpha = self.alpha
            return x + out + scaled_alpha * y
        else:
            return x + out


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
            for i in range(n_resblocks):
                use_dw_block = (i >= n_resblocks // 2)
                m_body.append(
                    SDWResidualBlock(
                        n_feats, kernel_size=kernel_size, res_scale=args.res_scale,
                        use_dwconv=use_dw_block, use_attention=use_att
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

        # 残差块堆栈
        res = self.body(x)
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
