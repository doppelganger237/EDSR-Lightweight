###############################################################################
# ===== EDSR变体：轻量级残差块与注意力模块 =====
###############################################################################
import torch.nn as nn
import torch
from model import common
import math


###############################################################################
# 深度可分离卷积模块（Depthwise Separable Convolution）
# - 该模块极大减少参数量与计算量，适合轻量化模型
###############################################################################
class DepthwiseSeparableConv(nn.Module):
    """精简版DWConv模块：DWConv + PWConv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


###############################################################################
# 全局特征聚合模块（Global Feature Aggregation, GFA）
# - 轻量级全局特征聚合。简化模式下仅用全局平均池化，否则平均+最大池化拼接
# - 生成全局通道权重，对输入特征进行加权
###############################################################################



###############################################################################
# MSGSA模块（Multi-Scale Global-Spatial Attention）
# - 融合MSGAv1（多尺度全局通道增强）与LESA（空间增强）
###############################################################################
class MSGSA(nn.Module):
    """
    多尺度全局-空间注意力模块（MSGSA）。
    - 通道增强: 融合ECA、CALayer、GFA（MSGAv1）。
    - 空间增强: LESA。
    - 输出: 输入特征乘以融合后的全局通道与空间权重。
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        # MSGAv1分支（通道增强）
        # ECA分支
        k = int(abs((math.log2(channel) / 2) + 1))
        if k % 2 == 0:
            k += 1  # 保证卷积核为奇数
        self.eca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # CALayer分支
        self.ca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv1 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)
        self.ca_relu = nn.ReLU(inplace=True)
        self.ca_conv2 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)
        # GFA分支（简化版）
        self.gfa_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gfa_conv = DepthwiseSeparableConv(channel, channel, kernel_size=1, bias=True)
        # LESA空间增强
        self.lesa = LESA(channel)
        # 融合
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道增强（MSGAv1融合）
        y_eca = self.eca_avg_pool(x)
        y_eca = y_eca.squeeze(-1).transpose(-1, -2)
        y_eca = self.eca_conv(y_eca)
        y_eca = self.sigmoid(y_eca).transpose(-1, -2).unsqueeze(-1)
        y_ca = self.ca_avg_pool(x)
        y_ca = self.ca_conv1(y_ca)
        y_ca = self.ca_relu(y_ca)
        y_ca = self.ca_conv2(y_ca)
        y_ca = self.sigmoid(y_ca)
        y_gfa = self.gfa_avg_pool(x)
        y_gfa = self.gfa_conv(y_gfa)
        y_gfa = self.sigmoid(y_gfa)
        # 通道融合
        y_channel = (y_eca + y_ca + y_gfa) / 3
        x_c = x * y_channel
        # 空间增强
        out = self.lesa(x_c)
        return out

###############################################################################
# LESA（Lightweight Enhanced Spatial Attention）模块
# - 轻量空间注意力，替代传统ESA
# - 通道池化后拼接，DWConv提取空间权重
###############################################################################
class LESA(nn.Module):
    """
    轻量增强空间注意力模块。
    - 输入: [B, C, H, W]
    - 通道最大池化与平均池化后拼接，通过DWConv提取空间权重，Sigmoid归一化。
    - 输出: x + x * att，实现空间增强。
    """
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        self.conv_1x1_a = DepthwiseSeparableConv(2, 2, kernel_size=1, bias=False)
        self.conv_1x1_b = DepthwiseSeparableConv(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入: [B, C, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # [B,1,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)        # [B,1,H,W]
        x_cat = torch.cat([max_out, avg_out], dim=1)        # [B,2,H,W]
        att = self.conv_1x1_a(x_cat)                        # [B,2,H,W]
        att = self.conv_1x1_b(att)                          # [B,1,H,W]
        att = self.sigmoid(att)                             # [B,1,H,W]
        return x + x * att                                  # 空间增强

# 移除SALayer、CALayer独立分支（已整合进MSGSA），保留LESA


###############################################################################
# 激活函数辅助函数
# - 支持silu、gelu、mish
###############################################################################
def get_activation(name):
    """
    获取激活函数模块。
    支持: 'silu', 'gelu', 'mish'，默认silu。
    """
    name = str(name).lower()
    if name == 'silu':
        return nn.SiLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'mish':
        class Mish(nn.Module):
            def forward(self, x):
                return x * torch.tanh(nn.functional.softplus(x))
        return Mish()
    else:
        return nn.SiLU(inplace=True)

###############################################################################
# HARB（Hybrid Attention Residual Block，混合注意力残差块）
# - 内含MSGAv1（通道增强）+ LESA（空间增强）
# - 主体卷积可选DWConv
# - 统一输出结构
###############################################################################
class HybridAttentionResidualBlock(nn.Module):
    """
    混合注意力残差块（HARB）：
    - 主体卷积可选DWConv
    - MSGSA: 通道+空间增强一体化
    - 残差缩放
    """
    def __init__(self, conv, block_feats, kernel_size, act, res_scale,
                 use_dwconv=False, use_attention=True):
        super().__init__()
        # 主体卷积函数
        conv_fn = (lambda in_c, out_c, k: DepthwiseSeparableConv(block_feats, block_feats, k)) if use_dwconv else (lambda in_c, out_c, k: conv(block_feats, block_feats, k))
        # 主体卷积序列
        self.body = nn.Sequential(
            conv_fn(block_feats, block_feats, kernel_size),
            nn.SiLU(inplace=True),
            conv_fn(block_feats, block_feats, kernel_size)
        )
        # MSGSA注意力
        self.use_attention = use_attention
        if use_attention:
            self.msgsa = MSGSA(block_feats)
        # 残差缩放
        self.res_scale = nn.Parameter(torch.ones(block_feats) * res_scale)

    def forward(self, x):
        res = self.body(x)
        if self.use_attention:
            res = self.msgsa(res)
        # 残差缩放
        if res.dim() == 4 and self.res_scale.dim() == 1:
            res = res * self.res_scale.view(1, -1, 1, 1)
        else:
            res = res.mul(self.res_scale)
        res = res + x
        return res

###############################################################################
# EDSR_Variant 主模型
# - 支持DWConv、轻量注意力、局部特征增强等
# - Head: 输入卷积
# - Body: 多个轻量残差块
# - GFA: 全局特征聚合
# - Tail: 上采样输出
###############################################################################
class EDSR_Variant(nn.Module):
    """
    EDSR模型变体，支持深度可分离卷积和MSGSA注意力机制。
    - Head: 输入卷积
    - Body: 多个轻量残差块
    - Tail: 上采样输出
    - 输入输出: [B, C, H, W]
    """
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]

        # 激活函数选择
        act_post = get_activation(getattr(args, 'act', 'silu'))

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # Head部分
        m_head = [conv(args.n_colors, n_feats, 3)]

        # Body部分
        kernel_size_list = getattr(args, 'kernel_size_list', None)
        if kernel_size_list is None:
            kernel_size_list = [3] * n_resblocks
        else:
            if len(kernel_size_list) != n_resblocks:
                raise ValueError("kernel_size_list length must be equal to n_resblocks")

        enable_dw = getattr(args, "use_dwconv", False)
        enable_attention = getattr(args, "use_attention", True)

        m_body = []
        for i in range(n_resblocks):
            m_body.append(
                HybridAttentionResidualBlock(
                    conv, n_feats, kernel_size_list[i], act=act_post,
                    res_scale=args.res_scale,
                    use_dwconv=enable_dw,
                    use_attention=enable_attention
                )
            )
        m_body.append(conv(n_feats, n_feats, 3))

        # Tail部分
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # 输入: [B, C, H, W]
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x
    
    # 这是自定义的权重加载函数：
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
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
###############################################################################
# 模型构建工厂函数
# - 根据args实例化EDSR_Variant并打印关键模块信息
###############################################################################
import torch
def make_model(args, parent=False):
    """
    构造EDSR变体模型的工厂函数。
    - 根据参数实例化EDSR_Variant模型。
    - 构建完成后调用check_modules函数，打印模型是否使用了关键轻量模块。
    """
    torch.manual_seed(42)
    net = EDSR_Variant(args)
    #check_modules(net)  # 打印模型中使用的关键模块信息
    return net

###############################################################################
# 关键模块检查工具
# - 检查模型是否包含DWConv、CA、SA及并行融合
# - 便于论文展示与模型对比
###############################################################################
def check_modules(net):
    """
    检查模型中是否包含关键轻量化模块，并打印结果。
    - 判断是否使用深度可分离卷积（groups等于输入通道数的Conv2d）。
    - 判断是否使用MSGSA（多尺度全局-空间注意力）。
    - 判断是否使用空间注意力模块（LESA）。
    - 判断是否在HARB中使用MSGSA。
    输出格式优化，适合论文展示。
    """
    has_dwconv = any(isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.kernel_size != (1,1)
                     for m in net.modules())
    has_msgsa = any(m.__class__.__name__ == "MSGSA" for m in net.modules())
    has_lesa = any(m.__class__.__name__ == "LESA" for m in net.modules())
    has_harb = any(m.__class__.__name__ == "HybridAttentionResidualBlock" for m in net.modules())
    print(f"[INFO] Use DWConv: {has_dwconv}")
    print(f"[INFO] Use MSGSA: {has_msgsa}")
    print(f"[INFO] Use LESA: {has_lesa}")
    print(f"[INFO] Use HARB (MSGSA): {has_harb}")