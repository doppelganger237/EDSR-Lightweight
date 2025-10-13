###############################################################################
# ===== EDSR变体：轻量级残差块与注意力模块 =====
###############################################################################
import torch.nn as nn
import torch
from model import common
import math


###############################################################################
# 深度可分离卷积模块（Depthwise Separable Convolution）
# - 支持1x1->DWConv->1x1结构（use_1x1_3x3），或DWConv->PWConv结构
# - 支持DWConv+PWConv融合为单个分组卷积以提升推理效率（fuse_dw_pw）
# - 该模块极大减少参数量与计算量，适合轻量化模型
###############################################################################
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块。
    - 可选1x1->DWConv->1x1结构（use_1x1_3x3=True），适合通道变换。
    - 默认DWConv->PWConv结构，先做深度卷积再做逐点卷积。
    - 支持DWConv+PWConv融合（fuse_dw_pw=True），用单个分组卷积近似两步卷积以加速推理。
    - 输入输出形状: [B, in_channels, H, W] -> [B, out_channels, H, W]
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, use_1x1_3x3=False, fuse_dw_pw=False):
        super().__init__()
        padding = kernel_size // 2
        self.use_1x1_3x3 = bool(use_1x1_3x3)
        self.fuse_dw_pw = bool(fuse_dw_pw)

        if self.use_1x1_3x3:
            # 1x1降维 -> DWConv -> 1x1升维，适合特征压缩/扩展
            mid_channels = in_channels
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=bias),
                nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, groups=mid_channels, bias=bias),
                nn.Conv2d(mid_channels, out_channels, 1, bias=bias)
            )
            self._use_seq = True
            self._use_fused = False
        elif self.fuse_dw_pw:
            # DWConv+PWConv融合：单个分组卷积近似（加速推理，训练时建议用分步结构）
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, groups=in_channels, bias=bias
            )
            self._use_seq = False
            self._use_fused = True
        else:
            # 标准DWConv+PWConv结构
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
            self._use_seq = False
            self._use_fused = False

    def forward(self, x):
        # 前向传播：根据配置选择不同路径
        if getattr(self, '_use_seq', False):
            # 1x1->DWConv->1x1结构
            return self.seq(x)
        elif getattr(self, '_use_fused', False):
            # 融合DWConv+PWConv路径（单分组卷积）
            return self.fused_conv(x)
        else:
            # 标准DWConv+PWConv路径
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x


###############################################################################
# 全局特征聚合模块（Global Feature Aggregation, GFA）
# - 轻量级全局特征聚合。简化模式下仅用全局平均池化，否则平均+最大池化拼接
# - 生成全局通道权重，对输入特征进行加权
###############################################################################
class GFA(nn.Module):
    """
    轻量级全局特征聚合模块。
    - 输入: [B, C, H, W]
    - 简化模式: 全局平均池化->DWConv->Sigmoid->加权输入
    - 非简化: 全局平均池化+最大池化->拼接->DWConv->Sigmoid->加权输入
    """
    def __init__(self, channels, reduction=16, simplify=False, out_channels=None):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.simplify = simplify
        self.out_channels = out_channels if out_channels is not None else channels
        if simplify:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
            self.conv = DepthwiseSeparableConv(channels, self.out_channels, kernel_size=1, bias=True)
            self.sigmoid = nn.Sigmoid()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
            self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
            self.conv = DepthwiseSeparableConv(channels * 2, self.out_channels, kernel_size=1, bias=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 前向传播：根据简化模式选择分支
        if self.simplify:
            avg_out = self.avg_pool(x)          # [B, C, 1, 1]
            w = self.conv(avg_out)              # [B, C, 1, 1]
            w = self.sigmoid(w)
            return x * w                        # 全局加权
        else:
            avg_out = self.avg_pool(x)          # [B, C, 1, 1]
            max_out = self.max_pool(x)          # [B, C, 1, 1]
            w = torch.cat([avg_out, max_out], dim=1)  # [B, 2C, 1, 1]
            w = self.conv(w)                    # [B, C, 1, 1]
            w = self.sigmoid(w)
            return x * w                        # 全局加权

###############################################################################
# Efficient Channel Attention（ECA）模块
# - 通过自适应卷积核大小的1D卷积建模通道间依赖
# - 轻量高效，适合轻量网络
###############################################################################
class ECA(nn.Module):
    """
    Efficient Channel Attention模块。
    - 利用自适应1D卷积捕获通道相关性，避免全连接带来的高复杂度。
    - 卷积核大小随通道数自适应调整。
    - 输出为输入特征乘以通道权重。
    """
    def __init__(self, channel):
        super().__init__()
        k = int(abs((math.log2(channel) / 2) + 1))
        if k % 2 == 0:
            k += 1  # 保证卷积核为奇数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)  # 1D卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)  # 通道加权

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
        self.conv_1x1_a = DepthwiseSeparableConv(2, 2, kernel_size=1, bias=False, use_1x1_3x3=False)
        self.conv_1x1_b = DepthwiseSeparableConv(2, 1, kernel_size=1, bias=False, use_1x1_3x3=False)
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

# SALayer为LESA别名，兼容旧代码
class SALayer(LESA):
    """
    SALayer为LESA模块的别名，用于兼容代码调用。
    """
    def __init__(self, channel):
        super().__init__(channel)

###############################################################################
# CALayer（通道注意力层）
# - 基于ECA的快速通道注意力，后接DWConv增强局部上下文
###############################################################################
class CALayer(nn.Module):
    """
    基于ECA的快速通道注意力模块。
    - 先ECA获取通道权重，再用3x3深度可分离卷积增强局部上下文信息。
    - 输出与输入形状一致。
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.eca = ECA(channel)
        self.dws_conv = DepthwiseSeparableConv(channel, channel, kernel_size=3, bias=True)

    def forward(self, x):
        x_eca = self.eca(x)
        out = self.dws_conv(x_eca)
        return out

###############################################################################
# LightGAF（轻量门控自适应融合模块）
# - 用于CA与SA特征动态融合
###############################################################################
class LightGAF(nn.Module):
    """
    轻量门控自适应融合模块。
    - 输入CA输出与SA输出，拼接后用DWConv生成门控权重，动态融合两分支特征。
    - 输出: gate * ca_out + (1 - gate) * sa_out
    """
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = DepthwiseSeparableConv(channels * 2, channels, kernel_size=3, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, ca_out, sa_out):
        # ca_out, sa_out: [B, C, H, W]
        combined = torch.cat([ca_out, sa_out], dim=1)      # [B, 2C, H, W]
        gate = self.sigmoid(self.gate_conv(combined) + self.gate_bias)      # [B, C, H, W]
        fused = gate * ca_out + (1 - gate) * sa_out        # [B, C, H, W]
        return fused

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
# ResidualBlock（轻量级残差块）
# - 主体卷积可选DWConv，支持CA/SA注意力，支持局部特征增强（LFE）
# - 支持前半块/后半块不同结构
# - 详细数据流见forward
###############################################################################
class ResidualBlock(nn.Module):
    """
    轻量级残差块设计：
    - 主体卷积可选DWConv或普通卷积
    - 支持通道注意力（CA）、空间注意力（SA）
    - 残差缩放，支持标量或向量缩放
    - 局部特征增强（LFE）：提升特征表达
    - 前半块/后半块可选不同结构
    """
    def __init__(self, conv, block_feats, kernel_size, act, res_scale,
                 use_ca=False, use_dwconv=False, use_sa=False, lfe_kernel=3, lfe_simplify=False,
                 front_block=False):
        super().__init__()
        # 主体卷积函数：可选DWConv或普通卷积
        conv_fn = (lambda in_c, out_c, k: DepthwiseSeparableConv(block_feats, block_feats, k)) if use_dwconv else (lambda in_c, out_c, k: conv(block_feats, block_feats, k))

        # 主体卷积序列：Conv-ACT-Conv
        self.body = nn.Sequential(
            conv_fn(block_feats, block_feats, kernel_size),
            nn.SiLU(inplace=True),
            conv_fn(block_feats, block_feats, kernel_size)
        )

        # 注意力模块
        self.use_ca = use_ca
        self.use_sa = use_sa
        self.ca = CALayer(block_feats) if use_ca else None
        self.sa = LESA(block_feats) if use_sa else None

        # 残差缩放参数
        if front_block:
            # 前半块：标量缩放
            self.res_scale = nn.Parameter(torch.ones(1) * res_scale)
        else:
            # 后半块：通道向量缩放
            self.res_scale = nn.Parameter(torch.ones(block_feats) * res_scale)

        # 局部特征增强（LFE）
        if front_block:
            # 前半块LFE：仅1x1 DWConv+激活
            self.lfe = nn.Sequential(
                DepthwiseSeparableConv(block_feats, block_feats, kernel_size=1),
                nn.SiLU(inplace=True)
            )
        else:
            if lfe_simplify:
                self.lfe = nn.Sequential(
                    DepthwiseSeparableConv(block_feats, block_feats, kernel_size=1),
                    nn.SiLU(inplace=True)
                )
            else:
                self.lfe = nn.Sequential(
                    DepthwiseSeparableConv(block_feats, block_feats, kernel_size=1),
                    DepthwiseSeparableConv(block_feats, block_feats, kernel_size=lfe_kernel),
                    nn.SiLU(inplace=True)
                )

    def forward(self, x):
        # 主体卷积
        res = self.body(x)  # [B, C, H, W]

        # 注意力分支及融合
        if self.use_ca and self.use_sa:
            ca_out = self.ca(res)  # 通道注意力输出
            sa_out = self.sa(res)  # 空间注意力输出
            res = (ca_out + sa_out) / 2  # 平均融合
        elif self.ca:
            res = self.ca(res)
        elif self.sa:
            res = self.sa(res)

        # 残差缩放
        if res.dim() == 4 and self.res_scale.dim() == 1:
            res = res * self.res_scale.view(1, -1, 1, 1)
        else:
            res = res.mul(self.res_scale)

        # 残差连接+LFE增强
        res = self.lfe(res + x)
        return res

    def extra_repr(self):
        return f"use_ca={self.use_ca}, use_sa={self.use_sa}, use_dwconv={'Yes' if isinstance(self.body[0], DepthwiseSeparableConv) else 'No'}"

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
    EDSR模型变体，支持深度可分离卷积和轻量注意力机制。
    - Head: 输入卷积
    - Body: 多个轻量残差块
    - GFA: 全局特征聚合
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
        act_pre = nn.SiLU(inplace=True)  # 前半块SiLU

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # Head部分
        m_head = [conv(args.n_colors, n_feats, 3)]

        # Body部分
        kernel_size_list = getattr(args, 'kernel_size_list', None)
        if kernel_size_list is None:
            kernel_size_list = [3] * (n_resblocks // 2) + [5] * (n_resblocks - n_resblocks // 2)
        else:
            if len(kernel_size_list) != n_resblocks:
                raise ValueError("kernel_size_list length must be equal to n_resblocks")

        enable_dw = getattr(args, "use_dwconv", False)
        enable_ca = getattr(args, "use_ca", False)
        enable_sa = getattr(args, "use_sa", False)
        lfe_simplify = getattr(args, "lfe_simplify", False)
        gfa_simplify = getattr(args, "gfa_simplify", False)
        front_ratio = getattr(args, "front_ratio", 0.5)
        front_blocks = int(n_resblocks * front_ratio)
        full_model = enable_ca and enable_sa and enable_dw

        m_body = []
        prev_feats = n_feats
        for i in range(n_resblocks):
            front_block = (i < front_blocks)
            if full_model:
                # 完整模型：前半块通道压缩、无注意力，后半块注意力稀疏、通道恢复
                if front_block:
                    use_ca_block = False
                    use_sa_block = False
                    use_dwconv_block = True
                    block_feats = n_feats // 2
                    act = act_pre
                    res_scale_val = 0.1
                    lfe_simplify_block = True
                else:
                    idx_in_back = i - front_blocks
                    back_blocks = n_resblocks - front_blocks
                    attention_enable_ratio = 0.5
                    enable_attention = (idx_in_back < int(back_blocks * attention_enable_ratio))
                    use_ca_block = enable_attention
                    use_sa_block = enable_attention
                    use_dwconv_block = True
                    block_feats = n_feats
                    act = act_post
                    res_scale_val = args.res_scale
                    lfe_simplify_block = gfa_simplify or lfe_simplify
            else:
                # 单独模块实验
                use_ca_block = enable_ca
                use_sa_block = enable_sa
                use_dwconv_block = enable_dw
                block_feats = n_feats
                act = act_post
                res_scale_val = args.res_scale
                lfe_simplify_block = False
                front_block = False

            kernel_size = kernel_size_list[i] if not (full_model and front_block) else 3

            # 通道对齐（如有必要）
            if prev_feats != block_feats:
                m_body.append(nn.Conv2d(prev_feats, block_feats, 1))
                prev_feats = block_feats

            m_body.append(
                ResidualBlock(
                    conv, block_feats, kernel_size, act=act,
                    res_scale=res_scale_val,
                    use_ca=use_ca_block,
                    use_dwconv=use_dwconv_block,
                    use_sa=use_sa_block,
                    lfe_kernel=kernel_size,
                    lfe_simplify=lfe_simplify_block,
                    front_block=(front_block if full_model else False)
                )
            )
            prev_feats = block_feats
        # Body后通道对齐
        if prev_feats != n_feats:
            m_body.append(nn.Conv2d(prev_feats, n_feats, 1))
        m_body.append(conv(n_feats, n_feats, 3))

        # Tail部分
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1)
        ]


        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        gfa_simplify_actual = gfa_simplify if not full_model else True
        self.gfa = GFA(n_feats, simplify=gfa_simplify_actual, out_channels=n_feats)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # 输入: [B, C, H, W]
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        res = res + self.gfa(res)
        x = self.tail(res)
        x = self.add_mean(x)
        return x
    
   # 这是自定义的权重加载函数：
    # 如果权重维度匹配 → 正常加载
    # 如果不匹配且不是 tail 层 → 报错
    # tail 层是最后输出层（不同倍数时大小可能不同），所以可以忽略不匹配
    # 👉 这就是为什么有时候看到 strict=False，它允许“部分加载”，比如迁移学习。
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
    check_modules(net)  # 打印模型中使用的关键模块信息
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
    - 判断是否使用通道注意力模块（CALayer）。
    - 判断是否使用空间注意力模块（SALayer）。
    - 判断是否在ResidualBlock中同时使用了CA和SA实现并行融合。
    输出格式优化，适合论文展示。
    """
    has_dwconv = any(isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.kernel_size != (1,1)
                     for m in net.modules())
    has_ca = any(m.__class__.__name__ == "CALayer" for m in net.modules())
    has_sa = any(m.__class__.__name__ in ["SALayer", "LESA"] for m in net.modules())
    has_parallel = False
    for m in net.modules():
        if isinstance(m, ResidualBlock):
            if m.ca is not None and m.sa is not None:
                has_parallel = True
                break
    print(f"[INFO] Use DWConv: {has_dwconv}")
    print(f"[INFO] Use CA: {has_ca}")
    print(f"[INFO] Use SA: {has_sa}")
    print(f"[INFO] Use parallel CA+SA in ResidualBlock: {has_parallel}")