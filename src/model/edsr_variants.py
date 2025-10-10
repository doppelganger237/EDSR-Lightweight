import torch.nn as nn
from model import common

# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    """
    实现深度可分离卷积（Depthwise Separable Convolution）模块。
    该模块将标准卷积分解为两步：
    1. 深度卷积（depthwise convolution）：对每个输入通道单独进行卷积，提取空间特征。
    2. 点卷积（pointwise convolution）：使用1x1卷积融合不同通道的信息。
    这样做可以显著减少参数量和计算复杂度，同时保持较好的特征表达能力。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        # 深度卷积：groups设置为输入通道数，保证每个通道独立卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        # 点卷积：1x1卷积，用于通道融合
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        # 先进行深度卷积，再进行点卷积
        return self.pointwise(self.depthwise(x))

# 通道注意力
class CALayer(nn.Module):
    """
    通道注意力机制模块（Channel Attention Layer）。
    通过自适应平均池化提取全局通道描述，然后通过两个全连接层（使用1x1卷积实现）
    形成通道权重，最后用Sigmoid激活，得到每个通道的注意力权重。
    该权重乘以输入特征，实现对重要通道的增强，从而提升模型表达能力。
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输出尺寸为1x1
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),  # 升维回原通道数
            nn.Sigmoid()  # 归一化权重到[0,1]
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 全局信息聚合
        y = self.conv_du(y)   # 生成通道注意力权重
        return x * y          # 加权输入特征，实现通道注意力

# 通用残差块
class ResidualBlock(nn.Module):
    """
    残差块模块，支持普通卷积或深度可分离卷积，并可选通道注意力机制。
    结构为：卷积 -> 激活 -> 卷积 -> (可选通道注意力) -> 残差连接。
    通过残差连接缓解深层网络训练困难，res_scale用于调整残差的强度。
    """
    def __init__(self, conv, n_feats, kernel_size, act, res_scale,
                 use_ca=False, use_dwconv=False):
        super().__init__()
        # 根据use_dwconv选择卷积类型：普通卷积或深度可分离卷积
        conv_fn = DepthwiseSeparableConv if use_dwconv else conv
        body = [
            conv_fn(n_feats, n_feats, kernel_size),  # 第一卷积层
            act,                                    # 激活函数（ReLU）
            conv_fn(n_feats, n_feats, kernel_size)  # 第二卷积层
        ]
        self.body = nn.Sequential(*body)
        # 是否使用通道注意力机制
        self.ca = CALayer(n_feats) if use_ca else None
        self.res_scale = res_scale  # 残差缩放因子

    def forward(self, x):
        res = self.body(x)          # 卷积和激活操作
        if self.ca:
            res = self.ca(res)      # 通道注意力加权
        res = res.mul(self.res_scale)  # 缩放残差
        return res + x              # 残差连接，输出

# EDSR 可选模块版本
class EDSR_Variant(nn.Module):
    """
    EDSR模型的变体版本，支持使用深度可分离卷积和通道注意力机制。
    结构包括：
    - 子均值处理（输入归一化）
    - Head：输入卷积层
    - Body：多个残差块组成的残差网络
    - Tail：上采样模块和输出卷积层
    - 加均值处理（输出反归一化）
    """
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        n_resblocks = args.n_resblocks  # 残差块数量
        n_feats = args.n_feats          # 特征通道数
        kernel_size = 3                 # 卷积核大小
        scale = args.scale[0]           # 放大倍数
        act = nn.ReLU(True)             # 激活函数

        self.sub_mean = common.MeanShift(args.rgb_range)          # 输入图像均值归一化
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)  # 输出图像反归一化

        # Head部分，输入卷积层
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # Body部分，由多个残差块组成
        m_body = [
            ResidualBlock(conv, n_feats, kernel_size, act=act,
                          res_scale=args.res_scale,
                          use_ca=getattr(args, "use_ca", False),         # 是否使用通道注意力
                          use_dwconv=getattr(args, "use_dwconv", False)) # 是否使用深度可分离卷积
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))  # 残差块后附加一个卷积层

        # Tail部分，上采样模块和输出卷积层
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)  # 头部模块
        self.body = nn.Sequential(*m_body)  # 主体残差模块
        self.tail = nn.Sequential(*m_tail)  # 尾部模块

    def forward(self, x):
        x = self.sub_mean(x)  # 输入归一化
        x = self.head(x)      # 头部卷积

        res = self.body(x)    # 主体残差网络
        res += x              # 残差连接

        x = self.tail(res)    # 上采样和输出卷积
        x = self.add_mean(x)  # 输出反归一化
        return x

def make_model(args, parent=False):
    net = EDSR_Variant(args)
    check_modules(net)  # 在构建模型时打印是否使用了 DWConv/CA
    return net

import torch.nn as nn

def check_modules(net):
    has_dwconv = any(isinstance(m, nn.Conv2d) and m.groups == m.in_channels for m in net.modules())
    has_ca = any(m.__class__.__name__ == "CALayer" for m in net.modules())
    print(f"Use DWConv: {has_dwconv}")
    print(f"Use CA: {has_ca}")