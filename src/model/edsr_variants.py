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
class EfficientFeatureExpand(nn.Module):
    """高效特征扩展模块（轻量化特征生成）
    功能：
    1) primary_conv: 生成初始通道特征
    2) cheap_operation: 使用深度卷积生成剩余特征
    最终输出通道数为 out_channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, relu=True, bias=False):
        super().__init__()
        import math as _math
        self.out_channels = out_channels
        adaptive_ratio = 2 if out_channels < 128 else 4
        self.init_channels = int(_math.ceil(out_channels / adaptive_ratio))
        new_channels = self.init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.init_channels, kernel_size, padding=kernel_size//2, bias=bias),
            nn.Hardswish(inplace=True) if relu else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.init_channels, new_channels, dw_kernel_size, padding=dw_kernel_size//2, groups=self.init_channels, bias=bias),
            nn.Hardswish(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)  # 主卷积生成初始特征
        x2 = self.cheap_operation(x1)  # 使用轻量操作生成额外特征
        out = torch.cat([x1, x2], dim=1)  # 拼接特征
        return out[:, :self.out_channels, :, :]  # 保持输出通道数一致




# EfficientDepthConvBlock: 极简轻量卷积模块
class EfficientDepthConvBlock(nn.Module):
    """
    轻量卷积模块
    - 使用深度卷积提取空间特征
    - 使用高效特征扩展生成通道
    - 激活函数采用 Hardswish
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, bias=True):
        super().__init__()
        padding = kernel_size // 2
        # 1) 深度卷积空间滤波
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        # 2) 高效特征扩展点卷积
        self.ghost = EfficientFeatureExpand(in_channels, out_channels, kernel_size=1, ratio=ratio, dw_kernel_size=3, relu=True, bias=bias)
        # 3) 激活函数
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.dw(x)  # 深度卷积提取空间信息
        x = self.ghost(x)  # 高效特征扩展
        return self.act(x)  # 激活输出


###############################################################################
# 新增轻量多尺度通道-空间注意力模块
###############################################################################
class LiteAttention(nn.Module):
    """
    轻量多尺度通道-空间注意力模块
    功能：
    - 通道注意力：自适应调整通道重要性
    - 空间注意力：深度卷积增强空间特征
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(channel, max(channel // reduction, 4), 1, bias=True),
            nn.Hardswish(),
            nn.Conv2d(max(channel // reduction, 4), channel, 1, bias=True),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(self.avg_pool(x))  # 通道注意力
        x = x * ca  # 对输入特征加权
        sa = self.spatial_att(x)  # 空间注意力
        return x * sa  # 输出加权特征

###############################################################################
# 新增统一轻量残差块
###############################################################################
class AdaptiveResidualBlock(nn.Module):
    """
    自适应残差块
    - 可选择是否使用轻量卷积 DWConv
    - 可选择是否使用 LiteAttention
    - 动态残差缩放 + skip 调整
    """
    def __init__(self, channels, kernel_size=3, res_scale=1.0,
                 use_attention=True, use_dwconv=True):
        super().__init__()
        conv_fn = (lambda in_c, out_c, k: EfficientDepthConvBlock(in_c, out_c, k)) \
            if use_dwconv else (lambda in_c, out_c, k: nn.Conv2d(in_c, out_c, k, padding=k//2, bias=True))
        self.conv1 = conv_fn(channels, channels, kernel_size)
        self.act = nn.Hardswish(inplace=True)
        self.conv2 = conv_fn(channels, channels, kernel_size)
        self.use_attention = use_attention
        if use_attention:
            self.att = LiteAttention(channels)
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale), dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.conv1(x)  # 第一层卷积
        res = self.act(res)  # 激活
        res = self.conv2(res)  # 第二层卷积
        if self.use_attention:
            res = self.att(res)  # 注意力加权
        res = res * (0.5 + 0.5 * self.sigmoid(self.res_scale))  # 残差缩放
        return x + 0.8 * res  # skip连接输出

###############################################################################
# EfficientEDSR 主模型
# - 支持DWConv、轻量注意力、局部特征增强等
# - Head: 输入卷积
# - Body: 多个轻量残差块
# - GFA: 全局特征聚合
# - Tail: 上采样输出
###############################################################################
class EfficientEDSR(nn.Module):
    """
    EfficientEDSR 主模型
    - Head: 输入卷积
    - Body: 多个轻量残差块
    - Tail: 上采样输出
    """
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # Head部分
        m_head = [
            nn.Conv2d(args.n_colors, args.n_colors, 3, padding=1, groups=args.n_colors, bias=True),
            nn.Hardswish(inplace=True),
            nn.Conv2d(args.n_colors, n_feats, 1, bias=True)
        ]

        # Body部分
        kernel_size_list = getattr(args, 'kernel_size_list', None)
        if kernel_size_list is None:
            kernel_size_list = [3] * n_resblocks
        else:
            if len(kernel_size_list) != n_resblocks:
                raise ValueError("kernel_size_list长度必须等于n_resblocks")

        enable_dw = getattr(args, "use_dwconv", True)
        enable_attention = getattr(args, "use_attention", True)

        m_body = []
        for i in range(n_resblocks):
            m_body.append(
                AdaptiveResidualBlock(
                    n_feats,
                    kernel_size_list[i],
                    res_scale=args.res_scale,
                    use_attention=enable_attention,
                    use_dwconv=enable_dw
                )
            )
        m_body.append(conv(n_feats, n_feats, 3))

        # Tail部分
        m_tail = [
            nn.Conv2d(n_feats, n_feats, 1, bias=True),  # 新增特征聚合层
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 1, padding=0)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # 输入: [B, C, H, W]
        x = self.sub_mean(x)  # 减去均值
        x = self.head(x)  # 头部卷积
        res = self.body(x)  # 主体残差块
        res += x  # 全局残差连接
        x = self.tail(res)  # 上采样输出
        x = self.add_mean(x)  # 加回均值
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
                        raise RuntimeError('复制参数时出现错误，参数名: {}, 模型维度: {}, 权重维度: {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('state_dict中存在意外的键 "{}"'
                                   .format(name))
###############################################################################
# 模型构建工厂函数
# - 根据args实例化EfficientEDSR并打印关键模块信息
###############################################################################
import torch
def make_model(args, parent=False):
    """
    构造EfficientEDSR模型的工厂函数。
    - 根据参数实例化EfficientEDSR模型。
    - 构建完成后调用check_modules函数，打印模型是否使用了关键轻量模块。
    """
    torch.manual_seed(42)
    net = EfficientEDSR(args)
    check_modules(net)  # 打印模型中使用的关键模块信息
    return net

###############################################################################
# 关键模块检查工具
# - 检查模型是否包含EfficientDepthConvBlock、LiteAttention及并行融合
# - 便于论文展示与模型对比
###############################################################################
def check_modules(net):
    """检查模型中是否包含关键轻量模块
    - EfficientDepthConvBlock
    - LiteAttention
    用于论文展示和 Ablation Study
    """
    has_dwconv = any(isinstance(m, EfficientDepthConvBlock) for m in net.modules())
    has_msgsa = any(isinstance(m, LiteAttention) for m in net.modules())
    print(f"[INFO] 使用 EfficientDepthConvBlock: {has_dwconv}")
    print(f"[INFO] 使用 LiteAttention: {has_msgsa}")