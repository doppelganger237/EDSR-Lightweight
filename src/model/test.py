# EDSR 的 PyTorch 实现，主要结构是：
# Head：卷积，把输入图像变成特征
# Body：残差块堆叠（主要学习能力在这里）
# Tail：上采样 + 卷积，输出高分辨率图像
# load_state_dict：能自动兼容不同放大倍数/通道数的预训练模型
from model import common

import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size,
                 bias=True, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        # 第一组深度卷积 + 点卷积 + 激活
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, groups=n_feats, bias=bias))
        m.append(nn.Conv2d(n_feats, n_feats, 1, bias=bias))
        m.append(act)
        # 第二组深度卷积 + 点卷积（无激活）
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, groups=n_feats, bias=bias))
        m.append(nn.Conv2d(n_feats, n_feats, 1, bias=bias))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class ECA(nn.Module):
    def __init__(self, n_feats, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                      # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)       # [B, 1, C]
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

def make_model(args, parent=False):
    net = EDSR(args)
    check_modules(net)
    return net

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks   # 残差块数量
        n_feats = args.n_feats           # 每层特征数（通道数）
        kernel_size = 3
        scale = args.scale[0]            # 放大倍数 (x2/x3/x4)
        act = nn.ReLU(True)              # 激活函数

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        # sub_mean / add_mean → 图像归一化/反归一化（减去均值再加回来），训练更稳定。

        # define head module
        # 第一层卷积：把输入 RGB 图像（3通道）映射到 n_feats 个特征通道。
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # 由一堆 残差块 ResBlock 组成（每个包含 2 个卷积+ReLU+残差连接）。
        # 最后再加一个卷积。
        m_body = []
        for _ in range(n_resblocks):
            m_body.append(ResBlock(n_feats, kernel_size, act=act, res_scale=args.res_scale))
            m_body.append(ECA(n_feats))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        # Upsampler：上采样模块（x2/x3/x4），用像素重排（pixel shuffle）实现。
        # 最后一层卷积：把特征图转回 RGB 图像。
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    # 输入图像 → 归一化 → 卷积映射 → 残差块堆叠 → 上采样 → 输出超分辨图像。
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

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
def check_modules(net):
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")
