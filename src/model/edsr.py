# EDSR 的 PyTorch 实现，主要结构是：
# Head：卷积，把输入图像变成特征
# Body：残差块堆叠（主要学习能力在这里）
# Tail：上采样 + 卷积，输出高分辨率图像
# load_state_dict：能自动兼容不同放大倍数/通道数的预训练模型
from model import common

import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}
# r16f64x2 → 16个残差块，每层64个特征，放大2倍
# r32f256x2 → 32个残差块，每层256个特征，放大2倍

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks   # 残差块数量
        n_feats = args.n_feats           # 每层特征数（通道数）
        kernel_size = 3
        scale = args.scale[0]            # 放大倍数 (x2/x3/x4)
        act = nn.ReLU(True)              # 激活函数
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        # sub_mean / add_mean → 图像归一化/反归一化（减去均值再加回来），训练更稳定。

        # define head module
        # 第一层卷积：把输入 RGB 图像（3通道）映射到 n_feats 个特征通道。
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # 由一堆 残差块 ResBlock 组成（每个包含 2 个卷积+ReLU+残差连接）。
        # 最后再加一个卷积。
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
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
    # 👉 这就是为什么有时候你看到 strict=False，它允许“部分加载”，比如迁移学习。
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

