# EDSR_DWConv: 继承自 EDSR，只替换 Body 为 DWConv
from model import common
from model.edsr import EDSR
import torch.nn as nn


# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride=1,
            padding=padding, groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ResBlock with DWConv
class ResBlockDW(nn.Module):
    def __init__(self, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1):
        super(ResBlockDW, self).__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            DepthwiseSeparableConv(n_feats, n_feats, kernel_size),
            act,
            DepthwiseSeparableConv(n_feats, n_feats, kernel_size)
        )

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res + x


# 继承自 EDSR，只改 body
class EDSR_DWConv(EDSR):
    def __init__(self, args):
        super().__init__(args, conv=common.default_conv)

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        # 替换 Body
        m_body = [
            ResBlockDW(n_feats, kernel_size, act=act, res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(DepthwiseSeparableConv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)


def make_model(args, parent=False):
    return EDSR_DWConv(args)