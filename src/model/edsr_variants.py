import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common



# ============================================================
# ECA (Efficient Channel Attention)
# ============================================================
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)  # B,1,C
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


# ============================================================
# MSIRB (Multi-Scale Interactive Residual Block)
# ============================================================
class MSIRB(nn.Module):
    def __init__(self, channels, res_scale=1.0):
        super(MSIRB, self).__init__()
        self.res_scale = res_scale
        self.dw3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.SiLU(inplace=True)
        )
        self.dw7 = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=True),
            nn.SiLU(inplace=True)
        )
        self.fuse = nn.Conv2d(channels, channels, 1, bias=True)
        self.eca = ECA(channels, k_size=3)
        self.reduce = nn.Conv2d(channels, channels // 2, 1, bias=True)
        self.expand = nn.Conv2d(channels // 2, channels, 1, bias=True)

    def forward(self, x):
        a = self.dw3(x)
        b = self.dw7(x)
        a = a + 0.5 * b
        b = b + 0.5 * a
        out = a + b
        out = self.fuse(out)
        out = self.reduce(out)
        out = self.expand(out)
        out = self.eca(out)
        return x + out * self.res_scale


# ============================================================
# Shallow Fusion Group
# ============================================================
class ShallowFusionGroup(nn.Module):
    def __init__(self, n_blocks, channels):
        super(ShallowFusionGroup, self).__init__()
        self.blocks = nn.Sequential(*[MSIRB(channels) for _ in range(n_blocks)])
        self.fuse = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        residuals = []
        for blk in self.blocks:
            x = blk(x)
            residuals.append(x)
        out = sum(residuals) / len(residuals)
        out = self.fuse(out)
        return out + x


# ============================================================
# 主网络 ULRNet-MSIRB
# ============================================================
class ULRNet(nn.Module):
    """统一轻量超分网络（Head → Body → Tail）"""
    def __init__(self, args, conv=common.default_conv):
        super(ULRNet, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # head
        self.head = nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=True)

        # body（使用 MSIRB + Shallow Fusion）
        body = []
        num_groups = max(1, n_resblocks // 3)
        for _ in range(num_groups):
            body.append(ShallowFusionGroup(3, n_feats))
        body.append(conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*body)

        # tail
        self.tail = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, 3)
        )

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


def make_model(args, parent=False):
    net = ULRNet(args)
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K (MSIRB + Shallow Fusion)")
    return net