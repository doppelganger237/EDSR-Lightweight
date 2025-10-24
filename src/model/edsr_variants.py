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
# （非门控多分支：DW 3x3 (d=1/2/3) 等权相加 → 1x1 → C/2→C + ECA）
# ============================================================
class MSIRB(nn.Module):
    def __init__(self, channels, res_scale=1.0):
        super(MSIRB, self).__init__()
        self.res_scale = res_scale

        # 多尺度分支：3x3 深度卷积，d=1/2/3（CUDA 上更快更稳）
        self.dw_d1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.SiLU(inplace=True)
        )
        self.dw_d2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=True),
            nn.SiLU(inplace=True)
        )
        self.dw_d3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=True),
            nn.SiLU(inplace=True)
        )

        # 融合与瓶颈
        self.fuse = nn.Conv2d(channels, channels, 1, bias=True)
        self.reduce = nn.Conv2d(channels, channels // 2, 1, bias=True)
        self.mid_act = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(channels // 2, channels, 1, bias=True)

        self.eca = ECA(channels, k_size=3)

    def forward(self, x):
        a = self.dw_d1(x)
        b = self.dw_d2(x)
        c = self.dw_d3(x)
        out = (a + b + c) / 3.0                      # 等权相加（无可学习门控）
        out = self.fuse(out)
        out = self.expand(self.mid_act(self.reduce(out)))
        out = self.eca(out)
        return x + out * self.res_scale

# ============================================================
# Shallow Fusion Group（级联残差 + 固定缩放）
# ============================================================
class ShallowFusionGroup(nn.Module):
    def __init__(self, n_blocks, channels, group_res_scale=0.2):
        super(ShallowFusionGroup, self).__init__()
        self.blocks = nn.ModuleList([MSIRB(channels, res_scale=1.0) for _ in range(n_blocks)])
        self.fuse = nn.Conv2d(channels, channels, 1, bias=True)
        self.group_res_scale = group_res_scale

    def forward(self, x):
        h = x
        for blk in self.blocks:
            h = blk(h)
        out = self.fuse(h)
        return x + self.group_res_scale * out        # 固定缩放，非可学习

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

        # body（多个组，每组含3个 MSIRB）
        body = []
        num_groups = max(1, n_resblocks // 3)
        for _ in range(num_groups):
            body.append(ShallowFusionGroup(3, n_feats, group_res_scale=0.2))
        body.append(conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*body)

        # tail（与 EDSR 一致，利于 PSNR）
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
    print(f"Params: {params/1e3:.1f}K (MSIRB-dilated + RIR Group)")
    return net