"""
Unified Lightweight Refinement Network (ULRNet) for Efficient Image Super-Resolution:
- Gated Asymmetric Depthwise Conv (GADWConv)
- Unified Lightweight Attention (ULAttention)
- Learnable Residual Scaling (LRS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


class GADWConv(nn.Module):
    """
    GADWConv: Asymmetric depthwise conv with grouped 1x1 gating conv.
    The gate conv uses groups=4 when possible to reduce parameter cost.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # asymmetric depthwise: (1x3) then (3x1)
        self.dw1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels, bias=True)
        self.dw2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=True)
        # use grouped 1x1 gating to reduce params when channels divisible by 4
        gate_groups = 4 if (channels % 4 == 0) else 1
        self.gate_conv = nn.Conv2d(channels, channels, 1, groups=gate_groups, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dw1(x)
        out = self.dw2(out)
        gate = self.sigmoid(self.gate_conv(x))
        return out * gate + x


class ULAttention(nn.Module):
    """
    ULAttention: Unified lightweight attention combining channel and spatial
    attention with a bottleneck. Uses grouped expansion on the channel
    re-projection to reduce parameter cost. Default reduction=16.
    This optimized variant uses ReLU and a lightweight spatial conv (k=3,p=1)
    to reduce FLOPs.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, channel // reduction)
        # first 1x1 reduces channel, second 1x1 restores; use groups for the restore when possible
        restore_groups = 2 if (channel % 2 == 0) else 1
        self.ca_conv = nn.Sequential(
            nn.Conv2d(channel, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channel, 1, groups=restore_groups, bias=True)
        )
        # lightweight spatial attention with small kernel (no dilation)
        self.sa_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        # learnable fusion weights (two scalars); will be softmaxed in forward
        self.fuse = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x):
        pooled = self.global_pool(x)
        ca = torch.sigmoid(self.ca_conv(pooled))                           # (B, C, 1, 1)
        sa = torch.sigmoid(self.sa_conv(torch.mean(x, dim=1, keepdim=True)))  # (B, 32, H, W) -> broadcast
        # compute normalized fusion weights
        w = torch.softmax(self.fuse, dim=0)
        f1, f2 = w[0], w[1]
        # broadcast CA to spatial shape
        ca_map = ca.expand_as(x)
        # if sa has channel != C, project it to C by simple repeat/interpolation
        if sa.size(1) != x.size(1):
            # use interpolation across channel dimension via repeat if small; this keeps it lightweight
            sa_map = F.interpolate(sa, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            # if sa channels != C, tile or reduce/expand along channel dim
            if sa_map.size(1) != x.size(1):
                repeat_times = x.size(1) // sa_map.size(1)
                if repeat_times >= 1 and x.size(1) % sa_map.size(1) == 0:
                    sa_map = sa_map.repeat(1, repeat_times, 1, 1)
                else:
                    # fallback: project via a 1x1 conv-like linear mapping using simple expansion
                    sa_map = sa_map.mean(dim=1, keepdim=True).repeat(1, x.size(1), 1, 1)
        else:
            sa_map = sa
        out = x * (f1 * ca_map + f2 * sa_map)
        return x + out


# NEW: StageULAttention class for stage-level attention
class StageULAttention(nn.Module):
    """
    Wrapper that applies a lightweight 1x1 projection around ULAttention and
    a learnable residual gating (alpha). This shrinks the parameter/FLOPs
    impact while keeping attention global and stable when applied once per
    body output (stage-level attention).
    """
    def __init__(self, channel, reduction=16, alpha=0.25):
        super().__init__()
        # small bottleneck projection to reduce compute and improve stability
        self.proj_in = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.att = ULAttention(channel, reduction=reduction)
        self.proj_out = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        # learnable scalar to control magnitude of the attention residual
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        y = self.proj_in(x)
        y = self.att(y)
        y = self.proj_out(y)
        return x + self.alpha * y


class GADWResidualBlock(nn.Module):
    """
    GADWResidualBlock: Residual block using GADWConv and optional StageULAttention.
    """
    def __init__(self, channels, kernel_size=3, res_scale=0.1,
                 use_dwconv=True, idx=0, use_attention=True):
        super().__init__()
        self.use_dwconv = use_dwconv
        self.use_attention = use_attention
        padding = kernel_size // 2

        if use_dwconv:
            self.conv1 = GADWConv(channels)
            self.conv2 = GADWConv(channels)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)

        # use lightweight ReLU6 to improve numerical stability on low-precision devices
        self.act = nn.ReLU6(inplace=True)

        # learnable residual scaling as a parameter
        self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))

        # Stage level attention with reduction=8 and alpha=0.25 by default
        if self.use_attention:
            self.stage_att = StageULAttention(channels, reduction=8, alpha=0.25)
        else:
            self.stage_att = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        out = out * self.res_scale
        if self.use_attention and self.stage_att is not None:
            out = self.stage_att(out)
            return x + out
        else:
            # skip attention (no additional attention module here, so just return residual)
            return x + out


class ULRNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [
            nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=True),
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.ReLU(inplace=True),
        ]

        use_dw = getattr(args, 'use_dwconv', True)
        use_att = getattr(args, 'use_attention', True)

        m_body = []
        total_blocks = n_resblocks
        for i in range(n_resblocks):
            use_dw_block = use_dw
            block = GADWResidualBlock(n_feats, kernel_size=3, res_scale=0.1,
                                           use_dwconv=use_dw_block, idx=i, use_attention=use_att)
            block.total_blocks = total_blocks
            m_body.append(block)
        m_body.append(conv(n_feats, n_feats, 3))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        # Remove global_att usage, keep None
        self.global_att = None

        m_tail = [
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1)
        ]

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        body_layers = list(self.body)
        last_conv = body_layers[-1]
        blocks = body_layers[:-1]

        out = x
        for block in blocks:
            out = block(out)
        res = last_conv(out)

        # self.global_att is None, so skip attention here
        res = res + x
        x = self.tail(res)
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
    has_gadw = any(isinstance(m, GADWConv) for m in net.modules())
    has_ulatt = any(isinstance(m, ULAttention) for m in net.modules())
    print(f"[INFO] GADWConv: {has_gadw}, ULAttention: {has_ulatt}")
