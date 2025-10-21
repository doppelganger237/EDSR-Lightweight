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
import math


# --- Helper: safe ECA kernel size computation ---
def _eca_kernel_size(channels, gamma=2, b=1):
    """Return an odd, valid 1D kernel size for ECA given channels. Always >=1 and <= channels (when channels>1).
    Handles small channel counts robustly."""
    if channels <= 1:
        return 1
    t = int(abs((math.log2(max(1, channels)) * gamma + b)))
    # make t odd and at least 1
    if t < 1:
        t = 1
    if t % 2 == 0:
        t += 1
    # bound to channels (make odd and >=1)
    if t > channels:
        k = channels if channels % 2 == 1 else (channels - 1 if channels > 1 else 1)
    else:
        k = t
    if k < 1:
        k = 1
    return k


# --- GADWConvLite: lightweight depthwise conv + ECA1D gating ---
class GADWConvLite(nn.Module):
    """
    GADWConvLite: simplified depthwise conv with lightweight ECA-style gating.
    - 仅使用单层depthwise + ECA1D全局通道注意力
    - 参数与计算量更低，适合与ULAttention共存
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=True)
        # adaptive ECA 1D kernel size (safe helper)
        k = _eca_kernel_size(channels, gamma=2, b=1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # learnable gating strength to avoid over-suppression in early training
        self.beta = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, x):
        out = self.dw(x)
        # Use post-conv statistics for ECA to align semantics
        y = out.mean(dim=(2, 3), keepdim=True)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.eca_conv(y)
        y = self.sigmoid(y.permute(0, 2, 1).unsqueeze(-1))
        # Residual-style gating to control modulation strength
        out = out + self.beta * (out * y - out)
        # return the processed residual chunk (do not add input here — outer block handles residual add)
        return out


class ULAttentionPlus(nn.Module):
    """
    Upgraded ULAttention (ULA++) for improved channel and spatial attention.
    - Channel: ECA1D + lightweight 1x1 Conv projection on descriptor.
    - Spatial: avg+max concat -> depthwise separable conv (3x3 DW + 1x1 PW).
    - Fusion: dynamic gate from input features replaces fixed softmax weights.
    - Residual output with learnable gamma scaling.
    """
    def __init__(self, channel, reduction=16, enable_spatial=True, gamma_init=0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.enable_spatial = enable_spatial

        # safe adaptive ECA kernel
        k = _eca_kernel_size(channel, gamma=2, b=1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.channel_proj = nn.Conv1d(1, 1, kernel_size=1, bias=True)

        # learnable residual scaling for attention output (smaller init for stability)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

        # Spatial attention: avg+max -> depthwise separable conv (3x3 DW + 1x1 PW)
        if self.enable_spatial:
            self.sa_dw = nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False)
            self.sa_pw = nn.Conv2d(2, 1, 1, bias=False)
        else:
            self.sa_dw = None
            self.sa_pw = None

        # Dynamic fusion gate replacing fixed softmax fusion weights
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(channel, 2, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel branch
        y = self.global_pool(x).view(b, c)
        y = y.unsqueeze(1)  # (B,1,C)
        y = self.eca_conv(y)
        y = self.channel_proj(y)
        ca_map = torch.sigmoid(y).view(b, c, 1, 1).expand_as(x)

        # Spatial branch (optional)
        if self.enable_spatial:
            avg = torch.mean(x, dim=1, keepdim=True)
            mx, _ = torch.max(x, dim=1, keepdim=True)
            sa_map = torch.sigmoid(self.sa_pw(self.sa_dw(torch.cat([avg, mx], dim=1)))).expand_as(x)
        else:
            sa_map = 0.0

        # Dynamic fusion gate
        gate = self.fuse_gate(x.mean(dim=(2,3), keepdim=True))
        out = x * (gate[:,0:1] * ca_map + (gate[:,1:2] * sa_map if isinstance(sa_map, torch.Tensor) else 0.0))

        # return attention-enhanced features (scaled). Outer block composes residuals.
        return self.gamma * out


# Keep old ULAttention name for compatibility
ULAttention = ULAttentionPlus


# NOTE: StageULAttention merged into GADWResidualBlock to simplify module hierarchy
class GADWResidualBlock(nn.Module):
    """
    GADWResidualBlock: Residual block using GADWConv and optional integrated
    ULAttention (previously StageULAttention). The stage-level attention wrapper
    has been merged into this block to simplify the codebase while preserving
    functionality. If use_attention is True, an internal 1x1 projection => ULAttention =>
    1x1 projection is applied and scaled by a learnable alpha before adding to the
    block input.
    """
    def __init__(self, channels, kernel_size=3, res_scale=0.1,
                 use_dwconv=True, idx=0, use_attention=True,
                 depth_ratio=1.0, depth_power=1.5, act='prelu',
                 att_alpha_init=0.1, att_gamma_init=0.1, att_spatial=True):
        super().__init__()
        self.use_dwconv = use_dwconv
        self.use_attention = use_attention
        padding = kernel_size // 2

        if use_dwconv:
            self.conv1 = GADWConvLite(channels)
            self.pw1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            self.conv2 = GADWConvLite(channels)
            self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.pw1 = None
            self.pw2 = None

        # lightweight activation (configurable)
        if act == 'prelu':
            self.act = nn.PReLU(num_parameters=channels)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        # learnable residual scaling
        self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))

        # depth-aware gate: (i/B)^p in [0,1]
        self.register_buffer('depth_gate', torch.tensor(float(max(1e-6, depth_ratio)) ** float(depth_power)))

        # Integrate StageULAttention functionality here
        if self.use_attention:
            self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            self.att = ULAttentionPlus(channels, reduction=8, enable_spatial=att_spatial, gamma_init=att_gamma_init)
            self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            # smaller init alpha for stability
            self.alpha = nn.Parameter(torch.tensor(att_alpha_init, dtype=torch.float32))
        else:
            self.proj_in = None
            self.att = None
            self.proj_out = None
            self.alpha = None

    def forward(self, x):
        out = self.conv1(x)
        if self.use_dwconv and self.pw1 is not None:
            out = self.pw1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.use_dwconv and self.pw2 is not None:
            out = self.pw2(out)

        out = out * self.res_scale

        if self.use_attention and self.att is not None:
            y_in = self.proj_in(out)
            # depth-gated attention response
            y_att = self.att(y_in) * self.depth_gate
            delta = y_att - y_in
            y = self.proj_out(delta)
            scaled_alpha = self.alpha * self.depth_gate
            return x + out + scaled_alpha * y
        else:
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

        # new knobs (read via getattr, keep defaults safe)
        att_start_ratio = float(getattr(args, 'att_start_ratio', 0.5))  # only last ~1/3 enable attention
        att_depth_power = float(getattr(args, 'att_depth_power', 1.5))
        att_alpha_init = float(getattr(args, 'att_alpha_init', 0.1))
        att_gamma_init = float(getattr(args, 'att_gamma_init', 0.1))
        att_spatial = bool(getattr(args, 'att_spatial', True))
        act_type = str(getattr(args, 'act', 'prelu'))

        m_body = []
        total_blocks = n_resblocks
        start_idx = int(total_blocks * att_start_ratio)
        for i in range(n_resblocks):
            use_dw_block = use_dw
            # enable attention only in the last K% blocks
            use_att_block = use_att and (i >= start_idx)
            depth_ratio = (i + 1) / float(total_blocks)
            block = GADWResidualBlock(
                n_feats, kernel_size=3, res_scale=0.1,
                use_dwconv=use_dw_block, idx=i, use_attention=use_att_block,
                depth_ratio=depth_ratio, depth_power=att_depth_power, act=act_type,
                att_alpha_init=att_alpha_init, att_gamma_init=att_gamma_init, att_spatial=att_spatial
            )
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
    has_gadw = any(isinstance(m, GADWConvLite) for m in net.modules())
    has_ulatt = any(isinstance(m, ULAttention) for m in net.modules())
    print(f"[INFO] GADWConvLite: {has_gadw}, ULAttention: {has_ulatt}")
