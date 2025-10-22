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
        # adaptive ECA 1D kernel size (safe helper) -- gamma changed to 1.5
        k = _eca_kernel_size(channels, gamma=1.5, b=1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # learnable gating strength to avoid over-suppression in early training (init changed to 0.15)
        self.beta = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))

    def forward(self, x):
        out = self.dw(x)
        # Use post-conv statistics for ECA to align semantics
        y = out.mean(dim=(2, 3), keepdim=True)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.eca_conv(y)
        y = self.sigmoid(y.permute(0, 2, 1).unsqueeze(-1))
        # Residual-style gating to control modulation strength
        beta = torch.clamp(self.beta, 0.0, 1.0)
        out = out + beta * (out * y - out)
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
    def __init__(self, channel, reduction=16, enable_spatial=True, gamma_init=0.1, alpha=0.1, expand=1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.enable_spatial = enable_spatial

        # safe adaptive ECA kernel
        k = _eca_kernel_size(channel, gamma=2, b=1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.channel_proj = nn.Conv1d(1, 1, kernel_size=1, bias=True)

        # learnable residual scaling for attention output (smaller init for stability)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        # Optionally allow alpha parameter (not used in this class, but for block)
        self.alpha = alpha
        self.expand = expand

        # Spatial attention: avg+max -> depthwise separable conv (3x3 DW + 1x1 PW)
        if self.enable_spatial:
            self.sa_dw = nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False)
            self.sa_pw = nn.Conv2d(2, 1, 1, bias=False)
        else:
            self.sa_dw = None
            self.sa_pw = None

        # Dynamic fusion gate replacing fixed softmax fusion weights
        self.fuse_gate = nn.Conv2d(channel, 2, 1, bias=True)
        nn.init.constant_(self.fuse_gate.weight, 0.0)
        nn.init.constant_(self.fuse_gate.bias, 0.0)
        with torch.no_grad():
            self.fuse_gate.bias[0] = math.log(0.7 / (1 - 0.7))
            self.fuse_gate.bias[1] = math.log(0.3 / (1 - 0.3))
        self.fuse_gate = nn.Sequential(self.fuse_gate, nn.Softmax(dim=1))

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
        gamma = torch.sigmoid(self.gamma) * 3.0  # 动态比例映射到 (0,3)
        return gamma * out


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
                 att_alpha_init=0.1, att_gamma_init=0.1, att_spatial=True,
                 expand=1, full_mode=False,
                 pw_groups=1, att_proj_ratio=0.5, shared_prelu=False):
        super().__init__()
        self.use_dwconv = use_dwconv
        self.use_attention = use_attention
        self.expand = int(max(1, expand))
        self.full_mode = full_mode
        self.pw_groups = max(1, int(pw_groups))
        self.att_proj_ratio = float(att_proj_ratio)
        self.shared_prelu = bool(shared_prelu)
        padding = kernel_size // 2

        if use_dwconv:
            if self.expand > 1:
                exp_ch = channels * self.expand
                self.pw_exp1 = nn.Conv2d(channels, exp_ch, 1, bias=True, groups=self.pw_groups)
                self.dw_e1   = GADWConvLite(exp_ch)
                self.pw_red1 = nn.Conv2d(exp_ch, channels, 1, bias=True, groups=self.pw_groups)

                self.pw_exp2 = nn.Conv2d(channels, exp_ch, 1, bias=True, groups=self.pw_groups)
                self.dw_e2   = GADWConvLite(exp_ch)
                self.pw_red2 = nn.Conv2d(exp_ch, channels, 1, bias=True, groups=self.pw_groups)
            else:
                self.conv1 = GADWConvLite(channels)
                self.pw1   = nn.Conv2d(channels, channels, kernel_size=1, bias=True, groups=self.pw_groups)
                self.conv2 = GADWConvLite(channels)
                self.pw2   = nn.Conv2d(channels, channels, kernel_size=1, bias=True, groups=self.pw_groups)
        else:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
            self.pw1 = self.pw2 = None

        if act == 'prelu':
            # 共享参数版进一步降参（可通过 shared_prelu 控制）
            self.act = nn.PReLU(num_parameters=(1 if self.shared_prelu else channels))
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        # Patch: set res_scale to 0.12 when attention is used (full/att-only), else 0.1
        if self.use_attention:
            if self.full_mode:
                res_scale_value = 0.12
            else:
                res_scale_value = 0.12
        else:
            res_scale_value = 0.1
        self.res_scale = nn.Parameter(torch.tensor(res_scale_value, dtype=torch.float32))
        self.register_buffer('depth_gate', torch.tensor(float(max(1e-6, depth_ratio)) ** float(depth_power)))

        if self.use_attention:
            # full_mode时ULAttention参数特殊
            if self.full_mode:
                att_alpha = 0.12
                att_gamma = 0.15
                att_expand = 1
                att_enable_spatial = True
                # 使用可配置的投影比例，full 模式上限 0.25 保持轻量
                proj_ratio = min(self.att_proj_ratio, 0.25)
                att_proj_channels = max(1, int(channels * att_expand * proj_ratio))
            else:
                att_alpha = att_alpha_init
                att_gamma = att_gamma_init
                att_expand = self.expand
                att_enable_spatial = att_spatial
                proj_ratio = max(0.0625, min(self.att_proj_ratio, 1.0))
                att_proj_channels = max(1, int(channels * att_expand * proj_ratio))
            self.proj_in  = nn.Conv2d(channels, att_proj_channels, kernel_size=1, bias=True)
            self.att      = ULAttentionPlus(
                att_proj_channels, reduction=8,
                enable_spatial=att_enable_spatial,
                gamma_init=att_gamma,
                alpha=att_alpha,
                expand=att_expand
            )
            self.proj_out = nn.Conv2d(att_proj_channels, channels, kernel_size=1, bias=True)
            self.alpha    = nn.Parameter(torch.tensor(att_alpha, dtype=torch.float32))
        else:
            self.proj_in = self.att = self.proj_out = self.alpha = None

    def forward(self, x):
        if self.use_dwconv and self.expand > 1:
            out = self.pw_exp1(x)
            out = self.act(self.dw_e1(out))
            out = self.pw_red1(out)
            out = self.act(out)
            out = self.pw_exp2(out)
            out = self.act(self.dw_e2(out))
            out = self.pw_red2(out)
        else:
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
            y_att = self.att(y_in) * self.depth_gate
            y = self.proj_out(y_att)
            scaled_alpha = self.alpha * self.depth_gate
            return x + out + scaled_alpha * y
        else:
            return x + out


# Define a simple fallback GlobalContext if not in common
class LocalGlobalContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LayerNorm([channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        mask = self.conv_mask(x).view(b, 1, -1)
        mask = self.softmax(mask)
        x = x.view(b, c, -1)
        context = torch.bmm(x, mask.permute(0, 2, 1))
        context = context.view(b, c, 1, 1)
        channel_add_term = self.channel_add_conv(context)
        return input_x + channel_add_term


class ULRNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        # --- Width scaling and grouped pointwise ---
        width_mult = 0.75  # 全局宽度缩放系数（固定常量，不引入新选项）
        pw_groups = 1      # 提升表达力：取消 1x1 分组，提升 PSNR，同时仍然控制总参
        att_proj_ratio = 0.5  # 放宽注意力投影，减少信息瓶颈
        n_resblocks = args.n_resblocks
        n_feats = int(args.n_feats * width_mult)
        scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # 标准更稳的 head：3x3 + PReLU（不做瓶颈）
        m_head = [
            nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=True),
            nn.PReLU(num_parameters=n_feats),
        ]

        # Change default values here; keep use_dwconv/use_attention defaults as False
        use_dw = getattr(args, 'use_dwconv', False)
        use_att = getattr(args, 'use_attention', False)

        # full_mode: 如果 use_dwconv 和 use_attention 同时为 True，则自动开启 full_mode
        full_mode = use_dw and use_att

        # new knobs (read via getattr, keep defaults safe)
        att_start_ratio = float(getattr(args, 'att_start_ratio', 0.67))  # only last ~1/3 enable attention
        att_depth_power = float(getattr(args, 'att_depth_power', 1.8))
        att_alpha_init = float(getattr(args, 'att_alpha_init', 0.1))
        att_gamma_init = float(getattr(args, 'att_gamma_init', 0.1))
        att_spatial = bool(getattr(args, 'att_spatial', False))
        act_type = str(getattr(args, 'act', 'prelu'))

        # full模式: 轻量增强版 full 模型
        # - att_expand=1
        # - att_alpha_init=0.12
        # - att_gamma_init=0.15
        # - att_spatial_local=True
        # - full_mode 启用GC模块与soft gate
        if full_mode:
            att_start_ratio_local = 0.6  # fewer attention blocks
            att_depth_power_local = min(att_depth_power, 1.6)
            att_alpha_init_local  = 0.12
            att_gamma_init_local  = 0.15
            att_spatial_local     = True
            expand_local          = 1
        elif use_att and not use_dw:
            # att-only模式轻量化增强
            att_start_ratio_local = min(att_start_ratio, 0.5)
            att_depth_power_local = att_depth_power
            att_alpha_init_local  = 0.12
            att_gamma_init_local  = 0.12
            att_spatial_local     = True
            expand_local          = 1
        else:
            att_start_ratio_local = att_start_ratio
            att_depth_power_local = att_depth_power
            att_alpha_init_local  = att_alpha_init
            att_gamma_init_local  = att_gamma_init
            att_spatial_local     = att_spatial
            expand_local          = 1

        m_body = []
        total_blocks = n_resblocks
        if full_mode:
            att_enable_blocks = total_blocks // 2
        else:
            att_enable_blocks = int(total_blocks * att_start_ratio_local)
        # 将最后 1/4 的残差块改为标准 3x3 卷积，提升重构质量（其余仍用 DWConv 控参）
        strong_tail_blocks = max(2, total_blocks // 4)
        for i in range(n_resblocks):
            use_dw_block = use_dw
            if i >= total_blocks - strong_tail_blocks:
                use_dw_block = False
            if full_mode:
                use_att_block = (i >= total_blocks // 2)
                expand_block = 1
                full_mode_block = True
            else:
                use_att_block = use_att and (i >= att_enable_blocks)
                expand_block = expand_local
                full_mode_block = False
            depth_ratio = (i + 1) / float(total_blocks)
            block = GADWResidualBlock(
                n_feats, kernel_size=3, res_scale=0.1,
                use_dwconv=use_dw_block, idx=i, use_attention=use_att_block,
                depth_ratio=depth_ratio, depth_power=att_depth_power_local, act=act_type,
                att_alpha_init=att_alpha_init_local, att_gamma_init=att_gamma_init_local, att_spatial=att_spatial_local,
                expand=expand_block,
                full_mode=full_mode_block,
                pw_groups=pw_groups,
                att_proj_ratio=att_proj_ratio
            )
            block.total_blocks = total_blocks
            m_body.append(block)
        m_body.append(conv(n_feats, n_feats, 3))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        # full模式下使用GC模块 (轻量 1x1 bottleneck)
        if full_mode:
            self.gc = nn.Conv2d(n_feats, n_feats // 4, 1, bias=False)
            self.gc_proj = nn.Conv2d(n_feats // 4, n_feats, 1, bias=False)
        else:
            self.gc = None
            self.gc_proj = None
        if self.gc is not None:
            # 更积极的初始门控，帮助 early gain
            self.gc_scale = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))

        self.global_att = None
        self.full_mode = full_mode

        # 标准更稳的 tail（不做瓶颈），利于 PSNR
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

        # full_mode下GC输出加到residual，增加gc_scale参数
        if self.full_mode and self.gc is not None:
            gc_out = self.gc(out)
            gc_out = self.gc_proj(gc_out)
            gate = torch.sigmoid(self.gc_scale)  # learnable soft gate
            res = res + gate * gc_out

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
    # 保留FLOPs和Params打印逻辑
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count
    dummy_input = torch.randn(1, 3, 48, 48).to(torch.device("cpu"))  # ⚠️ 必须是 CPU
    flops = FlopCountAnalysis(net.cpu(), dummy_input)
    flops.unsupported_ops_warnings(False)
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params / 1e3:.1f}K , FLOPs: {flops.total() / 1e9:.3f}G")