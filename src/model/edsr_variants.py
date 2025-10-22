# DIFRN: Dual-Interaction Feature Refinement Network
# Incorporating RepConv, SECA, ESALite and DW-ASPP for efficient super-resolution
import math
import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F
from collections import OrderedDict

# ---------------- helpers ----------------
def _safe_eca_k(channels, gamma=2.5, b=1):
    # dynamic ECA kernel size: slightly larger for higher channels
    if channels <= 1:
        return 1
    t = int(abs((math.log2(max(1, channels)) * gamma + b)))
    if t < 1: t = 1
    if t % 2 == 0: t += 1
    if t > channels:
        k = channels if channels % 2 == 1 else max(1, channels - 1)
    else:
        k = t
    if k < 1: k = 1
    return k

# ---------------- RepConvBlock ----------------
class RepConvBlock(nn.Module):
    """
    RepConvBlock: 3x3 conv + identity, no BN, SiLU/PReLU/ReLU.
    """
    def __init__(self, channels, kernel_size=3, act='relu', deploy=False):
        super().__init__()
        self.deploy = deploy
        padding = kernel_size // 2

        self.rbr_dense = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=True)
        self.rbr_identity = nn.Identity()

        if act == 'prelu':
            self.act = nn.SiLU(inplace=True)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.rbr_dense(x) + self.rbr_identity(x)
        return self.act(out)

    def convert_to_deploy(self):
        # No BN fusion needed, nothing to do
        self.deploy = True

# ---------------- SECA ----------------
class SECA(nn.Module):
    """Lightweight SECA but with aggressive projection (sparse) to save params."""
    def __init__(self, channels, proj_ratio=0.15, gamma=2, b=1):
        super().__init__()
        self.channels = channels
        self.proj_ch = max(1, int(channels * proj_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        k = _safe_eca_k(self.proj_ch, gamma=gamma, b=b)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        # small projection layers
        self.fc1 = nn.Conv2d(channels, self.proj_ch, 1, bias=True)
        self.fc2 = nn.Conv2d(self.proj_ch, channels, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)                             # (B, C, 1, 1)
        y = self.fc1(y)                              # (B, proj_ch, 1, 1)
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)   # (B, 1, proj_ch)
        y = self.conv1d(y)                           # (B, 1, proj_ch)
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1) # (B, proj_ch, 1, 1)
        y = self.fc2(y)                              # (B, C, 1, 1)
        att = self.sig(y)
        return x * att

# ---------------- ESALite (RFDN-inspired) ----------------
class ESALite(nn.Module):
    """
    Efficient Spatial Attention (lite): reduce -> DW conv stride2 -> conv -> upsample -> sigmoid gate.
    Lightweight variant using depthwise ops and bilinear upsample. Always bias=True to help small widths.
    """
    def __init__(self, channels, reduce=0.5):
        super().__init__()
        c_mid = max(8, int(channels * reduce))
        self.conv1 = nn.Conv2d(channels, c_mid, 1, bias=True)
        self.conv2 = nn.Conv2d(c_mid, c_mid, 3, stride=2, padding=1, groups=c_mid, bias=True)
        self.conv3 = nn.Conv2d(c_mid, c_mid, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(c_mid, channels, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.act(self.conv3(y))
        y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        y = self.conv4(y)
        return self.sig(y)

# ---------------- DW-ASPP (lightweight) ----------------
# ---------------- DW-ASPP (lightweight) ----------------
class DWASPP(nn.Module):
    """Depthwise ASPP: parallel depthwise dilated convs with small dilation rates, fuse by 1x1 conv."""
    def __init__(self, channels, rates=(1,2,3)):
        super().__init__()
        self.rates = list(rates)
        self.dw_convs = nn.ModuleList()
        for r in self.rates:
            self.dw_convs.append(
                nn.Conv2d(channels, channels, 3, padding=r, dilation=r, groups=channels, bias=False)
            )
        # fuse
        self.project = nn.Conv2d(channels * len(self.rates), channels, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        outs = []
        for conv in self.dw_convs:
            outs.append(self.act(conv(x)))
        out = torch.cat(outs, dim=1)
        out = self.project(out)
        return out

# ---------------- HybridASPP (DWASPP + pointwise channel interaction) ----------------
class HybridASPP(nn.Module):
    def __init__(self, channels, rates=(1,2,3,6)):
        super().__init__()
        # Parallel DWConv 3x3 and 5x5, then concat and project
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.project = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out3 = self.act(self.dw3(x))
        out5 = self.act(self.dw5(x))
        out = torch.cat([out3, out5], dim=1)
        out = self.project(out)
        return out

# ---------------- DIFRBlock ----------------
# context branch now uses progressive channel distillation (RFDN-inspired)
class DIFRBlock(nn.Module):
    def __init__(self, channels, distill_ratio=0.5, proj_ratio=0.15, use_eca=True, res_scale=0.1, act='prelu', enable_esa=False):
        super().__init__()
        self.channels = channels
        self.distill_ratio = distill_ratio
        c_d = int(channels * distill_ratio)
        c_c = channels - c_d

        # detail branch: 1x1 conv -> RepConvBlock
        self.detail = nn.Sequential(
            nn.Conv2d(c_d, c_d, 1, bias=False),
            RepConvBlock(c_d, act=act)
        )

        # context branch: progressive channel distillation (4-stage, slower decay)
        self.context = nn.Sequential(OrderedDict([
            ("c1_r", nn.Conv2d(c_c, c_c, 3, padding=1, groups=c_c, bias=False)),
            ("c1_act", nn.SiLU(inplace=True)),
            ("c1_d", nn.Conv2d(c_c, c_c // 2, 1, bias=False)),

            ("c2_r", nn.Conv2d(c_c // 2, c_c // 2, 3, padding=1, groups=c_c // 2, bias=False)),
            ("c2_act", nn.SiLU(inplace=True)),
            ("c2_d", nn.Conv2d(c_c // 2, c_c // 2, 1, bias=False)),

            ("c3_r", nn.Conv2d(c_c // 2, c_c // 2, 3, padding=1, groups=c_c // 2, bias=False)),
            ("c3_act", nn.SiLU(inplace=True)),
            ("c3_d", nn.Conv2d(c_c // 2, c_c // 4, 1, bias=False)),

            ("c4_r", nn.Conv2d(c_c // 4, c_c // 4, 3, padding=1, groups=c_c // 4, bias=False)),
            ("c4_act", nn.SiLU(inplace=True)),
            ("c4_d", nn.Conv2d(c_c // 4, c_c // 8, 1, bias=False)),
        ]))
        # Add learnable distillation weights
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.alpha3 = nn.Parameter(torch.tensor(1.0))
        self.alpha4 = nn.Parameter(torch.tensor(1.0))
        # Add gated DWConv1x1 fusion at the end of context
        self.fuse = nn.Sequential(
            nn.Conv2d((c_c // 2) + (c_c // 2) + (c_c // 4) + (c_c // 8), c_c, 1, bias=False, groups=1),
            nn.Conv2d(c_c, c_c, 1, groups=c_c, bias=True),
            nn.Sigmoid()
        )

        # fusion conv 1x1
        self.proj_f = nn.Conv2d(c_d + c_c, channels, 1, bias=False)
        self.proj_f2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.SiLU(inplace=True)
        )

        self.use_eca = use_eca
        if self.use_eca:
            self.eca = SECA(channels, proj_ratio=proj_ratio)
        else:
            self.eca = None

        # optional spatial attention for tail blocks
        self.esa = ESALite(channels) if enable_esa else None

        self.res_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.base_res_scale = 0.1

    def forward(self, x):
        c = x.size(1)
        c_d = int(c * self.distill_ratio)
        c_c = c - c_d
        xd = x[:, :c_d, :, :]
        xc = x[:, c_d:, :, :]
        # 引入来自 context 分支的轻反馈
        if hasattr(self, 'last_context'):
            yd = self.detail(xd + 0.2 * self.last_context)
        else:
            yd = self.detail(xd)

        # 判断 context 结构类型
        # context progressive distillation structure
        if len(self.context) >= 12:
            c1_r, c1_act, c1_d = self.context[0], self.context[1], self.context[2]
            c2_r, c2_act, c2_d = self.context[3], self.context[4], self.context[5]
            c3_r, c3_act, c3_d = self.context[6], self.context[7], self.context[8]
            c4_r, c4_act, c4_d = self.context[9], self.context[10], self.context[11]

            r1 = c1_act(c1_r(xc))
            d1 = self.alpha1 * c1_d(r1)
            r2 = c2_act(c2_r(r1 - d1))
            d2 = self.alpha2 * c2_d(r2)
            r3 = c3_act(c3_r(r2 - d2))
            d3 = self.alpha3 * c3_d(r3)
            r4 = c4_act(c4_r(r3 - d3))
            d4 = self.alpha4 * c4_d(r4)
            # gated DWConv1x1 fusion
            yc_fuse_input = torch.cat([d1, d2, d3, d4], dim=1)
            yc = self.fuse(yc_fuse_input) * yc_fuse_input
            yc = yc.mean(1, keepdim=True).expand_as(d1)
            self.last_context = yc.detach()
        else:
            # 非渐进结构（DWASPP或简单Conv）
            yc = self.context(xc)

        y = torch.cat([yd, yc], dim=1)
        if self.eca is not None:
            y = y + self.eca(y)
        y = self.proj_f(y)
        y = self.proj_f2(y)
        if self.esa:
            y = self.esa(y) * y
        # PFSC (Param-Free Spatial Contrast) gate
        contrast = (y - y.mean([2, 3], keepdim=True)) ** 2
        gate = torch.sigmoid(F.avg_pool2d(contrast, 3, 1, 1))
        y = y * gate
        res_scale = self.base_res_scale + torch.tanh(self.res_bias)
        return x + y * res_scale

# ---------------- TailRefine ----------------
class TailRefine(nn.Module):
    """A tiny residual refine: DW5x5 -> PW1x1 with GELU and gated residual."""
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.GELU()
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x):
        y = self.dw(x)
        y = self.act(self.pw(y))
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        return x + y * alpha

# ---------------- DIFRN network (modified with DW-ASPP and ESALite half-coverage) ----------------
class DIFRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_feats = args.n_feats
        n_resblocks = args.n_resblocks
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # head
        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=False),
            nn.SiLU(inplace=True)
        )

        # Cross-Block Feedback gate
        self.cbf_gate = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.Sigmoid()
        )

        # Multi-Scale Gating
        self.msg = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(n_feats, n_feats // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // 4, n_feats, 1, bias=True),
            nn.Sigmoid()
        )

        # body: stacked DIFRBlock blocks
        body = []
        enhanced_start = n_resblocks * 3 // 4
        esa_start = 0  # enable ESALite for all blocks
        for i in range(n_resblocks):
            enable_esa = (i >= esa_start)
            esa_reduce = 0.75 if i >= enhanced_start else 0.5
            if i >= enhanced_start:
                # enhanced tail blocks with res_scale=0.14 and standard conv in context branch
                distill_ratio = 0.5
                c_d = int(n_feats * distill_ratio)
                c_c = n_feats - c_d
                detail = nn.Sequential(
                    nn.Conv2d(c_d, c_d, 1, bias=False),
                    RepConvBlock(c_d, act=getattr(args,'act','prelu'))
                )
                context = nn.Sequential(
                    nn.Conv2d(c_c, c_c, 5, padding=2, groups=1, bias=False),
                    nn.Conv2d(c_c, c_c, 1, bias=False),
                    nn.ReLU(inplace=True)
                )
                # tail强化部分 proj_ratio=0.18
                res_scale_val = 0.14
                if i == n_resblocks - 1:
                    res_scale_val = 0.15
                block = DIFRBlock(n_feats, distill_ratio=distill_ratio, proj_ratio=0.18,
                                     use_eca=True, act=getattr(args,'act','prelu'), res_scale=res_scale_val,
                                     enable_esa=enable_esa)
                # replace detail and context with custom ones
                block.detail = detail
                block.context = context
                block.esa = ESALite(n_feats, reduce=esa_reduce)
            else:
                # non-tail blocks: use HybridASPP as context to enlarge receptive field with tiny cost and channel interaction
                block = DIFRBlock(n_feats, distill_ratio=0.5, proj_ratio=0.15,
                                     use_eca=True, act=getattr(args,'act','prelu'),
                                     res_scale=0.1, enable_esa=enable_esa)
                # replace default context with HybridASPP (operate on c_c channels)
                c_d = int(n_feats * 0.5)
                c_c = n_feats - c_d
                block.context = nn.Sequential(
                    HybridASPP(c_c, rates=(1,2,3,5))
                )
                block.esa = ESALite(n_feats, reduce=esa_reduce)
            body.append(block)
        body.append(conv(n_feats, n_feats, 3, bias=False))
        self.body = nn.Sequential(*body)
        # Add CLCF (Cross-Layer Context Fusion) after body
        self.clcf = SECA(n_feats, proj_ratio=0.20)

        # cross-stage feature fusion (lite)
        self.csff = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.SiLU(inplace=True),
            SECA(n_feats, proj_ratio=0.20)
        )

        # tail
        scale = args.scale[0] if isinstance(args.scale, (list,tuple)) else args.scale
        self.tail = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1, bias=False)
        )

        # final tiny refine
        self.tail_refine = TailRefine(n_feats)

        # Tail lightweight fusion after TailRefine
        self.tail_fuse = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.GELU()
        )

        # Tail Attention module
        self.tail_att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.SiLU(inplace=True),
            SECA(n_feats, proj_ratio=0.20)
        )
        # Add LKA-lite after tail_att
        self.lka_tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, padding=3, groups=n_feats, bias=True),
            nn.Conv2d(n_feats, n_feats, 5, padding=2, bias=True),
            nn.SiLU(inplace=True)
        )

        # lightweight pixel attention
        self.pixel_att = PixelAttention(n_feats)

        # Residual Group size (number of blocks per residual group)
        # default to 2, can be overridden by args.rg_size
        self.rg_size = getattr(args, 'rg_size', 2)

        # Multi-stage CSFF modules (lightweight)
        self.csff1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.SiLU(inplace=True),
            SECA(n_feats, proj_ratio=0.20)
        )
        self.csff2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=True),
            nn.SiLU(inplace=True),
            SECA(n_feats, proj_ratio=0.20)
        )

    def apply_msg(self, x):
        gate = self.msg(x)
        gate = F.interpolate(gate, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return gate

    def forward(self, x):
        x = self.sub_mean(x)
        h = self.head(x)
        body_layers = list(self.body)
        last = body_layers[-1]
        blocks = body_layers[:-1]
        out = h
        enhanced_start = len(blocks) * 3 // 4
        mid_feat = None
        mid1_feat = None
        mid2_feat = None

        total_blocks = len(blocks)
        mid1_idx = max(0, total_blocks // 2 - 1)
        mid2_idx = max(0, total_blocks * 3 // 4 - 1)

        feedback = None
        # SSA (Shallow Skip Aggregation): collect first and second block outputs
        ssa_feats = []
        ssa_weights = [0.65, 0.35]  # weights for aggregation
        for i, block in enumerate(blocks):
            # record group input at group start
            if (i % self.rg_size) == 0:
                group_input = out

            if i == len(blocks) - 3:
                feedback = self.cbf_gate(out)
            out = block(out)
            if i >= len(blocks) - 2 and feedback is not None:
                out = out * (1 + 0.5 * feedback)

            if i == mid1_idx:
                mid1_feat = out
            if i == mid2_idx:
                mid2_feat = out

            # SSA: collect first and second block outputs
            if i == 0 or i == 1:
                ssa_feats.append(out)

            # group residual add
            if ((i + 1) % self.rg_size) == 0 or (i == total_blocks - 1):
                out = group_input + out

            # capture the enhanced_start mid feature for original csff
            if i == enhanced_start - 1:
                mid_feat = out

        res = last(out)
        res = res + h

        # SSA: shallow skip aggregation (加权加至res)
        if len(ssa_feats) == 2:
            res = res + ssa_weights[0] * ssa_feats[0] + ssa_weights[1] * ssa_feats[1]
        elif len(ssa_feats) == 1:
            res = res + ssa_feats[0]

        # CLCF: cross-layer context fusion
        res = res + self.clcf(res)

        # apply multi-stage csff
        if mid1_feat is not None:
            res = res + self.csff1(mid1_feat)
        if mid2_feat is not None:
            res = res + self.csff2(mid2_feat)

        if mid_feat is not None:
            res = res + self.csff(mid_feat)

        res = self.tail_refine(res)
        res = self.tail_fuse(res)
        res = self.tail_att(res)
        res = self.lka_tail(res)
        res = res * (1 + 0.5 * self.apply_msg(res))
        res = self.pixel_att(res)
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def convert_to_deploy(self):
        # convert all RepConv inside detail branches, no BN fusion needed
        for m in self.modules():
            if isinstance(m, DIFRBlock):
                for subm in m.detail.modules():
                    if isinstance(subm, RepConvBlock):
                        subm.convert_to_deploy()

    # ---------------- lightweight pixel attention ----------------
 # PA-Gate: gated pixel attention for bidirectional control
class PixelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
    def forward(self, x):
        att = torch.tanh(self.conv(x))
        return x * (1 + 0.5 * att)

# ---------------- factory and diagnostics ----------------
def make_model(args, parent=False):
    net = DIFRN(args)
    check_modules(net)
    return net

def check_modules(net):
    has_rep = any(isinstance(m, RepConvBlock) for m in net.modules())
    has_eca = any(isinstance(m, SECA) for m in net.modules())
    print(f"[INFO] Model: DIFRN, RepConv: {has_rep}, SECA: {has_eca}")
    # print params (safe on CPU)
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")