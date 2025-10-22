# DFDN + RepConv + SparseECA implementation (drop-in for edsr_variants)
import math
import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F
# ---------------- helpers ----------------
def _safe_eca_k(channels, gamma=2, b=1):
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
    Training-time multi-branch block (3x3 conv + identity) with BN
    At inference we can fuse branches to a single conv. Provides strong training capacity,
    light inference cost after reparam.
    """
    def __init__(self, channels, kernel_size=3, act='relu', deploy=False):
        super().__init__()
        self.deploy = deploy
        padding = kernel_size // 2

        if deploy:
            self.rbr_reparam = nn.Conv2d(channels, channels, kernel_size,
                                         padding=padding, bias=True)
        else:
            # branch 1: kxk conv
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(channels)
            )
            # branch 3: identity (BN)
            self.rbr_identity = nn.BatchNorm2d(channels)

        if act == 'prelu':
            self.act = nn.PReLU(num_parameters=channels)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            out = self.rbr_reparam(x)
            return self.act(out)
        out = 0
        if hasattr(self, 'rbr_dense'):
            out = out + self.rbr_dense(x)
        if hasattr(self, 'rbr_identity'):
            out = out + self.rbr_identity(x)
        return self.act(out)

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        # fuse conv and bn to a new kernel and bias
        if conv is None:
            return 0, 0
        w = conv.weight
        if conv.bias is None:
            conv_bias = torch.zeros(w.size(0), device=w.device)
        else:
            conv_bias = conv.bias
        bn_w = bn.weight
        bn_b = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps

        scale = bn_w / torch.sqrt(running_var + eps)
        w_fold = w * scale.reshape(-1, 1, 1, 1)
        b_fold = bn_b + (conv_bias - running_mean) * scale
        return w_fold, b_fold

    def get_equivalent_kernel_bias(self):
        # produce fused kernel and bias for all branches
        # dense branch
        k_dense, b_dense = 0, 0
        if hasattr(self, 'rbr_dense'):
            conv = self.rbr_dense[0]
            bn = self.rbr_dense[1]
            k_dense, b_dense = self._fuse_conv_bn(conv, bn)
        k_id, b_id = 0, 0
        if hasattr(self, 'rbr_identity'):
            bn = self.rbr_identity
            # identity conv kernel is delta: shape (C,C,1,1)
            c = bn.num_features
            identity_kernel = torch.zeros((c, c, 1, 1), device=bn.weight.device)
            for i in range(c):
                identity_kernel[i, i, 0, 0] = 1.0
            # we'll recalc simply:
            # scale = bn.weight / sqrt(running_var+eps)
            # bias = bn.bias - running_mean*scale
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            k_id = identity_kernel * scale.reshape(-1,1,1,1)
            b_id = bn.bias - bn.running_mean * scale

        k = k_dense + k_id
        b = b_dense + b_id
        return k, b

    def convert_to_deploy(self):
        if self.deploy:
            return
        k, b = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(k.size(1), k.size(0), k.size(2), padding=k.size(2)//2, bias=True)
        self.rbr_reparam.weight.data = k
        self.rbr_reparam.bias.data = b
        # delete old branches
        for attr in ['rbr_dense', 'rbr_identity']:
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True

# ---------------- SparseECA ----------------
class SparseECA(nn.Module):
    """Lightweight ECA but with aggressive projection (sparse) to save params."""
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

# ---------------- DFDBlockLite ----------------
class DFDBlockLite(nn.Module):
    def __init__(self, channels, distill_ratio=0.5, proj_ratio=0.15, use_eca=True, res_scale=0.1, act='prelu'):
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

        # context branch: depthwise conv (3x3, groups=c_c) -> 1x1 conv -> ReLU
        self.context = nn.Sequential(
            nn.Conv2d(c_c, c_c, 3, padding=1, groups=c_c, bias=False),
            nn.Conv2d(c_c, c_c, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        # fusion conv 1x1
        self.proj_f = nn.Conv2d(c_d + c_c, channels, 1, bias=False)

        self.use_eca = use_eca
        if self.use_eca:
            self.eca = SparseECA(channels, proj_ratio=proj_ratio)
        else:
            self.eca = None

        self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))

    def forward(self, x):
        c = x.size(1)
        c_d = int(c * self.distill_ratio)
        c_c = c - c_d
        xd = x[:, :c_d, :, :]
        xc = x[:, c_d:, :, :]
        yd = self.detail(xd)
        yc = self.context(xc)
        y = torch.cat([yd, yc], dim=1)
        if self.eca is not None:
            y = y + self.eca(y)
        y = self.proj_f(y)
        return x + y * self.res_scale

# ---------------- DFDN network ----------------
class DFDN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()
        n_feats = args.n_feats
        n_resblocks = args.n_resblocks
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # head
        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, 3, padding=1, bias=False),
            nn.PReLU(num_parameters=n_feats)
        )

        # body: stacked DFDBlockLite blocks
        body = []
        enhanced_start = n_resblocks * 3 // 4
        for i in range(n_resblocks):
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
                    nn.Conv2d(c_c, c_c, 3, padding=1, groups=1, bias=False),
                    nn.Conv2d(c_c, c_c, 1, bias=False),
                    nn.ReLU(inplace=True)
                )
                # tail强化部分 proj_ratio=0.18
                res_scale_val = 0.14
                if i == n_resblocks - 1:
                    res_scale_val = 0.15
                block = DFDBlockLite(n_feats, distill_ratio=distill_ratio, proj_ratio=0.18,
                                     use_eca=True, act=getattr(args,'act','prelu'), res_scale=res_scale_val)
                # replace detail and context with custom ones
                block.detail = detail
                block.context = context
            else:
                block = DFDBlockLite(n_feats, distill_ratio=0.5, proj_ratio=0.15,
                                     use_eca=True, act=getattr(args,'act','prelu'),
                                     res_scale=0.1)
            body.append(block)
        body.append(conv(n_feats, n_feats, 3, bias=False))
        self.body = nn.Sequential(*body)

        # tail
        scale = args.scale[0] if isinstance(args.scale, (list,tuple)) else args.scale
        self.tail = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, 3, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.sub_mean(x)
        h = self.head(x)
        body_layers = list(self.body)
        last = body_layers[-1]
        blocks = body_layers[:-1]
        out = h
        for block in blocks:
            out = block(out)
        res = last(out)
        res = res + h
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def convert_to_deploy(self):
        # convert all RepConv inside detail branches
        for m in self.modules():
            if isinstance(m, DFDBlockLite):
                for subm in m.detail.modules():
                    if isinstance(subm, RepConvBlock):
                        subm.convert_to_deploy()

# ---------------- factory and diagnostics ----------------
def make_model(args, parent=False):
    net = DFDN(args)
    check_modules(net)
    return net

def check_modules(net):
    has_rep = any(isinstance(m, RepConvBlock) for m in net.modules())
    has_eca = any(isinstance(m, SparseECA) for m in net.modules())
    print(f"[INFO] RepConv: {has_rep}, SparseECA: {has_eca}")
    # print params (safe on CPU)
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")