import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# =========================================================
# basic functions
# =========================================================
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        groups=groups
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(f'padding layer [{pad_type}] is not implemented')
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'gelu':
        layer = nn.GELU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f'activation layer [{act_type}] is not found')
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(
        in_nc, out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups
    )
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


# =========================================================
# statistics for CCA
# =========================================================
def mean_channels(x):
    assert x.dim() == 4
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.size(2) * x.size(3))


def stdv_channels(x):
    assert x.dim() == 4
    x_mean = mean_channels(x)
    x_variance = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
    return x_variance.pow(0.5)


# =========================================================
# B2Conv
# =========================================================
class ACU_gate(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, rate=1, activation_layer=nn.ELU()):
        super().__init__()
        padding = int(rate * (ksize - 1) / 2)
        self.conv = nn.Conv2d(
            in_ch,
            2 * out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=rate
        )
        self.activation = activation_layer

    def forward(self, x):
        raw = self.conv(x)
        x1, x2 = raw.chunk(2, dim=1)
        gate = torch.sigmoid(x1)
        out = self.activation(x2) * gate
        return out


class B2Conv(nn.Module):
    """
    ACU-gated pointwise + depthwise spatial conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros"):
        super().__init__()
        self.pw = ACU_gate(
            in_ch=in_channels,
            out_ch=out_channels,
            ksize=1,
            stride=1,
            rate=1
        )
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode
        )

    def forward(self, x):
        x = self.pw(x)
        x = self.dw(x)
        return x


# =========================================================
# helper blocks
# =========================================================
class ChannelShuffle(nn.Module):
    def __init__(self, groups=4):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        if c % g != 0:
            return x
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        mid = max(4, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, mid, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = stdv_channels(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ESA(nn.Module):
    """
    spatial attention
    """
    def __init__(self, channels, esa_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, esa_channels, kernel_size=1)
        self.conv_f = nn.Conv2d(esa_channels, esa_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(esa_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        pooled = F.max_pool2d(c2, kernel_size=7, stride=3)
        c3 = self.conv3(pooled)
        c3 = F.interpolate(c3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cf = self.conv_f(c1)
        mask = self.sigmoid(self.conv4(c3 + cf))
        return x * mask


# =========================================================
# RP: Self-Rectified Blueprint Extraction Unit
# =========================================================
class SRBEU(nn.Module):
    """
    Conv1 -> GELU -> B2Conv3 -> DWConv5 -> Conv1
    single-path, stage-wise, no explicit multi-branch
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.b2 = B2Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.dw5 = nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels, bias=True)
        self.refine = nn.Conv2d(out_channels, out_channels, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.proj(x))
        x = self.act(self.b2(x))
        x = self.act(self.dw5(x))
        x = self.refine(x)
        return x


# =========================================================
# DP: Multi-scale Differential Rectification Module
# =========================================================
class CIRM(nn.Module):
    """
    Cross-stage Interactive Rectification Module
    Two-input rectification with local and contextual interaction.
    """
    def __init__(self, channels, shuffle_groups=4):
        super().__init__()
        self.shuffle = ChannelShuffle(groups=shuffle_groups)

        self.align = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.pool_conv = nn.Conv2d(channels, channels, 1, bias=True)

        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.out = nn.Conv2d(channels, channels, 1, bias=True)

        self.act = nn.GELU()

    def forward(self, fa, fb):
        x = torch.cat([fa, fb], dim=1)
        x = self.shuffle(x)
        base = self.act(self.align(x))

        local = self.act(self.dw3(base))

        context = F.max_pool2d(base, kernel_size=2, stride=2)
        context = self.act(self.pool_conv(context))
        context = F.interpolate(context, size=base.shape[-2:], mode='bilinear', align_corners=False)

        refine = self.act(self.fuse(torch.cat([local, context], dim=1)))
        out = self.out(base + refine)
        return out


# =========================================================
# BF: Hierarchical Distillation Fusion Refinement Module
# =========================================================
class THFM(nn.Module):
    """
    Tree-structured Hierarchical Fusion Module
    Group-wise fusion first, then global fusion.
    """
    def __init__(self, stage_channels, out_channels):
        super().__init__()

        self.fuse12_reduce = nn.Conv2d(stage_channels * 2, out_channels, 1, bias=True)
        self.fuse34_reduce = nn.Conv2d(stage_channels * 2, out_channels, 1, bias=True)

        self.fuse12_dw = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=True)
        self.fuse34_dw = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=True)

        self.group_out1 = nn.Conv2d(out_channels, out_channels, 1, bias=True)
        self.group_out2 = nn.Conv2d(out_channels, out_channels, 1, bias=True)

        self.global_shuffle = ChannelShuffle(groups=4)
        self.global_reduce = nn.Conv2d(out_channels * 2, out_channels, 1, bias=True)
        self.global_b2 = B2Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.global_out = nn.Conv2d(out_channels, out_channels, 1, bias=True)

        self.act = nn.GELU()

    def forward(self, p12, d2, p34, t):
        g1 = self.act(self.fuse12_reduce(torch.cat([p12, d2], dim=1)))
        g1 = self.group_out1(self.act(self.fuse12_dw(g1)))

        g2 = self.act(self.fuse34_reduce(torch.cat([p34, t], dim=1)))
        g2 = self.group_out2(self.act(self.fuse34_dw(g2)))

        x = torch.cat([g1, g2], dim=1)
        x = self.global_shuffle(x)
        x = self.act(self.global_reduce(x))
        x = self.act(self.global_b2(x))
        x = self.global_out(x)
        return x


# =========================================================
# Tail recalibration
# =========================================================
class SCRM(nn.Module):
    """
    Spatial-Channel Recalibration Module
    """
    def __init__(self, channels, esa_channels=16):
        super().__init__()
        self.esa = ESA(channels, esa_channels=esa_channels)

    def forward(self, x):
        x = self.esa(x)
        return x


# =========================================================
# block
# =========================================================
class B2RFDB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dc = in_channels // 2
        self.rc = in_channels

        # distillation path
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c2_d = conv_layer(self.rc, self.dc, 1)
        self.c3_d = conv_layer(self.rc, self.dc, 1)

        # remaining path
        self.c1_r = SRBEU(in_channels, self.rc)
        self.c2_r = SRBEU(self.rc, self.rc)
        self.c3_r = SRBEU(self.rc, self.rc)
        self.c4 = B2Conv(self.rc, self.dc, kernel_size=3, padding=1)

        # DP
        self.p12 = CIRM(self.dc)
        self.p34 = CIRM(self.dc)

        # BF
        self.hdfrm = THFM(stage_channels=self.dc, out_channels=in_channels)

        # tail recalibration
        self.scrm = SCRM(in_channels)

        self.act = nn.GELU()

    def forward(self, x):
        d1 = self.act(self.c1_d(x))
        r1 = self.act(self.c1_r(x) + x)

        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1) + r1)

        d3 = self.act(self.c3_d(r2))
        r3 = self.act(self.c3_r(r2) + r2)

        t = self.act(self.c4(r3))

        p12 = self.p12(d1, d2)
        p34 = self.p34(d3, t)

        fused = self.hdfrm(p12, d2, p34, t)
        out = self.scrm(fused)

        return out + x


# =========================================================
# network
# =========================================================
class B2RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super().__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.blocks = nn.ModuleList([B2RFDB(in_channels=nf) for _ in range(num_modules)])

        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='gelu')
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)

        block_outputs = []
        x = out_fea
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)

        out_B = self.c(torch.cat(block_outputs, dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


# =========================================================
# entry
# =========================================================
def make_model(args, parent=False):
    model = B2RFDN(
        num_modules=args.n_resblocks,
        nf=args.n_feats,
        upscale=args.scale[0],
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params / 1e3:.1f}K")
    return model
