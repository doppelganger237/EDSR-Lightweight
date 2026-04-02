from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


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


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def mean_channels(x):
    assert x.dim() == 4
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.size(2) * x.size(3))


def stdv_channels(x):
    assert x.dim() == 4
    x_mean = mean_channels(x)
    x_variance = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
    return x_variance.pow(0.5)


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


class MCCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        mid = max(8, channel // reduction)
        self.local_context = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=True),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel * 3, mid, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ctx = x + self.local_context(x)
        y_avg = self.avg_pool(x_ctx)
        y_std = stdv_channels(x_ctx)
        y_max = self.max_pool(x_ctx)
        y = torch.cat([y_avg, y_std, y_max], dim=1)
        y = self.conv_du(y)
        return x * y


class LKSCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        mid = max(8, channel // reduction)
        self.dw_h1 = nn.Conv2d(
            channel, channel, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=channel, bias=True
        )
        self.dw_v1 = nn.Conv2d(
            channel, channel, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=channel, bias=True
        )
        self.dw_h2 = nn.Conv2d(
            channel, channel, kernel_size=(1, 11), stride=1, padding=(0, 5), groups=channel, bias=True
        )
        self.dw_v2 = nn.Conv2d(
            channel, channel, kernel_size=(11, 1), stride=1, padding=(5, 0), groups=channel, bias=True
        )
        self.spatial_proj = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, mid, kernel_size=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_act = nn.Sigmoid()

    def forward(self, x):
        attn = self.dw_h1(x)
        attn = self.dw_v1(attn)
        attn = self.dw_h2(attn)
        attn = self.dw_v2(attn)
        attn = self.spatial_act(self.spatial_proj(attn))
        gate = self.channel_gate(x)
        return x * attn * gate


class MCBSConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros",
                 with_ln=False,
                 bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        if bn_kwargs is None:
            bn_kwargs = {}

        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2
        self.split_channels = (c1, c2, c3)

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.dw1 = None
        if c1 > 0:
            self.dw1 = nn.Conv2d(
                in_channels=c1,
                out_channels=c1,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=c1,
                bias=bias,
                padding_mode=padding_mode,
            )
        self.dw3 = None
        if c2 > 0:
            self.dw3 = nn.Conv2d(
                in_channels=c2,
                out_channels=c2,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=c2,
                bias=bias,
                padding_mode=padding_mode,
            )
        self.dw5 = None
        if c3 > 0:
            self.dw5 = nn.Conv2d(
                in_channels=c3,
                out_channels=c3,
                kernel_size=5,
                stride=stride,
                padding=2,
                groups=c3,
                bias=bias,
                padding_mode=padding_mode,
            )
        self.fuse = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.act = nn.ELU()

    def forward(self, x):
        x_proj = self.proj(x)
        x_gate, x_feat = torch.chunk(x_proj, 2, dim=1)
        x_gate = torch.sigmoid(x_gate)
        x_feat = self.act(x_feat)
        x_mod = x_gate * x_feat

        x1, x2, x3 = torch.split(x_mod, self.split_channels, dim=1)
        y1 = self.dw1(x1) if self.dw1 is not None else x1
        y2 = self.dw3(x2) if self.dw3 is not None else x2
        y3 = self.dw5(x3) if self.dw5 is not None else x3

        out = torch.cat([y1, y2, y3], dim=1)
        out = self.fuse(out)
        return out


class PairFuse(nn.Module):
    def __init__(self, channels, hidden_channels=None, kernel_size=5):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        padding = kernel_size // 2
        self.pw1 = conv_layer(channels, hidden_channels, 1)
        self.act1 = nn.GELU()
        self.dw5 = nn.Conv2d(hidden_channels,
                             hidden_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=padding,
                             groups=hidden_channels,
                             bias=True)
        self.act2 = nn.GELU()
        self.dw3 = nn.Conv2d(hidden_channels,
                             hidden_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             groups=hidden_channels,
                             bias=True)
        self.pw2 = conv_layer(hidden_channels, channels, 1)

    def forward(self, x):
        identity = x
        x = self.pw1(x)
        x = self.act1(x)
        x = self.dw5(x)
        x = self.act2(x)
        x = self.dw3(x)
        x = self.pw2(x)
        return x + identity


class MFDB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None):
        super(MFDB, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.conv = MCBSConv

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = self.conv(in_channels, self.rc, kernel_size=5, padding=2)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = self.conv(self.rc, self.rc, kernel_size=5, padding=2)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = self.conv(self.rc, self.rc, kernel_size=5, padding=2)
        self.c4 = self.conv(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.g12_fuse = PairFuse(self.dc * 2, kernel_size=5)
        self.g34_fuse = PairFuse(self.dc * 2, kernel_size=5)
        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1, 1, 0)
        self.c6 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.lksca = LKSCA(in_channels)
        self.cca = MCCALayer(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.act(self.c1_r(input))

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1))

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2))

        r_c4 = self.act(self.c4(r_c3))

        g12 = self.g12_fuse(torch.cat([distilled_c1, distilled_c2], dim=1))
        g34 = self.g34_fuse(torch.cat([distilled_c3, r_c4], dim=1))

        out = torch.cat([g12, g34], dim=1)
        out = self.c5(out)
        out = self.c6(out)
        out = self.lksca(out)
        out = self.cca(out)

        return out + input


class MFDN(nn.Module):
    def __init__(self,
                 args,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 upscale=2,
                 num_blocks=6):
        super(MFDN, self).__init__()

        num_blocks = args.n_resblocks
        feature_channels = args.n_feats
        upscale = args.scale[0]
        in_channels = args.n_colors
        out_channels = args.n_colors
        kernel_size = 3

        self.conv_1 = conv_layer(in_channels,
                                 feature_channels,
                                 kernel_size=kernel_size)

        self.blocks = nn.ModuleList([
            MFDB(feature_channels) for _ in range(num_blocks)
        ])

        self.fusion_conv = nn.Conv2d(feature_channels * num_blocks,
                                     feature_channels,
                                     kernel_size=1)
        self.GELU = nn.GELU()

        self.conv_2 = conv_layer(feature_channels,
                                 feature_channels,
                                 kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                            out_channels,
                                            upscale_factor=upscale)

    def forward(self, x):
        shallow = self.conv_1(x)

        out = shallow
        block_outs = []
        for block in self.blocks:
            out = block(out)
            block_outs.append(out)

        fused = self.fusion_conv(torch.cat(block_outs, dim=1))
        fused = self.GELU(fused)
        fused = self.conv_2(fused) + shallow
        output = self.upsampler(fused)
        return output

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1 and name.find('upsampler') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


def make_model(args, parent=False):
    model = MFDN(args)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params / 1e3:.1f}K")
    return model

