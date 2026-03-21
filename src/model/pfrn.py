from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lib import BSConvU


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True, stride=1, groups=1):
    kernel_size = _make_pair(kernel_size)
    padding = (
        int((kernel_size[0] - 1) / 2),
        int((kernel_size[1] - 1) / 2),
    )
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


def depthwise_conv(channels, kernel_size, bias=True):
    return conv_layer(channels, channels, kernel_size, bias=bias, groups=channels)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'gelu':
        layer = nn.GELU()
    elif act_type == 'silu':
        layer = nn.SiLU(inplace=inplace)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type)
        )
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.'
            )
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
    conv = conv_layer(
        in_channels,
        out_channels * (upscale_factor ** 2),
        kernel_size,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """Enhanced spatial attention used inside each PFRB."""

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


class DLFB(nn.Module):
    """Dual local feature block from the design figure."""

    def __init__(self, channels, act='silu'):
        super().__init__()
        left_channels = channels // 2
        right_channels = channels - left_channels
        self.left_channels = left_channels
        self.right_channels = right_channels

        self.left_conv = conv_layer(left_channels, left_channels, 3)
        self.right_bsconv = BSConvU(
            right_channels,
            right_channels,
            kernel_size=3,
            padding=1,
        )
        self.right_dwconv = depthwise_conv(right_channels, 5)
        self.concat_fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.refine = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = activation(act)

    def forward(self, x):
        left, right = torch.split(x, [self.left_channels, self.right_channels], dim=1)
        left = self.left_conv(left)
        right = self.right_dwconv(self.right_bsconv(right))
        out = torch.cat([left, right], dim=1)
        out = self.concat_fuse(out)
        out = self.act(out)
        out = self.refine(out)
        return x + out


class CFRM(nn.Module):
    """
    Cross-feature refinement module.

    The diagram shows a residual path over the concatenated tensor. We keep that
    behavior while projecting the residual back to the feature width so the block
    output matches the rest of the network.
    """

    def __init__(self, channels, act='silu'):
        super().__init__()
        merged_channels = channels * 2
        self.reduce = nn.Conv2d(merged_channels, channels, kernel_size=1, bias=True)
        self.act = activation(act)
        self.conv3 = conv_layer(channels, channels, 3)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.skip = nn.Conv2d(merged_channels, channels, kernel_size=1, bias=True)

    def forward(self, x1, x2):
        merged = torch.cat([x1, x2], dim=1)
        out = self.reduce(merged)
        out = self.act(out)
        out = self.conv3(out)
        out = self.conv1(out)
        return out + self.skip(merged)


class PFRB(nn.Module):
    """Progressive feature refinement block."""

    def __init__(self, channels, act='silu', esa_channels=16):
        super().__init__()
        self.pre = conv_layer(channels, channels, 3)
        self.pre_act = activation(act)

        self.local_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.local_dw = depthwise_conv(channels, 3)
        self.dlfb = DLFB(channels, act=act)
        self.cfrm = CFRM(channels, act=act)
        self.esa = ESA(channels, esa_channels=esa_channels)

    def forward(self, x):
        feat = self.pre_act(self.pre(x))
        local_feat = self.local_dw(self.local_proj(feat))
        dlfb_feat = self.dlfb(feat)
        out = self.cfrm(local_feat, dlfb_feat)
        out = self.esa(out)
        return x + out


class PFRN(nn.Module):
    """PFRN redesigned to follow the provided shallow/deep/reconstruction layout."""

    def __init__(
        self,
        args,
        in_channels=3,
        out_channels=3,
        feature_channels=52,
        upscale=2,
        num_blocks=6,
        esa_channels=16,
    ):
        super().__init__()

        num_blocks = args.n_resblocks
        feature_channels = args.n_feats
        upscale = args.scale[0]
        in_channels = args.n_colors
        out_channels = args.n_colors
        act = args.act

        self.shallow_conv = conv_layer(in_channels, feature_channels, 3)
        self.blocks = nn.ModuleList(
            [PFRB(feature_channels, act=act, esa_channels=esa_channels) for _ in range(num_blocks)]
        )

        self.deep_conv = conv_layer(feature_channels, feature_channels, 3)
        self.fusion_conv1 = nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1, bias=True)
        self.fusion_conv3 = conv_layer(feature_channels, feature_channels, 3)
        self.reconstruction = conv_layer(feature_channels, feature_channels, 3)
        self.upsampler = pixelshuffle_block(
            feature_channels,
            out_channels,
            upscale_factor=upscale,
        )

    def forward(self, x):
        shallow = self.shallow_conv(x)
        f0 = shallow

        deep = shallow
        f1 = None
        for idx, block in enumerate(self.blocks):
            deep = block(deep)
            if idx == 0:
                f1 = deep

        if f1 is None:
            f1 = deep

        f6 = deep
        f7 = self.deep_conv(f6)

        fused = self.fusion_conv1(torch.cat([f1, f6, f7], dim=1))
        fused = self.fusion_conv3(fused)
        lr_feat = self.reconstruction(fused + f0)
        return self.upsampler(lr_feat)

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
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'.format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


def make_model(args, parent=False):
    model = PFRN(args)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params / 1e3:.1f}K")
    return model


if __name__ == "__main__":
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Running on: {device}")

    class Args:
        n_resblocks = 6
        n_feats = 52
        scale = [2]
        n_colors = 3
        act = 'silu'

    args = Args()
    model = PFRN(args).to(device)
    model.eval()

    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        inputs = torch.rand(1, 3, 640, 360).to(device)
        print(flop_count_table(FlopCountAnalysis(model, inputs=(inputs,))))
    except ImportError:
        print("fvcore not installed; skipping FLOPs analysis.")
