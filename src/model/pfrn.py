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


class DDFB(nn.Module):
    """Distilled dual-path feature block."""

    def __init__(self, channels, act='silu'):
        super().__init__()
        distilled_channels = channels // 2

        self.main_bsconv = BSConvU(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.main_act = activation(act)

        self.reduce = nn.Conv2d(channels, distilled_channels, kernel_size=1, bias=True)
        self.aux_bsconv = BSConvU(
            distilled_channels,
            distilled_channels,
            kernel_size=3,
            padding=1,
        )
        self.aux_dwconv = depthwise_conv(distilled_channels, 5)
        self.expand = nn.Conv2d(distilled_channels, channels, kernel_size=1, bias=True)

        self.concat_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.act = activation(act)
        self.refine = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        main = self.main_act(self.main_bsconv(x))
        aux = self.reduce(x)
        aux = self.aux_dwconv(self.aux_bsconv(aux))
        aux = self.expand(aux)
        out = torch.cat([main, aux], dim=1)
        out = self.concat_fuse(out)
        out = self.act(out)
        out = self.refine(out)
        return x + out


class PCRM(nn.Module):
    """Pixel-wise channel recalibration module."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dwconv = depthwise_conv(channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.conv1(x)
        mask = self.dwconv(mask)
        mask = self.sigmoid(self.conv2(mask))
        return x * mask


class PFRB(nn.Module):
    """Progressive feature refinement block."""

    def __init__(self, channels, act='silu', esa_channels=16):
        super().__init__()
        self.pre = conv_layer(channels, channels, 3)
        self.pre_act = activation(act)
        self.ddfb = DDFB(channels, act=act)
        self.post = conv_layer(channels, channels, 3)
        self.post_act = activation(act)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.pcrm = PCRM(channels)
        self.esa = ESA(channels, esa_channels=esa_channels)

    def forward(self, x):
        identity = x
        feat = self.pre_act(self.pre(x))
        feat = self.ddfb(feat)
        feat = self.post_act(self.post(feat))
        feat = feat + identity
        feat = self.fuse(feat)
        feat = self.pcrm(feat)
        return self.esa(feat)


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

        self.fusion_conv1 = nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
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
        block_outputs = []
        for block in self.blocks:
            deep = block(deep)
            block_outputs.append(deep)

        if not block_outputs:
            block_outputs = [f0, f0, f0, f0]
        elif len(block_outputs) == 1:
            block_outputs = [block_outputs[0]] * 4
        elif len(block_outputs) == 2:
            block_outputs = [block_outputs[0], block_outputs[1], block_outputs[0], block_outputs[1]]
        elif len(block_outputs) == 3:
            block_outputs = [block_outputs[0], block_outputs[1], block_outputs[1], block_outputs[2]]
        else:
            block_outputs = [
                block_outputs[0],
                block_outputs[1],
                block_outputs[-2],
                block_outputs[-1],
            ]

        fused = self.fusion_conv1(torch.cat(block_outputs, dim=1))
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
