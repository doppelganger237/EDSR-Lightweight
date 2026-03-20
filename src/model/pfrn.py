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


class ESAB(nn.Module):
    """Enhanced spatial attention block."""

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


class AFRM(nn.Module):
    """Adaptive fusion refinement module following the new dual-input design."""

    def __init__(self, channels):
        super().__init__()
        self.shallow_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.deep_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.refine1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.refine2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, shallow, deep):
        shallow_feat = self.shallow_proj(shallow)
        deep_feat = self.deep_proj(deep)
        gate = self.sigmoid(self.gate_conv(torch.cat([shallow_feat, deep_feat], dim=1)))
        fused = shallow_feat + deep_feat * gate
        refined = self.refine2(self.act(self.refine1(fused)))
        return fused + refined


class DLFB(nn.Module):
    """Dual-path local feature block."""

    def __init__(self, channels):
        super().__init__()
        left_channels = channels // 2
        right_channels = channels - left_channels
        self.left_channels = left_channels
        self.right_channels = right_channels

        self.left_branch = conv_layer(left_channels, left_channels, 3)
        self.right_bsconv = BSConvU(
            right_channels,
            right_channels,
            kernel_size=3,
            padding=1,
        )
        self.right_dw = depthwise_conv(right_channels, 5)
        self.right_align = None
        if right_channels != left_channels:
            self.right_align = nn.Conv2d(right_channels, left_channels, kernel_size=1, bias=True)

        self.fuse = nn.Conv2d(left_channels, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        left, right = torch.split(x, [self.left_channels, self.right_channels], dim=1)
        left_feat = self.left_branch(left)
        right_feat = self.right_dw(self.right_bsconv(right))
        if self.right_align is not None:
            right_feat = self.right_align(right_feat)
        merged = left_feat + right_feat
        return x + self.act(self.fuse(merged))


class PFRB(nn.Module):
    """Progressive feature refinement block."""

    def __init__(self, channels, act='silu', esa_channels=16):
        super().__init__()
        self.branch1_pw1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.branch1_dw = depthwise_conv(channels, 3)
        self.branch1_pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.branch2_conv = conv_layer(channels, channels, 3)
        self.branch2_act = activation(act)
        self.dlfb = DLFB(channels)

        self.fuse = conv_layer(channels, channels, 3)
        self.esab = ESAB(channels, esa_channels=esa_channels)

    def forward(self, x):
        branch1 = self.branch1_pw2(self.branch1_dw(self.branch1_pw1(x)))
        branch2 = self.dlfb(self.branch2_act(self.branch2_conv(x)))
        out = self.fuse(branch1 + branch2)
        out = self.esab(out)
        return x + out


class PFRN(nn.Module):
    """Progressive feature refinement network."""

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

        self.fusion_conv = nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1, bias=True)
        self.afrm = AFRM(feature_channels)

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
        f1 = shallow
        for idx, block in enumerate(self.blocks):
            deep = block(deep)
            if idx == 0:
                f1 = deep
        fn = deep

        fused = self.fusion_conv(torch.cat([f0, f1, fn], dim=1))
        refined = self.afrm(f0, fused)
        final_feat = refined + fused
        lr_feat = self.reconstruction(final_feat)
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
