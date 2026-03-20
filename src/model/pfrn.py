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
    """ESAB / Enhanced Spatial Attention Block / 增强空间注意块"""

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
    """AFRM / Adaptive Fusion Refinement Module / 自适应融合细化模块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.sigmoid(self.conv2(self.act(self.conv1(x))))
        return x + x * weight


class DLFB(nn.Module):
    """DLFB / Dual-path Local Feature Modulation Block / 双路径局部特征调制块"""

    def __init__(self, channels):
        super().__init__()
        left_channels = channels // 2
        right_channels = channels - left_channels
        self.left_channels = left_channels

        self.left_branch = conv_layer(left_channels, left_channels, 3)
        self.right_bsconv = BSConvU(
            right_channels,
            right_channels,
            kernel_size=3,
            padding=1,
        )
        self.right_dw = depthwise_conv(right_channels, 5)
        self.fuse = nn.Conv2d(left_channels, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        left, right = torch.split(x, [self.left_channels, x.shape[1] - self.left_channels], dim=1)
        left_feat = self.left_branch(left)
        right_feat = self.right_dw(self.right_bsconv(right))
        merged = left_feat + right_feat
        return self.act(self.fuse(merged) + x)


class LFRB(nn.Module):
    """LFRB / Lightweight Feature Refinement Block / 轻量特征细化块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.dwconv = depthwise_conv(channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.dwconv(out)
        out = self.conv2(out)
        return out + x


class PFRB(nn.Module):
    """PFRB / Progressive Feature Refinement Block / 渐进式特征细化块"""

    def __init__(self, channels, act='silu', esa_channels=16):
        super().__init__()
        self.shortcut = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.pre = conv_layer(channels, channels, 3)
        self.act = activation(act)
        self.dlfb = DLFB(channels)
        self.lfrb = LFRB(channels)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.esab = ESAB(channels, esa_channels=esa_channels)

    def forward(self, x):
        local_shortcut = self.shortcut(x)
        out = self.pre(x)
        out = self.act(out)
        out = self.dlfb(out)
        out = self.lfrb(out)
        out = out + local_shortcut
        out = self.fuse(out)
        out = self.esab(out)
        return out + x


class PFRN(nn.Module):
    """PFRN / Progressive Feature Refinement Network / 渐进式特征细化网络"""

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

        # Shallow feature extraction / 浅层特征提取
        self.shallow_conv = conv_layer(in_channels, feature_channels, 3)
        self.blocks = nn.ModuleList(
            [PFRB(feature_channels, act=act, esa_channels=esa_channels) for _ in range(num_blocks)]
        )

        # Feature fusion / 特征融合
        self.fusion_conv = nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1, bias=True)
        self.afrm = AFRM(feature_channels)

        # Image reconstruction / 图像重建
        self.reconstruction = conv_layer(feature_channels, feature_channels, 3)
        self.upsampler = pixelshuffle_block(
            feature_channels,
            out_channels,
            upscale_factor=upscale,
        )

    def forward(self, x):
        shallow = self.shallow_conv(x)

        deep = shallow
        first_pfrb = None
        for idx, block in enumerate(self.blocks):
            deep = block(deep)
            if idx == 0:
                first_pfrb = deep

        fused = torch.cat([shallow, first_pfrb, deep], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.afrm(fused)

        lr_feat = self.reconstruction(fused + shallow)
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
