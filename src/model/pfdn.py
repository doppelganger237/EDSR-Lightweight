# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import torch.nn as nn
# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch    
from model.lib import BSConvU, simam_module


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), 
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].

    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    
    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class LCA(nn.Module):
    """Lightweight Channel Attention: GAP + 1x1 Conv + Sigmoid"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return x * y  # channel-wise scaling
    
class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class PFDB(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16,
                 lca_reduction=32):
        super(PFDB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = conv_layer(in_channels, mid_channels, 3)

        half_channels = mid_channels // 2
        self.c2_r_front = conv_layer(half_channels, half_channels, 3)
        self.c2_r_back = BSConvU(mid_channels - half_channels, mid_channels - half_channels, kernel_size=3)
        self.mix_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        
        self.c3_r = conv_layer(mid_channels, in_channels, 3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.lca = LCA(mid_channels, reduction=lca_reduction)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
        self.act = activation('gelu', neg_slope=0.05)


    def forward(self, x):
        out = self.c1_r(x)
        out = self.act(out)
        out = self.lca(out)

        half = out.shape[1] // 2
        front = out[:, :half]
        back = out[:, half:]

        front = self.c2_r_front(front)
        back = self.c2_r_back(back)
        out = torch.cat([front, back], dim=1)

        out = self.mix_conv(out)   # 通道交互融合
        out = self.act(out)
        out = self.lca(out)

        out = self.c3_r(out)
        out = self.act(out)
        out = self.lca(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out



class PFDN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 args,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 upscale=2,
                 num_blocks=6):
        super(PFDN, self).__init__()

        num_blocks = args.n_resblocks 
        feature_channels = args.n_feats
        upscale = args.scale[0]
        
        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.blocks = nn.ModuleList([PFDB(feature_channels) for _ in range(num_blocks)])

        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.fusion_conv = nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1)

        self.upsampler = pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        block_outputs = []
        out = out_feature
        for block in self.blocks:
            out = block(out)
            block_outputs.append(out)

        fused = self.fusion_conv(torch.cat([block_outputs[0], block_outputs[-1]], dim=1))

        out_low_resolution = self.conv_2(fused) + out_feature
        output = self.upsampler(out_low_resolution)

        return output
    
def make_model(args, parent=False):
    model =  PFDN(args)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e3:.1f}K")

    return model