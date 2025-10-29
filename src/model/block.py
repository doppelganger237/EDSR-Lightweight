import torch.nn as nn

import torch
import torch.nn.functional as F
from collections import OrderedDict
from model.ghost import GhostModule as GhostConv

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class BSConvU(nn.Module):
    """Blueprint Separable Convolution (Unshared)
       按照原论文定义：先 1×1 Conv，再 Depthwise Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=True):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        # 先 Pointwise 生成通道蓝图
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        # 再 Depthwise 应用空间模板
        self.dw = nn.Conv2d(out_channels, out_channels, kernel_size,
                            stride=stride, padding=padding,
                            groups=out_channels, bias=bias)

    def forward(self, x):
        return self.dw(self.pw(x))



# Alias: BSConv = BSConvU
class BSConv(BSConvU):
    """
    BSConv: Alias of BSConvU for convenience.
    默认使用Unshared Blueprint版本。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=True):
        super(BSConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
    


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
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
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


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
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

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

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


# class ESA(nn.Module):
#     def __init__(self, n_feats, conv):
#         super(ESA, self).__init__()
#         f = n_feats // 4
#         self.conv1 = conv(n_feats, f, kernel_size=1)
#         self.conv_f = conv(f, f, kernel_size=1)
#         self.conv_max = conv(f, f, kernel_size=3)
#         self.conv2 = conv(f, f, kernel_size=3, stride=2)
#         self.conv3 = conv(f, f, kernel_size=3)
#         self.conv3_ = conv(f, f, kernel_size=3)
#         self.conv4 = conv(f, n_feats, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         c1_ = self.conv1(x)
#         c1 = self.conv2(c1_)
#         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
#         v_range = self.relu(self.conv_max(v_max))
#         c3 = self.relu(self.conv3(v_range))
#         c3 = self.conv3_(c3)
#         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
#         cf = self.conv_f(c1_)
#         c4 = self.conv4(c3 + cf)
#         m = self.sigmoid(c4)
#         return x * m
    
# class ECA(nn.Module):
#     """Efficient Channel Attention (轻量通道注意力)"""
#     def __init__(self, channels, k_size=3):
#         super(ECA, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
        
class MESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(MESA, self).__init__()
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

class CCA(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CCA, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    

# class CSR(nn.Module):
#     """Contrast-aware Spatial Refinement (CSR) Module"""
#     def __init__(self, channels, kernel_size=5):
#         super(CSR, self).__init__()
#         self.dw = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels, bias=True)
#         self.pw = nn.Conv2d(channels, channels, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         att = self.sigmoid(self.pw(self.dw(x)))
#         return x * att + x

# class GLA(nn.Module):
#     """Global–Local Attention (融合全局通道与局部空间信息)"""
#     def __init__(self, channels, reduction=4, kernel_size=5):
#         super(GLA, self).__init__()
#         reduced_c = channels // reduction
#         # Global Channel Attention
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.global_conv = nn.Sequential(
#             nn.Conv2d(channels, reduced_c, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(reduced_c, channels, 1, bias=True),
#             nn.Sigmoid()
#         )
#         # Local Spatial Attention
#         self.local_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         g_att = self.global_conv(self.global_pool(x))
#         l_att = self.local_conv(x)
#         att = g_att * l_att
#         # Phase-Shifted Attention Enhancement
#         shift = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         shift = torch.sigmoid(shift)
#         att = att + 0.5 * shift
#         return x * att + x





# class RFDB(nn.Module):
#     def __init__(self, in_channels):
#         super(RFDB, self).__init__()
#         self.dc = self.distilled_channels = in_channels // 2
#         self.rc = self.remaining_channels = in_channels
#         self.c1_d = conv_layer(in_channels, self.dc, 1)
#         # DW+PW for c1_r
#         self.c1_r_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=True)
#         self.c1_r_pw = nn.Conv2d(in_channels, in_channels, 1, bias=True)
#         self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
#         self.c2_r_dw = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, padding=1, groups=self.remaining_channels, bias=True)
#         self.c2_r_pw = nn.Conv2d(self.remaining_channels, self.remaining_channels, 1, bias=True)
#         self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
#         self.c3_r_dw = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, padding=1, groups=self.remaining_channels, bias=True)
#         self.c3_r_pw = nn.Conv2d(self.remaining_channels, self.remaining_channels, 1, bias=True)
#         # DW+PW for c4
#         self.c4_dw = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, padding=1, groups=self.remaining_channels, bias=True)
#         self.c4_pw = nn.Conv2d(self.remaining_channels, self.dc, 1, bias=True)
#         self.act = activation('lrelu', neg_slope=0.05)
#         self.c5 = conv_layer(self.dc * 4, in_channels, 1)
#         self.gla = GLA(in_channels)

#     def forward(self, input):
#         distilled_c1 = self.act(self.c1_d(input))
#         # c1_r: DW+PW
#         r_c1 = self.c1_r_dw(input)
#         r_c1 = self.c1_r_pw(r_c1)
#         r_c1 = self.act(r_c1 + input)

#         distilled_c2 = self.act(self.c2_d(r_c1))
#         r_c2 = self.c2_r_dw(r_c1)
#         r_c2 = self.c2_r_pw(r_c2)
#         r_c2 = self.act(r_c2 + r_c1)

#         distilled_c3 = self.act(self.c3_d(r_c2))
#         r_c3 = self.c3_r_dw(r_c2)
#         r_c3 = self.c3_r_pw(r_c3)
#         r_c3 = self.act(r_c3 + r_c2)

#         r_c4 = self.c4_dw(r_c3)
#         r_c4 = self.c4_pw(r_c4)
#         r_c4 = self.act(r_c4)

#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
#         out_fused = self.c5(out)
#         out_fused = self.gla(out_fused)
#         return out_fused

class BFFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16):
        super(BFFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        # 前两层使用BSConv（先1x1再DWConv，保持与BSConvU一致）
        #self.c1_r = BSConv(in_channels, in_channels, kernel_size=3)
        #self.c2_r = BSConv(in_channels, in_channels, kernel_size=3)
        self.c1_r = BSConv(in_channels, in_channels, kernel_size=3)
        self.c2_r = BSConv(in_channels, in_channels, kernel_size=3)

        # 第三层保留标准Conv，保持通道融合能力
        #self.c3_r = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True)

        # 第三层GhostConv
        self.c3_r = GhostConv(in_channels, in_channels, kernel_size=3, ratio=0.5, stride=1, bias=True)

        #self.c3_r = BSConv(in_channels, in_channels, kernel_size=3)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        
        self.cca = CCA(out_channels, reduction=4)
        self.esa = MESA(esa_channels, out_channels, nn.Conv2d)
        #self.act = activation('lrelu', neg_slope=0.05)
        self.act = activation('gelu')

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.c5(out)
        # 先经过CCA通道注意力，再送入MESA
        out = self.cca(out)
        out = self.esa(out)

        return out


