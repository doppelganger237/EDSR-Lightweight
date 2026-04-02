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


# -----------------------------
# SLVR original-style modules
# -----------------------------
class ACU_gate(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, rate=1, activation=nn.ELU()):
        super(ACU_gate, self).__init__()
        padding = int(rate * (ksize - 1) / 2)
        self.conv = nn.Conv2d(in_ch,
                              2 * out_ch,
                              kernel_size=ksize,
                              stride=stride,
                              padding=padding,
                              dilation=rate)
        self.activation = activation

    def forward(self, x):
        raw = self.conv(x)
        x1 = raw.split(int(raw.shape[1] / 2), dim=1)
        gate = torch.sigmoid(x1[0])
        out = self.activation(x1[1]) * gate
        return out


class B2Conv(nn.Module):
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

        self.pw = ACU_gate(
            in_ch=in_channels,
            out_ch=out_channels,
            ksize=1,
            stride=1,
            rate=1,
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
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea



class PartialB2ConvU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros",
                 scale=2):
        super().__init__()
        self.remaining_channels = in_channels // scale
        self.other_channels = in_channels - self.remaining_channels
        # 仅对部分通道做大核 depthwise 卷积，保留其余通道作为旁路。
        self.pdw = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=self.remaining_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=self.remaining_channels,
            bias=bias,
            padding_mode=padding_mode,
        )
        # 生成“特征 + 门控”两路信号，用门控对 depthwise 输出做逐元素调制。
        self.gate_generator = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=2 * self.remaining_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )
        self.feature_activation = nn.ELU()
        # 对拼接后的全部通道做 1x1 融合并映射到目标通道数。
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, fea):
        fea1, fea2 = torch.split(fea, [self.remaining_channels, self.other_channels], dim=1)
        fea1 = self.pdw(fea1)
        gate_and_feature = self.gate_generator(fea1)
        gate, feat = gate_and_feature.split(self.remaining_channels, dim=1)
        fea1 = self.feature_activation(feat) * torch.sigmoid(gate)
        fea = torch.cat((fea1, fea2), 1)
        fea = self.pw(fea)
        return fea


class PairFuse(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels

        self.pre = nn.Conv2d(channels, hidden_channels, 1, 1, 0, bias=True)
        self.act1 = nn.GELU()

        self.branch3 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            1,
            1,
            groups=hidden_channels,
            bias=True
        )
        self.branch5 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            5,
            1,
            2,
            groups=hidden_channels,
            bias=True
        )
        self.branch1 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            1,
            1,
            0,
            bias=True
        )

        self.local_merge = nn.Conv2d(hidden_channels * 2, hidden_channels, 1, 1, 0, bias=True)
        self.act2 = nn.GELU()
        self.final_merge = nn.Conv2d(hidden_channels * 2, channels, 1, 1, 0, bias=True)

    def forward(self, x):
        identity = x

        f = self.act1(self.pre(x))

        f3 = self.branch3(f)
        f5 = self.branch5(f)
        f1 = self.branch1(f)

        fm = self.act2(self.local_merge(torch.cat([f3, f5], dim=1)))
        out = self.final_merge(torch.cat([fm, f1], dim=1))

        return identity + out


class FPA(nn.Module):
    def __init__(self, embed_dim, fft_norm="ortho"):
        super(FPA, self).__init__()
        self.conv_layer1 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_layer2 = nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.conv_layer3 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.real))))
        )
        imag = ffted.imag + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.imag))))
        )
        ffted = torch.complex(real, imag)
        output = torch.fft.irfftn(
            ffted,
            s=x.shape[-2:],
            dim=fft_dim,
            norm=self.fft_norm,
        )
        return x * output


class PFDB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None):
        super(PFDB, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = PartialB2ConvU(in_channels, self.rc, kernel_size=5, padding=2)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = PartialB2ConvU(self.rc, self.rc, kernel_size=5, padding=2)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = PartialB2ConvU(self.rc, self.rc, kernel_size=5, padding=2)
        self.c4 = B2Conv(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.g12_fuse = PairFuse(self.dc * 2)
        self.g34_fuse = PairFuse(self.dc * 2)
        self.fpa_channels = in_channels
        self.c5 = nn.Conv2d(self.dc * 4, self.fpa_channels, 1, 1, 0)
        self.fpa = FPA(self.fpa_channels)
        self.c6 = nn.Conv2d(self.fpa_channels, in_channels, 1, 1, 0)
        self.pixel_norm = nn.LayerNorm(in_channels)

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
        out = self.fpa(out)
        out = self.c6(out)
        out = out.permute(0, 2, 3, 1)
        out = self.pixel_norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out + input


class SPFDN(nn.Module):
    def __init__(self,
                 args,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 upscale=2,
                 num_blocks=6):
        super(SPFDN, self).__init__()

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
            PFDB(feature_channels) for _ in range(num_blocks)
        ])

        self.fusion_conv = nn.Conv2d(feature_channels * num_blocks,
                                     feature_channels,
                                     kernel_size=1)

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
    model = SPFDN(args)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params / 1e3:.1f}K")
    return model
