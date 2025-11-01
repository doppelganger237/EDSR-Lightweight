import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
#from model.OREPA import OREPA
from model.ecb import ECB

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
    


import torch
import torch.nn as nn
import torch.nn.functional as F

# Reparameterization: (3*3) U (3*1) U (1*3) U (1*1) U (identity) -> (3*3)
class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
       # self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                            stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                            stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')

    def forward(self, inputs):
        if self.deploy:
            return self.rbr_reparam(inputs)
        else:
            return (self.rbr_3x3_branch(inputs)) + \
                   (self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(inputs) + self.rbr_1x1_branch(inputs)) + \
                   (inputs)

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_3x3_branch')
        self.__delattr__('rbr_3x1_branch')
        self.__delattr__('rbr_1x3_branch')
        self.__delattr__('rbr_1x1_branch')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data
        # 1x1 1x3 3x1 branch
        kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
                                                                                       self.rbr_1x3_branch,
                                                                                       self.rbr_3x1_branch)
        # identity branch
        device = kernel_1x1_1x3_3x1_fuse.device  # just for getting the device
        kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            kernel_identity[i, i, 1, 1] = 1.0

        return kernel_3x3 + kernel_1x1_1x3_3x1_fuse + kernel_identity, \
               bias_3x3 + bias_1x1_1x3_3x1_fuse


    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
            conv3.weight.data, (1, 1, 0, 0))
        bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
        return weight, bias


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
    elif act_type == 'silu':
        layer = nn.SiLU(inplace=inplace)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


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
    


class BFFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels=16,
                 deploy=False,
                 use_esa=True):
        super(BFFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.use_esa = use_esa

        self.c1_r = ECB(
            inp_planes=in_channels,
            out_planes=mid_channels,
            depth_multiplier= 2,
            act_type="linear",with_idt=False)
        
        self.c2_r = ECB(
            inp_planes=mid_channels,
            out_planes=mid_channels,
            depth_multiplier= 2,
            act_type="linear",with_idt=False)
        
        self.c3_r = ECB(
            inp_planes=mid_channels,
            out_planes=out_channels,
            depth_multiplier= 2,
            act_type="linear",with_idt=False)
        
        # self.c1_r = OREPA(
        #     in_channels=in_channels,
        #     out_channels=mid_channels,
        #     kernel_size=3,
        #     padding=1,   # ✅ 一定要加
        #     deploy=deploy,
        #     nonlinear=None,
        #     weight_only=False
        # )
        # self.c2_r = OREPA(
        #     in_channels=mid_channels,
        #     out_channels=mid_channels,
        #     kernel_size=3,
        #     padding=1,   # ✅ 一定要加
        #     deploy=deploy,
        # )

        # self.c3_r = OREPA(
        #     in_channels=mid_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     padding=1,   # ✅ 一定要加
        #     deploy=deploy,
        # )

        self.fuse = conv_layer(in_channels, out_channels, 1)

        if self.use_esa:
            self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
        #self.cca = CCA(out_channels, reduction=4)
        self.act = activation('gelu')

    def forward(self, x):
        out = self.c1_r(x)
        out = self.act(out)

        out = self.c2_r(out)
        out = self.act(out)

        out = self.c3_r(out)
        out = self.act(out)

        out = out + x
        out = self.fuse(out)

        if self.use_esa:
            out = self.esa(out)
        #out=self.cca(out)

        return out



class BFFN(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3, deploy=False):
        super(BFFN, self).__init__()


        num_resblocks = args.n_resblocks   # 残差块数量
        num_features = args.n_feats           # 每层特征数（通道数）
        upscale_factor = args.scale[0]

        # 特征提取模块：初步提取浅层特征，使用Conv3
        #self.feature_extraction = conv_layer(in_channels, num_features, kernel_size=3)

        self.feature_extraction =  ECB(
            inp_planes=in_channels,
            out_planes=num_features,
            depth_multiplier= 2,
            act_type="linear")
        

        # 多个轻量特征块（RLFB变体）
        self.blocks = nn.ModuleList([
            BFFB(in_channels=num_features, use_esa=(i % 2 == 1))  # 偶数块和最后一个块使用 ESA
            for i in range(num_resblocks)
        ])

       #self.post_blocks_conv1 = BSConv(num_features, num_features, kernel_size=1)
        # 特征细化模块：对融合后的特征进行卷积增强，并与浅层特征做残差连接
        #self.refine_conv = conv_layer(num_features, num_features, kernel_size=3)

        self.refine_conv =  ECB(
            inp_planes=num_features,
            out_planes=num_features,
            depth_multiplier= 2,
            act_type="linear",with_idt=False)

        
        # 上采样模块：使用 PixelShuffle 实现分辨率提升
        self.upsampler = pixelshuffle_block(num_features, out_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        # 浅层特征提取

        shallow_features = self.feature_extraction(x)

        # 多块串联，每块输出作为下一块输入
        features = shallow_features
        for blk in self.blocks:
            features = blk(features)  # 串联，每块输入是上一块输出

        #features = self.post_blocks_conv1(features)

        # 特征细化 + 残差
        refined_features = self.refine_conv(features) + shallow_features

        # 上采样得到最终高分辨率图像
        sr_output = self.upsampler(refined_features)

        return sr_output

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



def check_modules(net):
    # try:
    #     from fvcore.nn import FlopCountAnalysis, flop_count_table
    #     print(flop_count_table(FlopCountAnalysis(net, inputs=(torch.rand(1, 3, 256, 256),))))
    # except ImportError:
    #     print("fvcore not installed; skipping FLOPs analysis.")
    count_deploy_parameters(net)
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")


def make_model(args, parent=False):
    model = BFFN(args)
    check_modules(model)
    return model

def count_deploy_parameters(model):
    import copy
    model_deploy = copy.deepcopy(model)

    # ② 遍历所有模块，执行 reparameterization 融合
    for m in model_deploy.modules():
        if hasattr(m, "switch_to_deploy"):
            try:
                m.switch_to_deploy()
            except Exception as e:
                print(f"[Warning] Failed to deploy {m.__class__.__name__}: {e}")

    # ③ 统计参数量
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        print(flop_count_table(FlopCountAnalysis(model_deploy, inputs=(torch.rand(1, 3, 256, 256),))))
    except ImportError:
        print("fvcore not installed; skipping FLOPs analysis.")


    total_params = 0
    for m in model_deploy.modules():
        if isinstance(m, nn.Conv2d):
            total_params += m.weight.numel()
            if m.bias is not None:
                total_params += m.bias.numel()

    print(f"\n[DEPLOY MODE] Total Conv Params: {total_params/1e6:.3f}M ({total_params:,})")
    
    return total_params
