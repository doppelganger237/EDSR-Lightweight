# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import copy
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
import os

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)




import torch.nn as nn
import torch
import numpy as np

'''
---- 1) FLOPs: floating point operations
---- 2) #Activations: the number of elements of all ‘Conv2d’ outputs
---- 3) #Conv2d: the number of ‘Conv2d’ layers
'''

def get_model_flops(model, input_res, print_per_layer_stat=True,
                              input_constructor=None):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        device = list(flops_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    flops_model.stop_flops_count()

    return flops_count

def get_model_activation(model, input_res, input_constructor=None):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    activation_model = add_activation_counting_methods(model)
    activation_model.eval().start_activation_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = activation_model(**input)
    else:
        device = list(activation_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = activation_model(batch)

    activation_count, num_conv = activation_model.compute_average_activation_cost()
    activation_model.stop_activation_count()

    return activation_count, num_conv


def get_model_complexity_info(model, input_res, print_per_layer_stat=True, as_strings=True,
                              input_constructor=None):
    assert type(input_res) is tuple
    assert len(input_res) >= 3
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        batch = torch.FloatTensor(1, *input_res)
        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    params_count = get_model_parameters_number(flops_model)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num):
    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + ' M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + ' k'
    else:
        return str(params_num)


def print_model_with_flops(model, units='GMac', precision=3):
    total_flops = model.compute_average_flops_cost()

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([flops_to_string(accumulated_flops_cost, units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    # embed()
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()
    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, (nn.BatchNorm2d)):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module,
                  (
                          nn.Conv2d, nn.ConvTranspose2d,
                          nn.BatchNorm2d,
                          nn.Linear,
                          nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6,
                  )):
        return True

    return False


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    # input = input[0]

    batch_size = output.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = np.prod(kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * np.prod(output_dims)
    overall_conv_flops = int(conv_per_position_flops) * int(active_elements_count)

    # overall_flops = overall_conv_flops

    conv_module.__flops__ += int(overall_conv_flops)
    # conv_module.__output_dims__ = output_dims


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)
    # print(module.__flops__, id(module))
    # print(module)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    if len(input.shape) == 1:
        batch_size = 1
        module.__flops__ += int(batch_size * input.shape[0] * output.shape[0])
    else:
        batch_size = input.shape[0]
        module.__flops__ += int(batch_size * input.shape[1] * output.shape[1])


def bn_flops_counter_hook(module, input, output):
    # input = input[0]
    # TODO: need to check here
    # batch_flops = np.prod(input.shape)
    # if module.affine:
    #     batch_flops *= 2
    # module.__flops__ += int(batch_flops)
    batch = output.shape[0]
    output_dims = output.shape[2:]
    channels = module.num_features
    batch_flops = batch * channels * np.prod(output_dims)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


# ---- Count the number of convolutional layers and the activation
def add_activation_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    # embed()
    net_main_module.start_activation_count = start_activation_count.__get__(net_main_module)
    net_main_module.stop_activation_count = stop_activation_count.__get__(net_main_module)
    net_main_module.reset_activation_count = reset_activation_count.__get__(net_main_module)
    net_main_module.compute_average_activation_cost = compute_average_activation_cost.__get__(net_main_module)

    net_main_module.reset_activation_count()
    return net_main_module


def compute_average_activation_cost(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Returns current mean activation consumption per image.

    """

    activation_sum = 0
    num_conv = 0
    for module in self.modules():
        if is_supported_instance_for_activation(module):
            activation_sum += module.__activation__
            num_conv += module.__num_conv__
    return activation_sum, num_conv


def start_activation_count(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Activates the computation of mean activation consumption per image.
    Call it before you run the network.

    """
    self.apply(add_activation_counter_hook_function)


def stop_activation_count(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Stops computing the mean activation consumption per image.
    Call whenever you want to pause the computation.

    """
    self.apply(remove_activation_counter_hook_function)


def reset_activation_count(self):
    """
    A method that will be available after add_activation_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    self.apply(add_activation_counter_variable_or_reset)


def add_activation_counter_hook_function(module):
    if is_supported_instance_for_activation(module):
        if hasattr(module, '__activation_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handle = module.register_forward_hook(conv_activation_counter_hook)
            module.__activation_handle__ = handle


def remove_activation_counter_hook_function(module):
    if is_supported_instance_for_activation(module):
        if hasattr(module, '__activation_handle__'):
            module.__activation_handle__.remove()
            del module.__activation_handle__


def add_activation_counter_variable_or_reset(module):
    if is_supported_instance_for_activation(module):
        module.__activation__ = 0
        module.__num_conv__ = 0


def is_supported_instance_for_activation(module):
    if isinstance(module,
                  (
                          nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.Linear, nn.ConvTranspose1d
                  )):
        return True

    return False

def conv_activation_counter_hook(module, input, output):
    """
    Calculate the activations in the convolutional operation.
    Reference: Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár, Designing Network Design Spaces.
    :param module:
    :param input:
    :param output:
    :return:
    """
    module.__activation__ += output.numel()
    module.__num_conv__ += 1


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def dconv_flops_counter_hook(dconv_module, input, output):
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    m_channels, in_channels, kernel_dim1, _, = dconv_module.weight.shape
    out_channels, _, kernel_dim2, _, = dconv_module.projection.shape
    # groups = dconv_module.groups

    # filters_per_channel = out_channels // groups
    conv_per_position_flops1 = kernel_dim1 ** 2 * in_channels * m_channels
    conv_per_position_flops2 = kernel_dim2 ** 2 * out_channels * m_channels
    active_elements_count = batch_size * np.prod(output_dims)

    overall_conv_flops = (conv_per_position_flops1 + conv_per_position_flops2) * active_elements_count
    overall_flops = overall_conv_flops

    dconv_module.__flops__ += int(overall_flops)
    # dconv_module.__output_dims__ = output_dims
    


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

def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
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

class RepRLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=48,
                 upscale=2,
                 deploy=False,
                 num_blocks=6):
        super(RepRLFN, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                 feature_channels,
                                 kernel_size=3)

        # Create num_blocks RepRLFB blocks in a Sequential
        blocks = []
        for _ in range(num_blocks):
            blocks.append(RepRLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy))
        self.blocks = nn.Sequential(*blocks)

        self.conv_2 = conv_layer(feature_channels,
                                 feature_channels,
                                 kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                            out_channels,
                                            upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)
        out = out_feature
        for block in self.blocks:
            out = block(out)
        out_low_resolution = self.conv_2(out) + out_feature
        output = self.upsampler(out_low_resolution)
        return output


class RepRLFB(nn.Module):
    """
    Reparameterized Residual Local Feature Block (RepRLFB).
    """
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 act_type = 'lrelu',
                 deploy = False,
                 # esa_channels=16):
                 esa_channels=15):
        super(RepRLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = RepBlock(in_channels=in_channels, out_channels=mid_channels, act_type=act_type, deploy=deploy)
        self.c2_r = RepBlock(in_channels=in_channels, out_channels=mid_channels, act_type=act_type, deploy=deploy)
        self.c3_r = RepBlock(in_channels=in_channels, out_channels=mid_channels, act_type=act_type, deploy=deploy)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
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

# Reparameterization: (3*3) U (3*1) U (1*3) U (1*1) U (identity) -> (3*3)
class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

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
            return self.activation(self.rbr_reparam(inputs))
        else:
            return self.activation( (self.rbr_3x3_branch(inputs)) +
                                   (self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(inputs) + self.rbr_1x1_branch(inputs)) +
                                   (inputs) )

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


def get_RepRLFN(checkpoint=None, deploy=False):
    model = RepRLFN(in_channels=3, out_channels=3, feature_channels=48, deploy=deploy)

    # param_key_g = 'params'
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)

    return model

def make_model(args, parent=False):
    model = RepRLFN(in_channels=3,
                    out_channels=3,
                    feature_channels=48,
                    upscale=args.scale[0],
                    deploy=args.test_only)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params/1e3:.1f}K")
    return model

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    seed_everything(2025)

    # mynet = get_RepRLFN(checkpoint=None, deploy=False)
    # img = torch.ones((1,3,40,40))
    # out = mynet(img)
    # state_dict = mynet.state_dict()
    # torch.save(state_dict, 'RepRLFN.pth')
    # print("size: {}".format(out.size()))
    # print(out[0][0][0][:4]) # [-0.3382, -0.1902,  0.2992,  0.0100]


    # mynet = get_RepRLFN(checkpoint='RepRLFN.pth', deploy=False)
    # img = torch.ones((1,3,40,40))
    # out = mynet(img)
    # print("size: {}".format(out.size()))
    # print(out[0][0][0][:4]) # [-0.3382, -0.1902,  0.2992,  0.0100]
    # # model convert
    # deploy_model = repvgg_model_convert(mynet, save_path='RepRLFN_deploy.pth')


    # mynet = get_RepRLFN(checkpoint='RepRLFN_deploy.pth', deploy=True)
    # img = torch.ones((1,3,40,40))
    # out = mynet(img)
    # print("size: {}".format(out.size()))
    # print(out[0][0][0][:4]) # [-0.3382, -0.1902,  0.2992,  0.0100]


    model = get_RepRLFN(checkpoint=None, deploy=True)
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))
    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

