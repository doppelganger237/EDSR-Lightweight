import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================
# 1. Gumbel Softmax (训练时随机掩码)
# ==============================
def sample_gumbel(shape, eps=1e-20, device="cpu"):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(weights, epoch):
    noise_temp = 0.97 ** (epoch - 1)
    noise = sample_gumbel(weights.shape, device=weights.device) * noise_temp
    y = weights + noise
    y_hard = torch.zeros_like(y)
    idx = y.abs().view(y.shape[0], -1).argmax(dim=1)
    y_hard.view(y.shape[0], -1)[torch.arange(y.shape[0]), idx] = 1.0
    return (y_hard - weights).detach() + weights


def hard_softmax(weights):
    y_hard = torch.zeros_like(weights)
    idx = weights.abs().view(weights.shape[0], -1).argmax(dim=1)
    y_hard.view(weights.shape[0], -1)[torch.arange(weights.shape[0]), idx] = 1.0
    return y_hard


# ==============================
# 2. ShiftConvGeneral（替代 MindSpore 的 ShiftConvGeneral）
# ==============================
class ShiftConvGeneral(nn.Module):
    def __init__(self, act_channel, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        self.epoch = 1
        self.act_channel = act_channel
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(act_channel))
        else:
            self.bias = None

    def forward(self, x):
        w = gumbel_softmax(self.weight, self.epoch) if self.training else hard_softmax(self.weight)
        w = w.to(x.dtype)
        # 每个通道单独卷积（group = C）
        w = w.repeat(x.shape[1], 1, 1, 1)
        out = F.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding, groups=x.shape[1])
        if self.bias_flag:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ==============================
# 3. GhostModule (主分支 + 线性cheap分支)
# ==============================
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=0.5, stride=1, bias=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup * ratio)
        new_channels = oup - init_channels

        self.primary_conv = nn.Conv2d(inp, init_channels, kernel_size, stride, padding=kernel_size // 2, bias=bias)
        self.cheap_conv = ShiftConvGeneral(new_channels, 1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.init_channels = init_channels
        self.new_channels = new_channels

    def forward(self, x):
        x1 = self.primary_conv(x)
        # 简化的 Ghost 分支：生成伪特征
        x2 = self.cheap_conv(x1[:, :self.new_channels, :, :])
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# ==============================
# 5. MeanShift（复制自 EDSR）
# ==============================
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1, padding=0, bias=True)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


