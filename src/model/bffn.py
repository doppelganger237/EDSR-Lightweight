import torch
import torch.nn as nn
from model import block as B



def make_model(args, parent=False):
    model = BFFN(args)
    check_modules(model)
    return model


class BFFN(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3):
        super(BFFN, self).__init__()


        num_resblocks = args.n_resblocks   # 残差块数量
        num_features = args.n_feats           # 每层特征数（通道数）
        upscale_factor = args.scale[0]

        # 特征提取模块：初步提取浅层特征，使用Conv3
        self.feature_extraction = B.conv_layer(in_channels, num_features, kernel_size=3)

        # 多个轻量特征块（RLFB变体）
        self.blocks = nn.ModuleList([B.BFFB(in_channels=num_features) for _ in range(num_resblocks)])

        # 特征细化模块：对融合后的特征进行卷积增强，并与浅层特征做残差连接
        self.refine_conv = B.conv_layer(num_features, num_features, kernel_size=3)

        # 上采样模块：使用 PixelShuffle 实现分辨率提升
        self.upsampler = B.pixelshuffle_block(num_features, out_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        # 浅层特征提取

        shallow_features = self.feature_extraction(x)

        # 多块串联，每块输出作为下一块输入
        features = shallow_features
        for blk in self.blocks:
            features = blk(features)  # 串联，每块输入是上一块输出

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
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        print(flop_count_table(FlopCountAnalysis(net, inputs=(torch.rand(1, 3, 256, 256),))))
    except ImportError:
        print("fvcore not installed; skipping FLOPs analysis.")
    
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")