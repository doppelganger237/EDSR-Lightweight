import torch
import torch.nn as nn
from model import block as B


def make_model(args, parent=False):
    model = BFFN(args)
    check_modules(model)
    return model


class BFFN(nn.Module):
    def __init__(self, args, in_channels=3, num_features=64, num_resblocks=8, out_channels=3):
        super(BFFN, self).__init__()
        upscale_factor = args.scale[0]

        # 特征提取模块：初步提取浅层特征
        self.feature_extraction = B.BSConv(in_channels, num_features, kernel_size=3)

        # 多个轻量特征块（RLFB变体）
        self.blocks = nn.ModuleList([B.BFFB(in_channels=num_features) for _ in range(num_resblocks)])

        # 通道拼接与融合：整合多块 RLFB 输出的通道信息
        self.fusion_conv = B.conv_block(num_features * num_resblocks, num_features, kernel_size=1, act_type='gelu')

        # 特征细化模块：对融合后的特征进行卷积增强，并与浅层特征做残差连接
        self.refine_conv = nn.Sequential(
            B.BSConv(num_features, num_features, kernel_size=3),
            B.activation("gelu")
        )
        # 上采样模块：使用 PixelShuffle 实现分辨率提升
        self.upsampler = B.pixelshuffle_block(num_features, out_channels, upscale_factor=upscale_factor)

    def forward(self, x):
        # 提取初始特征
        shallow_features = self.feature_extraction(x)

        # 多块串联提取特征，同时保存输出以用于融合
        features = shallow_features
        block_features = []
        for blk in self.blocks:
            features = blk(features)
            block_features.append(features)

        # 通道维度拼接所有 block 输出并融合
        fused_features = self.fusion_conv(torch.cat(block_features, dim=1))

        # 特征细化 + 残差增强
        refined_features = self.refine_conv(fused_features) + shallow_features

        # 上采样得到最终高分辨率图像
        sr_output = self.upsampler(refined_features)

        return sr_output


def check_modules(net):
   # params = sum(p.numel() for p in net.parameters())

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    print(flop_count_table(FlopCountAnalysis(net, inputs = (torch.rand(1, 3, 256, 256),))))

    #print(f"Params: {params/1e3:.1f}K")