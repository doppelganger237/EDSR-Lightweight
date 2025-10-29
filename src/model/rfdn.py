# import torch
# import torch.nn as nn
# from model import block as B

# # 该文件定义了RFDN模型，用于图像超分辨率重建任务
# # RFDN模型通过多层残差特征密集块提取图像特征，并融合多尺度信息，
# # 最终通过像素重排实现图像的上采样，提升图像分辨率和质量

# def make_model(args, parent=False):
#     model = RFDN(args)
#     check_modules(model)
#     return model

# # RFDN模型定义，包含特征提取、残差密集块、特征融合、特征细化和上采样模块
# class RFDN(nn.Module):
#     def __init__(self,args, in_nc=3, nf=64, num_modules=8, out_nc=3):
#         super(RFDN, self).__init__()
#         scale = args.scale[0] 

#         # 特征提取模块：融合通道信息并提取空间特征
#         self.feature_extraction = nn.Sequential(
#             nn.Conv2d(in_nc, nf, 1),              # 1x1卷积实现通道融合
#             nn.Conv2d(nf, nf, 3, padding=1, groups=nf),  # 3x3深度卷积提取空间特征
#         )

#         # 多个残差特征密集块（RFDB），逐层提取丰富的特征表示
#         self.rfdb_blocks = nn.ModuleList([B.RFDB(in_channels=nf) for _ in range(num_modules)])

#         # 特征融合卷积层，将多个RFDB输出的特征拼接后融合降维
#         self.fusion_conv = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

#         # 特征细化模块，进一步提升特征表达能力
#         self.refine_conv = nn.Sequential(
#             nn.Conv2d(nf, nf, 1),                        # 逐点卷积调整通道数
#             nn.Conv2d(nf, nf, 3, padding=1, groups=nf),  # 深度卷积提取空间信息
#             nn.LeakyReLU(0.05, inplace=True)             # 激活函数增加非线性
#         )

#         # 上采样模块，使用像素重排实现图像分辨率提升
#         upsample_block = B.pixelshuffle_block
#         self.upsampler = upsample_block(nf, out_nc, upscale_factor=scale)


#     def forward(self, input):
#         # 输入图像首先通过特征提取模块，获得初步的低级特征表示
#         out_feature = self.feature_extraction(input)

#         # 初始化特征变量，准备传入多个残差特征密集块
#         out = out_feature

#         # 保存每个RFDB模块的输出特征，用于后续融合
#         rfdb_outputs = []
#         for block in self.rfdb_blocks:
#             out = block(out)        # 逐个通过RFDB模块提取深层特征
#             rfdb_outputs.append(out)

#         # 将所有RFDB输出特征在通道维度拼接，并通过1x1卷积融合降维
#         out_fused = self.fusion_conv(torch.cat(rfdb_outputs, dim=1))

#         # 细化融合后的特征，增强特征表达能力
#         out_refined = self.refine_conv(out_fused) + out_feature  # 残差连接，保留低级特征

#         # 通过上采样模块将特征图转换为高分辨率输出图像
#         output = self.upsampler(out_refined)

#         return output


# def check_modules(net):
#     params = sum(p.numel() for p in net.parameters())
#     print(f"Params: {params/1e3:.1f}K")