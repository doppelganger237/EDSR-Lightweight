import torch
import torch.nn as nn
from model import block as B


def make_model(args, parent=False):
    model = RFDN(args)
    check_modules(model)
    return model

# RFDN Large 52 channels, 6 RFDBs
class RFDN(nn.Module):
    def __init__(self,args, in_nc=3, nf=64, num_modules=8, out_nc=3):
        super(RFDN, self).__init__()
        scale = args.scale[0] 
        #self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.fea_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, 1),              # 通道融合
            nn.Conv2d(nf, nf, 3, padding=1, groups=nf),  # 空间特征提取
        )
        self.RFDBs = nn.ModuleList([B.RFDB(in_channels=nf) for _ in range(num_modules)])
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 1),                        # PW
            nn.Conv2d(nf, nf, 3, padding=1, groups=nf),  # DW
            nn.LeakyReLU(0.05, inplace=True)
        )

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=scale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out = out_fea
        outs = []
        for block in self.RFDBs:
            out = block(out)
            outs.append(out)

        out_B = self.c(torch.cat(outs, dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output



def check_modules(net):
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")