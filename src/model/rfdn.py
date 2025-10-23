import torch
import torch.nn as nn
from model import block as B


def make_model(args, parent=False):
    model = RFDN(args)
    check_modules(model)
    return model

# RFDN Large 52 channels, 6 RFDBs
class RFDN(nn.Module):
    def __init__(self,args, in_nc=3, nf=52, num_modules=6, out_nc=3):
        super(RFDN, self).__init__()
        scale = args.scale[0] 
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.B5 = B.RFDB(in_channels=nf)
        self.B6 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=scale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output



def check_modules(net):
    params = sum(p.numel() for p in net.parameters())
    print(f"Params: {params/1e3:.1f}K")