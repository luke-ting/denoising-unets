import torch.nn as nn
from .att_unet_parts import DoubleConv, AttGate, DownSample, UpSample

class AttUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.att4 = AttGate(F_g=1024, F_l=512, F_int=512)
        self.att3 = AttGate(F_g=512,  F_l=256, F_int=256)
        self.att2 = AttGate(F_g=256,  F_l=128, F_int=128)
        self.att1 = AttGate(F_g=128,  F_l=64,  F_int=64)

        self.up4 = UpSample(1024, 512)
        self.up3 = UpSample(512, 256)
        self.up2 = UpSample(256, 128)
        self.up1 = UpSample(128, 64)

        self.final = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x):

        s1, p1 = self.down1(x)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)

        b = self.bottleneck(p4)

        a4 = self.att4(s4, b)
        d4 = self.up4(b, a4)
        a3 = self.att3(s3, d4)
        d3 = self.up3(d4, a3)
        a2 = self.att2(s2, d3)
        d2 = self.up2(d3, a2)
        a1 = self.att1(s1, d2)
        d1 = self.up1(d2, a1)

        residual = self.final(d1)
        return x + residual
