import torch.nn as nn
from .unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

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

        r4 = self.up4(b, s4)
        r3 = self.up3(r4, s3)
        r2 = self.up2(r3, s2)
        r1 = self.up1(r2, s1)

        residual = self.final(r1)
        denoised = x + residual
        
        return denoised
