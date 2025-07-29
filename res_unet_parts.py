import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        out = self.block(x)
        identity = self.shortcut(x)
        return self.act(out + identity)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels)
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        s = self.block(x)
        p = self.down(s)
        return s, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.PReLU(num_parameters=in_channels//2)
        )
        self.block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
