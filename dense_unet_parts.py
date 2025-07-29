import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        f = in_channels
        mid = f // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(f, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.PReLU(num_parameters=mid)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(f + mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.PReLU(num_parameters=mid)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(f + 2 * mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.PReLU(num_parameters=mid)
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(f + 3 * mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        features = [x]
        out1 = self.conv1(x)
        features.append(out1)

        out2 = self.conv2(torch.cat(features, 1))
        features.append(out2)

        out3 = self.conv3(torch.cat(features, 1))
        features.append(out3)

        out_final = self.conv_final(torch.cat(features, 1))

        return out_final

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = DenseBlock(in_channels, out_channels)
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
        self.block = DenseBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
