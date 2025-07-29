import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.theta = nn.Conv2d(F_l, F_int, kernel_size=1, stride=2)
        self.phi   = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1
        self.psi   = nn.Conv2d(F_int, 1, kernel_size=1, stride=1)
        self.relu  = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g   = self.phi(g)
        f       = self.relu(theta_x + phi_g) # combine
        psi_f   = self.psi(f)
        coef    = self.sigmoid(psi_f)
        coef_up = F.interpolate(coef, size=x.size()[2:], mode='bilinear')
        return x * coef_up

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        s = self.conv(x)
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
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
