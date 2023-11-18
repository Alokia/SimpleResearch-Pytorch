"""
Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//
Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference,
Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015: 234-241.

https://arxiv.org/pdf/1505.04597.pdf
"""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.down_sample(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, transpose=True):
        super().__init__()
        if transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, residual_x):
        x = self.up(x)

        diff_y = residual_x.shape[2] - x.shape[2]
        diff_x = residual_x.shape[3] - x.shape[3]

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([residual_x, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, transpose=True):
        super().__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.in_conv = DoubleConvBlock(in_channels, filters[0])

        self.down1 = DownSample(filters[0], filters[1])
        self.down2 = DownSample(filters[1], filters[2])
        self.down3 = DownSample(filters[2], filters[3])
        self.down4 = DownSample(filters[3], filters[4])

        self.up4 = UpSample(filters[4], filters[3], transpose)
        self.up3 = UpSample(filters[3], filters[2], transpose)
        self.up2 = UpSample(filters[2], filters[1], transpose)
        self.up1 = UpSample(filters[1], filters[0], transpose)

        self.out_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.in_conv(x)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u4 = self.up4(d4, d3)
        u3 = self.up3(u4, d2)
        u2 = self.up2(u3, d1)
        u1 = self.up1(u2, x)

        out = self.out_conv(u1)
        return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    model = UNet(3, 3)
    out = model(x)
    print(out.shape)  # [2, 3, 256, 256]
