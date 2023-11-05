"""
Woo S, Park J, Lee J Y, et al. Cbam: Convolutional block attention module[C]//
Proceedings of the European conference on computer vision (ECCV). 2018: 3-19.

https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
"""
import torch
from torch import nn


class ChannelGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Parameters:
            in_channels: number of input channels
            reduction: reduction ratio
        """
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.max_pool(x)  # (batch_size, channels, 1, 1)
        avg_result = self.avg_pool(x)  # (batch_size, channels, 1, 1)
        max_out = self.mlp(max_result)  # (batch_size, channels, 1, 1)
        avg_out = self.mlp(avg_result)  # (batch_size, channels, 1, 1)
        out = self.sigmoid(max_out + avg_out)  # (batch_size, channels, 1, 1)
        return x * out


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        avg_result = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        result = torch.cat([max_result, avg_result], dim=1)  # (batch_size, 2, height, width)
        out = self.conv(result)  # (batch_size, 1, height, width)
        out = self.sigmoid(out)  # (batch_size, 1, height, width)
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size=7,
                 channel_first: bool = True, no_spatial: bool = False):
        """
        Parameters:
            in_channels: number of input channels
            reduction: reduction ratio
            kernel_size: kernel size of the convolution operation
            channel_first: if True, Channel Attention Module is applied first, otherwise Spatial Attention Module is applied first
            no_spatial: if True, Spatial Attention Module is not applied
        """
        super().__init__()
        self.channel_attn = ChannelGate(in_channels, reduction)
        if not no_spatial:
            self.spatial_attn = SpatialGate(kernel_size)
        self.channel_first = channel_first
        self.no_spatial = no_spatial

    def forward(self, x):
        """
        Parameters:
            x: feature map with shape of (batch_size, channels, height, width)

        Returns:
            a tensor with shape of (batch_size, channels, height, width)
        """
        batch_size, channels, _, _ = x.shape
        residual = x  # (batch_size, channels, height, width)
        if self.no_spatial:
            out = self.channel_attn(x)
        else:
            if self.channel_first:
                out = self.channel_attn(x)  # (batch_size, channels, height, width)
                out = self.spatial_attn(out)  # (batch_size, channels, height, width)
            else:
                out = self.spatial_attn(x)  # (batch_size, channels, height, width)
                out = self.channel_attn(out)  # (batch_size, channels, height, width)
        out = out + residual  # (batch_size, channels, height, width)
        return out


if __name__ == '__main__':
    model = CBAM(in_channels=512, channel_first=True, no_spatial=False)
    x = torch.randn(2, 512, 14, 14)
    out = model(x)
    print(out.shape)  # (2, 512, 14, 14)
