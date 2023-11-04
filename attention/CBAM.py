"""
Woo S, Park J, Lee J Y, et al. Cbam: Convolutional block attention module[C]//
Proceedings of the European conference on computer vision (ECCV). 2018: 3-19.

https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
"""
import torch
from torch import nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
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
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        avg_result = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        result = torch.cat([max_result, avg_result], dim=1)  # (batch_size, 2, height, width)
        out = self.conv(result)  # (batch_size, 1, height, width)
        out = self.sigmoid(out)  # (batch_size, 1, height, width)
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7, channel_first=True):
        """
        Parameters:
            in_channels: number of input channels
            reduction: reduction ratio
            kernel_size: kernel size of the convolution operation
            channel_first: if True, Channel Attention Module is applied first, otherwise Spatial Attention Module is applied first
        """
        super().__init__()
        self.ca = ChannelAttentionModule(in_channels, reduction)
        self.sa = SpatialAttentionModule(kernel_size)
        self.channel_first = channel_first

    def forward(self, x):
        """
        Parameters:
            x: feature map with shape of (batch_size, channels, height, width)

        Returns:
            a tensor with shape of (batch_size, channels, height, width)
        """
        batch_size, channels, _, _ = x.shape
        residual = x  # (batch_size, channels, height, width)
        if self.channel_first:
            out = x * self.ca(x)  # (batch_size, channels, height, width)
            out = out * self.sa(out)  # (batch_size, channels, height, width)
        else:
            out = x * self.sa(x)  # (batch_size, channels, height, width)
            out = out * self.ca(out)  # (batch_size, channels, height, width)
        out = out + residual  # (batch_size, channels, height, width)
        return out


if __name__ == '__main__':
    model = CBAM(in_channels=512, channel_first=True)
    x = torch.randn(2, 512, 14, 14)
    out = model(x)
    print(out.shape)  # (2, 512, 14, 14)
