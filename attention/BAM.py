"""
Park J, Woo S, Lee J Y, et al. Bam: Bottleneck attention module[J]. arXiv preprint arXiv:1807.06514, 2018.

https://arxiv.org/pdf/1807.06514.pdf
"""
import torch
from torch import nn


class ChannelGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, num_layers: int = 1):
        """
        Parameters:
            in_channels: int, 输入的通道数
            reduction: int, 通道数缩小的倍数
            num_layers: int, 隐藏层的个数
        """
        super().__init__()
        # 隐藏层的个数
        self.gate_c = nn.Sequential()
        gate_channels = [in_channels] + [in_channels // reduction] * num_layers + [in_channels]
        for i in range(len(gate_channels) - 2):
            self.gate_c.append(nn.Conv2d(gate_channels[i], gate_channels[i + 1], kernel_size=1))
            self.gate_c.append(nn.BatchNorm2d(gate_channels[i + 1]))
            self.gate_c.append(nn.ReLU())
        self.gate_c.append(nn.Conv2d(gate_channels[-2], gate_channels[-1], kernel_size=1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_pool = self.avg_pool(x)  # (batch_size, channel, 1, 1)
        gate = self.gate_c(avg_pool)  # (batch_size, channel, 1, 1)
        return gate.expand_as(x)  # (batch_size, channel, height, width)


class SpatialGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, num_dilation_conv: int = 2, dilation: int = 4):
        """
        Parameters:
            in_channels: int, 输入的通道数
            reduction: int, 通道数缩小的倍数
            num_dilation_conv: int, 空洞卷积的个数
            dilation: int, 空洞卷积的扩张率
        """
        super().__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.append(nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1))
        self.gate_s.append(nn.BatchNorm2d(in_channels // reduction))
        self.gate_s.append(nn.ReLU())
        for i in range(num_dilation_conv):
            self.gate_s.append(nn.Conv2d(in_channels // reduction, in_channels // reduction,
                                         kernel_size=3, padding=dilation, dilation=dilation))
            self.gate_s.append(nn.BatchNorm2d(in_channels // reduction))
            self.gate_s.append(nn.ReLU())
        self.gate_s.append(nn.Conv2d(in_channels // reduction, out_channels=1, kernel_size=1))

    def forward(self, x):
        out = self.gate_s(x)  # (batch_size, 1, height, width)
        return out.expand_as(x)  # (batch_size, channel, height, width)


class BAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, num_layers: int = 1,
                 num_dilation_conv: int = 2, dilation: int = 4):
        super().__init__()
        self.channel_attn = ChannelGate(in_channels, reduction, num_layers)
        self.spatial_attn = SpatialGate(in_channels, reduction, num_dilation_conv, dilation)

    def forward(self, x):
        attn = 1 + torch.sigmoid(self.channel_attn(x) * self.spatial_attn(x))  # (batch_size, channel, height, width)
        return attn * x  # (batch_size, channel, height, width)


if __name__ == '__main__':
    x = torch.randn(3, 512, 7, 7)
    model = BAM(in_channels=512)
    out = model(x)
    print(out.shape)
