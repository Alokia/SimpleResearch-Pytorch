"""
Chollet F. Xception: Deep learning with depthwise separable convolutions[C]//
Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1251-1258.

https://arxiv.org/abs/1610.02357
"""
import torch
from torch import nn


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution

    相比于普通卷积，深度可分离卷积的参数量更小，可以搭建更深的模型。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    dsconv = DepthwiseSeparableConv2d(3, 64, kernel_size=3, stride=2)
    out = dsconv(x)
    print(out.shape)
