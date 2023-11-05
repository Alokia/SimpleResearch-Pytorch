"""
Wang Q, Wu B, Zhu P, et al. ECA-Net: Efficient channel attention for deep convolutional neural networks[C]//
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 11534-11542.

https://arxiv.org/pdf/1910.03151.pdf
"""
import torch
from torch import nn
import numpy as np


class ECA(nn.Module):
    def __init__(self, in_channels: int, gamma: float = 2, b: float = 1, k: int = None):
        """
        Parameters:
            in_channels: number of input channels
            gamma: parameter for mapping function
            b: parameter for mapping function
            k: kernel size of the conv layer, if `k` is None,
                it will be calculated automatically according to `gamma` and `b`.
        """
        super().__init__()
        if k is None:
            t = int(abs((np.log2(in_channels) + b) / gamma))
            k = t if t % 2 else t + 1  # 取一个奇数

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Parameters:
            x: feature map with shape of (batch_size, channels, height, width)

        Returns:
            a tensor with shape of (batch_size, channels, height, width)
        """
        avg_pool = self.avg_pool(x)  # (batch_size, channels, 1, 1)
        # (batch_size, channels, 1, 1) -> (batch_size, channels, 1) -> (batch_size, 1, channels)
        avg_pool = avg_pool.squeeze(-1).transpose(-1, -2)
        conv = self.conv(avg_pool)  # (batch_size, 1, channels)
        # (batch_size, 1, channels) -> (batch_size, channels, 1) -> (batch_size, channels, 1, 1)
        conv = conv.transpose(-1, -2).unsqueeze(-1)
        conv = self.sigmoid(conv)
        return x * conv.expand_as(x)  # (batch_size, channels, height, width)


if __name__ == '__main__':
    x = torch.randn(3, 512, 7, 7)
    model = ECA(512)
    out = model(x)
    print(out.shape)
