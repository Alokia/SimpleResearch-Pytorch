"""
Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//
Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.

https://arxiv.org/abs/1709.01507
"""
import torch
from torch import nn


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention

    通过全局平均池化得到通道维度的特征，然后通过两个全连接层得到每个通道的权重，
    本质上是在channel的层面上进行加权操作
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Parameters:
            in_channels: 输入的通道数
            reduction: 两个全连接层之间的隐藏层维度为 in_channels // reduction
        """
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            x: feature maps, with shape (batch_size, in_channels, height, width)

        Returns:
            feature maps after SE Attention, with shape (batch_size, in_channels, height, width)
        """
        batch_size, channels, _, _ = x.shape
        # (batch_size, channels, height, width) -> (batch_size, channels, 1, 1) -> (batch_size, channels)
        y = self.pool(x).view(batch_size, channels)
        # (batch_size, channels) -> (batch_size, channels) -> (batch_size, channels, 1, 1)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn(16, 256, 7, 7)
    se = SEAttention(in_channels=256, reduction=8)
    output = se(x)
    print(output.shape)  # [16, 256, 7, 7]
