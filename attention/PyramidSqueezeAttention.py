"""
Zhang H, Zu K, Lu J, et al. EPSANet: An efficient pyramid squeeze attention block on convolutional neural network[C]//
Proceedings of the Asian Conference on Computer Vision. 2022: 1161-1177.

https://arxiv.org/pdf/2105.14447.pdf
"""
import torch
from torch import nn
from typing import Tuple


class PSA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
                 stride: int = 1, groups: Tuple[int, ...] = (1, 4, 8, 16), reduction: int = 16):
        """
        Parameters:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_sizes: a tuple of `kernel_size` type, kernel sizes of each branch,
                `out_channels` must be divisible by len(kernel_sizes)
            stride: stride of the convolutional layers
            groups: a tuple of int, groups of the convolutional layers
            reduction: reduction ratio of the SE module
        """
        super().__init__()

        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by len(kernel_sizes)"
        self.split_channels = out_channels // len(kernel_sizes)
        self.num_branches = len(kernel_sizes)

        self.conv = nn.ModuleList()
        for i in range(self.num_branches):
            self.conv.append(nn.Conv2d(in_channels, self.split_channels, kernel_size=kernel_sizes[i],
                                       padding=kernel_sizes[i] // 2, stride=stride, groups=groups[i], bias=False))

        self.se_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.split_channels, self.split_channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.split_channels // reduction, self.split_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Parameters:
            x: input feature map, (batch_size, in_channels, height, width)

        Returns:
            out: output feature map, (batch_size, out_channels, height, width)
        """
        batch_size, _, height, width = x.shape

        conv_results = []
        for conv_layer in self.conv:
            conv_results.append(conv_layer(x))  # (batch_size, split_channels, height, width)

        se_weights = []
        for conv in conv_results:
            sew = self.se_weight(conv)  # (batch_size, split_channels, 1, 1)
            sew = self.softmax(sew)
            se_weights.append(sew)  # (batch_size, split_channels, 1, 1)

        for i in range(self.num_branches):
            conv_results[i] = conv_results[i] * se_weights[i]  # (batch_size, split_channels, height, width)

        out = torch.cat(conv_results, dim=1)  # (batch_size, out_channels, height, width)
        return out


if __name__ == '__main__':
    x = torch.randn(3, 512, 56, 56)
    model = PSA(in_channels=512, out_channels=1024)
    out = model(x)
    print(out.shape)  # (3, 1024, 56, 56)
