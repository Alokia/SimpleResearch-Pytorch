"""
Li X, Wang W, Hu X, et al. Selective kernel networks[C]//
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 510-519.

https://arxiv.org/abs/1903.06586
"""
import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict


class SKAttention(nn.Module):
    """
    Selective Kernel Attention
    """

    def __init__(self, in_channels: int, kernels: Tuple[int] = (1, 3, 5, 7),
                 reduction: int = 16, groups: int = 1, L: int = 32, bias=False):
        """
        Parameters:
            in_channels: input dimensions of feature map
            kernels: a tuple with each `Conv2d` layer's kernel size. default (1, 3, 5, 7)
            reduction: reduction ratio r, default 16. the true reduction ratio is max(L, in_channels // reduction)
            groups: the groups of feature map, default 1
            L: minimum reduction, default 32. The true reduction ratio is max(L, in_channels // reduction)
            bias: whether to add bias in `Linear` layer, default False
        """
        super().__init__()
        d = max(L, in_channels // reduction)

        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k // 2, groups=groups)),
                    ('bn', nn.BatchNorm2d(in_channels)),
                    ('relu', nn.ReLU())
                ]))
            )

        self.fc = nn.Linear(in_channels, d, bias=bias)

        self.fc_list = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fc_list.append(nn.Linear(d, in_channels, bias=bias))

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        Parameters:
            x: feature map with shape of (batch_size, channels, height, width)

        Returns:
            a tensor with same shape of input x
        """
        batch_size, channels, _, _ = x.shape

        # split
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, dim=0)  # (k, batch_size, channels, height, width)

        # fuse
        U = sum(conv_outs)  # (batch_size, channels, height, width)

        # reduction channel
        S = U.mean(-1).mean(-1)  # (batch_size, channels)
        Z = self.fc(S)  # (batch_size, d)

        # calculate attention weight
        weights = []
        for fc in self.fc_list:
            weight = fc(Z)  # (batch_size, channels)
            weights.append(weight.view(batch_size, channels, 1, 1))  # (batch_size, channels, 1, 1)
        attention_weights = torch.stack(weights, dim=0)  # (k, batch_size, channels, 1, 1)
        attention_weights = self.softmax(attention_weights)  # (k, batch_size, channels, 1, 1)

        # select
        V = (attention_weights * feats).sum(0)  # (batch_size, channels, height, width)
        return V


if __name__ == '__main__':
    x = torch.randn(50, 512, 7, 7)
    se = SKAttention(in_channels=512)
    output = se(x)
    print(output.shape)
