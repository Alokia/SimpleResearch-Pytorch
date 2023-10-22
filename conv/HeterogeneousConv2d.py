"""
Singh P, Verma V K, Rai P, et al. Hetconv: Heterogeneous kernel-based convolutions for deep cnns[C]//
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 4835-4844.

https://arxiv.org/pdf/1903.04120.pdf
"""
import torch
from torch import nn


class HetConv2d(nn.Module):
    """
    Heterogeneous Convolution
    """

    def __init__(self, in_channels, out_channels, p: int):
        """
        不会改变特征图的大小，只改变通道数

        Parameters:
            p: int, 卷积核为 kernel_size 的比例，也就是 groups 的个数
        """
        super().__init__()
        # Group-wise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=p, bias=False)
        # Point-wise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


if __name__ == '__main__':
    x = torch.rand(1, 16, 224, 125)
    conv = HetConv2d(16, 64, p=4)
    out = conv(x)
    print(out.shape)  # [1, 64, 224, 125]
