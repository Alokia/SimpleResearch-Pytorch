"""
Chen Y, Kalantidis Y, Li J, et al. A^ 2-nets: Double attention networks[J].
 Advances in neural information processing systems, 2018, 31.

https://arxiv.org/pdf/1810.11579.pdf
"""
import torch
from torch import nn


class DoubleAttention(nn.Module):
    def __init__(self, in_channels, out_channels: int = None, dn: int = None, reconstruct: bool = True):
        """
        Parameters:
            in_channels: 输入的通道数
            out_channels: 如果 `reconstruct=False`，则该参数为输出的通道数，否则为中间的通道数，真正的输出通道数会变为 `in_channels`
            dn: feature gating和feature distribution中的维度
            reconstruct: 是否重建特征图
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.dn = in_channels if dn is None else dn
        self.reconstruct = reconstruct

        self.convA = nn.Conv2d(in_channels, self.out_channels, 1)
        self.convB = nn.Conv2d(in_channels, self.dn, 1)
        self.convV = nn.Conv2d(in_channels, self.dn, 1)

        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        """
        Parameters:
            x: feature map with shape of (batch_size, channels, height, width)

        Returns:
            a tensor with shape of (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        A = self.convA(x)  # (batch_size, out_channels, height, width)
        B = self.convB(x)  # (batch_size, dn, height, width)
        V = self.convV(x)  # (batch_size, dn, height, width)
        tmpA = A.view(batch_size, self.out_channels, -1)  # (batch_size, out_channels, height * width)
        attn_maps = torch.softmax(B.view(batch_size, self.dn, -1), dim=-1)  # (batch_size, dn, height * width)
        attn_vectors = torch.softmax(V.view(batch_size, self.dn, -1), dim=-1)  # (batch_size, dn, height * width)

        # feature gating
        glob_desc = torch.bmm(tmpA, attn_maps.permute(0, 2, 1))  # (batch_size, out_channels, dn)
        # feature distribution
        tmpZ = glob_desc.matmul(attn_vectors)  # (batch_size, out_channels, height * width)
        tmpZ = tmpZ.view(batch_size, self.out_channels, height, width)  # (batch_size, out_channels, height, width)
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)  # (batch_size, channels, height, width)

        return tmpZ


if __name__ == '__main__':
    x = torch.randn(3, 128, 56, 56)
    model = DoubleAttention(128, 128, 128, True)
    out = model(x)
    print(out.shape)  # [3, 128, 56, 56]
