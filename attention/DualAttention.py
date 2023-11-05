"""
Fu J, Liu J, Tian H, et al. Dual attention network for scene segmentation[C]//
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 3146-3154.

https://arxiv.org/pdf/1809.02983.pdf
"""
import torch
from torch import nn
from SelfAttention import MultiHeadAttention
from SimplifiedSelfAttention import MultiHeadSimplifiedSelfAttention


class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels: int = 512, kernel_size=3, n_heads=1):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attention = MultiHeadAttention(d_model=in_channels, n_heads=n_heads, d_k=in_channels, d_v=in_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = self.cnn(x)  # (batch_size, channels, height, width)
        # (batch_size, channels, height, width) -> (batch_size, channels, height * width) -> (batch_size, height * width, channels)
        out = out.view(batch_size, channels, -1).permute(0, 2, 1)
        out = self.attention(out, out, out)  # (batch_size, height * width, channels)
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)  # (batch_size, channels, height, width)
        return x + out


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels: int = 512, kernel_size=3, height=7, width=7, n_heads=1):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attention = MultiHeadSimplifiedSelfAttention(d_model=height * width, n_heads=n_heads)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = self.cnn(x)  # (batch_size, channels, height, width)
        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        out = out.view(batch_size, channels, -1)
        out = self.attention(out, out, out)  # (batch_size, channels, height * width)
        out = out.view(batch_size, channels, height, width)  # (batch_size, channels, height, width)
        return x + out


class DualAttention(nn.Module):
    def __init__(self, in_channels: int, kernel_size=3, height: int = 7, width: int = 7, n_heads: int = 1):
        """
        Parameters:
            in_channels: 输入的通道数
            kernel_size: 卷积核大小
            height: 特征图的高度
            width: 特征图的宽度
            n_heads: 多头注意力的头数
        """
        super().__init__()
        self.position_attention_module = PositionAttentionModule(in_channels, kernel_size)
        self.channel_attention_module = ChannelAttentionModule(in_channels, kernel_size, height, width)

    def forward(self, x):
        """
        Parameters:
            x: feature map with shape of (batch_size, channels, height, width)

        Returns:
            a tensor with shape of (batch_size, channels, height, width)
        """
        p_out = self.position_attention_module(x)  # (batch_size, channels, height, width)
        c_out = self.channel_attention_module(x)  # (batch_size, channels, height, width)
        return p_out + c_out


if __name__ == '__main__':
    x = torch.randn(50, 512, 7, 7)
    model = DualAttention(in_channels=512, height=7, width=7)
    out = model(x)
    print(out.shape)
