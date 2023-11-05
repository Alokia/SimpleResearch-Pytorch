"""
Zhang Q, Yang Y B. Rest: An efficient transformer for visual recognition[J].
Advances in neural information processing systems, 2021, 34: 15475-15485.

https://arxiv.org/pdf/2105.13677.pdf
"""
import torch
from torch import nn
import numpy as np


class EMSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, qkv_bias: bool = False,
                 scale: float = None, attn_drop: float = 0., proj_drop: float = 0., sr_ratio: int = 1,
                 apply_transform=False):
        """
        Parameters:
            d_model: 前面层的输出维度，即q,k,v的维度 (batch_size, nq, d_model), 一般为上一层特征图的通道数
            n_heads: heads 的数量
            qkv_bias: 是否使用qkv全连接层的偏置
            scale: 缩放因子，默认为 None
            attn_drop: attention dropout 的概率
            proj_drop: 全连接层 dropout 的概率
            sr_ratio: spatial reduction ratio
            apply_transform: 是否使用 transform
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = scale if scale is not None else self.head_dim ** -0.5

        self.q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio + 1, stride=sr_ratio,
                                padding=sr_ratio // 2, groups=d_model)
            self.sr_norm = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and n_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.n_heads, self.n_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.n_heads)

    def forward(self, x, attn_mask=None, attn_weights=None):
        """
        Parameters:
            x: input tensor, shape: (batch_size, channels, height, width)
            attn_mask: attention mask, 当值为 True 时表示使用负无穷遮盖该处的值, shape: (batch_size, n_heads, nq, oq)
            attn_weights: attention weights, 与注意力相乘的权重矩阵, shape: (batch_size, n_heads, nq, oq)

        Returns:
            out: a tensor with shape (batch_size, channels, height, width)
        """
        batch_size, d_model, height, width = x.shape
        nq = height * width
        x = x.reshape(batch_size, d_model, nq).permute(0, 2, 1)  # (batch_size, nq, d_model)

        # (batch_size, nq, d_model) -> (batch_size, nq, n_heads, head_dim) -> (batch_size, n_heads, nq, head_dim)
        q = self.q(x).reshape(batch_size, nq, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(batch_size, d_model, height, width)
            x = self.sr(x).reshape(batch_size, d_model, -1).permute(0, 2, 1)
            x = self.sr_norm(x)  # (batch_size, oq, d_model)
        # (batch_size, n_heads, head_dim, oq)
        k = self.k(x).reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        # (batch_size, n_heads, oq, head_dim)
        v = self.v(x).reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale  # (batch_size, n_heads, nq, oq)

        if self.apply_transform:
            attn = self.transform_conv(attn)  # (batch_size, n_heads, nq, oq)
            attn = torch.softmax(attn, dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = torch.softmax(attn, dim=-1)

        if attn_weights is not None:
            attn = attn_weights * attn
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = self.attn_drop(attn)
        # (batch_size, n_heads, nq, head_dim) -> (batch_size, nq, n_heads, head_dim) -> (batch_size, nq, d_model)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, nq, d_model)
        out = self.proj(out)  # (batch_size, nq, d_model)
        out = self.proj_drop(out)
        # (batch_size, channels, height, width)
        out = out.reshape(batch_size, height, width, d_model).permute(0, 3, 1, 2)
        return out


if __name__ == '__main__':
    x = torch.randn(3, 512, 7, 7)
    model = EMSA(512, n_heads=8, sr_ratio=2, apply_transform=True)
    out = model(x)
    print(out.shape)
