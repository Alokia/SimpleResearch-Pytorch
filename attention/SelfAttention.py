"""
Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J].
Advances in neural information processing systems, 2017, 30.

https://arxiv.org/abs/1706.03762
"""
import torch
from torch import nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout: float = 0.1, scale: float = None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor = None, attn_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters:
            queries: queries, shape: (batch_size, nq, d_model)
            keys: keys, shape: (batch_size, seq_len, d_model)
            values: values, shape: (batch_size, seq_len, d_model)
            attn_mask: attention mask, 当值为 True 时表示使用负无穷遮盖该处的值, shape: (batch_size, nq, seq_len)
            attn_weights: attention weights, 与注意力相乘的权重矩阵, shape: (batch_size, nq, seq_len)

        Returns:
            out: a tensor with shape (batch_size, nq, d_model)
        """
        scale = 1 / np.sqrt(queries.shape[-1]) if self.scale is None else self.scale
        attn = torch.matmul(queries, keys.transpose(1, 2)) * scale  # (batch_size, nq, seq_len)

        if attn_weights is not None:
            attn = attn * attn_weights
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, values)  # (batch_size, nq, d_model)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int = None, d_v: int = None,
                 dropout: float = 0.1, qkv_bias: bool = False, scale: float = None):
        """
        Parameters:
            d_model: 前面层的输出维度，即 queries、keys、values的维度 (batch_size, *, d_model)
            n_heads: heads 的数量
            d_k: queries 和 keys 的隐藏层维度，如果为None，则默认为 d_model // n_heads
            d_v: values 的隐藏层维度，如果为None，则默认为 d_model // n_heads
            dropout: dropout 的概率
            qkv_bias: 是否使用qkv全连接层的偏置
            scale: 缩放因子，如果为None，则默认为 np.sqrt(d_k)
        """
        super().__init__()

        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads

        self.fc_q = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.fc_k = nn.Linear(d_model, self.d_k * n_heads, bias=qkv_bias)
        self.fc_v = nn.Linear(d_model, self.d_v * n_heads, bias=qkv_bias)
        self.fc_out = nn.Linear(self.d_v * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = np.sqrt(self.d_k) if scale is None else scale
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor = None, attn_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters:
            queries: queries, shape: (batch_size, nq, d_model)
            keys: keys, shape: (batch_size, seq_len, d_model)
            values: values, shape: (batch_size, seq_len, d_model)
            attn_mask: attention mask, 当值为 True 时表示使用负无穷遮盖该处的值, shape: (batch_size, n_heads, nq, seq_len)
            attn_weights: attention weights, 与注意力相乘的权重矩阵, shape: (batch_size, n_heads, nq, seq_len)

        Returns:
            out: a tensor with shape (batch_size, nq, d_model)
        """
        batch_size = queries.shape[0]
        nq = queries.shape[1]
        seq_len = keys.shape[1]

        # q: (batch_size, nq, d_model) -> (batch_size, nq, n_heads * d_k) -> (batch_size, nq, n_heads, d_k) -> (batch_size, n_heads, nq, d_k)
        q = self.fc_q(queries).view(batch_size, nq, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        # k: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads * d_k) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, d_k, seq_len)
        k = self.fc_k(keys).view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # v: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads * d_v) -> (batch_size, seq_len, n_heads, d_v) -> (batch_size, n_heads, seq_len, d_v)
        v = self.fc_v(values).view(batch_size, seq_len, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) * self.scale  # (batch_size, n_heads, nq, seq_len)

        if attn_weights is not None:
            attn = attn * attn_weights
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # out: (batch_size, n_heads, nq, d_v) -> (batch_size, nq, n_heads, d_v) -> (batch_size, nq, n_heads * d_v)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, nq, self.n_heads * self.d_v)
        # out: (batch_size, nq, n_heads * d_v) -> (batch_size, nq, d_model)
        out = self.fc_out(out)
        return out


if __name__ == "__main__":
    query = torch.randn(50, 20, 512)
    key = torch.randn(50, 49, 512)
    value = torch.randn(50, 49, 512)

    sdpa = ScaledDotProductAttention(dropout=0.2)
    mha = MultiHeadAttention(d_model=512, n_heads=8)

    output_sdpa = sdpa(query, key, value)
    output_mha = mha(query, key, value)

    print(output_sdpa.shape)  # (50, 20, 512)
    print(output_mha.shape)  # (50, 20, 512)
