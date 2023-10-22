"""
暂不知来源
"""
import torch
from torch import nn
import numpy as np


class MultiHeadSimplifiedSelfAttention(nn.Module):
    """
    Multi-Head Simplified Self-Attention

    相比于传统的 Multi-Head Self-Attention，该模块的参数量更小，且效果基本没有降低。
    该模块在语音识别问题中提出。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, scale: float = None):
        """
        Parameters:
            d_model: 前面层的输出维度，即 queries、keys、values的维度 (batch_size, *, d_model)
            n_heads: heads 的数量
            dropout: dropout 的概率
            scale: 缩放因子，如果为None，则默认为 np.sqrt(d_k)
        """
        super().__init__()

        assert d_model % n_heads == 0, f"d_model must divisible by n_heads, but get d_model: {d_model}, n_heads: {n_heads}"

        self.d_model = d_model
        self.d_k = self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.scale = scale if scale is not None else self.d_k ** -0.5

        self.fc_out = nn.Linear(n_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        # (batch_size, nq, d_model) -> (batch_size, nq, n_heads, d_k) -> (batch_size, n_heads, nq, d_k)
        q = queries.view(batch_size, nq, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, d_k, seq_len)
        k = keys.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_v) -> (batch_size, n_heads, seq_len, d_v)
        v = values.view(batch_size, seq_len, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) * self.scale  # (batch_size, n_heads, nq, seq_len)

        if attn_weights is not None:
            attn = attn * attn_weights
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # (batch_size, n_heads, nq, d_v) -> (batch_size, nq, n_heads, d_v) -> (batch_size, nq, n_heads * d_v)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, nq, self.n_heads * self.d_v)
        # (batch_size, nq, n_heads * d_v) -> (batch_size, nq, d_model)
        out = self.fc_out(out)
        return out


if __name__ == "__main__":
    query = torch.randn(50, 20, 512)
    key = torch.randn(50, 49, 512)
    value = torch.randn(50, 49, 512)

    mhssan = MultiHeadSimplifiedSelfAttention(d_model=512, n_heads=8)

    out = mhssan(query, key, value)

    print(out.shape)  # (50, 20, 512)
