"""
Guo M H, Liu Z N, Mu T J, et al. Beyond self-attention: External attention using two linear layers for visual tasks[J].
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022, 45(5): 5436-5447.

https://arxiv.org/abs/2105.02358
"""
import torch
from torch import nn
import torch.nn.functional as F


class ExternalAttention(nn.Module):
    """
    external attention

    与普通的 self-attention 相比，考虑了整个数据集层次的关系，并且复杂度降低为线性
    """

    def __init__(self, d_model: int, S: int = 64, dropout: float = 0.):
        """
        Parameters:
            d_model: the dimension of the input feature
            S: the dimension of the external memory unit, (S, d_model)
            dropout: the dropout rate
        """
        super().__init__()

        self.query_linear = nn.Linear(d_model, d_model, bias=False)

        self.d_model = d_model
        self.S = S

        self.M_k = nn.Linear(d_model, S, bias=False)
        self.M_v = nn.Linear(S, d_model, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor):
        """
        Parameters:
            queries: (batch_size, seq_len, d_model)
        """
        queries = self.query_linear(queries)  # (batch_size, seq_len, d_model)

        attn = self.M_k(queries)  # (batch_size, seq_len, S)
        attn = self.softmax(attn)
        attn = F.normalize(attn, p=1, dim=2)  # (batch_size, seq_len, S)
        attn = self.dropout(attn)

        out = self.M_v(attn)  # (batch_size, seq_len, d_model)
        return out


class MultiHeadExternalAttention(nn.Module):
    """
    Multi-head external attention
    """

    def __init__(self, d_model: int, n_heads: int, coef: int = 4, S: int = None, dropout: float = 0.1):
        """
        Parameters:
            d_model: the dimension of the input feature
            n_heads: the number of heads, d_model must be divisible by n_heads
            coef: 协调因子，用于控制 d_model 维度的放大倍数
            S: the dimension of the external memory unit，如果为 None，则默认为 256 // coef
            dropout: the dropout rate of attn
        """
        super().__init__()

        assert d_model % n_heads == 0, f"d_model must divisible by n_heads, but get d_model: {d_model}, n_heads: {n_heads}"

        self.query_linear = nn.Linear(d_model, d_model * coef, bias=False)

        self.coef = coef
        self.d_model = d_model
        self.n_heads = n_heads * coef
        self.S = 256 // self.coef if S is None else S

        self.M_k = nn.Linear(d_model * coef // self.n_heads, self.S, bias=False)
        self.M_v = nn.Linear(self.S, d_model * coef // self.n_heads, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.fc_o = nn.Linear(d_model * self.coef, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries: torch.Tensor):
        """
        Parameters:
            queries: (batch_size, seq_len, d_model)
        """
        batch_size = queries.shape[0]
        seq_len = queries.shape[1]

        queries = self.query_linear(queries)  # (batch_size, seq_len, d_model * coef)

        # (batch_size, seq_len, d_model * coef) -> (batch_size, seq_len, n_heads, d_model * coef / n_heads)
        #   -> (batch_size, n_heads, seq_len, d_model * coef / n_heads)
        queries = queries.view(batch_size, seq_len, self.n_heads, -1).permute(0, 2, 1, 3)

        attn = self.M_k(queries)  # (batch_size, n_heads, seq_len, S)
        attn = self.softmax(attn)
        attn = F.normalize(attn, p=1, dim=2)  # (batch_size, n_heads, seq_len, S)
        attn = self.attn_drop(attn)

        # (batch_size, n_heads, seq_len, S) -> (batch_size, n_heads, seq_len, d_model * coef / n_heads) ->
        #   (batch_size, seq_len, n_heads, d_model * coef / n_heads) -> (batch_size, seq_len, d_model * coef)
        out = self.M_v(attn).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model * self.coef)
        out = self.fc_o(out)  # (batch_size, seq_len, d_model)
        return out


if __name__ == '__main__':
    x = torch.randn(3, 49, 512)
    ea = ExternalAttention(d_model=512, S=64)
    mhea = MultiHeadExternalAttention(d_model=512, n_heads=4, coef=4, S=64)

    output_ea = ea(x)
    output_mhea = mhea(x)

    print(output_ea.shape)  # (3, 49, 512)
    print(output_mhea.shape)  # (3, 49, 512)
