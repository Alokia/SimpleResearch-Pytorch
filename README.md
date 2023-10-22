# 说明
这是一个记录所学习模块的仓库，实现了学习的深度学习模块，做到即插即用。

## 注意力模块

* [Self-Attention](https://arxiv.org/abs/1706.03762)  
Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.

```python
from attention.SelfAttention import ScaledDotProductAttention, MultiHeadAttention
import torch

query = torch.randn(50, 20, 512)
key = torch.randn(50, 49, 512)
value = torch.randn(50, 49, 512)

sdpa = ScaledDotProductAttention(dropout=0.2)
mha = MultiHeadAttention(d_model=512, d_k=512, d_v=256, n_heads=8)

output_sdpa = sdpa(query, key, value)
output_mha = mha(query, key, value)

print(output_sdpa.shape)  # [50, 20, 512]
print(output_mha.shape)  # [50, 20, 512]
```

* Simplified Self-Attention

```python
from attention.SimplifiedSelfAttention import MultiHeadSimplifiedSelfAttention
import torch

query = torch.randn(50, 20, 512)
key = torch.randn(50, 49, 512)
value = torch.randn(50, 49, 512)

mhssan = MultiHeadSimplifiedSelfAttention(d_model=512, n_heads=8)

out = mhssan(query, key, value)

print(out.shape)  # (50, 20, 512)
```



## 卷积模块

* [Depthwise Separable Convolution](https://arxiv.org/abs/1610.02357)  
Chollet F. Xception: Deep learning with depthwise separable convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1251-1258.

```python
from conv.DepthwiseSeparableConv2d import DepthwiseSeparableConv2d
import torch

x = torch.randn(2, 3, 224, 224)
dsconv = DepthwiseSeparableConv2d(3, 64, kernel_size=3, stride=2)
out = dsconv(x)
print(out.shape)  # [2, 64, 112, 112]
```

* [Heterogeneous Convolution](https://arxiv.org/pdf/1903.04120.pdf)  
Singh P, Verma V K, Rai P, et al. Hetconv: Heterogeneous kernel-based convolutions for deep cnns[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 4835-4844.

```python
from conv.HeterogeneousConv2d import HetConv2d
import torch

x = torch.rand(1, 16, 224, 125)
conv = HetConv2d(16, 64, p=4)
out = conv(x)
print(out.shape)  # [1, 64, 224, 125]
```

