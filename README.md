# 说明
这是一个记录学习的仓库，实现了学习的各种深度学习模型。欢迎提交PR，一起丰富这个仓库。


# 文件夹信息介绍

1. architecture：各种深度学习模型的架构，主要为创新模型。
2. **attention**：各种注意力机制，基本上做到即插即用。
3. backbone：各种基准模型，常用来作为创新的基础模型，如ResNet等。
4. **conv**：各种卷积结构，如深度可分离卷积等，基本上做到即插即用。
5. dataset：常用数据集的加载过程。
6. figure：各种图片，以及 `attention` 和 `conv` 的介绍。
7. loss_function：一些特殊的损失函数。
8. normalization：一些归一化方法。
9. pipeline：一些常用的训练流程。
10. position_embedding：一些位置编码方法。
11. utils：一些常用的工具函数。比如提前终止、指标计算、训练和测试的基本流程等。


# 核心：attention 和 conv

## attention
记录了所学习的各种注意力机制，基本做到了即插即用。详细的介绍见
[figure/attention.md](./figure/attention.md)。

## conv
记录了所学习的各种卷积结构，基本做到了即插即用。详细的介绍见
[figure/conv.md](./figure/conv.md)。

# 安装该库

```shell
git clone https://github.com/Alokia/SimpleResearch-Pytorch.git
cd SimpleResearch-Pytorch
pip install -e .
```

# 使用该库

```python
from SimpleResearch.attention.SelfAttention import MultiHeadAttention

...
```
