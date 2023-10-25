"""
Yang L, Zhang R Y, Li L, et al. Simam: A simple, parameter-free attention module for convolutional neural networks[C]//
International conference on machine learning. PMLR, 2021: 11863-11874.

https://proceedings.mlr.press/v139/yang21o/yang21o.pdf
"""
import torch
from torch import nn


class SimAM(nn.Module):
    """
    Sim Attention Module

    没有引入额外的参数，只是在原有的特征图上进行操作，所以可以直接在原有的网络上进行替换
    """

    def __init__(self, in_channels=None, e_lambda=1e-4):
        super().__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        """
        Parameters:
            x: feature map with shape of (batch_size, channels, height, width)

        Returns:
            a tensor with same shape of input x
        """
        _, _, height, width = x.size()

        n = width * height - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)


if __name__ == '__main__':
    x = torch.randn(3, 64, 7, 7)
    model = SimAM()
    outputs = model(x)
    print(outputs.shape)  # (3, 64, 7, 7)
