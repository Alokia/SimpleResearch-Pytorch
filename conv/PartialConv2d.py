"""
Chen J, Kao S, He H, et al. Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks[C]//
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 12021-12031.

https://arxiv.org/pdf/2303.03667.pdf
"""
import torch
from torch import nn


class PConv(nn.Module):
    def __init__(self, in_channels, n_div, method="split_cat"):
        """
        Parameters:
            in_channels: number of input channels
            n_div: number of divisions for the input channels. in_channels // n_div is the number of channels for the convolution
            method: method for partial convolution, can be "split_cat" or "slicing", default is "split_cat"
        """
        super().__init__()
        self.dim_conv3 = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv3

        assert method in ["split_cat", "slicing"]
        self.method = method

        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward_slicing(self, x):
        # slicing method, only used in inference
        x = x.clone()  # keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # split_cat method, used in training or inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat([x1, x2], dim=1)
        return x

    def forward(self, x):
        """
        Parameters:
            x: input tensor with shape of (batch_size, in_channels, height, width)

        Returns:
            output tensor with shape of (batch_size, in_channels, height, width)
        """
        if self.method == "split_cat":
            return self.forward_split_cat(x)
        elif self.method == "slicing":
            return self.forward_slicing(x)
        else:
            raise ValueError(f"Unknown method {self.method}")


if __name__ == '__main__':
    x = torch.randn(3, 128, 56, 56)
    model = PConv(128, 4)
    out = model(x)
    print(out.shape)  # (3, 128, 56, 56)
