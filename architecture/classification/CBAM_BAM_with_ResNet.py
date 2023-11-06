"""
He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//
Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

https://arxiv.org/pdf/1512.03385.pdf
"""
import torch
from torch import nn
from typing import Optional, List, Type, Union
from torch import Tensor
from attention.CBAM import CBAM
from attention.BAM import BAM


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, norm_layer=None,
                 method: str = 'BAM'):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.method = method
        if method == 'BAM':
            self.ext = BAM(planes, reduction=16)
        elif method == 'CBAM':
            self.ext = CBAM(planes, reduction=16)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.method == 'BAM' or self.method == 'CBAM':
            out = self.ext(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, norm_layer=None,
                 method: str = 'BAM'):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.method = method
        if method == 'BAM':
            self.ext = BAM(planes * self.expansion, reduction=16)
        elif method == 'CBAM':
            self.ext = CBAM(planes * self.expansion, reduction=16)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.method == 'BAM' or self.method == 'CBAM':
            out = self.ext(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
                 num_classes: int = 1000, norm_layer=None, method: str = 'BAM'):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.method = method

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(-1)

        if method == 'BAM':
            self.ext1 = BAM(64 * block.expansion)
            self.ext2 = BAM(128 * block.expansion)
            self.ext3 = BAM(256 * block.expansion)
        elif method == 'CBAM':
            self.ext1 = CBAM(64 * block.expansion)
            self.ext2 = CBAM(128 * block.expansion)
            self.ext3 = CBAM(256 * block.expansion)
        else:
            self.ext1 = None
            self.ext2 = None
            self.ext3 = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Bottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int,
                    blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer, method=self.method)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, method=self.method))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # 以输入 (batch_size, 3, 224, 224)，模型为 resnet18 为例
        x = self.conv1(x)  # (batch_size, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch_size, 64, 56, 56)

        x = self.layer1(x)  # (batch_size, 64, 56, 56)
        if self.method == 'BAM' or self.method == 'CBAM':
            x = self.ext1(x)
        x = self.layer2(x)  # (batch_size, 128, 28, 28)
        if self.method == 'BAM' or self.method == 'CBAM':
            x = self.ext2(x)
        x = self.layer3(x)  # (batch_size, 256, 14, 14)
        if self.method == 'BAM' or self.method == 'CBAM':
            x = self.ext3(x)
        x = self.layer4(x)  # (batch_size, 512, 7, 7)

        x = self.avgpool(x)  # (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 512)
        x = self.fc(x)  # (batch_size, num_classes)
        x = self.softmax(x)
        return x


def resnet18(num_classes: int, norm_layer=None, method='BAM'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=norm_layer, method=method)


def resnet34(num_classes: int, norm_layer=None, method='BAM'):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, norm_layer=norm_layer, method=method)


def resnet50(num_classes: int, norm_layer=None, method='BAM'):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, norm_layer=norm_layer, method=method)


def resnet101(num_classes: int, norm_layer=None, method='BAM'):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, norm_layer=norm_layer, method=method)


def resnet152(num_classes: int, norm_layer=None, method='BAM'):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, norm_layer=norm_layer, method=method)


def resnet_with_bam_and_cbam(num_classes: int, norm_layer=None, mode=50, method='BAM'):
    if mode == 18:
        return resnet18(num_classes, norm_layer, method)
    elif mode == 34:
        return resnet34(num_classes, norm_layer, method)
    elif mode == 50:
        return resnet50(num_classes, norm_layer, method)
    elif mode == 101:
        return resnet101(num_classes, norm_layer, method)
    elif mode == 152:
        return resnet152(num_classes, norm_layer, method)
    else:
        raise ValueError(f"mode should be one of [18, 34, 50, 101, 152], but got {mode}")


if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = resnet_with_bam_and_cbam(2, mode=50, method='BAM')
    out = model(x)
    print(out.shape)
