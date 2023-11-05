from torch import Tensor
from typing import Tuple
import torch


@torch.no_grad()
def accuracy(y_pred: Tensor, y_true: Tensor, topk: Tuple[int, ...] = (1,)):
    """
    Parameters:
        y_pred: Tensor of shape (batch_size, num_classes)
        y_true: Tensor of shape (batch_size)
        topk: Tuple of integers

    Returns:
        res: List of Tensors of shape (len(topk), )
    """
    max_k = max(topk)
    batch_size = y_true.size(0)

    _, pred = y_pred.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
