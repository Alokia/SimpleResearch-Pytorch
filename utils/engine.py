import torch
from typing import Callable, Iterable, Tuple
from tqdm import tqdm
from utils.metrics import accuracy
from utils.utils import AverageMeter
from utils.logging import save_training_log


def classification_train_one_epoch(loader: Iterable, model, criterion: Callable, optimizer,
                                   device, epoch: int = 0, log_freq: int = 0, tqdm_desc: bool = True,
                                   topk: Tuple[int, ...] = (1,)):
    """
    Parameters:
        loader: 训练集的dataloader
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        epoch: 当前的训练轮数
        log_freq: 日志记录的频率，如果为None，则不记录日志，如果为0，则记录当前epoch的日志
        tqdm_desc: 是否显示tqdm的描述信息
        topk: 计算topk准确率指标, 默认为(1,), 不得超过类别数
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = [AverageMeter() for _ in topk]

    loader = tqdm(loader, colour="#f09199", dynamic_ncols=True)
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 计算准确率指标，记录loss
        acc = accuracy(outputs, labels, topk=topk)
        loss_meter.update(loss.item(), batch_size)
        for i, meter in enumerate(acc_meter):
            meter.update(acc[i].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tqdm_desc:
            desc = f"Epoch: {epoch} --> loss: {loss_meter.avg:.4f}"
            for i, meter in enumerate(acc_meter):
                desc += f" | top{topk[i]}_acc: {meter.avg:.4f}%"
            loader.desc = desc
        # 记录日志
        save_training_log(log_freq, step, batch_size, prefix=f'Train Epoch: {epoch} - ',
                          loss=loss_meter.avg)

    return loss_meter.avg


@torch.no_grad()
def classification_evaluate(loader: Iterable, model, criterion: Callable, device,
                            log_freq: int = 0, tqdm_desc: bool = True, topk: Tuple[int, ...] = (1,)):
    """
    Parameters:
        loader: 验证集的dataloader
        model: 模型
        criterion: 损失函数
        device: 训练设备
        log_freq: 日志记录的频率，如果为None，则不记录日志，如果为0，则记录当前epoch的日志
        tqdm_desc: 是否显示tqdm的描述信息
        topk: 计算topk准确率指标, 默认为(1,), 不得超过类别数
    """
    loss_meter = AverageMeter()
    acc_meter = [AverageMeter() for _ in topk]

    loader = tqdm(loader, colour="#a0d8ef", dynamic_ncols=True)
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 计算top1和top5准确率指标，记录loss
        acc = accuracy(outputs, labels, topk=topk)
        loss_meter.update(loss.item(), batch_size)
        for i, meter in enumerate(acc_meter):
            meter.update(acc[i].item(), batch_size)

        if tqdm_desc:
            desc = f"      --> loss: {loss_meter.avg:.4f}"
            for i, meter in enumerate(acc_meter):
                desc += f" | top{topk[i]}_acc: {meter.avg:.4f}%"
            loader.desc = desc
        # 记录日志
        save_training_log(log_freq, step, batch_size, prefix="Evaluate: ",
                          loss=loss_meter.avg)

    return loss_meter.avg
