import torch
from typing import Callable, Iterable
from tqdm import tqdm
from utils.metrics import accuracy
from utils.utils import AverageMeter
from utils.logging import save_training_log


def classification_train_one_epoch(loader: Iterable, model, criterion: Callable, optimizer,
                                   device, epoch: int = 0, log_freq: int = 0, tqdm_desc: bool = True):
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
    """
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    loader = tqdm(loader, colour="#f09199")
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 计算top1和top5准确率指标，记录loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1.item(), batch_size)
        top5_meter.update(acc5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tqdm_desc:
            loader.desc = (f"Epoch: {epoch} --> loss: {loss_meter.avg:.4f}"
                           f" | top1_acc: {top1_meter.avg:.4f}"
                           f" | top5_acc: {top5_meter.avg:.4f}")
        # 记录日志
        save_training_log(log_freq, step, batch_size, prefix=f'Train Epoch: {epoch} - ',
                          loss=loss_meter.avg, top1=top1_meter.avg, top5=top5_meter.avg)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


@torch.no_grad()
def classification_evaluate(loader: Iterable, model, criterion: Callable, device,
                            log_freq: int = 0, tqdm_desc: bool = True):
    """
    Parameters:
        loader: 验证集的dataloader
        model: 模型
        criterion: 损失函数
        device: 训练设备
        log_freq: 日志记录的频率，如果为None，则不记录日志，如果为0，则记录当前epoch的日志
        tqdm_desc: 是否显示tqdm的描述信息
    """
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    loader = tqdm(loader, colour="#a0d8ef")
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 计算top1和top5准确率指标，记录loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1.item(), batch_size)
        top5_meter.update(acc5.item(), batch_size)

        if tqdm_desc:
            loader.desc = (f"       --> loss: {loss_meter.avg:.4f}"
                           f" | top1_acc: {top1_meter.avg:.4f}"
                           f" | top5_acc: {top5_meter.avg:.4f}")

        # 记录日志
        save_training_log(log_freq, step, batch_size, prefix="Evaluate: ",
                          loss=loss_meter.avg, top1=top1_meter.avg, top5=top5_meter.avg)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg
