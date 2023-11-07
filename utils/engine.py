import torch
from typing import Callable, Iterable, Tuple
from tqdm import tqdm
from utils.metrics import accuracy
from utils.utils import AverageMeter
from utils.logging import save_training_log
from utils.distributed_setting import is_main_process, reduce_value
import sys


def classification_train_one_epoch(loader: Iterable, model, criterion: Callable, optimizer,
                                   device, epoch: int = 0, log_freq: int = 0, tqdm_desc: bool = True,
                                   topk: Tuple[int, ...] = (1,), use_tqdm: bool = True, tqdm_row: int = 0,
                                   is_distributed: bool = False):
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
        use_tqdm: 是否使用tqdm
        tqdm_row: tqdm显示的行数
        is_distributed: 是否是分布式训练
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = [AverageMeter() for _ in topk]

    if use_tqdm:
        loader = tqdm(loader, colour="#f09199", dynamic_ncols=True, position=tqdm_row)
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_distributed:
            reduce_value(loss, average=True)

        # 损失是无穷，则终止所有训练
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 计算准确率指标，记录loss
        acc = accuracy(outputs, labels, topk=topk)
        loss_meter.update(loss.item(), batch_size)
        for i, meter in enumerate(acc_meter):
            meter.update(acc[i].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_tqdm:
            if tqdm_desc:
                desc = f"Epoch: {epoch} - GPU sequence: {tqdm_row} --> loss: {loss_meter.avg:.4f}"
                for i, meter in enumerate(acc_meter):
                    desc += f" | top{topk[i]}_acc: {meter.avg:.4f}%"
                loader.desc = desc
        # 记录日志
        save_training_log(log_freq, step, batch_size, prefix=f'Train Epoch: {epoch} - ',
                          loss=loss_meter.avg)

    if is_distributed:
        # 等待所有进程计算完毕
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

    return loss_meter.avg, [meter.avg for meter in acc_meter]


@torch.no_grad()
def classification_evaluate(loader: Iterable, model, criterion: Callable, device,
                            log_freq: int = 0, tqdm_desc: bool = True, topk: Tuple[int, ...] = (1,),
                            use_tqdm: bool = True, tqdm_row: int = 0, is_distributed: bool = False):
    """
    Parameters:
        loader: 验证集的dataloader
        model: 模型
        criterion: 损失函数
        device: 训练设备
        log_freq: 日志记录的频率，如果为None，则不记录日志，如果为0，则记录当前epoch的日志
        tqdm_desc: 是否显示tqdm的描述信息
        topk: 计算topk准确率指标, 默认为(1,), 不得超过类别数
        use_tqdm: 是否使用tqdm
        tqdm_row: tqdm显示的行数
        is_distributed: 是否是分布式训练
    """
    loss_meter = AverageMeter()
    acc_meter = [AverageMeter() for _ in topk]

    if use_tqdm:
        loader = tqdm(loader, colour="#a0d8ef", dynamic_ncols=True, position=tqdm_row)
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

        if use_tqdm:
            if tqdm_desc:
                desc = f"      --> loss: {loss_meter.avg:.4f}"
                for i, meter in enumerate(acc_meter):
                    desc += f" | top{topk[i]}_acc: {meter.avg:.4f}%"
                loader.desc = desc
        # 记录日志
        save_training_log(log_freq, step, batch_size, prefix="Evaluate: ",
                          loss=loss_meter.avg)

    if is_distributed:
        # 等待所有进程计算完毕
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

    return loss_meter.avg, [meter.avg for meter in acc_meter]
