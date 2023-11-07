from utils.engine import classification_train_one_epoch, classification_evaluate
from argparse import ArgumentParser
from utils.callbacks import ModelCheckpoint, EarlyStopping
from utils.utils import seed_everything
import torch


def argument_parser(return_parser=True):
    parser = ArgumentParser()

    # 分布式训练相关参数
    # 单机多卡不要改该参数，系统会自动分配
    parser.add_argument('--device', type=str, default='cuda', help='use which device to train')
    parser.add_argument('--distributed', action='store_true', default=False, help='whether to use distributed training')
    #  开启的进程数(注意不是线程), 不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend, win: gloo, linux: nccl')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    # 是否启用SyncBatchNorm
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync_bn')
    # 该参数不用设置，用于控制分布式训练模型当前的设备，以及控制进度条显示的位置
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--topk', type=int, nargs='+', default=(1,), help='topk accuracy')
    parser.add_argument('--val_every_epoch', type=int, default=1, help='validate every n epoch')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--no_tqdm', action='store_false', default=True, help='do not use tqdm')

    # 恢复训练相关参数
    parser.add_argument('--resume', action="store_true", default=False, help='resume from checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default='checkpoints.pth', help='resume from checkpoint path')
    parser.add_argument('--resume_model_only', action='store_true', default=False, help='resume model only')
    parser.add_argument('--resume_callback_reset', action='store_true', default=False,
                        help='whether to reset callback state')

    # 回调函数相关参数
    parser.add_argument('--use_cp', action='store_true', default=False, help='whether to use ModelCheckpoint')
    parser.add_argument('--save_path', type=str, default='checkpoints.pth', help='ModelCheckpoint save path')
    parser.add_argument('--cp_monitor', type=str, default='train_top1_acc', help='ModelCheckpoint monitor')
    parser.add_argument('--cp_mode', type=str, default='max', choices=['min', 'max'], help='ModelCheckpoint mode')
    parser.add_argument('--not_save_best_only', action='store_false', default=True,
                        help='ModelCheckpoint save best only')
    parser.add_argument('--save_freq', type=int, default=1, help='ModelCheckpoint save frequency')
    parser.add_argument('--save_model_only', action='store_true', default=False, help='ModelCheckpoint save model only')

    parser.add_argument('--use_early_stop', action='store_true', default=False, help='whether to use EarlyStopping')
    parser.add_argument('--early_stop_monitor', type=str, default='val_loss', help='EarlyStopping monitor')
    parser.add_argument('--early_stop_mode', type=str, default='min', choices=['min', 'max'], help='EarlyStopping mode')
    parser.add_argument('--patience', type=int, default=5, help='EarlyStopping patience')

    # 日志相关参数
    parser.add_argument('--log_freq', type=int, default=0, help='log frequency')
    parser.add_argument('--no_tqdm_desc', action='store_false', default=True, help='whether to show tqdm description')

    if return_parser:
        return parser
    return parser.parse_args()


def resume_checkpoint(args, model, device, optimizer, model_cp, early_stop):
    # 如果恢复训练，加载相关权重
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        if not args.resume_model_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['start_epoch'] + 1
            model_cp.load_state_dict(checkpoint['model_cp'])
            early_stop.load_state_dict(checkpoint['early_stop'])
            if args.resume_callback_reset:
                model_cp.reset()
                early_stop.reset()
    return start_epoch


def classification_step(train_loader, model, criterion, optimizer, device, val_loader=None, args=None):
    # 是否固定随机种子
    seed_everything(args.seed)
    # 自动保存训练过程
    model_cp = ModelCheckpoint(filepath=args.save_path, monitor=args.cp_monitor, mode=args.cp_mode,
                               save_best_only=args.not_save_best_only, save_freq=args.save_freq)
    # 提前终止
    early_stop = EarlyStopping(monitor=args.early_stop_monitor, mode=args.early_stop_mode, patience=args.patience)

    # 恢复训练
    start_epoch = resume_checkpoint(args, model, device, optimizer, model_cp, early_stop)

    # 开始训练
    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            # 在每个epoch开始前打乱数据顺序
            train_loader.sampler.set_epoch(epoch)

        metrics = {}
        # 训练一个epoch
        train_loss, train_meters = classification_train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch,
            log_freq=args.log_freq, tqdm_desc=args.no_tqdm_desc, topk=args.topk,
            use_tqdm=args.no_tqdm, tqdm_row=args.gpu, is_distributed=args.distributed,
        )
        # 记录训练集的loss和topk准确率
        metrics['train_loss'] = train_loss
        for i in range(len(args.topk)):
            metrics[f'train_top{args.topk[i]}_acc'] = train_meters[i]

        if val_loader is not None and epoch % args.val_every_epoch == 0:
            # 验证
            val_loss, val_meters = classification_evaluate(
                val_loader, model, criterion, device,
                log_freq=args.log_freq, tqdm_desc=args.no_tqdm_desc, topk=args.topk,
                use_tqdm=args.no_tqdm, tqdm_row=args.gpu, is_distributed=args.distributed,
            )
            # 记录验证集的loss和topk准确率
            metrics['val_loss'] = val_loss
            for i in range(len(args.topk)):
                metrics[f'val_top{args.topk[i]}_acc'] = val_meters[i]

        # 保存模型
        if args.use_cp:
            save_dict = {'model': model.state_dict()}
            if not args.save_model_only:
                save_dict.update({'optimizer': optimizer.state_dict(), 'start_epoch': epoch, 'args': args,
                                  'model_cp': model_cp.state_dict(), 'early_stop': early_stop.state_dict()})
            model_cp.step(metrics, **save_dict)
        # 提前终止
        if args.use_early_stop:
            if early_stop.step(metrics):
                break

    # 清理进程
    torch.distributed.destroy_process_group()
