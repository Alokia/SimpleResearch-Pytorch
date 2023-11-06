from utils.engine import classification_train_one_epoch, classification_evaluate
from argparse import ArgumentParser
from utils.callbacks import ModelCheckpoint, EarlyStopping


def argument_parser(return_parser=False):
    parser = ArgumentParser()

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--topk', type=int, nargs='+', default=(1,), help='topk accuracy')
    parser.add_argument('--val_every_epoch', type=int, default=1, help='validate every n epoch')

    # 回调函数相关参数
    parser.add_argument('--use_cp', action='store_true', default=False, help='whether to use ModelCheckpoint')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='ModelCheckpoint save path')
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


def classification_step(train_loader, model, criterion, optimizer, device, val_loader=None,
                        start_epoch=1, args=None):
    # 自动保存训练过程
    model_cp = ModelCheckpoint(filepath=args.save_path, monitor=args.cp_monitor, mode=args.cp_mode,
                               save_best_only=args.not_save_best_only, save_freq=args.save_freq)
    # 提前终止
    early_stop = EarlyStopping(monitor=args.early_stop_monitor, mode=args.early_stop_mode, patience=args.patience)

    for epoch in range(start_epoch, args.epochs + 1):
        metrics = {}
        # 训练一个epoch
        train_loss, train_meters = classification_train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch,
            log_freq=args.log_freq, tqdm_desc=args.no_tqdm_desc, topk=args.topk
        )
        # 记录训练集的loss和topk准确率
        metrics['train_loss'] = train_loss
        for i in range(len(args.topk)):
            metrics[f'train_top{args.topk[i]}_acc'] = train_meters[i]

        if val_loader is not None and epoch % args.val_every_epoch == 0:
            # 验证
            val_loss, val_meters = classification_evaluate(
                val_loader, model, criterion, device,
                log_freq=args.log_freq, tqdm_desc=args.no_tqdm_desc, topk=args.topk
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
                return
