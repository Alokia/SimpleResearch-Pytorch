import torch
from torch import nn
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from architecture.classification.CBAM_BAM_with_ResNet import resnet_with_bam_and_cbam
from pipeline.classification_pipeline import argument_parser, classification_step
from utils.distributed_setting import init_distributed_mode, cleanup
import os
import tempfile
from utils.distributed_setting import dist


def main(args):
    if args.distributed:
        if torch.cuda.is_available() is False:
            raise EnvironmentError("not find GPU device for distributed training.")
        # 初始化各进程环境
        init_distributed_mode(args=args)

    device = torch.device(args.device)
    if args.distributed:
        args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

    # 数据集
    train_dataset = ImageFolder(
        root=args.train_folder,
        transform=transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    val_dataset = ImageFolder(
        root=args.val_folder,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

    if args.distributed:
        # 给每个rank对应的进程分配训练的样本索引
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        # 将样本索引每batch_size个元素组成一个list
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                                   num_workers=args.num_workers, pin_memory=True, )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True,
                                                 sampler=val_sampler, num_workers=args.num_workers, )
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True,
                                                 num_workers=args.num_workers, )

    # 模型
    model = resnet_with_bam_and_cbam(num_classes=args.num_classes, mode=args.model_mode, method='BAM').to(device)

    if args.distributed:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if args.rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if args.distributed and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 转换为同步BN层
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    classification_step(train_loader, model, criterion, optimizer, device, val_loader=val_loader,
                        args=args, sampler=train_sampler)

    # 删除临时缓存文件
    if args.rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argument_parser(return_parser=True)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--model_mode', type=int, default=50, help='model mode')
    parser.add_argument('--train_folder', type=str, default='G:\\datasets\\Intel Image Classification\\seg_train',
                        help='train folder')
    parser.add_argument('--val_folder', type=str, default='G:\\datasets\\Cat and Dog\\test_set', help='val folder')

    args = parser.parse_args()

    main(args)
