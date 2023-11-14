import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from architecture.classification.CBAM_BAM_with_ResNet import resnet_with_bam_and_cbam
from pipeline.classification_pipeline import parser_args, create_trainer
from utils.engine import ClassificationLightningModel
from utils.utils import load_yaml_with_omegaconf
import warnings

warnings.filterwarnings("ignore")


def main(args):
    omega_conf = load_yaml_with_omegaconf(args.config)

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True,
                                             num_workers=args.num_workers, )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 模型
    model = resnet_with_bam_and_cbam(num_classes=args.num_classes, mode=args.model_mode, method='BAM')
    lightning_model = ClassificationLightningModel(model, criterion, opt_params=omega_conf.fit.optimizer)

    trainer = create_trainer(omega_conf)
    trainer.logger.log_hyperparams(args)  # 保存超参数

    # 训练模型
    trainer.fit(lightning_model, train_loader, val_loader)


if __name__ == '__main__':
    parser = parser_args(return_args=False)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--model_mode', type=int, default=50, help='model mode')
    parser.add_argument('--train_folder', type=str, default='G:\\datasets\\Intel Image Classification\\seg_train',
                        help='train folder')
    parser.add_argument('--val_folder', type=str, default='G:\\datasets\\Cat and Dog\\test_set', help='val folder')

    args = parser.parse_args()

    main(args)
