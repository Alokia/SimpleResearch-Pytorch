import torch
import lightning as L
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def configure_optimizers(opt_params, parameters):
    optimizer = create_optimizer(opt_params.optimizer, parameters)
    if opt_params.scheduler.sched == "cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=opt_params.scheduler.warmup_epochs,
            max_epochs=opt_params.scheduler.max_epochs,
            warmup_start_lr=opt_params.scheduler.warmup_start_lr,
            eta_min=opt_params.scheduler.eta_min
        )
    else:
        scheduler, _ = create_scheduler(opt_params.scheduler, optimizer)
    return [optimizer], [scheduler]


class ClassificationLightningModel(L.LightningModule):
    def __init__(self, model, criterion, opt_params):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.opt_params = opt_params

    def forward(self, x):
        return self.model(x)

    def _calculate_metrics(self, predicts, labels, mode, sync_dist=True):
        num_classes = predicts.shape[1]
        if num_classes <= 5:
            acc1 = accuracy(predicts, labels, topk=(1,))
            self.log(f"{mode}_acc1", acc1, prog_bar=True, sync_dist=sync_dist)
        else:
            acc1, acc5 = accuracy(predicts, labels, topk=(1, 5))
            self.log(f"{mode}_acc1", acc1, prog_bar=True, sync_dist=sync_dist)
            self.log(f"{mode}_acc5", acc5, prog_bar=True, sync_dist=sync_dist)

    def _calculate_loss(self, batch, mode="train"):
        images, labels = batch
        predicts = self.model(images)
        loss = self.criterion(predicts, labels)

        sync_dist = True if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else False
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=sync_dist)

        # 计算指标
        if mode != "train":
            self._calculate_metrics(predicts, labels, mode, sync_dist=sync_dist)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")
        return loss

    def predict_step(self, images: torch.Tensor) -> torch.Tensor:
        # images: (batch_size, channels, height, width)
        predicts = self.model(images)  # (batch_size, num_classes)
        predicts = torch.softmax(predicts, dim=1)
        predicts = torch.argmax(predicts, dim=1)
        return predicts  # (batch_size, )

    def configure_optimizers(self):
        return configure_optimizers(self.opt_params, self.parameters())


class DetectionLightningModel(L.LightningModule):
    def __init__(self, model, criterion, opt_params):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.opt_params = opt_params

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch, mode="train"):
        images, boxes, labels = batch
        boxes = [box.to(images) for box in boxes]
        labels = [label.to(images) for label in labels]

        pred_loc, pred_score = self.model(images)
        loss = self.criterion(pred_loc, pred_score, boxes, labels)

        sync_dist = True if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else False
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=sync_dist)

        # 计算指标
        if mode != "train":
            self._calculate_metrics(pred_loc, pred_score, boxes, labels, mode, sync_dist=sync_dist)

        return loss

    def _calculate_metrics(self, pred_loc, pred_score, boxes, labels, mode, sync_dist=True):
        pass

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="test")
        return loss

    def predict_step(self, images: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        return configure_optimizers(self.opt_params, self.parameters())


class SegmentLightningModel(L.LightningModule):
    def __init__(self, model, criterion, opt_params):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.opt_params = opt_params

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch, mode="train"):
        pass

    def _calculate_metrics(self, predicts, labels, mode, sync_dist=True):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, images: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        return configure_optimizers(self.opt_params, self.parameters())
