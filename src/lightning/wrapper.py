from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
import torchvision.transforms.v2 as transforms


class LightningImageClassificationWrapper(pl.LightningModule):
    """
    A general purpose Lightning wrapper for Image Classification.
    Allows for the use mixup and cutmix.
    Uses the Adam optimizer, and OneCycleLR as the learning rate scheduler with a warmup of 10%.
    """
    def __init__(self, model: nn.Module, total_steps: int, learning_rate: float = 0.05, n_classes=10,
                 use_mixup_cutmix=True):
        super().__init__()
        self.model = model
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()

        # TorchMetrics gets a little bit funky with Lightning.
        # This is done to keep these metrics visible to be reset in the training/validation/test epoch end.
        self.accuracies = nn.ModuleDict({
            "train_acc": Accuracy(task='multiclass', num_classes=10),
            "val_acc": Accuracy(task='multiclass', num_classes=10),
            "test_acc": Accuracy(task='multiclass', num_classes=10),
        })

        self.use_mixup_cutmix = use_mixup_cutmix

        # Settings taken from EfficientNet:
        #   (https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2)
        self.mixup_cutmix = transforms.RandomChoice([
            transforms.CutMix(num_classes=n_classes, alpha=1.0),
            transforms.MixUp(num_classes=n_classes, alpha=0.2)
        ])

    def log_metrics(self, loss, x_hat, y, step='train'):
        log_on_step = step == "train"
        self.log(f"{step}_loss", loss, on_step=log_on_step, on_epoch=True, prog_bar=True)
        # Get the actual predictions
        preds = torch.softmax(x_hat, dim=-1)

        # Get the corresponding step Accuracy Metric
        accuracy = self.accuracies[f"{step}_acc"]
        # Update accuracy
        accuracy(preds, y)
        # Log accuracy
        self.log(f"{step}_acc", accuracy, on_step=log_on_step, on_epoch=True, prog_bar=True)

    def shared_step(self, batch):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        return x_hat, y, loss

    def training_step(self, batch):
        if self.use_mixup_cutmix:
            x, y = batch
            batch = self.mixup_cutmix(x, y)

        x_hat, y, loss = self.shared_step(batch)
        self.log_metrics(loss, x_hat, y, step='train')
        return loss

    def validation_step(self, batch):
        x_hat, y, loss = self.shared_step(batch)
        self.log_metrics(loss, x_hat, y, step='val')
        return loss

    def test_step(self, batch):
        x_hat, y, loss = self.shared_step(batch)
        self.log_metrics(loss, x_hat, y, step='test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.total_steps,
                                                           pct_start=0.1,
                                                           max_lr=self.learning_rate)

        return [optimizer], {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
