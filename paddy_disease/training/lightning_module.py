from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from paddy_disease.models.resnet34 import build_resnet34


@dataclass
class ModelConfig:
    """
    Temporary config
    """

    num_classes: int = 10
    dropout: float = 0.1
    pretrained: bool = True


@dataclass
class OptimConfig:
    """
    Temporary config
    """

    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0


class PaddyLightningModule(pl.LightningModule):
    """
    PaddyLightningModule for lightning
    """

    def __init__(self, model_cfg: ModelConfig, optim_cfg: OptimConfig) -> None:
        super().__init__()
        self.save_hyperparameters()  # оставим для логгирования

        self.model: nn.Module = build_resnet34(
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
            pretrained=model_cfg.pretrained,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=model_cfg.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=model_cfg.num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=model_cfg.num_classes, average="macro")

        self.optim_cfg = optim_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function

        :param self: Description
        :param x: Description
        :type x: torch.Tensor
        :return: Description
        :rtype: Tensor
        """
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step function

        :param self: Description
        :param batch: Description
        :param batch_idx: Description
        :type batch_idx: int
        :return: Description
        :rtype: Tensor
        """
        x, y, _ids = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Validation step function

        :param self: Description
        :param batch: Description
        :param batch_idx: Description
        :type batch_idx: int
        """
        x, y, _ids = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.optim_cfg.lr,
            momentum=self.optim_cfg.momentum,
            weight_decay=self.optim_cfg.weight_decay,
        )
        return optimizer
