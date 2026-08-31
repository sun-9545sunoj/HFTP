"""
PyTorch Lightning Module for DeepLOB Training, Validation, and Metric Logging.
"""

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchmetrics

from src.models.deeplob import DeepLOB, SpatialTemporalTransformerLOB
from src.models.loss import ClassWeightedCrossEntropy, FocalLoss


class DeepLOBModule(pl.LightningModule):
    """
    PyTorch Lightning Module wrapper for DeepLOB models.
    Supports multi-class directional forecasting, metric tracking, and loss balancing.
    """

    def __init__(
        self,
        model_type: str = "deeplob",
        in_features: int = 40,
        seq_len: int = 100,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        loss_type: str = "focal",
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        scheduler_type: str = "cosine",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Instantiate backbone architecture
        if model_type.lower() == "deeplob":
            self.model = DeepLOB(
                in_features=in_features,
                seq_len=seq_len,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif model_type.lower() == "transformer":
            self.model = SpatialTemporalTransformerLOB(
                in_features=in_features,
                seq_len=seq_len,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'deeplob' or 'transformer'.")

        # Loss function setup
        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif loss_type == "weighted_ce":
            self.criterion = ClassWeightedCrossEntropy(weights=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Performance Metrics Setup
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(preds, y)
        self.val_conf_mat.update(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_acc = self.val_acc.compute()
        f1_macro = self.val_f1_macro.compute()

        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1_macro", f1_macro, prog_bar=True)

        self.val_acc.reset()
        self.val_f1_macro.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_conf_mat.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif self.hparams.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        else:
            return {"optimizer": optimizer}
