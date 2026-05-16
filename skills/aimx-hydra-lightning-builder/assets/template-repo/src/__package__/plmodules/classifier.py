from __future__ import annotations

import hydra
import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig


class ClassificationModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.net = hydra.utils.instantiate(cfg.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch, mode: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        on_step = mode == "train"
        self.log(f"{mode}/loss", loss, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/acc", acc, on_step=on_step, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.opt.optimizer, params=self.parameters())
