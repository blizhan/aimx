from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from {{ package_name }}.plmodules import BaseLitModule


class ClassificationModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _parse_batch(self, batch: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["input"]["x"], batch["target"]["label"]

    def _shared_step(self, batch, mode: str) -> dict[str, torch.Tensor]:
        x, y = self._parse_batch(batch)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        res = {
            "y_hat": preds,
            "y": y,
        }
        if mode in ["train", "val"]:
            loss = F.cross_entropy(logits, y)
            acc = (preds == y).float().mean()
            on_step = mode == "train"
            self.log(f"{mode}/loss", loss, on_step=on_step, on_epoch=True, prog_bar=True)
            self.log(f"{mode}/acc", acc, on_step=on_step, on_epoch=True, prog_bar=True)
            res["loss"] = loss
        return res

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        res = self._shared_step(batch, "train")
        return res["loss"]

    def validation_step(self, batch, batch_idx: int) -> dict[str, torch.Tensor]:
        return self._shared_step(batch, "val")
