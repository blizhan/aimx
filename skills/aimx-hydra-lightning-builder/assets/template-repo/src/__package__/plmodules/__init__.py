from __future__ import annotations

import hydra
import lightning as L
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from omegaconf import DictConfig


class BaseLitModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.net = hydra.utils.instantiate(cfg.model)
        self._net_compiled = False

        sdpa_map = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
        }

        self.sdpa_backends = [sdpa_map[backend] for backend in self.cfg.accelerate.get("sdpa", ["math"])]

    def forward(self, *args, **kwargs):
        return self._model_forward(*args, **kwargs)

    def _model_forward(self, *args, **kwargs):
        with sdpa_kernel(self.sdpa_backends):
            return self.net(*args, **kwargs)

    def setup(self, stage: str) -> None:
        if self.cfg.accelerate.compile and stage == "fit" and hasattr(torch, "compile") and not self._net_compiled:
            self.net = torch.compile(self.net)
            self._net_compiled = True

    def get_lr_scheduler(self, optimizer):
        scheduler = hydra.utils.instantiate(self.cfg.opt.scheduler)(optimizer=optimizer)
        kwargs = {
            key: value for key, value in self.cfg.opt.items() if key not in ["optimizer", "scheduler"]
        }
        return {
            "scheduler": scheduler,
            **kwargs,
        }

    def get_optimizer(self):
        if self.cfg.opt.optimizer._target_ == "torch.optim.AdamW":
            optimizer = hydra.utils.instantiate(
                self.cfg.opt.optimizer,
                params=filter(lambda p: p.requires_grad, self.net.parameters()),
            )
        elif self.cfg.opt.optimizer._target_ == "colossalai.nn.optimizer.HybridAdam":
            optimizer = hydra.utils.instantiate(
                self.cfg.opt.optimizer,
                model_params=filter(lambda p: p.requires_grad, self.net.parameters()),
            )
        else:
            optimizer = hydra.utils.instantiate(
                self.cfg.opt.optimizer,
                params=filter(lambda p: p.requires_grad, self.net.parameters()),
            )
        return optimizer

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        if not self.cfg.opt.get("scheduler"):
            return optimizer

        lr_scheduler = self.get_lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def _aim_experiments(self):
        for logger in self.loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "track"):
                yield experiment

    def _instantiate_metric(self, name: str, defaults: dict[str, dict[str, object]]):
        metrics_cfg = self.cfg.get("metrics", {})
        metric_cfg = metrics_cfg[name] if name in metrics_cfg else defaults[name]
        return hydra.utils.instantiate(metric_cfg)


from {{ package_name }}.plmodules.classifier import ClassificationModule

__all__ = ["BaseLitModule", "ClassificationModule"]
