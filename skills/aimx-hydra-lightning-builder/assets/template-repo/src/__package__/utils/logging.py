from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def log_hyperparameters(objects: dict[str, Any]) -> None:
    cfg = objects["cfg"]
    model = objects["model"]
    trainer = objects["trainer"]
    if not trainer.logger:
        return

    hparams = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "datamodule": OmegaConf.to_container(cfg.datamodule, resolve=True),
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
        "opt": OmegaConf.to_container(cfg.opt, resolve=True),
        "plmodule": OmegaConf.to_container(cfg.plmodule, resolve=True),
        "autoresearch": OmegaConf.to_container(cfg.autoresearch, resolve=True),
        "task_name": cfg.get("task_name"),
        "tags": list(cfg.get("tags") or []),
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
