from __future__ import annotations

from typing import Any

import hydra
import rootutils
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from {{ package_name }}.utils.instantiators import instantiate_callbacks, instantiate_loggers
from {{ package_name }}.utils.logging import log_hyperparameters

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def instantiate_runtime(cfg: DictConfig) -> dict[str, Any]:
    datamodule: LightningDataModule = instantiate(cfg.datamodule)
    model: LightningModule = instantiate(cfg.plmodule)(cfg=cfg)
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    logger: list[Logger] | bool = False
    if cfg.trainer.get("logger") is not False:
        logger = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    return {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> dict[str, float]:
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, workers=True)

    objects = instantiate_runtime(cfg)
    trainer: Trainer = objects["trainer"]
    model: LightningModule = objects["model"]
    datamodule: LightningDataModule = objects["datamodule"]

    if objects["logger"]:
        log_hyperparameters(objects)

    trainer.fit(model=model, datamodule=datamodule)
    return {
        key: float(value)
        for key, value in trainer.callback_metrics.items()
        if hasattr(value, "numel") and value.numel() == 1
    }


if __name__ == "__main__":
    main()
