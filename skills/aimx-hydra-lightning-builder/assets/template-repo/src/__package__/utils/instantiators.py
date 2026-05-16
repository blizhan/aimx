from __future__ import annotations

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    callbacks: list[Callback] = []
    if not callbacks_cfg:
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig.")
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    loggers: list[Logger] = []
    if not logger_cfg:
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig.")
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers
