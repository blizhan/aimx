from __future__ import annotations

import lightning as L


class AutoResearchMarker(L.Callback):
    """Log a small completion marker before Lightning finalizes loggers."""

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.logger:
            return
        for logger in trainer.loggers:
            logger.log_metrics({"autoresearch/complete": 1.0}, step=int(trainer.global_step))
