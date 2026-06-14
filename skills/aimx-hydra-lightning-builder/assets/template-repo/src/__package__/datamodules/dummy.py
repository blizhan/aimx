from __future__ import annotations

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class PytreeClassificationDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "input": {"x": self.x[index]},
            "target": {"label": self.y[index]},
        }


class RandomClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        num_samples: int = 64,
        num_features: int = 8,
        num_classes: int = 2,
        batch_size: int = 16,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(int(self.hparams.seed))
        x = torch.randn(int(self.hparams.num_samples), int(self.hparams.num_features), generator=generator)
        weights = torch.randn(int(self.hparams.num_features), int(self.hparams.num_classes), generator=generator)
        y = torch.argmax(x @ weights, dim=1)
        dataset = PytreeClassificationDataset(x, y)
        train_len = max(1, int(0.8 * len(dataset)))
        val_len = len(dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.hparams.batch_size),
            num_workers=int(self.hparams.num_workers),
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.hparams.batch_size),
            num_workers=int(self.hparams.num_workers),
            shuffle=False,
        )
