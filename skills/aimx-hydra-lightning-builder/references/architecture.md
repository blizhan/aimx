# Hydra Lightning Aim Architecture

The core relationship is:

`Hydra config -> instantiated Lightning runtime -> Aim evidence -> aimx analysis`

## Config Layer

Use a primary config such as `configs/train.yaml` with defaults for:

- `datamodule`
- `model`
- `plmodule`
- `trainer`
- `callbacks`
- `logger`
- `paths`
- `accelerate`
- optional `experiment`

Experiment configs should override choices and hyperparameters, not duplicate the whole tree.

## Runtime Layer

`src/train.py` should:

- call `rootutils.setup_root` or otherwise make local imports stable;
- seed through Lightning;
- instantiate datamodule, plmodule, callbacks, loggers, and trainer from config;
- log hyperparameters before training when loggers exist;
- run `trainer.fit`, `trainer.validate`, or `trainer.test` based on config;
- return compact metrics when used programmatically.

## Module Layer

`BaseLitModule` should own common behavior:

- store the full `cfg`;
- instantiate `cfg.model`;
- configure optimizer and scheduler from `cfg.opt`;
- apply compile, precision, or SDPA settings from `cfg.accelerate`;
- provide helper methods for Aim experiments when explicit traces are needed.

Task subclasses should own only domain logic: batch parsing, forward call, loss, metrics, prediction outputs, and optional qualitative trace artifacts.

## Data Layer

`LightningDataModule` classes own data preparation and dataloaders. Keep user data paths in config. Use dummy/random data in templates so fast validation does not depend on private datasets.

## Evidence Layer

Use Lightning `self.log` for scalar metrics. Use Aim `experiment.track(...)` for images and distributions. Keep evidence names stable and context-rich so `aimx` can query them later.
