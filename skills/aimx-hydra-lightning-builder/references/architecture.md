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
- `opt`
- `experiment`


Baseline defaults live in the regular config groups, such as `datamodule/default.yaml`, `model/default.yaml`, `trainer/default.yaml`, `logger/default.yaml`, and `opt/default.yaml`. Keep the primary config as the reproducible baseline and select experiment deltas explicitly:

```bash
uv run python src/train.py experiment=exp
```

Experiment files live under `configs/experiment/<name>.yaml`, use `# @package _global_`, and override config-group choices with Hydra defaults such as `override /model: mlp` or `override /opt: default`. They should override choices and hyperparameters, not duplicate the whole tree.

Keep optimizer and scheduler policy in `opt`. Override learning rates, weight decay, scheduler settings, and optimizer choices through `opt` in the experiment yaml instead of placing optimizer state under `model`, `plmodule`, or `trainer`.

Config defines how an experiment runs:

- which datamodule, model, plmodule, callbacks, logger, and trainer are instantiated;
- paths, batch sizes, worker counts, optimizer settings, precision, and accelerator choices;
- experiment names, tags, objective metadata, and evidence switches.

Code defines what the operation means:

- how a domain batch is parsed;
- how targets, losses, metrics, and predictions are computed;
- what qualitative artifacts or distribution traces mean for the domain.

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

Task subclasses should inherit from `BaseLitModule` and own only domain logic: batch parsing, forward call, loss, metrics, prediction outputs, and optional qualitative trace artifacts. Do not duplicate optimizer setup, model instantiation, logger access, compile handling, or trainer construction in each task module.

Keep inheritance trees shallow and explicit:

- use one shared base for contracts and mechanics;
- use one child adapter for the domain task;
- avoid framework-like hierarchies where behavior is spread across several parent classes.

Prefer composition when behavior varies. Swap models, losses, metrics, callbacks, data sources, and loggers through Hydra config instead of adding inheritance layers.

## Domain Adapter Pattern

Use domain adapters when multiple datasets or modalities should share the same AutoResearch contract but differ in domain semantics. Shared bases define stable contracts and common mechanics; child adapters translate domain data into that contract.

Examples:

- radar adapters parse radar tensors, lead times, geospatial masks, and forecast targets;
- satellite adapters parse channels, tiles, projections, and cloud or retrieval targets;
- vision-frame adapters parse images, labels, boxes, masks, or frame windows;
- tabular adapters parse feature tables, categorical encodings, sample weights, and targets;
- sequence adapters parse token, sensor, event, or time-series windows.

A domain adapter should own:

- batch parsing and validation;
- target construction and masking;
- domain metrics and loss inputs;
- prediction formatting;
- optional qualitative artifacts and Aim traces.

A domain adapter should not own:

- trainer construction;
- logger construction;
- config composition;
- filesystem layout;
- sweep or experiment orchestration.

## Data Layer

`LightningDataModule` classes own data preparation and dataloaders. Keep user data paths in config. Use dummy/random data in templates so fast validation does not depend on private datasets.

Keep data modules cohesive: they prepare datasets, splits, dataloaders, sampling, and collation. Keep model math and task losses out of data modules.

Prefer dataset samples as pytrees, such as nested dictionaries or dataclasses containing tensors, arrays, masks, metadata, or target leaves. Pytrees keep model and task design flexible because new leaves can be added without changing every positional tuple unpack. Let the DataLoader collate the pytree when possible, and let the task adapter parse named leaves into model inputs, targets, masks, and metadata.

Use stable leaf names that express domain meaning:

- `inputs` or `input` for tensors passed to the model;
- `targets` or `target` for supervised labels or forecast targets;
- `metadata` for ids, timestamps, coordinates, horizon, or source provenance;
- `mask` or domain-specific masks for valid regions, sample weights, or loss masks.

Avoid positional tuple batches in templates and migrations unless an upstream dataset API forces them. If the upstream API returns tuples, adapt them into a pytree at the dataset or collate boundary.

## Evidence Layer

Use Lightning `self.log` for scalar metrics. Use Aim `experiment.track(...)` for images and distributions. Keep evidence names stable and context-rich so `aimx` can query them later.
