# {{ project_name }}

Hydra + Lightning + Aim template for Aimx AutoResearch.

## Quick Start

```bash
uv sync
uv run python src/train.py trainer.fast_dev_run=true trainer.logger=false
uv run python src/train.py experiment=exp trainer.fast_dev_run=true trainer.logger=false
uv run pytest
```

Use `experiment=<name>` to apply a file from `configs/experiment/<name>.yaml`.
Experiment yaml files should override config groups and values such as
`model`, `datamodule`, `trainer`, `opt`, `accelerate`, and `logger`.

Enable Aim logging by leaving `trainer.logger=true` and using `logger=aim`.

```bash
uv run python src/train.py experiment=exp
aimx query params "run.hash != ''" --repo .
aimx query metrics "metric.name != ''" --repo .
aimx query metrics "metric.name == 'acc'" --repo . --json
```

System parameters are not logged by default to avoid storing environment
variables in experiment evidence. Opt in only for safe environments:

```bash
uv run python src/train.py logger.aim.log_system_params=true
```
