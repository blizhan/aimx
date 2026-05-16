---
name: aimx-hydra-lightning-builder
description: Use when creating, auditing, or planning migration for Hydra + PyTorch Lightning ML repositories that should log to Aim, expose Aimx-readable experiment evidence, or use configurable datamodules/models/plmodules/callbacks/loggers.
---

# Aimx Hydra Lightning Builder

## Overview

Build or migrate Hydra + Lightning repositories that satisfy the Aimx AutoResearch contract.

The pattern is distilled from production Hydra + Lightning research repositories: Hydra composes the experiment, Lightning owns the training lifecycle, and Aim/Aimx preserve evidence for agentic iteration.

## Workflows

### Scaffold a new repository

Run:

```bash
python skills/aimx-hydra-lightning-builder/scripts/scaffold_repo.py \
  --stack hydra-lightning \
  --name <repo-name> \
  --package <package_name> \
  --preset classification \
  --output <target-dir>
```

Then validate inside the generated repository:

```bash
uv sync
uv run pytest
uv run python src/train.py trainer.fast_dev_run=true trainer.logger=false
```

### Audit an existing repository

Run read-only:

```bash
python skills/aimx-hydra-lightning-builder/scripts/audit_repo.py \
  --stack hydra-lightning \
  --repo <repo-path> \
  --format json > audit.json
python skills/aimx-hydra-lightning-builder/scripts/plan_migration.py \
  --audit audit.json \
  --target-stack hydra-lightning
```

Never edit, format, sync dependencies, generate files, or run mutation/codegen commands inside audit targets unless the user separately approves migration execution.

## Architecture Pattern

Read `references/architecture.md` before scaffold or migration work.

- `configs/<task>.yaml` composes `data`, `datamodule`, `model`, `plmodule`, `trainer`, `callbacks`, `logger`, `paths`, `accelerate`, and optional `experiment`.
- `src/train.py` seeds, instantiates configured objects, logs hyperparameters, and calls `trainer.fit/validate/test`.
- `BaseLitModule` owns `cfg`, `cfg.model` instantiation, optimizer/scheduler, compile/SDPA options, and shared trace helpers.
- Task modules own batch parsing, loss, metrics, and prediction/evaluation outputs.
- DataModules own splits, dataloaders, sampler/collate policy, and data preparation boundaries.
- Aim trace uses Lightning loggers for scalars and explicit `experiment.track(...)` for images/distributions.

## References

- `references/architecture.md`: core relationships and file layout.
- `references/aim-trace.md`: evidence naming, context, and Aimx query conventions.
- `references/migration-audit.md`: read-only audit checklist and migration staging.

## Scripts And Assets

- `scripts/scaffold_repo.py`: copies `assets/template-repo`.
- `scripts/audit_repo.py`: scans a repository without writing to it.
- `scripts/plan_migration.py`: generates a staged migration plan from audit JSON.
- `assets/template-repo`: minimal runnable Hydra + Lightning + Aim template.
