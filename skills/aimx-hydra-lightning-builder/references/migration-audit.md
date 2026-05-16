# Migration Audit

Migration starts with a read-only audit. Do not edit the target repository during audit.

## Audit Checklist

Inspect:

- dependency manager and Python version;
- training entrypoints;
- Hydra config root and defaults composition;
- model, datamodule, plmodule, trainer, callback, logger config groups;
- LightningModule and LightningDataModule classes;
- metric logging through `self.log`;
- AimLogger config and direct `experiment.track(...)` traces;
- hyperparameter logging;
- fast validation command;
- project-local Aim repo path.

## Migration Stages

1. Establish the AutoResearch contract.
2. Add or normalize Hydra config groups.
3. Move runtime orchestration into `src/train.py`.
4. Adapt model/data/task code into Lightning boundaries.
5. Add Aim/Aimx evidence conventions.
6. Add fast validation tests.

Keep migration patches small and reversible.
