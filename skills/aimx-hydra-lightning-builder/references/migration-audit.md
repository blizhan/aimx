# Migration Audit

Migration starts with a read-only audit. Do not edit the target repository during audit.

## Audit Checklist

Inspect:

- dependency manager and Python version;
- training entrypoints;
- Hydra config root and defaults composition;
- model, datamodule, plmodule, trainer, callback, logger config groups;
- `opt` config group for optimizer and scheduler policy;
- `experiment` config group with explicit Hydra override files;
- LightningModule and LightningDataModule classes;
- dataset item and batch shape, preferring named pytree leaves over positional tuples;
- domain adapter boundaries for radar, satellite, vision-frame, tabular, sequence, or other domain-specific logic;
- shared bases that hold only contracts and common mechanics;
- shallow inheritance trees with explicit child adapters;
- high-cohesion modules with low cross-module coupling;
- baseline defaults separated from experiment deltas;
- metric logging through `self.log`;
- AimLogger config and direct `experiment.track(...)` traces;
- hyperparameter logging;
- fast validation command;
- project-local Aim repo path.

## Migration Stages

1. Establish the AutoResearch contract.
2. Add or normalize Hydra config groups.
3. Move runtime orchestration into `src/train.py`.
4. Separate baseline defaults from experiment deltas.
5. Move optimizer and scheduler choices into `opt`.
6. Adapt datasets and collate outputs into named pytrees.
7. Adapt model/data/task code into Lightning boundaries.
8. Introduce domain adapters only where domain semantics differ.
9. Add Aim/Aimx evidence conventions.
10. Add fast validation tests.

Keep migration patches small and reversible.
