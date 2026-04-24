# aimx CLI Reference for log_experiment

Use this reference when building an `autoresearch` `log_experiment` step from a
local Aim repository. All commands below are read-only inspection commands.

## Installation Requirement

`aimx` must be installed in the environment that runs the collection commands.
Verify one of these before collecting:

```bash
aimx --help
python -m aimx --help
```

If it is missing, inspect the current project first and use the dependency
manager it already uses. Check project instructions, `pyproject.toml`, lockfiles,
README setup docs, and CI config before choosing a command.

Examples:

```bash
# uv-managed project
uv add aimx
uv sync

# Poetry-managed project
poetry add aimx

# Pipenv-managed project
pipenv install aimx

# requirements.txt project
python -m pip install aimx
python -m pip freeze > requirements.txt
```

Use the project's normal lockfile/update flow. For this `aimx` repository,
prefer `uv sync`, `uv run ...`, and `uv add aimx` because repo policy requires
`uv`. For a standalone CLI with no project dependency to modify, use the user's
preferred tool manager; use `uv tool install aimx` only when `uv` is already the
chosen or accepted workflow.

## Repository Paths

`--repo` accepts either the Aim repository root or the `.aim` metadata
directory. Prefer explicit repo paths so automation does not accidentally read a
user-level or unrelated Aim repository.

Examples:

```bash
aimx query params "run.hash != ''" --repo data --json
aimx query params "run.hash != ''" --repo data/.aim --json
```

## AimQL Scoping

Use AimQL in the expression argument. Common run scopes:

```text
run.hash != ''
run.hash == 'eca37394'
run.experiment == 'cloud-segmentation'
run.name == 'resnet-ft-0420'
```

Metric expressions combine run fields with metric fields:

```text
(run.experiment == 'cloud-segmentation') and metric.name == 'loss'
(run.hash == 'eca37394') and metric.name != ''
```

Short run hashes are expanded by `aimx` where supported.

## Params

Use params first to understand the design of each run.

```bash
aimx query params "<run-scope>" --repo <repo> --json
aimx query params "<run-scope>" --repo <repo> --json --param hparam.lr --param model
```

JSON shape:

```json
{
  "target": "params",
  "repo": "data",
  "expression": "run.hash != ''",
  "runs_count": 2,
  "param_keys": ["hparam.lr", "model"],
  "runs": [
    {
      "hash": "full-run-hash",
      "experiment": "experiment-name",
      "name": "run-name",
      "params": {"hparam.lr": 0.0001, "model": "ResNet"},
      "missing_params": []
    }
  ]
}
```

Use `missing_params` as a confidence signal when comparing runs.

## Metric Summaries

Use metric summaries to rank runs cheaply.

```bash
aimx query metrics "<metric-expr>" --repo <repo> --json
aimx query metrics "<metric-expr>" --repo <repo> --json --steps 100:500
aimx query metrics "<metric-expr>" --repo <repo> --json --epochs 1:10
```

JSON shape:

```json
{
  "target": "metrics",
  "repo": "data",
  "expression": "metric.name == 'loss'",
  "runs_count": 1,
  "metrics_count": 2,
  "runs": [
    {
      "hash": "full-run-hash",
      "experiment": "experiment-name",
      "name": "run-name",
      "metrics": [
        {
          "name": "loss",
          "context": {"subset": "val"},
          "steps": 110,
          "last": {"value": 0.43, "step": 110},
          "min": {"value": 0.32, "step": 60},
          "max": {"value": 0.48, "step": 107}
        }
      ]
    }
  ]
}
```

Use `context` to distinguish train, val, test, dataset split, seed, or other
metric dimensions.

## Traces

Use traces when a summary hides important behavior such as late overfitting,
instability, divergence, or plateauing.

```bash
aimx trace "<metric-expr>" --repo <repo> --json
aimx trace "<metric-expr>" --repo <repo> --json --tail 50
aimx trace "<metric-expr>" --repo <repo> --json --steps 100:500 --every 5
```

JSON shape:

```json
[
  {
    "run": {
      "hash": "full-run-hash",
      "experiment": "experiment-name",
      "name": "run-name"
    },
    "metric": "loss",
    "context": {"subset": "val"},
    "count": 50,
    "steps": [1, 2, 3],
    "epochs": [1.0, 2.0, 3.0],
    "values": [0.9, 0.7, 0.5]
  }
]
```

If no metrics match, current `aimx trace --json` may print a text message
instead of JSON. Treat that as no trace evidence rather than a parsing failure.

## Images

Use images for qualitative checks such as sample predictions, masks, generated
outputs, confusion examples, or visual regressions.

```bash
aimx query images "images" --repo <repo> --json
aimx query images "images" --repo <repo> --json --head 20
aimx query images "images" --repo <repo> --json --epochs 10:50
```

JSON shape:

```json
{
  "target": "images",
  "repo": "data",
  "expression": "images",
  "count": 1,
  "rows": [
    {
      "run_hash": "full-run-hash",
      "experiment": "experiment-name",
      "name": "example",
      "context": {"epoch": 10, "subset": "val"}
    }
  ]
}
```

Use image metadata in automated logs; render images manually only when the user
asks for visual inspection.

## log_experiment Evidence Fields

Recommended fields for autoresearch output:

- `run_scope`: AimQL expression and repo path used for evidence.
- `params`: selected hyperparameters and model identifiers per run.
- `metric_summary`: objective metric summaries per run and context.
- `trace_evidence`: sampled value arrays for decisive metrics.
- `image_evidence`: image row counts and representative contexts.
- `ranking`: best run per objective, objective direction, and tie-breakers.
- `regressions`: runs worse than baseline, incomplete runs, missing metrics, or
  suspicious curves.
- `next_experiments`: concrete parameter changes grounded in the evidence.

Keep conclusions tied to the collected data. When metric direction is unknown,
state the assumption before ranking.
