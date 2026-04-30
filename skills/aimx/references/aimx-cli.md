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

Distribution expressions combine run fields with distribution fields:

```text
(run.experiment == 'cloud-segmentation') and distribution.name != ''
distribution.name == 'head/gradients/head.0.bias'
distribution.context.kind == 'weights'
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

## Analysis Patterns

Treat `aimx` output as structured evidence, not as text to paste wholesale.
Discover shape first, narrow the query, load JSON locally, then report compact
aggregates. This mirrors the large-project rule from experiment trackers: never
fetch or print every run, metric, or trace point unless discovery is the task.

### Rank runs from summaries

1. Collect params for the run scope.
2. Collect metric summaries for candidate objective metrics.
3. Choose the objective direction before ranking:
   - loss, error, perplexity: smaller is usually better; use `min.value`.
   - accuracy, F1, AUC, IoU, reward: larger is usually better; use `max.value`.
   - final-checkpoint objectives: use `last.value` and say why.
4. Prefer validation/test contexts over train contexts. If contexts are missing
   or mixed, lower confidence instead of forcing a ranking.

### Inspect curve health from traces

Use traces for the smallest set of runs that can answer the question:

```bash
aimx trace "(run.hash == 'baseline') and metric.name == 'loss'" --repo <repo> --json --tail 200
aimx trace "(run.hash == 'candidate') and metric.name == 'loss'" --repo <repo> --json --steps 1000:5000 --every 10
```

Check these before writing conclusions:

- final-window mean and standard deviation
- best value and best step
- sustained increases near the end of training
- large one-step spikes or NaN/Inf values
- plateaus where absolute step-to-step change is near zero
- train-vs-validation gap when both contexts exist

### Compact local analysis

Use Python to reduce trace JSON to a few numbers. Keep output small.

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


def finite_floats(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                result.append(number)
    return result


def curve_summary(values: list[float]) -> dict[str, float | int | bool]:
    if not values:
        return {"points": 0, "has_values": False}

    final_window = values[-max(1, len(values) // 5) :]
    diffs = [b - a for a, b in zip(values, values[1:])]
    mean_abs_change = fmean(abs(diff) for diff in diffs) if diffs else 0.0

    return {
        "points": len(values),
        "has_values": True,
        "first": values[0],
        "last": values[-1],
        "best_min": min(values),
        "best_min_index": values.index(min(values)),
        "final_mean": fmean(final_window),
        "final_std": pstdev(final_window) if len(final_window) > 1 else 0.0,
        "mean_abs_change": mean_abs_change,
        "increased_at_end": len(final_window) > 1 and final_window[-1] > final_window[0],
    }


payload: list[dict[str, Any]] = json.loads(Path("trace.json").read_text())
for series in payload:
    values = finite_floats(series.get("values", []))
    print(
        json.dumps(
            {
                "run": series.get("run", {}).get("hash"),
                "metric": series.get("metric"),
                "context": series.get("context", {}),
                "summary": curve_summary(values),
            },
            sort_keys=True,
        )
    )
```

### Compare runs side by side

For comparisons, report only selected params and selected metrics:

- run identity: hash, experiment, name
- controlled variables: model, seed, dataset, optimizer, learning rate, batch size
- objective summary: best value, best step, final value
- curve health: final-window stability, spikes, divergence, plateau
- missing evidence: missing params, missing validation metrics, short traces

When the next experiment is requested, tie it to evidence. Examples:
increase regularization only if validation worsens while training improves;
reduce learning rate only if traces show spikes or divergence; extend training
only if the final window is still improving without validation degradation.

### Find best run by objective

Input command:

```bash
aimx query metrics "(<run-scope>) and metric.name == '<metric>'" --repo <repo> --json > metrics.json
```

Output: top-ranked rows `(run_hash, run_name, context, objective_value, steps)`.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Direction = Literal["min", "max", "last"]


def metric_value(metric: dict[str, Any], direction: Direction) -> float | None:
    if direction == "last":
        value = metric.get("last", {}).get("value")
    elif direction == "max":
        value = metric.get("max", {}).get("value")
    else:
        value = metric.get("min", {}).get("value")
    return float(value) if isinstance(value, (int, float)) else None


payload = json.loads(Path("metrics.json").read_text())
direction: Direction = "min"
rows: list[dict[str, Any]] = []
for run in payload.get("runs", []):
    for metric in run.get("metrics", []):
        value = metric_value(metric, direction)
        if value is None:
            continue
        rows.append(
            {
                "run_hash": run.get("hash"),
                "run_name": run.get("name"),
                "metric": metric.get("name"),
                "context": metric.get("context", {}),
                "objective": value,
                "steps": metric.get("steps", 0),
            }
        )

top = sorted(rows, key=lambda item: item["objective"], reverse=(direction == "max"))[:5]
for item in top:
    print(item)
```

### Spike / divergence / plateau / NaN detection

Input command:

```bash
aimx trace "(<run-scope>) and metric.name == '<metric>'" --repo <repo> --json --tail 500 > trace.json
```

Output: one compact anomaly summary per series.

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def finite_array(values: list[Any]) -> np.ndarray:
    cleaned = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    return np.asarray(cleaned, dtype=float)


payload: list[dict[str, Any]] = json.loads(Path("trace.json").read_text())
window = 20
eps = 1e-5
for series in payload:
    values = finite_array(series.get("values", []))
    if values.size < window + 1:
        continue
    diffs = np.diff(values)
    roll_mean = np.convolve(values, np.ones(window) / window, mode="valid")
    aligned = values[window - 1 :]
    centered = aligned - roll_mean
    roll_std = np.sqrt(np.convolve(centered**2, np.ones(window) / window, mode="same"))
    spikes = int(np.sum(np.abs(centered) > (3.0 * np.maximum(roll_std, 1e-12))))
    trend = np.convolve(diffs, np.ones(window) / window, mode="valid")
    divergence = bool(np.sum(trend > 0.0) > window)
    plateau = bool(np.sum(np.abs(trend) < eps) > window)
    non_finite = len(series.get("values", [])) - int(values.size)
    print(
        {
            "run": series.get("run", {}).get("hash"),
            "metric": series.get("metric"),
            "context": series.get("context", {}),
            "spikes": spikes,
            "diverging": divergence,
            "plateau": plateau,
            "non_finite_points": non_finite,
        }
    )
```

### Overfitting detection (train vs val)

Input commands:

```bash
aimx trace "(<run-scope>) and metric.name == '<metric>' and metric.context.subset == 'train'" --repo <repo> --json --tail 300 > train.json
aimx trace "(<run-scope>) and metric.name == '<metric>' and metric.context.subset == 'val'" --repo <repo> --json --tail 300 > val.json
```

Output: per-run train/val tail means and an overfitting flag.

```python
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any


def tail_mean(values: list[float], ratio: float = 0.2) -> float:
    if not values:
        return float("nan")
    start = max(0, int(len(values) * (1.0 - ratio)))
    return fmean(values[start:])


def index_by_run(payload: list[dict[str, Any]]) -> dict[str, list[float]]:
    index: dict[str, list[float]] = {}
    for series in payload:
        run_hash = str(series.get("run", {}).get("hash", ""))
        values = [
            float(value)
            for value in series.get("values", [])
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if run_hash and values:
            index[run_hash] = values
    return index


train_payload: list[dict[str, Any]] = json.loads(Path("train.json").read_text())
val_payload: list[dict[str, Any]] = json.loads(Path("val.json").read_text())
train_by_run = index_by_run(train_payload)
val_by_run = index_by_run(val_payload)
threshold = 0.05
for run_hash in sorted(set(train_by_run) & set(val_by_run)):
    train_values = train_by_run[run_hash]
    val_values = val_by_run[run_hash]
    train_tail = tail_mean(train_values)
    val_tail = tail_mean(val_values)
    gap = val_tail - train_tail
    train_trend_down = train_values[-1] <= train_values[0]
    overfit = gap > threshold and train_trend_down
    print(
        {
            "run_hash": run_hash,
            "train_tail_mean": train_tail,
            "val_tail_mean": val_tail,
            "gap": gap,
            "overfitting": overfit,
        }
    )
```

### Sweep ranking (params x metric summary)

Input commands:

```bash
aimx query params "<run-scope>" --repo <repo> --json > params.json
aimx query metrics "(<run-scope>) and metric.name == '<metric>'" --repo <repo> --json > metrics.json
```

Output: top-5 rows with objective value and selected control variables.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def param_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(run.get("hash")): dict(run.get("params", {}))
        for run in payload.get("runs", [])
        if run.get("hash")
    }


params_payload = json.loads(Path("params.json").read_text())
metrics_payload = json.loads(Path("metrics.json").read_text())
params_by_run = param_index(params_payload)
rows: list[dict[str, Any]] = []
for run in metrics_payload.get("runs", []):
    run_hash = str(run.get("hash", ""))
    if not run_hash:
        continue
    for metric in run.get("metrics", []):
        objective = metric.get("min", {}).get("value")
        if not isinstance(objective, (int, float)):
            continue
        run_params = params_by_run.get(run_hash, {})
        rows.append(
            {
                "run_hash": run_hash,
                "run_name": run.get("name"),
                "objective": float(objective),
                "model": run_params.get("model"),
                "lr": run_params.get("hparam.lr"),
                "batch_size": run_params.get("hparam.batch_size"),
            }
        )

for item in sorted(rows, key=lambda row: row["objective"])[:5]:
    print(item)
```

## Distribution Traces

Use distribution traces when histogram shape matters, such as weights,
activations, gradients, or other Aim distribution sequences.

For automation, keep using explicit structured modes:

```bash
aimx trace distribution "<distribution-expr>" --repo <repo> --json
aimx trace distribution "<distribution-expr>" --repo <repo> --csv --tail 5
aimx trace distribution "<distribution-expr>" --repo <repo> --table --head 2
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
    "distribution": "head/gradients/head.0.bias",
    "context": {"kind": "gradients"},
    "count": 2,
    "points": [
      {
        "step": 300,
        "epoch": 0.0,
        "bin_edges": [-1.0, 0.0, 1.0],
        "weights": [0.0, 2.0]
      }
    ]
  }
]
```

For human terminal inspection, omit the output-mode flag:

```bash
aimx trace distribution "distribution.name != ''" --repo <repo>
aimx trace distribution "distribution.name != ''" --repo <repo> --step 12300
```

Default distribution output is deterministic, non-interactive text. It lists
matched distribution names, selects the first non-empty series, renders a
current-step histogram, and renders a step-by-bin heatmap. `--step N` selects
the visual histogram step; if the step is absent, `aimx` labels the nearest
tracked step used. `--step` does not filter `--table`, `--csv`, or `--json`
outputs.

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
- `distribution_evidence`: selected histogram payloads, visual inspection notes,
  or structured distribution rows for weight/gradient analysis.
- `image_evidence`: image row counts and representative contexts.
- `ranking`: best run per objective, objective direction, and tie-breakers.
- `regressions`: runs worse than baseline, incomplete runs, missing metrics, or
  suspicious curves.
- `next_experiments`: concrete parameter changes grounded in the evidence.

Keep conclusions tied to the collected data. When metric direction is unknown,
state the assumption before ranking.
