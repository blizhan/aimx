---
name: aimx
description: Use when autoresearch, log_experiment, experiment analysis, or automatic iteration workflows need to inspect local Aim repositories with aimx, collect run params, metric summaries, traces, image evidence, compare training runs, or summarize model results without mutating Aim data.
---

# Aimx

## Overview

Use `aimx` as a read-only evidence collector for `autoresearch` `log_experiment`
steps. Prefer JSON output so downstream agents can compare runs, explain model
effects, and propose the next experiment from concrete Aim data.

## Fast Recipes

Use these first for common analysis tasks. Keep `--repo` explicit and prefer
`--json` for machine-readable output.

### Discover run scope and available params

```bash
aimx query params "run.hash != ''" --repo <repo> --json
```

### Inspect one run quickly

```bash
aimx query params "run.hash == '<run-hash>'" --repo <repo> --json
aimx query metrics "(run.hash == '<run-hash>') and metric.name != ''" --repo <repo> --json
```

### Rank runs by an objective metric

```bash
aimx query metrics "(<run-scope>) and metric.name == '<metric>'" --repo <repo> --json > metrics.json
python - <<'PY'
from __future__ import annotations
import json
from pathlib import Path

payload = json.loads(Path("metrics.json").read_text())
rows = []
for run in payload.get("runs", []):
    for metric in run.get("metrics", []):
        value = metric.get("min", {}).get("value")
        if value is not None:
            rows.append((value, run.get("hash"), run.get("name"), metric.get("context", {})))
for value, run_hash, run_name, context in sorted(rows)[:5]:
    print(f"{value:.6f}\t{run_hash}\t{run_name}\t{context}")
PY
```

### Compare two runs side by side

```bash
aimx query params "run.hash == '<baseline-hash>' or run.hash == '<candidate-hash>'" --repo <repo> --json
aimx query metrics "((run.hash == '<baseline-hash>') or (run.hash == '<candidate-hash>')) and metric.name == '<metric>'" --repo <repo> --json
```

### Check curve health with bounded trace evidence

```bash
aimx trace "(<run-scope>) and metric.name == '<metric>'" --repo <repo> --json --tail 200 > trace.json
```

Then reduce `trace.json` with the `curve_summary` snippet from
`references/aimx-cli.md` instead of pasting raw series.

### Sanity-check distribution traces

```bash
aimx trace distribution "<distribution-expr>" --repo <repo> --json --tail 5
aimx trace distribution "distribution.name != ''" --repo <repo> --step 12300
```

### Capture one snapshot bundle for logs

```bash
uv run python skills/aimx/scripts/collect_experiment_snapshot.py \
  --repo data \
  --base-expr "run.hash != ''" \
  --metric loss \
  --trace-metric loss \
  --pretty
```

## When to use what

| Need | Use |
| --- | --- |
| Discover runs and key hyperparameters | `aimx query params "<run-scope>" --repo <repo> --json` |
| Rank runs cheaply by objective | `aimx query metrics "<metric-expr>" --repo <repo> --json` and compare `min.value` or `max.value` |
| Inspect curve shape and late stability | `aimx trace "<metric-expr>" --repo <repo> --json --tail N` |
| Focus on a step or epoch window | `--steps a:b` or `--epochs a:b` on query/trace commands |
| Analyze weight or gradient histograms | `aimx trace distribution "<distribution-expr>" --repo <repo> --json` |
| Collect qualitative image evidence | `aimx query images "<image-expr>" --repo <repo> --json --head N` |
| Check native Aim passthrough readiness | `aimx doctor` |

## Requirements

- Require `aimx` in the Python environment that runs `log_experiment`.
  Verify with `aimx --help` or `python -m aimx --help` before collecting data.
- When `aimx` is missing, first identify and follow the user's current
  dependency-management workflow from project files and instructions
  (`pyproject.toml`, lockfiles, README, AGENTS/CLAUDE/GEMINI notes, CI config).
  Use that manager to add or install `aimx`. In this repository, prefer
  `uv sync`, `uv run ...`, and `uv add aimx` because repo policy requires `uv`.
- If there is no project environment to modify and the user only needs a
  standalone CLI, install `aimx` as a tool using the user's preferred tool
  manager. Use `uv tool install aimx` only when `uv` is already the chosen or
  accepted tool workflow.
- Require read access to a local Aim repository path. Pass `--repo` explicitly
  and keep collection commands read-only.
- If native Aim availability matters for passthrough checks, run `aimx doctor`;
  do not initialize, repair, migrate, or rewrite Aim repositories from this
  skill.

## Workflow

For common tasks, start from **Fast Recipes** and only switch to this full
workflow when the scope is unclear or the question is complex.

1. Locate the Aim repository. Pass `--repo <repo-root-or-.aim>` explicitly; in
   this repository, use `--repo data` or `--repo data/.aim` for local checks.
2. Define the run scope as an AimQL expression. Start broad with
   `run.hash != ''`, then narrow by `run.hash`, `run.experiment`, or `run.name`.
3. Collect run parameters:

   ```bash
   aimx query params "<run-scope>" --repo <repo> --json
   aimx query params "<run-scope>" --repo <repo> --json --param hparam.lr --param model
   ```

4. Collect metric summaries for candidate objective metrics:

   ```bash
   aimx query metrics "(<run-scope>) and metric.name == 'loss'" --repo <repo> --json
   aimx query metrics "(<run-scope>) and metric.name != ''" --repo <repo> --json
   ```

5. Collect traces only for decisive metrics or suspected anomalies:

   ```bash
   aimx trace "(<run-scope>) and metric.name == 'loss'" --repo <repo> --json --tail 50
   ```

6. Inspect distribution traces when weight, activation, or gradient histograms
   matter. Prefer JSON/CSV for automation; use the default visual output for
   human terminal inspection.

   ```bash
   aimx trace distribution "<distribution-expr>" --repo <repo> --json --tail 5
   aimx trace distribution "distribution.name != ''" --repo <repo> --step 12300
   ```

7. Collect image metadata when qualitative outputs matter:

   ```bash
   aimx query images "images" --repo <repo> --json --head 20
   ```

8. Emit a compact `log_experiment` record containing:

   ```json
   {
     "repo": "<repo>",
     "run_scope": "<AimQL>",
     "params": {},
     "metric_summary": {},
     "trace_evidence": {},
     "distribution_evidence": {},
     "image_evidence": {},
     "interpretation": {
       "best_runs": [],
       "regressions": [],
       "confidence": "low|medium|high",
       "next_experiments": []
     }
   }
   ```

## Analysis Workflow

Use the same discipline as large experiment trackers: inspect structure first,
query only the fields needed for the question, then reduce evidence into compact
statistics before writing conclusions.

1. Start with params and metric summaries to discover candidate runs, objective
   metrics, contexts, and missing fields. Avoid dumping full JSON payloads into
   conversation context.
2. Choose the objective direction explicitly. Rank cheaply from summaries first:
   `min.value` for loss/error, `max.value` for accuracy/F1/AUC/IoU, and
   `last.value` only when the final checkpoint is the real objective.
3. Pull bounded traces only for the baseline, top candidates, and suspicious
   runs. Prefer `--tail`, `--steps`, `--epochs`, and `--every` before collecting
   full curves.
4. Compute local stats before interpreting: best step, final-window mean/std,
   train-vs-val gap, NaN/Inf counts, sustained increases, spikes, and plateaus.
5. Compare runs side by side with selected params plus selected metrics. Do not
   iterate every param or every metric unless discovery is the goal.
6. Escalate evidence by modality: use distribution traces for weights,
   activations, or gradients; use image metadata for qualitative regressions.
7. Keep the final analysis small: state objective, run scope, top runs, curve
   health, anomalies, confidence, and the next experiment suggested by evidence.

## Critical Rules

- Discover scope first with `aimx query params "<run-scope>" --repo <repo> --json`.
  Do not assume metric or param names.
- Treat `aimx` output as data: parse JSON and report aggregates, not raw payloads.
- Slice traces aggressively with `--tail`, `--head`, `--steps`, `--epochs`, or
  `--every` before computing local statistics.
- Always pass `--repo` explicitly to avoid reading an unintended repository.
- For automation, use `aimx trace distribution` with `--json`, `--csv`, or
  `--table`. Unflagged mode is terminal visualization for human inspection.
- Always finish with a compact conclusion: objective, top runs, curve health,
  anomalies, confidence, and next experiment.

## Interpretation Rules

- Prefer validation, test, or held-out contexts over training contexts when
  ranking runs.
- Treat `aimx query metrics` as summary data: `last`, `min`, `max`, and step
  counts. Use `aimx trace --json` when shape, stability, divergence, or late
  improvement matters.
- Use `aimx trace distribution --json` or `--csv` for automated histogram
  evidence. The unflagged distribution command is a non-interactive terminal
  visual that lists matched distributions, selects the first non-empty series,
  and renders a current-step histogram plus step-by-bin heatmap. `--step N`
  affects only this visual mode and falls back to the nearest tracked step.
- For minimization metrics such as loss or error, compare `min.value` and the
  corresponding step. For maximization metrics such as accuracy, F1, AUC, or
  IoU, compare `max.value`.
- Separate incomplete or failed runs from strong results before drawing
  conclusions. Very low step counts, missing params, or missing validation
  metrics should reduce confidence.
- Preserve read-only behavior. Do not run commands that initialize, repair,
  migrate, delete, or rewrite Aim repositories during `log_experiment`.

## Gotchas

| Gotcha | Wrong | Right |
| --- | --- | --- |
| Missing `aimx` in environment | Assume `aimx` is available | Verify with `aimx --help` or `python -m aimx --help`, then follow project install workflow |
| Repository targeting | Rely on current directory | Pass `--repo <repo>` explicitly on every collection command |
| Summary vs curve confusion | Treat `query metrics` output as full history | Use `query metrics` for summary (`last/min/max`) and `trace --json` for curve shape |
| Raw payload dumping | Paste full JSON into conversation | Parse and compute compact aggregates before reporting |
| AimQL string quoting | `metric.name == "loss"` | `metric.name == 'loss'` |
| Short hash assumptions | Assume short hash is canonical identity | Let `aimx` expand it, but compare/store full run hash |
| Distribution output mode | Use default distribution mode in scripts | Use `--json`, `--csv`, or `--table` for automation |
| `--step` expectation | Expect `--step` to filter JSON/CSV/table exports | Use `--step` only for visual histogram mode |
| Empty trace handling | Treat non-JSON message as fatal parsing error | Treat it as no trace evidence and continue analysis |
| Full trace collection | Pull all runs and all points first | Rank by summary, then trace only baseline, top candidates, and suspicious runs |

## Helper Script

Use `scripts/collect_experiment_snapshot.py` when an agent needs one structured
JSON bundle instead of several manual commands.

```bash
uv run python skills/aimx/scripts/collect_experiment_snapshot.py \
  --repo data \
  --base-expr "run.experiment == 'cloud-segmentation'" \
  --metric loss \
  --trace-metric loss \
  --param hparam.lr \
  --param model \
  --pretty
```

The script uses the current Python interpreter as `python -m aimx` by default.
Pass `--aimx "aimx"` or `--aimx "uv run aimx"` when a different launcher is
needed. It writes only to stdout.

## Reference

Read `references/aimx-cli.md` for command details, JSON envelope shapes, and
suggested `log_experiment` evidence fields. For deeper experiment analysis
patterns, see "Analysis Patterns", "Find best run by objective", "Spike /
divergence / plateau / NaN detection", "Overfitting detection", and "Sweep
ranking".
