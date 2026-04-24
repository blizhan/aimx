---
name: aimx
description: Use when autoresearch, log_experiment, experiment analysis, or automatic iteration workflows need to inspect local Aim repositories with aimx, collect run params, metric summaries, traces, image evidence, compare training runs, or summarize model results without mutating Aim data.
---

# Aimx

## Overview

Use `aimx` as a read-only evidence collector for `autoresearch` `log_experiment`
steps. Prefer JSON output so downstream agents can compare runs, explain model
effects, and propose the next experiment from concrete Aim data.

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

6. Collect image metadata when qualitative outputs matter:

   ```bash
   aimx query images "images" --repo <repo> --json --head 20
   ```

7. Emit a compact `log_experiment` record containing:

   ```json
   {
     "repo": "<repo>",
     "run_scope": "<AimQL>",
     "params": {},
     "metric_summary": {},
     "trace_evidence": {},
     "image_evidence": {},
     "interpretation": {
       "best_runs": [],
       "regressions": [],
       "confidence": "low|medium|high",
       "next_experiments": []
     }
   }
   ```

## Interpretation Rules

- Prefer validation, test, or held-out contexts over training contexts when
  ranking runs.
- Treat `aimx query metrics` as summary data: `last`, `min`, `max`, and step
  counts. Use `aimx trace --json` when shape, stability, divergence, or late
  improvement matters.
- For minimization metrics such as loss or error, compare `min.value` and the
  corresponding step. For maximization metrics such as accuracy, F1, AUC, or
  IoU, compare `max.value`.
- Separate incomplete or failed runs from strong results before drawing
  conclusions. Very low step counts, missing params, or missing validation
  metrics should reduce confidence.
- Preserve read-only behavior. Do not run commands that initialize, repair,
  migrate, delete, or rewrite Aim repositories during `log_experiment`.

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
suggested `log_experiment` evidence fields.
