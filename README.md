# aimx

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10%2C%3C3.13-blue.svg)](./pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/aimx.svg?color=blue)](https://pypi.org/project/aimx/)
[![CI](https://github.com/blizhan/aimx/actions/workflows/CI.yaml/badge.svg)](https://github.com/blizhan/aimx/actions/workflows/CI.yaml)
[![Publish](https://github.com/blizhan/aimx/actions/workflows/publish.yaml/badge.svg)](https://github.com/blizhan/aimx/actions/workflows/publish.yaml)

![aimx trace output preview](static/trace.png)

`aimx` is a safe, additive, CLI-first companion for native [Aim](https://github.com/aimhubio/aim).

It keeps a small owned command surface for diagnostics and guidance, and
delegates everything else to the native [`aim`](https://github.com/aimhubio/aim) executable already available in
the user's environment.

## Installation

```bash
# Using uv (recommended)
uv add aimx

# Or using pip
pip install aimx
```

## What aimx owns

- `aimx`
- `aimx --help`
- `aimx help`
- `aimx version`
- `aimx doctor`
- `aimx query`
- `aimx trace`

These commands explain how `aimx` works, show the `aimx` version, and report
whether native Aim is available for passthrough.

`--repo` is optional for owned `query` and `trace` commands and defaults to the
current directory `.`. When provided, it accepts either the repository root,
such as `data`, or the metadata directory itself, such as `data/.aim`.

Both `aimx query` and `aimx trace` accept **AimQL** expressions (Aim's native
Python-like query language) as their filter argument — e.g.
`"metric.name == 'loss' and run.hparams.learning_rate > 0.001"`. For the full
syntax, supported properties (`run.*`, `metric.*`, `images.*`), and security
restrictions, see the upstream docs:
[Aim — Query language basics](https://aimstack.readthedocs.io/en/latest/using/search.html).

### `aimx query` — discover and summarise metrics

Queries an Aim repository and shows a grouped table with per-metric statistics
(step count, last value, min/max with step).

![aimx query output preview](static/metrics.png)

```bash
# If your current working directory is the Aim repo root, --repo can be omitted
aimx query metrics "metric.name == 'loss'"

# Rich table (default, colored in terminal)
aimx query metrics "metric.name == 'loss'" --repo data

# Short run hashes are transparently expanded to full hashes
aimx query metrics "run.hash=='eca37394' and metric.name=='loss'" --repo data

# Tab-separated plain text, suitable for awk/grep
aimx query metrics "metric.name == 'loss'" --repo data --oneline

# Structured JSON (nested by run)
aimx query metrics "metric.name == 'loss'" --repo data --json

# Step range filter — statistics recomputed within the window
aimx query metrics "metric.name == 'loss'" --repo data --steps 100:500
aimx query metrics "metric.name == 'loss'" --repo data --steps :50     # first 50 steps
aimx query metrics "metric.name == 'loss'" --repo data --steps 100:    # from step 100 onwards

# Epoch range filter (mutually exclusive with --steps)
aimx query metrics "metric.name == 'loss'" --repo data --epochs 1:10
aimx query metrics "metric.name == 'loss'" --repo data --epochs :5

# Density subsampling: first N / last N / every K-th point per series
aimx query metrics "metric.name == 'loss'" --repo data --head 20
aimx query metrics "metric.name == 'loss'" --repo data --tail 20
aimx query metrics "metric.name == 'loss'" --repo data --every 5

# Combine short hash + step range + head
aimx query metrics "run.hash=='eca37394' and metric.name=='loss'" --repo data --steps 100:300 --head 10

# Images — metadata table only (--json / --plain / redirected stdout)
aimx query images "images" --repo data --json
aimx query images "images" --repo data --plain

# Images — filter by epoch range (affects all output modes)
aimx query images "images" --repo data --epochs 10:50 --plain
aimx query images "images" --repo data --epochs :30 --json

# Images — global row subsampling (applied to the sorted result list)
aimx query images "images" --repo data --head 5
aimx query images "images" --repo data --tail 5
aimx query images "images" --repo data --every 3

# Images — inline preview in a modern terminal (iTerm2 / Kitty / WezTerm / Ghostty)
aimx query images "images" --repo data              # default: renders up to 6 images inline
aimx query images "images" --repo data --max-images 20   # render more
aimx query images "images" --repo data --max-images 0    # no cap (render all)

# Combine epoch filter + head + TTY cap
aimx query images "images" --repo data --epochs 10:50 --head 10 --max-images 4
```

Output modes: `--json` (nested runs→metrics), `--oneline` / `--plain` (tab-separated),
default (rich table with inline image preview).
Filter/sampling flags (affect all output modes): `--steps start:end | --epochs start:end`
(mutually exclusive), `--head N`, `--tail N`, `--every K`.
Additional flags: `--no-color`, `--verbose`, `--max-images N` (images TTY cap only).

#### Inline image preview

![aimx query images output preview](static/images.png)

When stdout is a TTY and `aimx` detects a graphics-capable terminal, `aimx query images`
renders matched images directly in the terminal. On plain ANSI terminals it falls back
to half-block character art — exit code is always `0`.

Terminal support is provided by [`textual-image`](https://github.com/lnqs/textual-image/tree/main#support-matrix-1).
Confirmed working terminals include: iTerm2, Kitty, Konsole, WezTerm, foot, tmux (Sixel),
xterm (Sixel), Windows Terminal, and VS Code integrated terminal. Warp and GNOME Terminal
are not supported.

To disable inline rendering without changing flags, redirect stdout `aimx query images > out.txt` or use `--plain` / `--json`.

### `aimx trace` — plot or export a metric time series

Fetches the full value sequence for one or more metrics and renders a curve,
table, or structured export. Multiple matching runs are overlaid on the same plot.

![aimx trace output preview](static/trace.png)

```bash
# If your current working directory is the Aim repo root, --repo can be omitted
aimx trace "metric.name=='loss'"

# Plot loss curve for a specific run — short hash transparently expanded
aimx trace "run.hash=='eca37394' and metric.name=='loss'" --repo data

# Compare train vs val loss across all runs
aimx trace "metric.name=='loss'" --repo data

# Step-by-step table
aimx trace "metric.name=='loss'" --repo data --table

# CSV export
aimx trace "metric.name=='loss'" --repo data --csv > loss.csv

# JSON with full value arrays
aimx trace "metric.name=='loss'" --repo data --json

# Step range filter (hard constraint, applied before sampling)
aimx trace "metric.name=='loss'" --repo data --steps 100:500
aimx trace "metric.name=='loss'" --repo data --steps :50      # first 50 steps
aimx trace "metric.name=='loss'" --repo data --steps 100:     # step 100 onwards

# Combine step filter + JSON
aimx trace "run.hash=='eca37394' and metric.name=='loss'" --repo data --steps 1:200 --json

# Limit to first 50 points per series (density subsampling, applied after --steps)
aimx trace "metric.name=='loss'" --repo data --head 50

# Sample every 10th point
aimx trace "metric.name=='loss'" --repo data --every 10
```

Output modes: default (plotext chart), `--table`, `--csv`, `--json`.
Step filtering: `--steps start:end` (inclusive, open-ended sides allowed).
Sampling: `--head N`, `--tail N`, `--every K`.
Display: `--width W`, `--height H`, `--no-color`.

## What aimx delegates

Any unowned command path is passed through to native `aim`.

Examples:

```bash
aimx up
aimx init --help
aimx runs --help
aimx runs ls
```

## Runtime contract

- `aimx` does not replace the `aim` executable.
- `aimx` does not modify the installed `aim` package.
- `aimx` does not mutate `.aim` data during help, version, doctor, or
passthrough flows.
- Native Aim remains an external runtime prerequisite for delegated commands.
- The repo's development dependency on Aim is only for local development and
testing convenience.

## Local development

```bash
uv sync --group dev
uv run pytest
```

## Quick checks

```bash
uv run aimx --help
uv run aimx version
uv run aimx doctor
uv run aimx query metrics "metric.name == 'loss'" --repo data
uv run aimx query images "images" --repo data/.aim --json
```

## TODO

- [ ] Introduce `skills` — composable, reusable workflow modules that layer higher-level experiment
  analysis and auto-research capabilities on top of `aimx`.
