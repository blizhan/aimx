# Quickstart: Inline Terminal Image Rendering for `aimx query images`

**Feature**: `003-query-images-terminal-render`
**Date**: 2026-04-22

This quickstart shows how to try the new inline image rendering end-to-end
against the sample Aim repository at `data/` in this workspace.

---

## 1. Prerequisites

No new installs are required. The runtime deps (`rich`, `textual-image`)
are already in `pyproject.toml`, and the sample repo under `data/` already
contains image-tracked runs (used by feature 002 tests).

Sync the environment once:

```bash
uv sync
```

## 2. Run the query in a modern terminal (iTerm2 / Kitty / WezTerm / Ghostty)

```bash
uv run aimx query images "images" --repo data
```

Expected:

- The existing rich summary table is printed at the top (same columns as
  feature 002: `RUN`, `EXPERIMENT`, `NAME`, `CONTEXT`).
- Up to 6 images are rendered inline as **section blocks** under the table,
  each preceded by a single-line metadata header.
- If more than 6 matches exist, a footer line appears:
  `... rendered 6 of M images, use --max-images=0 for all`.

## 3. Raise or remove the render cap

```bash
# Render up to 20 images
uv run aimx query images "images" --repo data --max-images 20

# Render all matching images (use with care on large repos)
uv run aimx query images "images" --repo data --max-images 0
```

## 4. Verify scripting paths are still byte-clean

```bash
# JSON output MUST contain zero image bytes
uv run aimx query images "images" --repo data --json | jq .

# Plain / oneline output MUST stay one line per match
uv run aimx query images "images" --repo data --plain

# Redirecting stdout disables inline rendering automatically
uv run aimx query images "images" --repo data > /tmp/out.txt
grep -c $'\x1b' /tmp/out.txt   # should print 0 for the default (rich goes through isatty)
```

## 5. Verify graceful fallback on a plain ANSI terminal

In a non-graphics terminal (e.g. stock `xterm`, `tmux` without
passthrough, or any SSH session without Kitty/iTerm2 protocol support):

```bash
uv run aimx query images "images" --repo data
```

Expected:

- The summary table prints exactly as in modern terminals.
- The section blocks render via half-block characters (coarser, but legible).
- Exit code is `0`.

## 6. Verify failure isolation

- If `textual_image` is uninstalled (for example inside an unusual
  environment), the first invocation prints one line to stderr
  (`aimx: inline image rendering unavailable: <reason>`) and continues to
  print the summary table with exit `0`. Subsequent invocations in the
  same Python process do not re-warn.
- If a single image's blob is corrupted, the matching section block shows
  `[image unavailable: <short reason>]` while every other block renders
  normally. Exit code remains `0`.

## 7. Automated validation

```bash
uv run pytest tests/unit/test_image_render.py -q
uv run pytest tests/unit/test_query_helpers.py -q
uv run pytest tests/integration/test_query_command.py -q
uv run pytest tests/contract/test_query_contract.py -q
```

The contract suite is the guardrail for SC-003: any accidental image-byte
leak into `--json` / `--plain` / non-TTY output will be caught there.
