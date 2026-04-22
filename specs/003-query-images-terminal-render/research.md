# Phase 0 Research: Inline Terminal Image Rendering for `aimx query images`

**Feature**: `003-query-images-terminal-render`
**Date**: 2026-04-22

This document captures the Phase 0 decisions that resolve every implicit
unknown in the Technical Context of `plan.md`. All `[NEEDS CLARIFICATION]`
items from the spec were already resolved in the 2026-04-22 clarify session.

---

## R-001 — Rendering library

- **Decision**: Use `textual_image.renderable.Image` driven by a
  `rich.Console` instance (`force_terminal=True, width=<cols>`).
- **Rationale**: `textual_image` already probes and selects the best
  available image protocol (Kitty graphics, iTerm2 inline, Sixel) and
  automatically falls back to half-block ANSI on plain terminals. Using it
  as a `rich` Renderable lets us reuse the existing `Console` plumbing in
  `src/aimx/rendering/`, keep styling consistent, and avoid a second output
  abstraction. Both `rich>=13.7` and `textual-image>=0.12.0` are already
  declared in `pyproject.toml`, so no new runtime dependency is introduced.
- **Alternatives considered**:
  - `term-image` / `climage`: older APIs, weaker Kitty/iTerm2 coverage,
    extra dependency.
  - Hand-rolled half-block renderer on top of PIL + ANSI: works only on the
    fallback path, doesn't give us iTerm2/Kitty upgrades "for free".
  - Writing Sixel directly: narrow terminal support, brittle on tmux.

## R-002 — Terminal capability detection

- **Decision**: Detect capability **passively** using `sys.stdout.isatty()`
  plus a small env-var classification (`TERM_PROGRAM`, `KITTY_WINDOW_ID`,
  `WEZTERM_EXECUTABLE`, `GHOSTTY_RESOURCES_DIR`, `VTE_VERSION`, `TERM`).
  No active Device-Attribute (DA) queries.
- **Rationale**: DA queries block on stdin reads and can hang over SSH or
  in CI runners with no controlling TTY. Passive detection is deterministic,
  testable, and matches what `textual_image` itself does internally.
- **Alternatives considered**:
  - DA query: rejected for SSH/CI hang risk (violates CA-003).
  - "Always try graphics, let terminal gracefully ignore": Sixel garbage
    characters leak onto plain terminals and tmux.

## R-003 — Image decoding source

- **Decision**: Consume Aim images via `aim.Image.to_pil_image()` (returns a
  decoded `PIL.Image.Image`); do **not** touch the raw `BLOB` storage
  directly. Resize with PIL before handing the bitmap to `textual_image`.
- **Rationale**: `to_pil_image()` is the stable public path; it handles
  multi-frame / palette / RGBA / grayscale transparently. PIL is already a
  transitive dependency of Aim, so no additional package is required.
- **Alternatives considered**:
  - Read `storage['data']` BLOB directly and decode manually: duplicates
    Aim internals, risks breakage across Aim minor versions.
  - Skip PIL resize and rely on `textual_image` downsampling only: loses
    fine-grained control over the row-count cap from clarify Q5.

## R-004 — Sizing strategy (confirmed by clarify Q5)

- **Decision**: Equal-ratio downscale so that both of the following hold:
  `target_width ≤ terminal_columns` and
  `target_height ≤ terminal_rows // 3` (height measured in character cells,
  with one cell ≈ two pixels of vertical resolution when the half-block
  fallback is active).
- **Rationale**: A 1/3-screen-height cap guarantees that, even with the
  default `--max-images=6`, at most 2 images fully share a screen with
  their metadata headers, so users can visually compare adjacent steps
  without scrolling. It also bounds the worst-case Sixel payload.
- **Alternatives considered**: no height cap (risk of a single tall image
  pushing metadata off-screen); 1/2 cap (too generous for 24-row SSH
  sessions); fixed pixel cap (ignores terminal cell size, unstable across
  resolutions).

## R-005 — Output layout (confirmed by clarify Q3)

- **Decision**: Emit the existing `render_image_rich_table` summary **first**;
  follow it with one section block per rendered image consisting of a
  single-line metadata header (`▌ <hash8>  <experiment>  <name>  <ctx>`)
  and the image renderable beneath it; close with a one-line footer when
  the `--max-images` cap truncates the list.
- **Rationale**: Keeping today's summary table intact preserves the
  existing machine-friendly eye-scan and the measured columns layout. The
  per-item section blocks are simpler to implement than embedding
  Renderables inside a rich `Table` cell (the `textual_image` Renderable
  does not currently negotiate cell sizing reliably inside tables).
- **Alternatives considered**: inline as a "Preview" column (blocked on
  `textual_image` + rich `Table` cell-size interaction); replace the
  summary table entirely (regresses on SC-003 because existing users may
  have grown accustomed to the table output).

## R-006 — Default `--max-images` cap (confirmed by clarify Q2)

- **Decision**: Default to **6**; `--max-images 0` means unlimited.
- **Rationale**: Fits 80–200-column terminals without scrolling; leaves
  room for the metadata table above and the 1/3-row image cap below.
  Keeps SC-004 (≤ 5 s on ≥ 50-match queries) comfortably in reach because
  worst-case decoding + rendering cost scales with the cap, not with the
  total match count.

## R-007 — Opt-out policy (confirmed by clarify Q1)

- **Decision**: No new toggle. Users who want "no inline images" use any
  of the existing mechanisms: `--plain`, `--json`, or redirect stdout to a
  file/pipe (breaks the `isatty()` check). `--max-images 0` is for
  unlimited rendering, not for disabling it.
- **Rationale**: Keeps the CLI surface minimal; avoids the tri-state
  `--images={auto,always,never}` pattern that invites drift from the
  actual TTY capability.

## R-008 — Warning channel (confirmed by clarify Q4)

- **Decision**: Dependency-missing or decode warnings go to **stderr**
  exactly once per process; de-duplicated via a module-level flag.
- **Rationale**: Keeps stdout byte-clean for SC-003, matches
  `aimx`'s existing error channel (`_run_*_query` writes errors to stderr
  at the CLI entrypoint), and follows Unix convention.
- **Alternatives considered**: stdout with `# warning:` prefix (pollutes
  pipes); silent-only behind `--verbose` (fails discoverability: users
  would not know why the feature did nothing on a bad install).

## R-009 — Failure isolation

- **Decision**: Per-row decoding failures render as
  `[image unavailable: <short reason>]` in the section block; the
  surrounding command continues and still exits 0 provided the metadata
  path succeeded. Only true command-level failures (unparseable args, repo
  not found) keep the existing non-zero exit.
- **Rationale**: FR-004 requires one bad image not to break the run.
  Localizing the placeholder inline makes it obvious which row failed
  without forcing users to cross-reference logs.

## R-010 — Scope boundaries

- **Decision**: First implementation covers **single-frame still images
  only**. If an Aim image sequence contains multi-frame / animated data,
  render only the first frame; document this in quickstart.
- **Rationale**: Per spec Assumptions. Animated rendering has different
  capability requirements (timing, clearing) and would inflate scope.
