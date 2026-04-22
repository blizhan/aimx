# Implementation Plan: Inline Terminal Image Rendering for `aimx query images`

**Branch**: `003-query-images-terminal-render` | **Date**: 2026-04-22 | **Spec**: [spec.md](/Users/blizhan/data/code/github/aimx/specs/003-query-images-terminal-render/spec.md)
**Input**: Feature specification from `/specs/003-query-images-terminal-render/spec.md`

## Summary

Extend the existing `aimx`-owned `query images` command so that, when stdout is a
TTY and the rich rendering path is in effect, `aimx` not only prints the
summary metadata table but also decodes each matched Aim image and renders it
**inline** in the terminal via `rich.Console` + `textual_image.renderable.Image`.
For `--json`, `--plain`/`--oneline`, and non-TTY stdout, behavior MUST remain
byte-for-byte identical to today — no graphics escapes, no Sixel/Kitty/iTerm2
bytes, no extra warnings on the main output stream. The feature is purely
additive on the rendering layer: no write-capable paths, no new passthrough
coupling, no change to the Aim SDK surface we use.

## Technical Context

**Language/Version**: Python 3.12 for development, runtime support `>=3.10,<3.13`
**Primary Dependencies**: `rich>=13.7`, `textual-image>=0.12.0` (already declared in
`pyproject.toml`), Aim SDK from the `dev` group for tests; no new runtime deps.
**Storage**: Existing local Aim repositories (read-only). Image bytes are read
via `aim.Image.to_pil_image()` which already returns a decoded `PIL.Image`.
**Testing**: pytest with unit + integration suites; rely on the existing sample
repository rooted at `data` for end-to-end verification.
**Target Platform**: Local shell, SSH, and CI. Inline rendering only activates
on TTY stdout; everything else falls back to the current metadata-only path.
**Project Type**: Single-project CLI application (same layout as feature 002).
**Performance Goals**: Per SC-004, a query matching ≥ 50 images MUST return
within 5 seconds on a normal dev machine under default settings (6-image
render cap).
**Constraints**: Read-only; no mutation of `.aim`; MUST preserve existing
`--json` / `--plain` byte output; image rendering failures MUST NOT fail the
command (exit 0 still required when metadata output succeeds).
**Scale/Scope**: One new rendering helper module + minimal wiring in
`_run_images_query`, one new CLI flag `--max-images`, plus terminal-capability
detection. No changes to the passthrough layer.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Safe coexistence: feature only adds terminal output inside the `aimx`
      rendering layer; does not touch the installed `aim` package, the native
      `aim` executable, or `.aim` repository data.
- [x] Ownership boundary: scope is strictly within the already-owned
      `aimx query images` surface; no new passthrough claims, no removal of
      existing passthrough behavior.
- [x] Read-only default: rendering decodes image blobs in-memory only; no
      writes, no caching to disk, no mutation of user data.
- [x] CLI-first contract: `--json` / `--plain` / non-TTY paths remain
      unchanged and scriptable; the new TTY rendering is opt-out via those
      existing mechanisms (no new toggle, per clarify Q1).
- [x] Compatibility plan: validated against the sample repo under `data` and
      against Aim ≥ 3.29 (matches existing `dev` dependency); `textual-image`
      is already pinned in `pyproject.toml`.

## Project Structure

### Documentation (this feature)

```text
specs/003-query-images-terminal-render/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── cli-output.md
├── checklists/
│   └── requirements.md
└── tasks.md            # created later by /speckit.tasks
```

### Source Code (repository root)

```text
src/
└── aimx/
    ├── commands/
    │   └── query.py                     # extend: parse --max-images, pass capability into renderer
    ├── aim_bridge/
    │   └── metric_stats.py              # extend: return image sequence object (lazy access to PIL bytes)
    └── rendering/
        ├── query_views.py               # unchanged public shape; render_image_rich_table keeps today's columns
        └── image_render.py              # NEW: terminal capability probe + inline image rendering pipeline

tests/
├── unit/
│   ├── test_query_helpers.py            # extend: --max-images parsing, default = 6, 0 = unlimited
│   └── test_image_render.py             # NEW: capability probe, size clamp (1/3 rows), graceful fallback
├── integration/
│   └── test_query_command.py            # extend: --json/--plain byte-stability, TTY vs non-TTY branches
└── contract/
    └── test_query_contract.py           # extend: no graphics bytes in --json / --plain output
```

**Structure Decision**: Keep the existing single-project CLI layout. Add a new
`src/aimx/rendering/image_render.py` module so that `query.py` stays thin and
`query_views.py` stays pure (string-returning, no side effects). All inline
rendering flows through the new module and is invoked only on the rich TTY
branch inside `_run_images_query`.

## Phase 0: Research Summary

Phase 0 decisions are captured in [research.md](/Users/blizhan/data/code/github/aimx/specs/003-query-images-terminal-render/research.md). Key outcomes:

- Use `textual_image.renderable.Image` over a `rich.Console` with
  `force_terminal=True, width=<cols>` for inline draw; let `textual_image`
  pick Kitty / iTerm2 / Sixel / half-block automatically based on env.
- Detect "can we draw images at all" from `sys.stdout.isatty()` +
  env hints (`TERM_PROGRAM`, `KITTY_WINDOW_ID`, `WEZ_*`, etc.) once per
  invocation; no active DA queries (SSH/CI safe).
- Decode image bytes via `aim.Image.to_pil_image()` (already pillow), then
  resize in PIL to respect both column cap and `rows/3` cap before handing off.
- Render ordering: keep today's rich summary table, then emit one "section
  block" per rendered image (header line with run · step · name · context,
  blank line, image renderable). Non-rendered overflow entries print only
  the header line (no image), followed by a single summary footer.
- Dependency-absence / decoding failure: print a **one-shot** warning to
  stderr and continue with metadata output; exit status stays 0.

## Phase 1: Design Summary

- Introduce `TerminalCapability` dataclass capturing `is_tty`, `columns`,
  `rows`, and a coarse protocol tag (`auto | fallback_text | disabled`).
- Introduce `ImageRenderPlan` dataclass: `(rows_with_image, skipped_count,
  max_images, capability)` — pure data, easy to unit test.
- Extend `collect_image_series` to include the raw Aim image sequence handle
  (or a thin accessor returning bytes / PIL image lazily) without changing
  the existing dict keys consumed by JSON/plain renderers.
- Add `--max-images N` to `parse_query_invocation` (default 6, `0` = unlimited);
  no other new flags per clarify Q1.
- `_run_images_query` branches:
  - `--json` / `--plain`: unchanged (existing renderers only read metadata
    keys, not image handles).
  - rich + non-TTY stdout: unchanged (metadata table only).
  - rich + TTY: call `query_views.render_image_rich_table(...)` first, then
    call `image_render.render_inline(rows, capability, max_images)` and
    concatenate the two strings; the latter writes image bytes via its own
    `Console(file=sys.stdout, force_terminal=True)`, keeping stderr clean.
- Dependency / decode failures are converted into
  `[image unavailable: <reason>]` placeholders at the per-row level, with a
  one-shot stderr warning that is de-duplicated via a module-level flag.

## Post-Design Constitution Check

- [x] Safe coexistence: rendering path is additive; no edits to the `aim`
      package, the native executable, or `.aim` data.
- [x] Ownership boundary: all changes remain within `aimx query images`;
      passthrough code paths are untouched.
- [x] Read-only default: only `to_pil_image()` on in-memory blobs; no file
      writes, no SDK mutation calls.
- [x] CLI-first contract: `--json` / `--plain` / non-TTY paths produce the
      same bytes as before (SC-003); the new TTY behavior degrades to
      text-only on plain ANSI terminals (SC-002).
- [x] Compatibility: relies only on already-declared deps (`rich`,
      `textual-image`, Aim ≥ 3.29 via dev group) and on `PIL` which is a
      transitive dep of Aim.

## Complexity Tracking

No constitution violations; no exceptional complexity requires justification.
The only nuance is the graceful degradation matrix, which is addressed via
a single new module (`image_render.py`) with focused unit tests.
