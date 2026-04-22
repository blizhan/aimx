# Phase 1 Data Model: Inline Terminal Image Rendering for `aimx query images`

**Feature**: `003-query-images-terminal-render`
**Date**: 2026-04-22

This feature is a rendering-layer extension; it does not introduce new
persistent state, storage schemas, or Aim SDK mutations. The "data model"
below captures the **in-memory value objects** added on the rendering path
and clarifies how they interact with the existing `collect_image_series`
output.

---

## 1. `TerminalCapability`

Captures everything the renderer needs to know about the current terminal.
Resolved once per `aimx query images` invocation, before any image is
decoded.

| Field          | Type                                   | Notes |
|----------------|----------------------------------------|-------|
| `is_tty`       | `bool`                                 | Equals `sys.stdout.isatty()`. |
| `columns`      | `int`                                  | From `shutil.get_terminal_size(fallback=(120, 24))`. |
| `rows`         | `int`                                  | Same source as `columns`. |
| `protocol`     | `Literal["auto", "fallback_text", "disabled"]` | `"auto"` when `is_tty` and `textual_image` is available; `"fallback_text"` for the half-block path; `"disabled"` when non-TTY or deps missing. |
| `reason`       | `str \| None`                          | When `protocol == "disabled"`, human-readable reason used in the one-shot stderr warning. |

Validation rules:

- `columns >= 20` — below this width the renderer MUST skip image drawing
  and revert to metadata-only output (edge case in spec).
- `protocol == "disabled"` iff rendering is suppressed entirely (non-TTY,
  `--plain`, `--json`, or missing dependency); in that case
  `image_render.render_inline(...)` MUST return an empty string.

State transitions: none — the value is immutable for the lifetime of the
invocation.

## 2. `ImageRow` (extension, not a new type)

The existing `collect_image_series` dict is kept as-is for JSON/plain
renderers. The rendering path reads the same dict, plus an **optional**
lazy accessor:

| Key              | Type                              | Notes |
|------------------|-----------------------------------|-------|
| `run`            | `RunMeta`                         | Unchanged. |
| `name`           | `str`                             | Unchanged. |
| `context`        | `dict[str, Any]`                  | Unchanged. |
| `_image_accessor`| `Callable[[], PIL.Image.Image] \| None` | **New**, private; present only when rendering is possible; invoked lazily inside `image_render.render_inline`. `--json` and `--plain` MUST NOT touch this key. |

Validation rules:

- `_image_accessor` is `None` in non-TTY / `--json` / `--plain` runs so the
  existing renderers remain byte-identical.
- Calling the accessor MUST NOT raise across the whole renderer; all
  exceptions are caught and converted to `[image unavailable: …]`.

## 3. `ImageRenderPlan`

Pure data passed between "decide what to render" and "actually render".
Makes the cap logic (Q2/Q5) unit-testable.

| Field           | Type                               | Notes |
|-----------------|------------------------------------|-------|
| `capability`    | `TerminalCapability`               | Borrowed from §1. |
| `rendered_rows` | `list[ImageRow]`                   | The first `max_images` rows (or all if `max_images == 0`). |
| `skipped_rows`  | `list[ImageRow]`                   | Rows whose metadata will still be printed but no image drawn. |
| `target_width`  | `int`                              | `capability.columns` (or slightly less for margins). |
| `max_height`    | `int`                              | `capability.rows // 3`; lower bound 3 cells. |
| `max_images`    | `int`                              | Effective cap; `0` means unlimited. |

Derived rules:

- If `capability.protocol == "disabled"`, both `rendered_rows` and
  `skipped_rows` MUST be empty and the renderer MUST short-circuit.
- `len(rendered_rows) + len(skipped_rows) == len(all_rows)`.
- `max_height >= 3` — any smaller and we skip image drawing entirely and
  move the row from `rendered_rows` to `skipped_rows`.

## 4. Invocation flag addition

| Field on `QueryInvocation` | Type | Default | Notes |
|----------------------------|------|---------|-------|
| `max_images`               | `int`  | `6` | `0` → unlimited; negative values rejected at parse time. |

Parsing rules (enforced in `parse_query_invocation`):

- Token: `--max-images <N>`.
- `<N>` MUST parse as a non-negative integer; otherwise raise
  `ValueError("Invalid --max-images value: <token>")`.
- No other new flags are introduced (per clarify Q1).

## 5. Relationships

```
QueryInvocation ──────────────► ImageRenderPlan ◄────── TerminalCapability
       │                                │
       │ max_images, target=images      │ rendered_rows/skipped_rows
       ▼                                ▼
  collect_image_series  ────►  list[ImageRow] (with _image_accessor on TTY)
                                       │
                                       ▼
                          render_image_rich_table  ──►  summary (stdout)
                                       │
                                       ▼
                             image_render.render_inline  ──►  inline images (stdout)
```

No persisted state. No cross-invocation cache. No shared mutable singleton
aside from the one-shot stderr warning flag guarded by a module-level
`bool`.
