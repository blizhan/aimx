# CLI Output Contract: `aimx query images` (003)

**Feature**: `003-query-images-terminal-render`
**Date**: 2026-04-22

This contract defines the observable CLI output of `aimx query images` after
feature 003 is implemented. It supersedes the relevant portions of the 002
contract only for the `images` target; the `metrics` target is unchanged.

---

## 1. Command surface

```
aimx query images <expression> [--repo <path>] [--json] [--oneline | --plain]
                               [--no-color] [--verbose] [--steps start:end]
                               [--max-images <N>]
```

Changes vs. feature 002:

- New flag `--max-images <N>` (default: `6`; `0` = unlimited).
  - Only affects the default rich TTY path.
  - Has **no effect** on `--json` or `--plain` output bytes.

No other new flags are added (clarify Q1).

## 2. Exit codes (unchanged from 002)

| Exit | Meaning |
|------|---------|
| `0`  | Query executed successfully (even if 0 matches, even if some images failed to render). |
| `2`  | Usage error (bad args, bad `--repo`, bad `--max-images`), or underlying Aim error. |

Image-rendering failures MUST NOT change the exit code. Missing
`textual_image` or PIL MUST NOT change the exit code.

## 3. Output mode matrix

Let `TTY := sys.stdout.isatty()`.

| Mode                                   | stdout bytes        | stderr bytes                         |
|----------------------------------------|---------------------|--------------------------------------|
| `--json`                               | **identical to 002**| empty (unless query errors)          |
| `--plain` / `--oneline`                | **identical to 002**| empty (unless query errors)          |
| rich + `TTY == False`                  | **identical to 002**| empty (unless query errors)          |
| rich + `TTY == True`                   | 002 summary table **followed by** §4 block(s) | one-shot dep/decoding warning if any |

"**identical to 002**" is a hard contract (SC-003): byte-for-byte equal to
the pre-003 output for the same inputs. This is enforced by
`tests/contract/test_query_contract.py`.

## 4. Rich TTY output layout

When rendered on a TTY, the stdout stream is the concatenation of:

1. **Summary table** — produced by the existing
   `render_image_rich_table(...)`. Unchanged columns: `RUN`, `EXPERIMENT`,
   `NAME`, `CONTEXT`. Unchanged header.
2. **Zero or more section blocks**, one per row in
   `ImageRenderPlan.rendered_rows`:
   ```
   <blank line>
   ▌ <hash8>  <experiment>  <name>  <ctx-kv-string>
   <blank line>
   <image renderable — width ≤ cols, height ≤ rows//3>
   ```
   The header line reuses the existing rich styles (`colors.RUN_HASH`,
   `colors.EXPERIMENT`, `colors.METRIC_NAME`, `colors.CONTEXT_VAL`). The
   image renderable is produced by `textual_image.renderable.Image`.
3. **Truncation footer** (only when `len(skipped_rows) > 0`):
   ```
   ... rendered <K> of <M> images, use --max-images=0 for all
   ```
   Exactly one line, printed after the last section block.

Edge cases:

- `capability.columns < 20` or `capability.rows < 9`
  (so `rows//3 < 3`): skip §4 entirely; stdout matches the non-TTY mode
  bytes exactly.
- Per-row decoding failure: replace the image renderable with a single
  line `[image unavailable: <short reason>]`; the section block otherwise
  looks the same.

## 5. `--max-images` semantics

- `--max-images 6` (default): render the first 6 rows; remaining rows
  contribute **only** to §4's truncation footer (no section block, no
  image bytes).
- `--max-images 0`: unlimited; render every row.
- `--max-images N` with `N < 0` or non-integer: usage error, exit `2`.

Only the default-rich-TTY path consults `--max-images`. It is a hard error
to observe any `--max-images`-driven difference in `--json`, `--plain`, or
non-TTY output.

## 6. Warning contract (stderr)

At most one line may be written to stderr during an otherwise-successful
invocation, in either of the following shapes:

```
aimx: inline image rendering unavailable: <reason>
```

Triggered when `textual_image` / PIL cannot be imported, or when
terminal-capability detection disables image drawing on a TTY (e.g.
unrecognised `$TERM`). De-duplicated within a single process via a
module-level flag so batched invocations in Python never produce more
than one warning.

No stderr output is produced on the non-TTY / `--json` / `--plain` paths.

## 7. Compatibility expectations

- Aim SDK: `>= 3.29` (covered by the existing `dev` dependency group).
- Python: `>= 3.10, < 3.13` (matches `pyproject.toml`).
- `rich >= 13.7`, `textual-image >= 0.12.0` (already declared).
- Terminal targets explicitly validated: iTerm2, Kitty, WezTerm, Ghostty
  (all "auto" protocol), plus tmux-over-xterm and plain `xterm`
  (fallback half-block path), plus non-TTY stdout (no image bytes at all).
