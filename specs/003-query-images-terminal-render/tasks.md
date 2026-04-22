# Tasks: Inline Terminal Image Rendering for `aimx query images`

**Input**: Design documents from `/specs/003-query-images-terminal-render/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/cli-output.md, quickstart.md

**Tests**: Test tasks are included because feature 003 carries hard, testable
CLI contracts that the constitution requires us to guard (read-only
behavior, CLI-first output stability under `--json`/`--plain`/non-TTY,
graceful failure isolation). Contract tests are the primary safety net for
SC-003 byte-stability.

**Organization**: Tasks are grouped by user story. US1 and US2 are both
priority P1 (US2 is the "do no harm" guardrail that keeps scripting
workflows working); US3 is P2 (bounded output for large result sets).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single-project CLI layout (matches feature 002):
- Source: `src/aimx/`
- Tests: `tests/unit/`, `tests/integration/`, `tests/contract/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm toolchain and dependencies are ready. No new runtime
packages are introduced; both `rich>=13.7` and `textual-image>=0.12.0` are
already declared in `pyproject.toml`.

- [x] T001 Run `uv sync` at repo root to confirm `rich`, `textual-image`, `aim`, and `pytest` are resolvable; verify `uv run python -c "import textual_image, PIL, aim"` exits 0

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Ground the new rendering module and the shared invocation
surface. Every user story depends on this phase.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T002 [P] Create `src/aimx/rendering/image_render.py` with module skeleton: frozen `TerminalCapability` dataclass (fields per data-model.md §1), frozen `ImageRenderPlan` dataclass (fields per data-model.md §3), public function stubs `detect_capability() -> TerminalCapability`, `plan_render(rows, capability, max_images) -> ImageRenderPlan`, `render_inline(plan) -> str`, and a module-level `_WARNED = False` flag for one-shot stderr warnings. All stubs raise `NotImplementedError` for now.
- [x] T003 [P] Extend `collect_image_series` in `src/aimx/aim_bridge/metric_stats.py` so each returned dict also carries a private key `_image_accessor: Callable[[], PIL.Image.Image] | None` that lazily calls `image.to_pil_image()` on the underlying Aim image object; keep the existing keys (`run`, `name`, `context`) byte-identical so `render_image_json` / `render_image_oneline` / `render_image_rich_table` remain untouched.
- [x] T004 Extend `parse_query_invocation` in `src/aimx/commands/query.py` to accept `--max-images <N>`: store on `QueryInvocation` as `max_images: int` with default `6`; reject `N < 0` or non-integer values with a `ValueError` ("Invalid --max-images value: …") so `run_query_command` maps it to exit status `2`. Update the `Usage:` string printed on bad args.

**Checkpoint**: Foundation ready — user story implementation can now proceed.

---

## Phase 3: User Story 1 — View Matching Images Inline In Terminal (Priority: P1) 🎯 MVP

**Goal**: When stdout is a TTY and the default rich path is in effect,
render each matched image inline under the existing summary table, using
`rich.Console` + `textual_image.renderable.Image`, with graceful half-block
fallback on plain ANSI terminals.

**Independent Test**: `uv run aimx query images "images" --repo data` in
iTerm2/Kitty/WezTerm/Ghostty prints the existing summary table followed by
per-row section blocks whose inline image renderables are observable (visible
pixels + non-empty bytes). The same command in a dumb terminal still exits
`0` with half-block fallback output.

### Tests for User Story 1

- [x] T005 [P] [US1] Unit tests in `tests/unit/test_image_render.py`: cover `detect_capability()` (envvar matrix: iTerm2/Kitty/WezTerm/Ghostty → `protocol=="auto"`; plain xterm → `"fallback_text"`; non-TTY / tiny width → `"disabled"`), `plan_render()` clamp rules (`max_height = max(rows // 3, 3)`, `columns < 20` short-circuits to `disabled`), and lazy `_image_accessor` failure → `[image unavailable: ...]` placeholder.
- [x] T006 [P] [US1] Integration test in `tests/integration/test_query_command.py`: run `run_query_command(["images", "images", "--repo", "data"])` with `sys.stdout` monkey-patched to a pseudo-TTY; assert output contains the existing rich table header AND at least one `▌ <hash8>` section header AND at least one byte that is ESC-prefixed (or half-block `▀`/`▄`); exit status is `0`.

### Implementation for User Story 1

- [x] T007 [US1] Implement `detect_capability()` in `src/aimx/rendering/image_render.py`: `is_tty = sys.stdout.isatty()`, read `shutil.get_terminal_size(fallback=(120, 24))`, classify protocol by env vars (`TERM_PROGRAM` in {iTerm.app}, `KITTY_WINDOW_ID`, `WEZTERM_EXECUTABLE`, `GHOSTTY_RESOURCES_DIR`, `VTE_VERSION`, fallback to `TERM`); return `"disabled"` when `not is_tty`, when `columns < 20`, or when `import textual_image.renderable` fails. Populate `reason` on disabled paths.
- [x] T008 [US1] Implement pure `plan_render(rows, capability, max_images)` in `src/aimx/rendering/image_render.py`: short-circuit to empty plan when `capability.protocol == "disabled"`; compute `max_height = max(capability.rows // 3, 3)`; split `rows` into `rendered_rows[:N]` / `skipped_rows[N:]` using `max_images` (0 = unlimited). Do NOT invoke `_image_accessor` here — this is a pure data step.
- [x] T009 [US1] Implement `render_inline(plan) -> str` in `src/aimx/rendering/image_render.py`: construct a `Console(file=StringIO(), force_terminal=True, width=capability.columns, highlight=False)`; for each `rendered_rows` item, print a blank line, then a single-line header `▌ <hash8>  <experiment>  <name>  <ctx>` using the same `colors.*` palette as `query_views.py`, then the `textual_image.renderable.Image` built from `_image_accessor()` resized to `(target_width, max_height * 2)` pixels (half-block convention). Per-row exceptions → `[image unavailable: <short-reason>]` single line; do NOT raise.
- [x] T010 [US1] Wire the rich+TTY branch in `_run_images_query` in `src/aimx/commands/query.py`: call the existing `render_image_rich_table(...)` for the summary string; if `capability.protocol != "disabled"`, call `render_inline(plan_render(...))` and concatenate; return both as the `output` of `QueryCommandResult`. Non-TTY / `--json` / `--plain` branches remain literally unchanged (no new code path).

**Checkpoint**: US1 complete — inline rendering works end-to-end on modern terminals and half-block fallback, summary table still visible above.

---

## Phase 4: User Story 2 — Keep Scripting & Machine-Readable Paths Byte-Clean (Priority: P1)

**Goal**: Guarantee that `--json`, `--plain`/`--oneline`, `--no-color`, and
any non-TTY stdout path produce **byte-identical** output to the pre-003
baseline, and that dependency/decoding issues surface as **one** single
line on stderr without ever leaking into stdout.

**Independent Test**: `aimx query images "images" --repo data --json | jq .`
produces valid JSON with no ANSI/image bytes. `aimx query images ... --plain`
produces only tab-separated lines. `aimx query images ... > out.txt` (i.e.
non-TTY) contains zero ESC / Kitty / iTerm2 / Sixel markers. If
`textual_image` fails to import, the command still exits `0` and emits
exactly one line on stderr.

### Tests for User Story 2

- [x] T011 [P] [US2] Contract test in `tests/contract/test_query_contract.py`: run `run_query_command(["images", "images", "--repo", "data", "--json"])` and assert the returned `output` is valid JSON, byte-identical to a captured pre-003 baseline (store baseline as a constant at top of the test), and contains no `\x1b`, `\x90`, `\x9d`, Kitty `_G` sequences, or iTerm2 `]1337;File=` markers.
- [x] T012 [P] [US2] Contract test in `tests/contract/test_query_contract.py`: run the same command with `--plain`; assert each non-empty line is a single row, no `\x1b`, no binary bytes outside printable ASCII + tab.
- [x] T013 [P] [US2] Contract test in `tests/contract/test_query_contract.py`: run `run_query_command(["images", "images", "--repo", "data"])` with `sys.stdout` swapped to a plain `io.StringIO` (simulating redirection / non-TTY); assert the produced string contains zero image-protocol byte patterns AND matches the pre-003 baseline for the rich summary exactly.
- [x] T014 [P] [US2] Contract test in `tests/contract/test_query_contract.py`: run the command with `--max-images 3`, `--max-images 0`, and `--max-images 100` under `--json` and `--plain`; assert all three produce the **same** bytes as the default (proves `--max-images` has zero effect off the TTY path).
- [x] T015 [P] [US2] Unit test in `tests/unit/test_image_render.py`: simulate `textual_image` import failure (monkey-patch `sys.modules`); call `detect_capability()` → `protocol == "disabled"`, `reason` is populated; call a public `warn_once(message)` helper twice in the same process and assert `sys.stderr` receives exactly one line.

### Implementation for User Story 2

- [x] T016 [US2] Implement `warn_once(message: str) -> None` in `src/aimx/rendering/image_render.py`: module-level `_WARNED` flag gates a single `print(f"aimx: inline image rendering unavailable: {message}", file=sys.stderr)`; idempotent across repeated calls within one process. Wire it into both the import-failure branch in `detect_capability()` and the catch-all exception handler in `render_inline()`.
- [x] T017 [US2] Harden the TTY gate in `_run_images_query` (`src/aimx/commands/query.py`): explicitly short-circuit BEFORE calling into `image_render` when `invocation.output_json` or `invocation.plain` is True, or when `detect_capability().protocol == "disabled"`, ensuring no `textual_image` import happens on those paths. Add an assertion in the `--json` and `--plain` branches that the returned `output` string contains no `\x1b` byte (wrapped in `__debug__` so it's stripped from `-O` builds).

**Checkpoint**: US2 complete — SC-003 enforced by contract tests; stderr
warning contract (R-008) enforced by unit tests.

---

## Phase 5: User Story 3 — Bounded Output For Large Result Sets (Priority: P2)

**Goal**: When a query matches many images, render only up to `--max-images`
by default (6), print metadata for the rest, and emit a single-line
truncation footer so users aren't surprised by a flood of output.

**Independent Test**: Point `aimx query images` at a fabricated result set
with ≥ 8 matches (helper fixture fabricates `ImageRow` dicts with synthetic
PIL images); default run draws 6 and prints
`... rendered 6 of 8 images, use --max-images=0 for all` at the end;
`--max-images 0` draws all 8 and omits the footer; `--max-images 3` draws 3
and footer reads `... rendered 3 of 8 images, ...`.

### Tests for User Story 3

- [x] T018 [P] [US3] Unit tests in `tests/unit/test_query_helpers.py`: `--max-images` parsing — default `6`, `0` → unlimited marker, `--max-images 12` → `12`, `--max-images -1` → `ValueError`, `--max-images abc` → `ValueError`, missing value → `ValueError`.
- [x] T019 [P] [US3] Integration test in `tests/integration/test_query_command.py`: build an in-memory list of 8 synthetic `ImageRow` dicts with tiny 4×4 PIL images; call `plan_render(rows, fake_capability, max_images=6)` → `len(rendered_rows) == 6`, `len(skipped_rows) == 2`; call with `max_images=0` → all 8 rendered, 0 skipped; call with `max_images=3` → 3/5.
- [x] T020 [P] [US3] Integration test in `tests/integration/test_query_command.py`: assert `render_inline(plan_with_skipped)` ends with exactly one line matching `... rendered <K> of <M> images, use --max-images=0 for all` and that `render_inline(plan_with_zero_skipped)` does NOT contain that pattern.

### Implementation for User Story 3

- [x] T021 [US3] Finalize `plan_render` split in `src/aimx/rendering/image_render.py` per data-model.md §3: when `max_images == 0` or `len(rows) <= max_images`, `skipped_rows = []`; otherwise `rendered_rows = rows[:max_images]`, `skipped_rows = rows[max_images:]`. Treat `max_images == 0` as the "unlimited" sentinel explicitly (not as "zero rendered").
- [x] T022 [US3] In `render_inline` in `src/aimx/rendering/image_render.py`, when `plan.skipped_rows` is non-empty, append after the last section block exactly one line: `... rendered {len(plan.rendered_rows)} of {len(plan.rendered_rows) + len(plan.skipped_rows)} images, use --max-images=0 for all`. The line MUST NOT be emitted when `skipped_rows` is empty. No duplicate emission on repeat calls.
- [x] T023 [US3] Pass `invocation.max_images` through from `_run_images_query` in `src/aimx/commands/query.py` to `plan_render(...)`. Keep the default (`6`) consistent with `parse_query_invocation`; do NOT hardcode `6` in `image_render`.

**Checkpoint**: All three user stories deliverable independently. MVP = US1 + US2; US3 adds the safety cap for large repositories.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T024 [P] Update `README.md` with a short "Inline image preview" section (2–4 sentences) referencing `specs/003-query-images-terminal-render/quickstart.md`; mention default `--max-images=6` and that `--json` / `--plain` paths are unchanged.
- [x] T025 [P] Update `specs/003-query-images-terminal-render/checklists/requirements.md` to mark the `No [NEEDS CLARIFICATION] markers remain` item fully closed (already closed in clarify, but re-verify), and confirm the Feature Readiness section all check.
- [x] T026 Run `uv run pytest -q` and ensure the full suite is green (new and existing). Any regression in the 002 query contract counts as a blocker for this feature.
- [x] T027 Execute `specs/003-query-images-terminal-render/quickstart.md` sections 2–6 manually against the repository under `data/`, in a modern terminal; record any deviations as follow-up issues. Smoke-check SSH / `TERM=dumb` fallback if accessible.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup. **Blocks all user stories.**
- **US1 (Phase 3, P1)**: Depends on Foundational (T002, T003, T004).
- **US2 (Phase 4, P1)**: Depends on Foundational; may run in parallel with
  US1 by a second developer (US2 implementation touches the same query.py
  section, so merge-coordination is needed — see parallel notes).
- **US3 (Phase 5, P2)**: Depends on Foundational **and** US1 (needs
  `plan_render`/`render_inline` scaffolding from US1).
- **Polish (Phase 6)**: Depends on US1+US2 at minimum; T024–T025 can start
  once US1+US2 are green even before US3 lands.

### User Story Dependencies

- US1 (view inline): no inter-story dep beyond Foundational.
- US2 (byte-clean scripting): no inter-story dep; contract tests can land
  before or alongside US1 implementation.
- US3 (bounded output): reuses US1's `plan_render` + `render_inline`; must
  land after US1 to avoid churn.

### Within Each User Story

- Unit tests (for US1 and US2) SHOULD be written first and observed to fail
  before the corresponding implementation task lands.
- Contract tests (US2) MUST be red against a pre-003 reference first, then
  green against the final implementation without changes to the test file.
- Within US3, `plan_render` split (T021) before `render_inline` footer (T022).

### Parallel Opportunities

- Foundational: T002, T003 are different files and can run in parallel; T004
  depends on nothing new and can run in parallel with T002/T003.
- US1 tests (T005, T006) can be authored in parallel.
- US2 contract tests T011–T015 all live in `tests/contract/test_query_contract.py`
  (with T015 in a different file) — T011–T014 share a file so write them in
  one edit session; T015 in parallel.
- US3 tests (T018, T019, T020) across two files — parallel.
- Polish T024 and T025 are independent docs changes — parallel.

---

## Parallel Example: User Story 1

```bash
# Independent tests for US1 can be authored in parallel:
Task: "Unit tests in tests/unit/test_image_render.py (T005)"
Task: "Integration test in tests/integration/test_query_command.py (T006)"

# Implementation tasks T007/T008 are same file (image_render.py) — sequential.
# Integration wiring T010 must come after T007+T008+T009.
```

---

## Implementation Strategy

### MVP First (US1 + US2)

1. Phase 1 Setup → Phase 2 Foundational.
2. Phase 3 (US1): deliver inline image rendering on rich+TTY path.
3. Phase 4 (US2): contract tests guard SC-003 byte-stability; stderr
   warning contract landed.
4. **STOP and VALIDATE**: `uv run pytest -q` green; quickstart sections 2–4
   pass manually. This is the releasable MVP.

### Incremental Delivery

1. MVP (US1 + US2) → release.
2. US3 (bounded output + truncation footer) → release.
3. Polish (docs, README, final sweep).

### Parallel Team Strategy

With two developers after Foundational:

1. Dev A: US1 (T005 → T006 → T007 → T008 → T009 → T010).
2. Dev B: US2 (T011 → T012 → T013 → T014 → T015 → T016 → T017).
   Dev B's T017 edits `query.py` — coordinate with Dev A's T010 via a
   shared feature branch and resolve at merge (same function body).
3. Once US1 merges, either dev takes US3.

---

## Notes

- [P] tasks = different files, no dependencies.
- Every task carries an exact file path.
- Constitution-driven guardrails (SC-003 byte-stability, stderr warning
  contract, per-row failure isolation) are expressed as US2 tests and
  enforced on every run of `pytest`.
- Commit after each checkpoint or logical group (e.g. end of Phase 2, end
  of US1, end of US2, end of US3).
- Any regression in existing 002 tests is a blocker — do not proceed past
  the affected task.
