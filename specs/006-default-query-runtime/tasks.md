# Tasks: Default Query Expression And Run Duration

**Input**: Design documents from `/Users/blizhan/data/code/github/aimx/specs/006-default-query-runtime/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-output.md, quickstart.md

**Tests**: Test tasks are included because this feature changes owned CLI
defaults and params output contracts, and the constitution requires validation
for scriptable output, safe failure modes, read-only behavior, and native Aim
passthrough non-regression.

**Organization**: Tasks are grouped by user story so each story can be
implemented and tested as an independently useful increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or depends
  only on completed foundation work
- **[Story]**: Maps task to a user story (`US1`, `US2`, `US3`)
- Every task includes exact repository-relative file paths

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify the existing CLI, bridge, renderer, docs, and test touch
points before changing behavior.

- [X] T001 Review current query parser, dispatch, and params rendering touch points in `src/aimx/commands/query.py`, `src/aimx/aim_bridge/run_params.py`, and `src/aimx/rendering/params_views.py`
- [X] T002 [P] Review current trace parser and dispatch touch points in `src/aimx/commands/trace.py` and `src/aimx/rendering/trace_views.py`
- [X] T003 [P] Review existing CLI documentation examples that mention required expressions in `README.md` and `src/aimx/commands/help.py`
- [X] T004 [P] Review current parser, contract, and integration test coverage in `tests/unit/test_query_helpers.py`, `tests/unit/test_trace_helpers.py`, `tests/contract/test_query_contract.py`, `tests/contract/test_trace_contract.py`, `tests/integration/test_query_command.py`, and `tests/integration/test_trace_command.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add shared parser defaults and run-duration primitives required by
all user stories.

**Critical**: No user story work should begin until this phase is complete.

- [X] T005 [P] Add query parser unit tests for omitted expression, option-first expression, blank expression, explicit expression precedence, and missing target behavior in `tests/unit/test_query_helpers.py`
- [X] T006 [P] Add trace parser unit tests for omitted metric expression, option-first metric expression, omitted distribution expression, option-first distribution expression, and explicit expression precedence in `tests/unit/test_trace_helpers.py`
- [X] T007 [P] Add run-duration unit tests for `duration`, `end_time - creation_time`, negative values, non-numeric values, missing metadata, and running status in `tests/unit/test_run_params.py`
- [X] T008 Define shared `DEFAULT_QUERY_EXPRESSION = "run.hash != ''"` and update `parse_query_invocation()` to support optional expressions without changing explicit-expression behavior in `src/aimx/commands/query.py`
- [X] T009 Import the shared default expression and update `parse_trace_invocation()` to support optional metric and distribution expressions without changing explicit-expression behavior in `src/aimx/commands/trace.py`
- [X] T010 Add `RunDuration` metadata and duration extraction helpers for Aim run objects in `src/aimx/aim_bridge/run_params.py`
- [X] T011 Run `uv run pytest tests/unit/test_query_helpers.py tests/unit/test_trace_helpers.py tests/unit/test_run_params.py -q` and fix foundational failures in `src/aimx/commands/query.py`, `src/aimx/commands/trace.py`, and `src/aimx/aim_bridge/run_params.py`

**Checkpoint**: Parser defaults and duration primitives are ready; user story implementation can start.

---

## Phase 3: User Story 1 - Query Runs Without Typing The Default Expression (Priority: P1) MVP

**Goal**: Users can run `aimx query metrics`, `aimx query images`, and
`aimx query params` without a query-language expression, and each command uses
`run.hash != ''` as the effective expression.

**Independent Test**: Run `uv run aimx query metrics --repo data`,
`uv run aimx query images --repo data --plain`, and
`uv run aimx query params --repo data`; confirm each behaves as though
`"run.hash != ''"` had been supplied.

### Tests for User Story 1

- [X] T012 [P] [US1] Add contract tests for omitted-expression `query metrics`, `query images`, and `query params` JSON/rich output effective expression in `tests/contract/test_query_contract.py`
- [X] T013 [P] [US1] Add integration tests comparing omitted-expression query commands to explicit `run.hash != ''` commands against `data` and `data/.aim` in `tests/integration/test_query_command.py`
- [X] T014 [P] [US1] Add unit tests proving option-first query invocations preserve existing option parsing for `--repo`, `--json`, `--plain`, `--steps`, `--epochs`, `--head`, `--tail`, `--every`, `--max-images`, and `--param` in `tests/unit/test_query_helpers.py`

### Implementation for User Story 1

- [X] T015 [US1] Update query usage and validation messages so expressions are optional but query target remains required in `src/aimx/commands/query.py`
- [X] T016 [US1] Ensure `run_query_command()` reports the effective default expression in query headers and JSON payloads through existing `header_info` in `src/aimx/commands/query.py`
- [X] T017 [US1] Preserve query option validation and target-specific restrictions for params, images, and metrics in `src/aimx/commands/query.py`
- [X] T018 [US1] Run `uv run pytest tests/unit/test_query_helpers.py tests/contract/test_query_contract.py tests/integration/test_query_command.py -q` and fix US1 failures in `src/aimx/commands/query.py`, `src/aimx/rendering/query_views.py`, and `src/aimx/rendering/params_views.py`

**Checkpoint**: User Story 1 is fully functional and independently testable.

---

## Phase 4: User Story 2 - Trace Runs Without Typing The Default Expression (Priority: P1)

**Goal**: Users can run `aimx trace` and `aimx trace distribution` without a
query-language expression, and each command uses `run.hash != ''` as the
effective expression.

**Independent Test**: Run `uv run aimx trace --repo data --head 5` and
`uv run aimx trace distribution --repo data --head 2 --no-color`; confirm both
behave as though `"run.hash != ''"` had been supplied.

### Tests for User Story 2

- [X] T019 [P] [US2] Add contract tests for omitted-expression metric trace default plot, JSON mode, and explicit-expression preservation in `tests/contract/test_trace_contract.py`
- [X] T020 [P] [US2] Add contract tests for omitted-expression distribution trace default visual mode and explicit `--table`, `--csv`, and `--json` modes in `tests/contract/test_trace_contract.py`
- [X] T021 [P] [US2] Add integration tests comparing omitted-expression trace commands to explicit `run.hash != ''` commands against `data` in `tests/integration/test_trace_command.py`
- [X] T022 [P] [US2] Add unit tests proving option-first trace invocations preserve existing option parsing for `--repo`, `--json`, `--table`, `--csv`, `--steps`, `--head`, `--tail`, `--every`, `--width`, `--height`, `--no-color`, and `--step` in `tests/unit/test_trace_helpers.py`

### Implementation for User Story 2

- [X] T023 [US2] Update trace usage and validation messages so metric and distribution expressions are optional while preserving the distribution subtarget in `src/aimx/commands/trace.py`
- [X] T024 [US2] Ensure `_execute_trace_pipeline()` receives the effective default expression for metric and distribution collectors in `src/aimx/commands/trace.py`
- [X] T025 [US2] Preserve trace no-match, no-data-in-step-range, invalid expression, invalid repo, and explicit mode behavior in `src/aimx/commands/trace.py`
- [X] T026 [US2] Run `uv run pytest tests/unit/test_trace_helpers.py tests/contract/test_trace_contract.py tests/integration/test_trace_command.py -q` and fix US2 failures in `src/aimx/commands/trace.py` and `src/aimx/rendering/trace_views.py`

**Checkpoint**: User Stories 1 and 2 both work independently.

---

## Phase 5: User Story 3 - Compare Params With Run Duration Visible (Priority: P2)

**Goal**: `aimx query params` displays each returned run's duration in rich,
plain, and JSON output, while keeping runs visible when duration is unavailable.

**Independent Test**: Run `uv run aimx query params --repo data`,
`uv run aimx query params --repo data --plain`, and
`uv run aimx query params --repo data --json`; confirm every returned run shows
a calculated duration or explicit unavailable/running status.

### Tests for User Story 3

- [X] T027 [P] [US3] Add unit tests for params duration formatting in rich/plain renderers and JSON serialization in `tests/unit/test_run_params.py`
- [X] T028 [P] [US3] Add contract tests for `query params --json` duration object shape and rich/plain duration visibility in `tests/contract/test_query_contract.py`
- [X] T029 [P] [US3] Add integration tests for sample-repo params duration output with omitted expression and explicit expression in `tests/integration/test_query_command.py`
- [X] T030 [P] [US3] Add unit tests proving selected params and missing params remain independent from duration metadata in `tests/unit/test_run_params.py`

### Implementation for User Story 3

- [X] T031 [US3] Extend `RunParams` to carry `RunDuration` without changing selected-key or missing-key semantics in `src/aimx/aim_bridge/run_params.py`
- [X] T032 [US3] Populate `RunDuration` during `collect_run_params()` from raw Aim run timing attributes before sorting rows in `src/aimx/aim_bridge/run_params.py`
- [X] T033 [US3] Add duration display helpers and a `DURATION` column to params rich output in `src/aimx/rendering/params_views.py`
- [X] T034 [US3] Add duration cells to params plain output and stable `duration` objects to params JSON output in `src/aimx/rendering/params_views.py`
- [X] T035 [US3] Run `uv run pytest tests/unit/test_run_params.py tests/contract/test_query_contract.py tests/integration/test_query_command.py -q` and fix US3 failures in `src/aimx/aim_bridge/run_params.py` and `src/aimx/rendering/params_views.py`

**Checkpoint**: All user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Complete discoverability, safety validation, and full regression
coverage across the feature.

- [X] T036 [P] Update owned-command help text with omitted-expression defaults for query and trace examples in `src/aimx/commands/help.py`
- [X] T037 [P] Update README query and trace examples to show omitted-expression forms and params duration output in `README.md`
- [X] T038 [P] Update quickstart verification notes if implementation behavior differs from planned examples in `specs/006-default-query-runtime/quickstart.md`
- [X] T039 Run quickstart sections 2-6 manually and record any deviations in `specs/006-default-query-runtime/quickstart.md`
- [X] T040 Run passthrough and owned-command regression tests with `uv run pytest tests/contract/test_cli_contract.py tests/integration/test_passthrough_behavior.py tests/integration/test_missing_native_aim.py tests/integration/test_missing_python_aim_package.py -q` and fix regressions in `src/aimx/router.py`, `src/aimx/cli.py`, `src/aimx/commands/query.py`, or `src/aimx/commands/trace.py`
- [X] T041 Run the full suite with `uv run pytest -q` and fix regressions in touched files under `src/aimx/` and `tests/`
- [X] T042 Update final implementation verification notes in `specs/006-default-query-runtime/checklists/requirements.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies; can start immediately.
- **Phase 2 Foundational**: Depends on Phase 1; blocks every user story.
- **Phase 3 US1**: Depends on Phase 2; MVP scope.
- **Phase 4 US2**: Depends on Phase 2; can proceed after shared default-expression parsing is stable.
- **Phase 5 US3**: Depends on Phase 2; can proceed after params collection and rendering paths from existing features are understood.
- **Phase 6 Polish**: Depends on whichever user stories are included in the delivery.

### User Story Dependencies

- **US1 (P1)**: First independently valuable slice for query commands; no dependency on US2 or US3.
- **US2 (P1)**: First independently valuable slice for trace commands; no dependency on US1 except the shared default-expression constant from Phase 2.
- **US3 (P2)**: Enhances params output with duration metadata; no dependency on US2 and can be validated independently through `query params`.

### Within Each User Story

- Write tests first and confirm they fail for the missing behavior.
- Parser behavior before command dispatch assertions.
- Bridge/data extraction before renderer assertions for params duration.
- Story-specific pytest command before moving to the next priority.

---

## Parallel Opportunities

- T002, T003, and T004 can run in parallel after T001 ownership is clear.
- T005, T006, and T007 can run in parallel because they touch different test files.
- T012, T013, and T014 can run in parallel for US1 test coverage.
- T019, T020, T021, and T022 can run in parallel for US2 test coverage.
- T027, T028, T029, and T030 can run in parallel for US3 test coverage.
- T036, T037, and T038 can run in parallel during polish because they touch different documentation files.

---

## Parallel Example: User Story 1

```text
Task: "T012 [P] [US1] Add contract tests for omitted-expression `query metrics`, `query images`, and `query params` JSON/rich output effective expression in tests/contract/test_query_contract.py"
Task: "T013 [P] [US1] Add integration tests comparing omitted-expression query commands to explicit `run.hash != ''` commands against `data` and `data/.aim` in tests/integration/test_query_command.py"
Task: "T014 [P] [US1] Add unit tests proving option-first query invocations preserve existing option parsing for `--repo`, `--json`, `--plain`, `--steps`, `--epochs`, `--head`, `--tail`, `--every`, `--max-images`, and `--param` in tests/unit/test_query_helpers.py"
```

## Parallel Example: User Story 2

```text
Task: "T019 [P] [US2] Add contract tests for omitted-expression metric trace default plot, JSON mode, and explicit-expression preservation in tests/contract/test_trace_contract.py"
Task: "T020 [P] [US2] Add contract tests for omitted-expression distribution trace default visual mode and explicit `--table`, `--csv`, and `--json` modes in tests/contract/test_trace_contract.py"
Task: "T021 [P] [US2] Add integration tests comparing omitted-expression trace commands to explicit `run.hash != ''` commands against `data` in tests/integration/test_trace_command.py"
Task: "T022 [P] [US2] Add unit tests proving option-first trace invocations preserve existing option parsing for `--repo`, `--json`, `--table`, `--csv`, `--steps`, `--head`, `--tail`, `--every`, `--width`, `--height`, `--no-color`, and `--step` in tests/unit/test_trace_helpers.py"
```

## Parallel Example: User Story 3

```text
Task: "T027 [P] [US3] Add unit tests for params duration formatting in rich/plain renderers and JSON serialization in tests/unit/test_run_params.py"
Task: "T028 [P] [US3] Add contract tests for `query params --json` duration object shape and rich/plain duration visibility in tests/contract/test_query_contract.py"
Task: "T029 [P] [US3] Add integration tests for sample-repo params duration output with omitted expression and explicit expression in tests/integration/test_query_command.py"
Task: "T030 [P] [US3] Add unit tests proving selected params and missing params remain independent from duration metadata in tests/unit/test_run_params.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational parser and duration primitives.
3. Complete Phase 3 US1.
4. Stop and validate `uv run aimx query metrics --repo data`, `uv run aimx query images --repo data --plain`, and `uv run aimx query params --repo data`.
5. Run US1 unit, contract, and integration tests before adding trace defaults or params duration rendering.

### Incremental Delivery

1. US1: deliver omitted-expression defaults for query commands.
2. US2: deliver omitted-expression defaults for trace commands.
3. US3: add params duration metadata and display.
4. Polish: update docs, run quickstart, and run full regression suite.

### Multi-Developer Coordination

- One developer owns `src/aimx/commands/query.py` while US1 parser and dispatch tasks are active.
- One developer owns `src/aimx/commands/trace.py` while US2 parser and dispatch tasks are active.
- One developer owns `src/aimx/aim_bridge/run_params.py`, `src/aimx/rendering/params_views.py`, and `tests/unit/test_run_params.py` while US3 duration tasks are active.
- Contract and integration tests can be split by `tests/contract/test_query_contract.py`, `tests/contract/test_trace_contract.py`, `tests/integration/test_query_command.py`, and `tests/integration/test_trace_command.py`.

---

## Notes

- `[P]` means the task can be parallelized only after its stated phase dependencies are satisfied.
- Story labels map directly to the spec user stories.
- Keep every command read-only; do not call Aim mutation APIs such as `run.set`, `track`, artifact logging, migration, or repair operations.
- Preserve existing `query metrics`, `query images`, `query params`, `trace`, `trace distribution`, and native Aim passthrough contracts throughout implementation.
- Commit after each phase or a small coherent task group when using the git hook workflow.
