# Tasks: Run Params Query And Experiment Comparison

**Input**: Design documents from `/Users/blizhan/data/code/github/aimx/specs/004-run-params-query/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-output.md, quickstart.md

**Tests**: Test tasks are included because the feature changes an owned CLI
surface and the constitution requires validation for output contracts,
read-only behavior, safe failure modes, and passthrough non-regression.

**Organization**: Tasks are grouped by user story so each story can be
implemented and tested as an independently useful increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or depends
  only on completed foundation work
- **[Story]**: Maps task to a user story (`US1`, `US2`, `US3`)
- Every task includes exact repository-relative file paths

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the new params-specific modules and documentation touch
points without changing behavior yet.

- [X] T001 Create empty implementation modules with module docstrings in `src/aimx/aim_bridge/run_params.py` and `src/aimx/rendering/params_views.py`
- [X] T002 [P] Create the params unit-test placeholder file `tests/unit/test_run_params.py`
- [X] T003 [P] Review and preserve current owned-query behavior references in `src/aimx/commands/query.py`, `src/aimx/rendering/query_views.py`, and `tests/contract/test_query_contract.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add shared parser and params-data primitives required by every
user story.

**Critical**: No user story work should begin until this phase is complete.

- [X] T004 [P] Add parser unit tests for `params` target defaults, repeatable `--param`, missing `--param` value, empty `--param` value, duplicate `--param`, and `--param` rejection on `metrics`/`images` in `tests/unit/test_query_helpers.py`
- [X] T005 Extend `QueryInvocation` and `parse_query_invocation` with `params` target and `param_keys` validation in `src/aimx/commands/query.py`
- [X] T006 [P] Add unit tests for nested params flattening, scalar preservation, non-scalar preservation, and deterministic key ordering in `tests/unit/test_run_params.py`
- [X] T007 Implement `RunParams`, `flatten_params`, and deterministic params key helpers in `src/aimx/aim_bridge/run_params.py`
- [X] T008 Run `uv run pytest tests/unit/test_query_helpers.py tests/unit/test_run_params.py -q` and fix failures in `src/aimx/commands/query.py` and `src/aimx/aim_bridge/run_params.py`

**Checkpoint**: Parser and params primitives are ready; user story implementation can start.

---

## Phase 3: User Story 1 - Query Run Parameters From CLI (Priority: P1) MVP

**Goal**: Users can run a read-only params query and see matching run identity,
experiment label, run name, and parameter values.

**Independent Test**: Run `uv run aimx query params "run.hash != ''" --repo data`
and confirm the output lists matching runs with params or an explicit no-params
marker.

### Tests for User Story 1

- [X] T009 [P] [US1] Add contract tests for default `query params` rich output and JSON envelope shape in `tests/contract/test_query_contract.py`
- [X] T010 [P] [US1] Add integration tests for sample-repo `query params "run.hash != ''" --repo data` and `--repo data/.aim` equivalence in `tests/integration/test_query_command.py`
- [X] T011 [P] [US1] Add integration tests for zero-match params queries and runs with no params in `tests/integration/test_query_command.py`

### Implementation for User Story 1

- [X] T012 [US1] Implement read-only `collect_run_params()` using `Repo.query_runs(..., QueryReportMode.DISABLED)` and existing short-hash expansion in `src/aimx/aim_bridge/run_params.py`
- [X] T013 [US1] Implement params rich, plain, and JSON renderers for default all-param output in `src/aimx/rendering/params_views.py`
- [X] T014 [US1] Wire `params` dispatch through `run_query_command()` and a new `_run_params_query()` branch in `src/aimx/commands/query.py`
- [X] T015 [US1] Ensure missing repositories and invalid params query expressions return exit code `2` with actionable stderr messages via `src/aimx/commands/query.py`
- [X] T016 [US1] Run `uv run pytest tests/contract/test_query_contract.py tests/integration/test_query_command.py -q` and fix US1 failures in `src/aimx/aim_bridge/run_params.py`, `src/aimx/rendering/params_views.py`, and `src/aimx/commands/query.py`

**Checkpoint**: User Story 1 is fully functional and independently testable.

---

## Phase 4: User Story 2 - Compare Selected Parameters Across Runs (Priority: P1)

**Goal**: Users can focus params output on selected parameter names and see
explicit missing-value markers across runs.

**Independent Test**: Run `uv run aimx query params "run.experiment == 'cloud-segmentation'" --repo data --param hparam.lr --param hparam.optimizer --param hparam.weight_decay`
and confirm the selected keys align across matching runs with `-` for missing
values.

### Tests for User Story 2

- [X] T017 [P] [US2] Add unit tests for selected-key filtering and missing-key tracking in `tests/unit/test_run_params.py`
- [X] T018 [P] [US2] Add contract tests for `--param` JSON `param_keys`, `params`, and `missing_params` fields in `tests/contract/test_query_contract.py`
- [X] T019 [P] [US2] Add integration tests for three selected params across at least three sample-repo runs in `tests/integration/test_query_command.py`

### Implementation for User Story 2

- [X] T020 [US2] Implement selected-key filtering and `missing_keys` calculation in `src/aimx/aim_bridge/run_params.py`
- [X] T021 [US2] Update params renderers to align selected keys and render missing selected values as `-` in `src/aimx/rendering/params_views.py`
- [X] T022 [US2] Pass `invocation.param_keys` into `collect_run_params()` from `_run_params_query()` in `src/aimx/commands/query.py`
- [X] T023 [US2] Run `uv run pytest tests/unit/test_run_params.py tests/contract/test_query_contract.py tests/integration/test_query_command.py -q` and fix US2 failures in `src/aimx/aim_bridge/run_params.py`, `src/aimx/rendering/params_views.py`, and `src/aimx/commands/query.py`

**Checkpoint**: User Stories 1 and 2 both work independently.

---

## Phase 5: User Story 3 - Compare Runs By Experiment Name (Priority: P2)

**Goal**: Params results include experiment labels, support Aim expression
filtering by experiment name, and sort/group results so experiment comparisons
are easy to scan.

**Independent Test**: Run `uv run aimx query params "run.experiment == 'cloud-segmentation'" --repo data --param hparam.lr`
and confirm only matching experiment rows are returned, with stable experiment
labels in human-readable and JSON output.

### Tests for User Story 3

- [X] T024 [P] [US3] Add unit tests for experiment-aware sorting with missing, empty, duplicate, and case-varied experiment labels in `tests/unit/test_run_params.py`
- [X] T025 [P] [US3] Add integration tests for experiment-name filtering and experiment-label presence in params JSON/plain/rich outputs in `tests/integration/test_query_command.py`
- [X] T026 [P] [US3] Add contract tests proving `query metrics` and `query images` reject `--param` while existing output envelopes remain unchanged in `tests/contract/test_query_contract.py`

### Implementation for User Story 3

- [X] T027 [US3] Implement experiment-aware params result sorting by experiment label, run name, then run hash in `src/aimx/aim_bridge/run_params.py`
- [X] T028 [US3] Update params renderers to keep experiment labels visible in rich, plain, and JSON output in `src/aimx/rendering/params_views.py`
- [X] T029 [US3] Update owned-command help text with `query params` and experiment-filter examples in `src/aimx/commands/help.py`
- [X] T030 [US3] Run `uv run pytest tests/unit/test_run_params.py tests/integration/test_query_command.py tests/contract/test_query_contract.py -q` and fix US3 failures in `src/aimx/aim_bridge/run_params.py`, `src/aimx/rendering/params_views.py`, `src/aimx/commands/query.py`, and `src/aimx/commands/help.py`

**Checkpoint**: All user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Complete discoverability, safety validation, and full regression
coverage across the feature.

- [X] T031 [P] Update README params-query examples for rich, `--plain`, `--json`, `--param`, and experiment filtering in `README.md`
- [X] T032 [P] Update quickstart verification notes if implementation behavior differs from planned examples in `specs/004-run-params-query/quickstart.md`
- [X] T033 Run quickstart sections 2-7 manually and record any deviations in `specs/004-run-params-query/quickstart.md`
- [X] T034 Run passthrough and owned-command regression tests with `uv run pytest tests/contract/test_cli_contract.py tests/integration/test_missing_native_aim.py tests/integration/test_missing_python_aim_package.py -q` and fix regressions in `src/aimx/router.py`, `src/aimx/cli.py`, or `src/aimx/commands/query.py`
- [X] T035 Run the full suite with `uv run pytest -q` and fix any regressions in touched files under `src/aimx/` and `tests/`
- [X] T036 Update `specs/004-run-params-query/checklists/requirements.md` with final implementation verification notes for params query readiness

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies; can start immediately.
- **Phase 2 Foundational**: Depends on Phase 1; blocks every user story.
- **Phase 3 US1**: Depends on Phase 2; MVP scope.
- **Phase 4 US2**: Depends on Phase 2 and can start after US1 render/collection shapes exist.
- **Phase 5 US3**: Depends on Phase 2 and benefits from US1/US2 renderer coverage.
- **Phase 6 Polish**: Depends on whichever user stories are included in the delivery.

### User Story Dependencies

- **US1 (P1)**: First independently valuable slice; no dependency on US2 or US3.
- **US2 (P1)**: Builds on the same params target and parser foundation; requires `param_keys` plumbing from Phase 2 and can be implemented after or alongside late US1 renderer work with coordination.
- **US3 (P2)**: Uses the params rows from US1 and selected-key behavior from US2 for richer comparison, but experiment filtering itself is independently testable through Aim expressions.

### Within Each User Story

- Write tests first and confirm they fail for the missing behavior.
- Implement bridge/data extraction before renderers.
- Implement renderers before command dispatch assertions.
- Run the story-specific pytest command before moving to the next story.

---

## Parallel Opportunities

- T002 and T003 can run in parallel with T001 once file ownership is clear.
- T004 and T006 can run in parallel because they touch different test files.
- T009, T010, and T011 can run in parallel for US1 test coverage.
- T017, T018, and T019 can run in parallel for US2 test coverage.
- T024, T025, and T026 can run in parallel for US3 test coverage.
- T031 and T032 can run in parallel during polish because they touch different docs.

---

## Parallel Example: User Story 1

```text
Task: "T009 [P] [US1] Add contract tests for default `query params` rich output and JSON envelope shape in tests/contract/test_query_contract.py"
Task: "T010 [P] [US1] Add integration tests for sample-repo `query params \"run.hash != ''\" --repo data` and `--repo data/.aim` equivalence in tests/integration/test_query_command.py"
Task: "T011 [P] [US1] Add integration tests for zero-match params queries and runs with no params in tests/integration/test_query_command.py"
```

## Parallel Example: User Story 2

```text
Task: "T017 [P] [US2] Add unit tests for selected-key filtering and missing-key tracking in tests/unit/test_run_params.py"
Task: "T018 [P] [US2] Add contract tests for `--param` JSON `param_keys`, `params`, and `missing_params` fields in tests/contract/test_query_contract.py"
Task: "T019 [P] [US2] Add integration tests for three selected params across at least three sample-repo runs in tests/integration/test_query_command.py"
```

## Parallel Example: User Story 3

```text
Task: "T024 [P] [US3] Add unit tests for experiment-aware sorting with missing, empty, duplicate, and case-varied experiment labels in tests/unit/test_run_params.py"
Task: "T025 [P] [US3] Add integration tests for experiment-name filtering and experiment-label presence in params JSON/plain/rich outputs in tests/integration/test_query_command.py"
Task: "T026 [P] [US3] Add contract tests proving `query metrics` and `query images` reject `--param` while existing output envelopes remain unchanged in tests/contract/test_query_contract.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational parser and params primitives.
3. Complete Phase 3 US1.
4. Stop and validate `uv run aimx query params "run.hash != ''" --repo data`.
5. Run US1 contract and integration tests before adding selection or experiment-specific polish.

### Incremental Delivery

1. US1: deliver default params discovery and all-output-mode support.
2. US2: add focused comparison via repeatable `--param`.
3. US3: improve experiment-name comparison, sorting, and discoverability.
4. Polish: update docs, run quickstart, and run full regression suite.

### Multi-Developer Coordination

- One developer owns `src/aimx/commands/query.py` during parser/dispatch tasks to avoid conflicts.
- One developer can own `src/aimx/aim_bridge/run_params.py` and `tests/unit/test_run_params.py`.
- One developer can own `src/aimx/rendering/params_views.py` and output contract tests.
- Documentation tasks can proceed after the CLI shape is stable.

---

## Notes

- `[P]` means the task can be parallelized only after its stated phase
  dependencies are satisfied.
- Story labels map directly to the spec user stories.
- Keep the command read-only; do not call Aim mutation APIs such as `run.set`,
  `track`, artifact logging, migration, or repair operations.
- Preserve existing `query metrics`, `query images`, `trace`, and native Aim
  passthrough contracts throughout implementation.
- Commit after each phase or a small coherent task group when using the git
  hook workflow.
