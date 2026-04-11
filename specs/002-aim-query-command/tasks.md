# Tasks: Aim Query Command

**Input**: Design documents from `/specs/002-aim-query-command/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include constitution-driven validation for owned-command routing, read-only query behavior, output contracts, repo-path compatibility, and passthrough non-regression.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the owned query feature documentation and help surface for implementation.

- [X] T001 Update feature-facing documentation baseline in /Users/blizhan/data/code/github/aimx/README.md to reserve space for the new `aimx query` command
- [X] T002 [P] Add query command placeholders and owned-command wording updates in /Users/blizhan/data/code/github/aimx/src/aimx/commands/help.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add the shared routing and query-support scaffolding required by all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Extend owned-command routing for `query` in /Users/blizhan/data/code/github/aimx/src/aimx/router.py
- [X] T004 [P] Add router coverage for the reserved `query` command in /Users/blizhan/data/code/github/aimx/tests/unit/test_router.py
- [X] T005 Create shared query invocation and repo-normalization helpers in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T006 [P] Add unit coverage for repo-path normalization and query argument validation in /Users/blizhan/data/code/github/aimx/tests/unit/test_query_helpers.py
- [X] T007 Wire the owned query command dispatch path into /Users/blizhan/data/code/github/aimx/src/aimx/cli.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Query Metric Runs From CLI (Priority: P1) 🎯 MVP

**Goal**: Let users query metric runs from the CLI with Aim-style expressions against a local repo.

**Independent Test**: `uv run aimx query metrics "metric.name == 'loss'" --repo data` and `--repo data/.aim` both succeed and report the same metric-run match count on the sample repository.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T008 [P] [US1] Add contract coverage for `query metrics` human-readable and JSON output in /Users/blizhan/data/code/github/aimx/tests/contract/test_query_contract.py
- [X] T009 [P] [US1] Add integration coverage for metric queries against `data` and `data/.aim` in /Users/blizhan/data/code/github/aimx/tests/integration/test_query_command.py

### Implementation for User Story 1

- [X] T010 [US1] Implement metric-query repository loading and result collection in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T011 [US1] Implement human-readable metric result rendering in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T012 [US1] Implement JSON metric result rendering with the shared output envelope in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T013 [US1] Update owned-command help text for the `query metrics` workflow in /Users/blizhan/data/code/github/aimx/src/aimx/commands/help.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Query Image Sequences And Structured Output (Priority: P2)

**Goal**: Let users query image sequences and consume stable machine-readable output from the CLI.

**Independent Test**: `uv run aimx query images "images" --repo data --json` succeeds and returns a stable structured envelope with non-empty rows from the sample repository.

### Tests for User Story 2 ⚠️

- [X] T014 [P] [US2] Extend structured-output and image-query contract coverage in /Users/blizhan/data/code/github/aimx/tests/contract/test_query_contract.py
- [X] T015 [P] [US2] Extend integration coverage for image queries against the sample repo in /Users/blizhan/data/code/github/aimx/tests/integration/test_query_command.py

### Implementation for User Story 2

- [X] T016 [US2] Implement image-query repository loading and result collection in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T017 [US2] Implement human-readable image result rendering in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T018 [US2] Reuse the structured query envelope for image results in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T019 [US2] Document image-query usage examples in /Users/blizhan/data/code/github/aimx/README.md

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Get Clear Guidance For Invalid Query Inputs (Priority: P3)

**Goal**: Fail invalid query invocations clearly and safely without mutating repository data.

**Independent Test**: Invalid expressions, unsupported targets, and invalid repo paths all fail with actionable messages while valid passthrough commands remain unaffected.

### Tests for User Story 3 ⚠️

- [X] T020 [P] [US3] Add contract coverage for query error messaging in /Users/blizhan/data/code/github/aimx/tests/contract/test_query_contract.py
- [X] T021 [P] [US3] Add integration coverage for invalid query expressions and invalid repo paths in /Users/blizhan/data/code/github/aimx/tests/integration/test_query_command.py

### Implementation for User Story 3

- [X] T022 [US3] Implement actionable query parse and repository-open failures in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T023 [US3] Add zero-result success messaging and unsupported-mode validation in /Users/blizhan/data/code/github/aimx/src/aimx/commands/query.py
- [X] T024 [US3] Update CLI help and usage guidance for query failures in /Users/blizhan/data/code/github/aimx/src/aimx/commands/help.py

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate non-regression, documentation completeness, and feature quickstart flow across stories.

- [X] T025 [P] Preserve passthrough contract coverage while adding query ownership in /Users/blizhan/data/code/github/aimx/tests/contract/test_cli_contract.py
- [X] T026 [P] Preserve missing-native-Aim behavior for non-query owned and passthrough commands in /Users/blizhan/data/code/github/aimx/tests/integration/test_missing_native_aim.py
- [X] T027 [P] Preserve passthrough argument and exit-code fidelity in /Users/blizhan/data/code/github/aimx/tests/integration/test_passthrough_behavior.py
- [X] T028 Run the feature quickstart verification commands from /Users/blizhan/data/code/github/aimx/specs/002-aim-query-command/quickstart.md and capture any needed documentation fixes in /Users/blizhan/data/code/github/aimx/README.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational - no dependency on other stories
- **User Story 2 (P2)**: Starts after Foundational - depends on shared query scaffolding from Phases 1-2 but not on US1 completion
- **User Story 3 (P3)**: Starts after Foundational - may reuse query implementation from US1/US2 but remains independently testable

### Within Each User Story

- Tests MUST be written and fail before implementation
- Shared query helpers and routing must exist before story-specific command behavior
- Result loading must precede rendering refinements
- Story help/documentation updates follow working command behavior

### Parallel Opportunities

- `T001` and `T002` can proceed in parallel
- `T004` and `T006` can proceed in parallel with their corresponding implementation tasks once stubs exist
- Within each story, contract and integration tests can be written in parallel
- Polish non-regression tasks `T025`, `T026`, and `T027` can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch User Story 1 tests together:
Task: "Add contract coverage for `query metrics` human-readable and JSON output in tests/contract/test_query_contract.py"
Task: "Add integration coverage for metric queries against `data` and `data/.aim` in tests/integration/test_query_command.py"

# After shared scaffolding exists, implement and document in parallel:
Task: "Implement metric-query repository loading and result collection in src/aimx/commands/query.py"
Task: "Update owned-command help text for the `query metrics` workflow in src/aimx/commands/help.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run metric-query checks against `data` and `data/.aim`

### Incremental Delivery

1. Complete Setup + Foundational
2. Add User Story 1 and validate metric queries
3. Add User Story 2 and validate image queries plus JSON output
4. Add User Story 3 and validate failure handling
5. Finish with passthrough non-regression and quickstart validation

### Parallel Team Strategy

1. One contributor completes Phase 2 routing and shared query scaffolding
2. After Phase 2:
   - Contributor A: User Story 1
   - Contributor B: User Story 2
   - Contributor C: User Story 3 test design and error handling path
3. Merge into Phase 6 non-regression validation

---

## Notes

- [P] tasks = different files, no dependencies on unfinished work
- [Story] labels map tasks directly to the feature specification stories
- The feature must remain read-only and must not mutate Aim repository data
- Repo-path compatibility with both `data` and `data/.aim` is part of done-ness
- Existing passthrough behavior is a protected contract and must be re-verified before completion
