# Tasks: Native Aim Passthrough

**Input**: Design documents from `/specs/001-aim-command-passthrough/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include the validation work that the feature and constitution
require. Even if broad test coverage is not requested, add tasks for
safety-critical checks such as passthrough fidelity, read-only guarantees,
failure handling, and CLI output contracts when relevant.

**Organization**: Tasks are grouped by user story to enable independent
implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths shown below assume single-project Python CLI structure from `plan.md`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Update packaging metadata, console entrypoint, and test/dev settings in `pyproject.toml`
- [X] T002 [P] Create the base package module in `src/aimx/__init__.py`
- [X] T003 [P] Create the owned-command package marker in `src/aimx/commands/__init__.py`
- [X] T004 [P] Create the native Aim integration package marker in `src/aimx/native_aim/__init__.py`
- [X] T005 [P] Create shared pytest fixtures for CLI invocation helpers in `tests/conftest.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create the process entrypoint module in `src/aimx/__main__.py`
- [X] T007 [P] Create the top-level CLI shell and exit handling skeleton in `src/aimx/cli.py`
- [X] T008 [P] Create the routing model and reserved-command registry in `src/aimx/router.py`
- [X] T009 [P] Create integration-test helpers for fake native Aim executables in `tests/integration/conftest.py`
- [X] T010 Create router-focused unit test scaffolding in `tests/unit/test_router.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Run Existing Aim Commands (Priority: P1) 🎯 MVP

**Goal**: Let existing Aim users run familiar native Aim commands through `aimx`
without changing the observable command workflow.

**Independent Test**: With native `aim` available or simulated, run
`aimx up`, `aimx init --help`, and `aimx runs --help` and confirm that
delegated argument order, output streams, and exit status stay aligned with the
native command behavior.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T011 [P] [US1] Add passthrough contract coverage for delegated invocation behavior in `tests/contract/test_cli_contract.py`
- [X] T012 [P] [US1] Add integration coverage for passthrough output and exit-status fidelity in `tests/integration/test_passthrough_behavior.py`

### Implementation for User Story 1

- [X] T013 [US1] Implement passthrough route classification and delegated argument forwarding in `src/aimx/router.py`
- [X] T014 [US1] Implement delegated native Aim process execution and status propagation in `src/aimx/native_aim/passthrough.py`
- [X] T015 [US1] Wire unowned command execution through the passthrough path in `src/aimx/cli.py`
- [X] T016 [US1] Finalize the CLI entrypoint handoff for delegated commands in `src/aimx/__main__.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Understand aimx Behavior Quickly (Priority: P2)

**Goal**: Make `aimx` self-explanatory by exposing a small owned command
surface and a clear mixed help experience.

**Independent Test**: Run `aimx`, `aimx --help`, `aimx version`, and
`aimx doctor` with native `aim` available and confirm the CLI explains owned
commands, passthrough behavior, and the resolved native Aim environment.

### Tests for User Story 2 ⚠️

- [X] T017 [P] [US2] Add owned-command contract coverage for help, version, and doctor in `tests/contract/test_cli_contract.py`
- [X] T018 [P] [US2] Add unit coverage for mixed help and version output in `tests/unit/test_owned_commands.py`

### Implementation for User Story 2

- [X] T019 [P] [US2] Implement the mixed help command output in `src/aimx/commands/help.py`
- [X] T020 [P] [US2] Implement Aimx and native Aim version reporting in `src/aimx/commands/version.py`
- [X] T021 [P] [US2] Implement the initial diagnostic report for available native Aim in `src/aimx/commands/doctor.py`
- [X] T022 [US2] Implement owned-command dispatch and reserved-command handling in `src/aimx/router.py`
- [X] T023 [US2] Integrate owned-command execution into the top-level CLI flow in `src/aimx/cli.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Get Actionable Errors When Aim Is Missing (Priority: P3)

**Goal**: Keep owned commands usable and make delegated-command failures clear
when native `aim` is missing or unusable.

**Independent Test**: Run `aimx --help`, `aimx version`, `aimx doctor`, and a
delegated command such as `aimx up` in an environment where native `aim` is
missing or unusable and confirm that owned commands still work while
passthrough fails fast with actionable guidance.

### Tests for User Story 3 ⚠️

- [X] T024 [P] [US3] Add unit coverage for missing and unusable native Aim resolution in `tests/unit/test_locator.py`
- [X] T025 [P] [US3] Add integration coverage for missing-Aim owned-command and passthrough behavior in `tests/integration/test_missing_native_aim.py`

### Implementation for User Story 3

- [X] T026 [US3] Implement native Aim discovery, availability states, and diagnostics in `src/aimx/native_aim/locator.py`
- [X] T027 [US3] Implement missing/unusable passthrough failures and exit-code mapping in `src/aimx/native_aim/passthrough.py`
- [X] T028 [US3] Update doctor diagnostics for missing or unusable native Aim in `src/aimx/commands/doctor.py`
- [X] T029 [US3] Keep owned commands available while blocking delegated commands when native Aim is unavailable in `src/aimx/cli.py`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T030 [P] Document the user-facing runtime contract and example usage in `README.md`
- [X] T031 [P] Expand router edge-case coverage for reserved commands and empty invocation behavior in `tests/unit/test_router.py`
- [X] T032 [P] Verify package metadata and entrypoint behavior stay aligned with the CLI contract in `pyproject.toml`
- [X] T033 Run the quickstart validation flow against `specs/001-aim-command-passthrough/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel if team capacity allows
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - no dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational - benefits from US1 CLI shell but remains independently testable
- **User Story 3 (P3)**: Can start after Foundational - extends diagnostic and passthrough behavior while remaining independently testable

### Within Each User Story

- Tests MUST be written and fail before implementation
- Routing and locator changes come before CLI integration work
- CLI integration comes before final story validation
- Story completion should be validated before moving to lower-priority polish work

### Parallel Opportunities

- Setup tasks `T002`-`T005` can run in parallel after `T001`
- Foundational tasks `T007`-`T009` can run in parallel after `T006`
- US1 test tasks `T011` and `T012` can run in parallel
- US2 implementation tasks `T019`-`T021` can run in parallel once tests are in place
- US3 test tasks `T024` and `T025` can run in parallel
- Polish tasks `T030`-`T032` can run in parallel after the stories are complete

---

## Parallel Example: User Story 1

```bash
# Launch the US1 verification work together:
Task: "Add passthrough contract coverage for delegated invocation behavior in tests/contract/test_cli_contract.py"
Task: "Add integration coverage for passthrough output and exit-status fidelity in tests/integration/test_passthrough_behavior.py"

# Then split the implementation by responsibility:
Task: "Implement passthrough route classification and delegated argument forwarding in src/aimx/router.py"
Task: "Implement delegated native Aim process execution and status propagation in src/aimx/native_aim/passthrough.py"
```

---

## Parallel Example: User Story 2

```bash
# Build the owned commands independently once the tests exist:
Task: "Implement the mixed help command output in src/aimx/commands/help.py"
Task: "Implement Aimx and native Aim version reporting in src/aimx/commands/version.py"
Task: "Implement the initial diagnostic report for available native Aim in src/aimx/commands/doctor.py"
```

---

## Parallel Example: User Story 3

```bash
# Separate native Aim detection from CLI-facing diagnostics:
Task: "Implement native Aim discovery, availability states, and diagnostics in src/aimx/native_aim/locator.py"
Task: "Add integration coverage for missing-Aim owned-command and passthrough behavior in tests/integration/test_missing_native_aim.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run the passthrough contract and integration checks for US1
5. Demo the companion-CLI passthrough behavior before moving on

### Incremental Delivery

1. Complete Setup + Foundational → routing foundation ready
2. Add User Story 1 → validate delegated native Aim behavior
3. Add User Story 2 → validate the mixed help and owned-command experience
4. Add User Story 3 → validate missing-Aim failure semantics
5. Finish polish tasks and quickstart validation

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 passthrough behavior
   - Developer B: User Story 2 owned commands
   - Developer C: User Story 3 missing-Aim diagnostics
3. Rejoin for final polish and quickstart validation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps each task to a specific user story
- Each user story remains independently completable and testable
- Constitution-driven validation is included for passthrough fidelity,
  read-only guarantees, ownership boundaries, and CLI output contracts
- Tests are scheduled before implementation within each story
- Avoid broad refactors that blur the owned-versus-delegated command boundary
