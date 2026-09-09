# Tasks: Agent-First AutoResearch Control Plane

**Input**: Design documents from `/Users/blizhan/data/code/github/aimx/specs/007-research-state/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: This feature adds an explicit write surface and stable agent-facing contracts, so the constitution-driven safety, atomicity, read-only, conflict, deterministic-output, and passthrough tests below are required. Story tests should be written before the corresponding implementation and observed failing for the intended reason.

**Organization**: Tasks are grouped by user story so durable state can ship as an MVP before context, frontier, agenda, and the full AutoResearch loop are layered on top.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it changes a different file and does not depend on incomplete work in the same phase.
- **[Story]**: Maps directly to the numbered user stories in `spec.md`.
- Every task names the file it creates or changes.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the research subsystem package and reusable test scaffolding without changing existing Aim behavior.

- [X] T001 Create the research subsystem package exports and public module boundary in `src/aimx/research/__init__.py`
- [X] T002 [P] Add reusable temporary-repository, empty-state, revision, and ResearchUpdate fixture helpers that do not pre-create `.aimx` in `tests/conftest.py`
- [X] T003 [P] Add shared research error/result types for invalid input, unreadable store, revision conflict, and budget failures in `src/aimx/research/errors.py`

**Checkpoint**: The package imports cleanly and test fixtures can represent both a fresh repo with no `.aimx` and an explicitly initialized research store.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement the storage/event/domain primitives and CLI ownership boundary required by every user story.

**⚠️ CRITICAL**: No user-story implementation should begin until these primitives are in place.

- [X] T004 [P] Define immutable IDs, prefixes, local-reference parsing, and ID allocation helpers for commits/events/findings/relations/annotations/frontier/agenda entities in `src/aimx/research/ids.py`
- [X] T005 [P] Define V1 domain enums and immutable dataclasses for Author, EvidenceReference, Finding, Relation, Annotation, FrontierLane, FrontierItem, AgendaItem, ResearchState, and ResearchUpdate in `src/aimx/research/models.py`
- [X] T006 Implement supported ResearchUpdate operation/event types plus canonical JSON serialization/deserialization in `src/aimx/research/events.py`
- [X] T007 Implement SQLite schema creation-on-explicit-write, read-only existing-store open, missing-store empty-state detection, metadata revision access, commit/event reads, and transaction primitives in `src/aimx/research/store.py`
- [X] T008 Implement ordered commit/event replay into a deterministic `ResearchState` projection with revision tracking and unknown-event/schema failure handling in `src/aimx/research/replay.py`
- [X] T009 Implement the shared V1 validation pipeline skeleton for schema version, operation shape, durable/local entity references, duplicate local IDs, and state-transition dispatch in `src/aimx/research/validation.py`
- [X] T010 [P] Implement stable human/JSON success and error envelope helpers, UTC timestamp formatting, and deterministic entity ordering utilities in `src/aimx/rendering/research_views.py`
- [X] T011 Reserve `research`, `finding`, `lineage`, and `frontier` as Aimx-owned command roots while preserving all other passthrough routing in `src/aimx/router.py`
- [X] T012 Extend owned-command dispatch plumbing for the four new roots without changing query/trace/help/version/doctor behavior in `src/aimx/cli.py`
- [X] T013 Add ownership-boundary regression tests proving the new roots are owned and unknown roots preserve delegated argv unchanged in `tests/unit/test_router.py`

**Checkpoint**: Foundation can open/replay empty or populated Aimx research state, but no story-specific command needs to be usable yet.

---

## Phase 3: User Story 1 - Preserve Research Knowledge Across Agent Sessions (Priority: P1) 🎯 MVP

**Goal**: Persist evidence-grounded Findings, Lineage, Annotations, assessments, and governance as atomic, immutable-history ResearchUpdates that survive agent/session changes.

**Independent Test**: Commit a multi-operation update grounded in Aim runs, start a new process/session, read the same durable state, then prove invalid/stale updates leave no partial state and changed claim meaning is represented by a new Finding plus lineage rather than overwrite.

### Tests for User Story 1

- [X] T014 [P] [US1] Add store tests for missing-store reads, explicit-write initialization, contiguous revisions, `BEGIN IMMEDIATE` atomic rollback, stale `base_revision`, idempotent `client_update_id`, and read-only open semantics in `tests/unit/test_research_store.py`
- [X] T015 [P] [US1] Add replay tests for finding creation, independent epistemic/governance changes, annotations, relation creation/retraction, immutable claim history, and deterministic ordering in `tests/unit/test_research_replay.py`
- [X] T016 [P] [US1] Add ResearchUpdate semantic-validation tests for local refs, unknown refs, duplicate local IDs, illegal transitions, claim-overwrite rejection, and all-or-nothing operation validation in `tests/unit/test_research_validation.py`
- [X] T017 [P] [US1] Add Aim evidence tests for full hashes, unambiguous short hashes, unknown hashes, ambiguous prefixes, canonical persistence, and read-only repository access in `tests/unit/test_research_evidence.py`
- [X] T018 [P] [US1] Add contract tests for `research state`/`research update`, dry-run, created-ID maps, schema/revision fields, exit codes 0/2/3, and stable JSON error codes in `tests/contract/test_research_contract.py`
- [X] T019 [P] [US1] Add end-to-end command tests proving a second CLI process sees prior Findings/Lineage/Annotations and proving research reads/writes never modify `.aim` in `tests/integration/test_research_commands.py`

### Implementation for User Story 1

- [X] T020 [US1] Implement read-only Aim run-reference canonicalization and existence/ambiguity validation without copying run payloads in `src/aimx/aim_bridge/research_evidence.py`
- [X] T021 [US1] Complete V1 Finding/Relation/Annotation/assessment/governance validation, local-reference resolution, evidence canonicalization, and immutable-claim rules in `src/aimx/research/validation.py`
- [X] T022 [US1] Implement atomic ResearchUpdate commit application with revision conflict detection, client-update idempotency, ordered event insertion, rollback, and dry-run no-write behavior in `src/aimx/research/store.py`
- [X] T023 [US1] Complete Finding/Relation/Annotation event replay, relation retraction, assessment/governance evolution, and provenance/history projection in `src/aimx/research/replay.py`
- [X] T024 [US1] Implement state/update/dry-run/revision-conflict human and JSON renderers matching `contracts/cli-output.md` in `src/aimx/rendering/research_views.py`
- [X] T025 [US1] Implement parsing and execution for `aimx research state` and `aimx research update --stdin|--file [--dry-run] [--json]` in `src/aimx/commands/research.py`
- [X] T026 [US1] Implement `finding ls|show|comment|accept|reject|assess` as read operations or one-operation ResearchUpdates using the latest observed revision without silent conflict retry in `src/aimx/commands/finding.py`
- [X] T027 [US1] Implement `lineage show|link|retract` with typed relation validation, preserved provenance, and explicit retraction history in `src/aimx/commands/lineage.py`
- [X] T028 [US1] Wire the Research/Finding/Lineage handlers into owned CLI dispatch and preserve result/stdout/stderr/exit-status behavior in `src/aimx/cli.py`

**Checkpoint**: User Story 1 is deployable as the MVP: an external agent can persist and recover durable research knowledge while Aim remains read-only.

---

## Phase 4: User Story 2 - Resume Research From Bounded Shared Context (Priority: P1)

**Goal**: Compile a deterministic objective-specific Research Context that fits an exact UTF-8 byte budget and never hides a materially linked contradiction.

**Independent Test**: Populate relevant/irrelevant/supporting/challenging findings plus human annotations, request the same objective/budget twice, and verify identical substantive items, exact budget accounting, canonical evidence refs, and explicit budget failure when mandatory contradiction closure cannot fit.

### Tests for User Story 2

- [X] T029 [P] [US2] Add deterministic lexical relevance, governance/epistemic weighting, human-steering priority, graph-neighborhood, and stable tie-break tests in `tests/unit/test_research_context.py`
- [X] T030 [US2] Add exact UTF-8 `items` budget accounting, challenge/refine/supersede closure, minimum-required-budget error, and evidence-ref deduplication cases in `tests/unit/test_research_context.py`
- [X] T031 [US2] Extend context contract tests for `compiler_version`, revision/objective/budget fields, deterministic item ordering, and `budget_too_small_for_required_context` error shape in `tests/contract/test_research_contract.py`
- [X] T032 [US2] Extend integration coverage so context includes durable human annotations, materially relevant contradictions, and Aim run refs while irrelevant history is omitted under budget in `tests/integration/test_research_commands.py`

### Implementation for User Story 2

- [X] T033 [US2] Implement normalized lexical tokenization/scoring plus governance, epistemic, annotation, relation, and stable-ID ranking signals in `src/aimx/research/context.py`
- [X] T034 [US2] Implement one-hop structural closure, mandatory contradiction/evolution packing groups, exact UTF-8 byte packing, budget failure reporting, and evidence-ref deduplication in `src/aimx/research/context.py`
- [X] T035 [US2] Add Research Context JSON/human rendering with selection notes and exact budget metadata in `src/aimx/rendering/research_views.py`
- [X] T036 [US2] Implement `aimx research context --objective <text> --budget <bytes> [--json]` parsing, empty-state behavior, compiler invocation, and expected error mapping in `src/aimx/commands/research.py`

**Checkpoint**: User Stories 1 and 2 allow an external agent to persist research knowledge and later resume from a bounded deterministic context.

---

## Phase 5: User Story 3 - Steer The Research Frontier (Priority: P2)

**Goal**: Persist project-defined frontier lanes and research directions so agent proposals and human steering affect subsequent research context.

**Independent Test**: Create custom lanes, add a finding/direction to the frontier, move/pause/retire it as a human, and verify the frontier retains rationale/provenance plus the updated active state appears in subsequent context.

### Tests for User Story 3

- [X] T037 [P] [US3] Add frontier validation/replay tests for custom lane uniqueness, lane retirement, finding/direction subjects, priority range, status changes, move rules, provenance, and historical retention in `tests/unit/test_research_frontier.py`
- [X] T038 [P] [US3] Add contract tests for `frontier show|lane-add|add|move|pause|retire`, deterministic JSON output, invalid lane/entity errors, and revision conflicts in `tests/contract/test_frontier_contract.py`
- [X] T039 [P] [US3] Add integration coverage proving a human frontier move/pause/retire persists across processes and changes later Research Context selection without re-entering prompt instructions in `tests/integration/test_research_frontier.py`

### Implementation for User Story 3

- [X] T040 [US3] Implement FrontierLane/FrontierItem operation validation, active-lane rules, priority/status constraints, and linked finding checks in `src/aimx/research/validation.py`
- [X] T041 [US3] Implement frontier lane/item create/move/priority/status/update event replay with retired-history preservation in `src/aimx/research/replay.py`
- [X] T042 [US3] Implement `frontier show|lane-add|add|move|pause|retire` and compile every write command through the shared ResearchUpdate path in `src/aimx/commands/frontier.py`
- [X] T043 [US3] Add active frontier relevance/provenance signals and selected frontier artifacts to bounded context scoring/packing in `src/aimx/research/context.py`
- [X] T044 [US3] Add deterministic human/JSON frontier rendering for lanes, active/paused/retired items, rationale, priority, provenance, and linked knowledge in `src/aimx/rendering/research_views.py`
- [X] T045 [US3] Wire the Frontier handler into owned CLI dispatch with expected conflict/error exit codes in `src/aimx/cli.py`

**Checkpoint**: Frontier becomes a durable human-intervention surface; context can now reflect search-policy steering rather than only accumulated findings.

---

## Phase 6: User Story 4 - Produce A Durable Research Agenda (Priority: P2)

**Goal**: Persist agent/human-proposed Experiment Contracts and expose deterministic agenda/next selection without Aimx inventing experiments or executing them.

**Independent Test**: Create Agenda Items motivated by Findings or Frontier Items, exercise valid lifecycle transitions and evidence attachment, then verify `research next` deterministically selects active before proposed work and returns a successful no-work envelope when nothing actionable remains.

### Tests for User Story 4

- [X] T046 [P] [US4] Add Agenda Item validation/replay tests for required motivation, optional hypothesis/question, controls/factors/constraints/success criteria, priority range, lifecycle transitions, evidence attachment, and result provenance in `tests/unit/test_research_agenda.py`
- [X] T047 [US4] Add deterministic `research next` tests for active-before-proposed priority, priority ordering, creation-revision/ID tie breaks, and no-actionable-agenda behavior in `tests/unit/test_research_agenda.py`
- [X] T048 [P] [US4] Add agenda/next contract tests for status filters, full Experiment Contract JSON shape, no-work success, stable ordering, and invalid transition errors in `tests/contract/test_agenda_contract.py`
- [X] T049 [P] [US4] Add integration coverage proving agenda lifecycle/evidence links remain traceable and frontier/human steering changes later agenda inspection/selection in `tests/integration/test_research_agenda.py`

### Implementation for User Story 4

- [X] T050 [US4] Implement Agenda Item create/transition/evidence/priority semantic validation including motivation and lifecycle transition rules in `src/aimx/research/validation.py`
- [X] T051 [US4] Implement agenda create/transition/evidence/priority event replay and result-commit provenance projection in `src/aimx/research/replay.py`
- [X] T052 [US4] Implement deterministic agenda filtering/sorting and `next` selection policy without hypothesis generation or experiment execution in `src/aimx/research/agenda.py`
- [X] T053 [US4] Add agenda/next human and JSON rendering including successful `no_actionable_agenda` envelope in `src/aimx/rendering/research_views.py`
- [X] T054 [US4] Implement `aimx research agenda [--status]` and `aimx research next` command parsing, read-only behavior, and deterministic selection in `src/aimx/commands/research.py`
- [X] T055 [US4] Include relevant active Agenda Items and motivation links in Research Context without changing deterministic budget semantics in `src/aimx/research/context.py`

**Checkpoint**: An external agent can now read durable next-experiment contracts and track their lifecycle while execution remains outside Aimx.

---

## Phase 7: User Story 5 - Complete A Multi-Round AutoResearch Loop (Priority: P1 Integration)

**Goal**: Prove the state/context/frontier/agenda contracts compose into a tool-neutral two-round AutoResearch protocol using existing Aim evidence observation.

**Independent Test**: Complete two rounds of `context -> next/agenda -> external experiment -> Aim evidence observation -> research update`; verify round two consumes round-one knowledge/steering, then switch to a second agent adapter/process using the same JSON contracts without product-specific state.

### Tests for User Story 5

- [X] T056 [P] [US5] Add a two-round end-to-end AutoResearch integration test covering context, agenda selection, simulated external experiment evidence, ResearchUpdate, human intervention, and second-round state reuse in `tests/integration/test_autoresearch_loop.py`
- [X] T057 [P] [US5] Add agent-neutral contract fixtures proving two distinct caller identities can consume the same state/context/agenda JSON and submit compatible ResearchUpdates without product-specific fields in `tests/contract/test_autoresearch_protocol.py`
- [X] T058 [P] [US5] Add regression coverage proving the existing snapshot collector still performs read-only evidence collection and does not write Research State in `tests/integration/test_autoresearch_snapshot.py`

### Implementation for User Story 5

- [X] T059 [US5] Document the agent-neutral `context -> next/agenda -> execute externally -> observe Aim evidence -> update -> repeat` protocol, revision-conflict handling, and handoff between agent products in `skills/aimx/references/autoresearch-protocol.md`
- [X] T060 [US5] Update the Aimx agent skill to use Research Context/Agenda/ResearchUpdate as durable memory while retaining `collect_experiment_snapshot.py` as the read-only Observe step in `skills/aimx/SKILL.md`
- [X] T061 [US5] Align snapshot documentation/output guidance with the new ResearchUpdate handoff without making the collector write state in `skills/aimx/scripts/collect_experiment_snapshot.py`

**Checkpoint**: The complete two-round AutoResearch loop works across sessions/caller identities without Aimx hosting an agent or scheduler.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Finish documentation, safety regression, performance validation, and existing-command compatibility across all stories.

- [X] T062 [P] Document all new owned command families, explicit-write/read-only boundaries, empty-state behavior, and AutoResearch workflow examples in `src/aimx/commands/help.py`
- [X] T063 [P] Add user-facing AutoResearch quickstart, `.aimx` ownership/removability, and Agent/Aimx/Aim architecture guidance in `README.md`
- [X] T064 [P] Add scale tests for approximately 10,000 commits / 50,000 events covering replay and bounded-context compilation against the plan's interactive target in `tests/unit/test_research_scale.py`
- [X] T065 Add CLI contract regression assertions that legacy query/trace/help/version/doctor commands keep their existing behavior and new research roots do not capture unrelated native Aim commands in `tests/contract/test_cli_contract.py`
- [X] T066 Run and fix the complete existing passthrough/query/trace integration regression suite without weakening assertions in `tests/integration/test_passthrough_behavior.py`
- [X] T067 Execute every command sequence and safety assertion in the feature quickstart, updating any drift between documented and implemented behavior in `specs/007-research-state/quickstart.md`
- [X] T068 Reconcile final implemented CLI/error/schema behavior with the design contract and update only intentional contract changes in `specs/007-research-state/contracts/cli-output.md`
- [X] T069 Validate `research-update.schema.json` remains synchronized with accepted V1 operation names/fields and examples in `specs/007-research-state/contracts/research-update.schema.json`
- [X] T070 Run the full `uv run pytest` suite and resolve all regressions without adding runtime dependencies or `.aim` mutation in `pyproject.toml`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational**: Depends on Phase 1 and blocks every user story.
- **Phase 3 US1**: Depends on Phase 2 and is the MVP foundation for all durable Research State consumers.
- **Phase 4 US2**: Depends on US1 because context compiles persisted Research State.
- **Phase 5 US3**: Core frontier persistence depends on US1; the required “frontier affects context” acceptance path also depends on US2, so implement after US2 in the default sequence.
- **Phase 6 US4**: Core agenda persistence depends on US1 and can model motivation from Findings alone; frontier-aware agenda acceptance additionally uses US3, so the default sequence places US4 after US3.
- **Phase 7 US5**: Depends on US1-US4 because it is the cross-story two-round integration proof.
- **Phase 8 Polish**: Depends on all stories selected for release.

### User Story Dependency Graph

```text
Foundational
    |
    v
US1 Durable Research State (MVP)
    |
    v
US2 Bounded Context
    |
    v
US3 Frontier Steering
    |
    v
US4 Durable Agenda
    |
    v
US5 Multi-Round AutoResearch Integration
```

The graph shows the recommended delivery order. After US1, teams may build the core US3 frontier persistence and core US4 agenda persistence in parallel with US2, but their cross-story acceptance tests must wait for the referenced context/frontier capability.

### Within Each User Story

- Write the listed story tests first and confirm they fail for the intended missing behavior.
- Implement model/validation/replay primitives before command handlers that depend on them.
- Implement services/selection logic before rendering and command integration where applicable.
- Finish each story's contract and integration tests before moving its checkpoint to complete.
- Do not silently retry revision conflicts in convenience commands; surface exit status `3` and require caller refresh/reconsideration.

### Parallel Opportunities

- Phase 1 T002 and T003 can run in parallel after T001 is understood.
- Phase 2 T004/T005/T010 can run in parallel; T006 then depends on the model/ID vocabulary, while store/replay/validation converge afterward.
- US1 test files T014-T019 can be authored in parallel before implementation; T020 evidence work can proceed in parallel with store/replay internals after foundational types exist.
- US2 contract/integration tests can be authored while context unit tests are developed; implementation in the single `context.py` file is intentionally sequential.
- US3 unit/contract/integration tests T037-T039 can run in parallel; command, validation, replay, and rendering work touches separate files until context integration T043.
- US4 T046, T048, and T049 can run in parallel; T047 follows T046 in the same unit-test file, while agenda selection in `agenda.py` can proceed alongside replay/validation work once Agenda models exist.
- US5 T056-T058 and protocol documentation T059 can be developed in parallel after US1-US4 contracts stabilize.
- Polish documentation T062/T063 and performance test T064 can run in parallel before final full-regression tasks.

---

## Parallel Example: User Story 1

```text
Task T014: store transaction/revision tests in tests/unit/test_research_store.py
Task T015: replay/history tests in tests/unit/test_research_replay.py
Task T016: validation/local-ref tests in tests/unit/test_research_validation.py
Task T017: Aim evidence-reference tests in tests/unit/test_research_evidence.py
Task T018: CLI JSON/exit-code contract tests in tests/contract/test_research_contract.py
Task T019: end-to-end persistence/read-only tests in tests/integration/test_research_commands.py
```

## Parallel Example: User Story 3

```text
Task T037: frontier domain tests in tests/unit/test_research_frontier.py
Task T038: frontier CLI contract tests in tests/contract/test_frontier_contract.py
Task T039: human-steering integration tests in tests/integration/test_research_frontier.py
```

## Parallel Example: User Story 4

```text
Task T046: agenda lifecycle tests in tests/unit/test_research_agenda.py
Task T048: agenda CLI contract tests in tests/contract/test_agenda_contract.py
Task T049: agenda evidence/steering integration tests in tests/integration/test_research_agenda.py
```

---

## Implementation Strategy

### MVP First: User Story 1

1. Complete Setup and Foundational phases.
2. Write US1 safety/contract/integration tests.
3. Implement atomic ResearchUpdate + Finding/Lineage/Annotation state.
4. Validate cross-session persistence, invalid-update rollback, revision conflict, and `.aim` read-only guarantees.
5. Stop here if a durable Agent/Human shared research memory is the immediate release goal.

### Incremental Delivery

1. **US1** → durable knowledge/control state.
2. **US2** → bounded deterministic memory retrieval.
3. **US3** → durable human/agent search-policy steering.
4. **US4** → durable executable Experiment Contracts and deterministic next selection.
5. **US5** → prove the entire two-round tool-neutral AutoResearch loop.
6. **Polish** → docs, scale target, quickstart, and full legacy compatibility gate.

### Parallel Team Strategy

After Foundational + US1 stabilize:

- One developer can implement US2 context.
- A second can implement core US3 frontier persistence/CLI.
- A third can implement core US4 agenda persistence/selection.
- Cross-story frontier/context and frontier/agenda acceptance tasks merge after the dependencies land.
- US5 remains the final integration stream because it intentionally exercises all prior capabilities together.

## Notes

- No task adds a new runtime package; use Python standard library SQLite/JSON plus existing Aim/Rich dependencies.
- `.aim` is an evidence source only. All explicit research writes target Aimx-owned `.aimx/research/state.sqlite3`.
- Read commands must not create `.aimx` or initialize SQLite.
- Finding claim meaning is immutable; later meaning changes create a new Finding and lineage relation.
- `research next` selects persisted Agenda Items deterministically and never invents experiments.
- Every machine-readable research envelope remains agent-product-neutral and includes stable schema/revision information where specified.
