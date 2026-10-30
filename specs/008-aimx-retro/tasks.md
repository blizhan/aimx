# Tasks: Aimx Research Retro Skill

**Input**: Design artifacts in `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/`  
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/skill-interface.md`, `contracts/report-interface.md`, `quickstart.md`  
**Organization**: Shared report foundation, then independently verifiable user-story increments. US1 and US2 are P1; US3 is P2.  
**Tests**: Include the specification's Independent Tests and SC-001–SC-007 verification, plus constitution-required source-preservation and output-contract checks from the plan. No unrelated coverage expansion or new dependency. Deterministic tests cannot certify semantic reasoning.

## Format and working rules

Each task uses `- [ ] Tnnn [P?] [USn?] Description with absolute file path`. `[P]` means independent work within the stated ready wave, not permission to ignore prerequisites or edit shared files concurrently. Paths identify planned files where they do not yet exist. Use `uv` and Python 3.12; preserve user changes. No `src/aimx` change, new CLI command, database entity, checkpoint schema change or runtime/model dependency is planned.

## Phase 1: Setup

**Purpose**: Establish the installable package and evaluation criteria without changing research behavior.

- [x] T001 Create skill discovery metadata for both user invocation and Lead invocation in `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/SKILL.md` and `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/agents/openai.yaml`; follow existing package conventions, keep the body ready for the shared protocol, and introduce no project dependency.
- [x] T002 [P] Define source-entailment, hindsight, attribution, next-step usefulness, source-preservation and report-consistency acceptance criteria mapped to SC-001–SC-007 in `/Users/blizhan/data/code/github/aimx/tests/fixtures/retro/scenarios/README.md`; distinguish fresh-Agent semantic review from deterministic pytest and specify how to record inputs, actual model/settings, output paths and reviewer results.

## Phase 2: Foundational report contract

**Purpose**: Supply the shared report model and safe publication used by every story. Complete this phase before publishing story reports.

- [x] T003 Translate all version-1 entities, required fields, null rules, enums, IDs, provenance links and disposition rules from the design into the self-contained installed reference `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/references/report-format.md`; include validate/publish examples and a minimal valid example without requiring access to `specs/`.
- [x] T004 Create complete and minimal valid report fixtures in `/Users/blizhan/data/code/github/aimx/tests/fixtures/retro/reports/complete.json` and `/Users/blizhan/data/code/github/aimx/tests/fixtures/retro/reports/minimal.json`, then contract tests in `/Users/blizhan/data/code/github/aimx/tests/unit/test_retro_report.py` for strict JSON, schema/type/field errors, time ordering, unique/resolved references, evidence basis, tested-advice evidence, array limits and experiment branches; include legitimate null/empty cases and confirm missing helper behavior fails before implementation.
- [x] T005 Implement pure report loading and validation plus `validate --file` in `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/scripts/report.py` using only the standard library; reject duplicate keys, non-finite numbers and bool-as-integer values, return bounded JSON receipts with field paths and exit 0/2, and create no files or directories during validation.
- [x] T006 Extend `/Users/blizhan/data/code/github/aimx/tests/unit/test_retro_report.py` with deterministic full-field rendering, visible unknowns, inert rendering of embedded directives, default/explicit output roots, protected-path and symlink rejection, collision preservation, injected write failure cleanup and partial-directory recovery cases; hash source sentinels before and after each relevant path.
- [x] T007 Implement `publish --file [--output-root]` and canonical `retro.json` → `RETRO.md` rendering in `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/scripts/report.py`; validate before writes, preserve all analytical fields, exclusively create a fresh report directory, reject existing/protected destinations, clean up only invocation-owned partial files on normal failure, and return the documented receipt/error codes; pass T004/T006 tests.

**Checkpoint**: Contract tests pass, validation is read-only, and publication creates only a new report pair. Report validation checks structure and provenance references, not truth or causal proof.

## Phase 3: User Story 1 — 新 Agent 独立复盘既有研究 (P1, MVP)

**Goal**: A new Agent reviews one research objective from persisted records without the original conversation, Lead or checkpoint.

**Independent Test**: Give a fresh Agent current Research State, experiment evidence and optional dated artifacts but no checkpoint/chat. It must state scope, reconstruct supported decisions, identify credible conclusions and unresolved questions, and publish an actionable or explicitly evidence-limited report without modifying sources.

### Acceptance preparation

- [x] T008 [P] [US1] Create standalone-history, ambiguous-goals and obsolete-agenda evidence packets with specific source selectors and expected rubric outcomes in `/Users/blizhan/data/code/github/aimx/tests/fixtures/retro/scenarios/standalone.md`; cover explicit direction/range selection and justified references outside that range without embedding suggested conclusions in the Agent-facing packet.
- [x] T009 [P] [US1] Add isolated-install integration tests in `/Users/blizhan/data/code/github/aimx/tests/integration/test_retro_skill.py` that copy only `skills/aimx-retro`, invoke its helper with `python -I`, verify validate/publish receipts and exit codes, resolve installed references, and prove absence of sibling skills or repository imports is harmless.
### Implementation and independent verification

- [x] T010 [US1] Write scope selection, current-state/run evidence reads, exact locators, provenance capture, historical-context limits, bounded selection and no-checkpoint fallback in `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/references/evidence-guide.md`; use supported commands only, distinguish revision 0 from unavailable state, and avoid internal replay or automatic source discovery outside the requested scope.
- [x] T011 [US1] Complete the common scope → evidence → reconstruct → analyze → advise → challenge/publish workflow in `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/SKILL.md`, linking the two references at the relevant steps; include explicit unknowns, six review dimensions, alternative explanations, 0–3 reflections/options, result-dependent next decisions, scope clarification and source-write boundaries from the start.
- [x] T012 [US1] Run the standalone packets in a fresh Agent context using only the installed skill and packet inputs, validate/publish the resulting reports, and record report paths, actual coverage, source-preservation checks and rubric results in `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/validation/standalone.md`; correct supported failures in the owning skill files and rerun affected cases before marking complete.

**Checkpoint**: US1 works independently, including absent checkpoint, ambiguity and historical/current distinction. A standalone retrospective is the first demoable increment; evidence limitations remain mandatory even before US3's extended evaluation.

## Phase 4: User Story 2 — Lead 在轮次之间调整研究决策 (P1)

**Goal**: Reuse the same report protocol after a completed round and before next-round allocation, preserving Lead authority and original constraints.

**Independent Test**: Supply a completed round, its reconciled research update and remaining constraints. The report must compare next-step choices with outcome branches, allow justified continuation/stopping, and require rechecking stale assumptions without starting another round or changing checkpoint/state.

### Acceptance preparation

- [x] T013 [P] [US2] Create round-handoff packets for confounded improvement, a sound process, an already-satisfied goal and changed revision/user steering in `/Users/blizhan/data/code/github/aimx/tests/fixtures/retro/scenarios/between-rounds.md`; specify independent expected behavior for each case, including paused/budget-exhausted state preservation.

### Implementation and independent verification

- [x] T014 [US2] Add a conditional `$aimx-retro` pointer after result/integration/update reconciliation and before next-round allocation in `/Users/blizhan/data/code/github/aimx/skills/aimx-autoresearch/SKILL.md`; keep invocation optional and preserve original completion, budget and terminal-state handling without introducing helper or checkpoint mutations.
- [x] T015 [P] [US2] Document the supplied goal, revision, round evidence and remaining constraints, report consumption and stale-recommendation recheck in `/Users/blizhan/data/code/github/aimx/skills/aimx-autoresearch/references/research-handoff.md`; Lead alone decides and submits any later ResearchUpdate, and sibling skill absence never becomes an execution failure.
- [x] T016 [US2] Complete the optional checkpoint evidence branch in `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/references/evidence-guide.md`, using read-only `control.py status` when available or a clearly labeled direct artifact read otherwise; inspect referenced task/result/update evidence rather than treating checkpoint summaries as verified claims, and preserve report-only completion semantics.
- [x] T017 [US2] Evaluate the round packets in fresh contexts, inspect rendered/structured agreement and unchanged source checkpoints/state, and record decision usefulness, valid stopping/continuation, stale-advice handling and result paths in `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/validation/between-rounds.md`; fix and rerun failing cases before marking complete.

**Checkpoint**: A Lead can consume the report without a new research command or scheduling mechanism. US2 uses the shared protocol but has its own completed-round acceptance packet and does not require the US1 scenario run to succeed first.

## Phase 5: User Story 3 — 在证据不完整时获得可信的分析 (P2)

**Goal**: Verify and refine bounded conclusions when motives, knowledge, comparisons or outcomes cannot be established.

**Independent Test**: Give fresh Agents interrupted experiments, mixed comparisons, later success, missing context and conflicting/missing sources. They must identify limitations, separate observation/cause/validation, preserve competing explanations and propose specific evidence collection where appropriate.

### Acceptance preparation

- [x] T018 [P] [US3] Create incomplete-evidence packets covering unknown awareness, execution failure, incomparable splits/metrics, hindsight, conflicting summaries, missing references, unknown costs and an instruction embedded in source material in `/Users/blizhan/data/code/github/aimx/tests/fixtures/retro/scenarios/incomplete-evidence.md`; include empty/Aim-only research and a declared large-history sample rather than a full scan.
- [x] T019 [P] [US3] Extend `/Users/blizhan/data/code/github/aimx/tests/unit/test_retro_report.py` with valid `clarify`/`gather_evidence`/`stop` reports, unknown objective/revisions/costs, missing/unreadable source catalogs, absent historical reasons and invalid observed/tested claims; ensure valid uncertainty is accepted while unsupported provenance is rejected, without pretending to automate semantic judgment.

### Implementation and independent verification

- [x] T020 [US3] Refine `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/references/evidence-guide.md` with the degraded-evidence decision rules: available-vs-known, reported-vs-observed, execution before hypothesis conclusions, source capture time vs historical knowledge, unknown resource cost, revision-start/end drift with bounded reread, unfinished runs and actionable gaps; keep historical replay out of scope.
- [x] T021 [US3] Strengthen the final counterargument and bounded-conclusion step in `/Users/blizhan/data/code/github/aimx/skills/aimx-retro/SKILL.md` using the incomplete-evidence cases; require observation, causal assessment and advice validation to remain separate, keep conflicting explanations where unresolved, and explicitly allow zero problems or evidence-first advice without forcing criticism.
- [x] T022 [US3] Evaluate incomplete-evidence packets in fresh contexts, verify every substantive observation against its source and each limitation against the recommendation, and record outputs plus rubric results in `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/validation/incomplete-evidence.md`; rerun any corrected scenario and do not report semantic acceptance solely from passing JSON validation.

**Checkpoint**: All four SC-004 reasoning traps are evaluated successfully; missing evidence is visible and advice remains bounded. Core safety/provenance rules were implemented in the foundation/US1, not postponed until this phase.

## Phase 6: Polish and cross-cutting validation

- [x] T023 [P] Register the new skill and document standalone/between-round use, report locations, optional checkpoint and Lead ownership in `/Users/blizhan/data/code/github/aimx/README.md`, distinguishing skill helper commands from Aimx CLI commands.
- [x] T024 [P] Add equivalent Chinese installation/discovery and two-entry guidance in `/Users/blizhan/data/code/github/aimx/README.zh-CN.md`, with the same report and source-write boundaries as the English guide.
- [x] T025 Run the real repo-scoped reads from `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/quickstart.md` against available `/Users/blizhan/data/code/github/aimx/data/.aim`, record the actual Aim version, commands, coverage and source-integrity evidence in `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/validation/compatibility.md`; if unavailable, record the concrete gap and use isolated synthetic fixtures without initializing or repairing user data.
- [x] T026 Run focused helper/isolated-install tests followed by `uv run pytest` including existing passthrough coverage, verify documentation links and no new dependency/core ownership change, and record exact commands/results and any remaining limitations in `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/validation/summary.md`; mark SC-001–SC-007 separately using actual semantic and deterministic evidence, not inferred success.
- [x] T027 Reconcile the delivered commands, artifact behavior and verification instructions in `/Users/blizhan/data/code/github/aimx/specs/008-aimx-retro/quickstart.md` with the implemented skill, and update only the planned Retro context entry in `/Users/blizhan/data/code/github/aimx/AGENTS.md` to describe delivered behavior while preserving manual instructions.

## Dependencies and execution order

```text
Setup T001 + T002
  -> Contract T003 -> tests T004 -> validate T005 -> tests T006 -> publish T007
  -> US1 T008 + T009 + T010 -> T011 -> T012 (standalone MVP)
                         |
                         +-> US2 T013 + T014 + T015 -> T016 -> T017
                         |
                         +-> US3 T018 + T019 -> T020 -> T021 -> T022
  -> Documentation T023 + T024
  -> Real-read validation T025 -> acceptance/regressions T026 -> reconcile T027
```

- T001 and T002 are independent; T003 requires package setup. Foundation completion gates story publication.
- T008, T009 and T010 may start together after the foundation; they own different files. T011 waits for the evidence guide. T012 waits for the complete skill and isolated-install checks.
- US2 and US3 reuse the common protocol established by T011, but each is evaluated independently with its own supplied evidence. Neither requires the other story's outputs. Default sequence follows phase order to avoid concurrent edits to the evidence guide and SKILL.md.
- T013–T015 may run together after T011. T016 must not overlap T010 or T020; T017 waits for T013–T016 and the common helper.
- T018 and T019 may run together after foundation. T020 waits for prior evidence-guide edits (T010 and, if executing in phase order, T016). T021 waits for T020 and prior SKILL edits. T022 waits for T018–T021.
- T023/T024 run together once story behavior is final. T025 follows assembled skill and docs; T026 requires story reviews and deterministic checks, and T027 follows final validation.
- All edits to `report.py`, `test_retro_report.py`, `SKILL.md`, and `evidence-guide.md` are serialized per file. `[P]` applies only after the declared wave's prerequisites complete.

## Parallel execution examples

**US1 (foundation complete)**: One worker prepares `scenarios/standalone.md` (T008); another writes isolated-install tests (T009); the owner writes `references/evidence-guide.md` (T010). Integrate these before completing the shared workflow and running T012.

**US2 (T011 complete)**: Prepare `scenarios/between-rounds.md` (T013), add the autoresearch invocation pointer (T014), and document Lead consumption in `research-handoff.md` (T015) independently. Then finish the optional evidence branch and run the round evaluation.

**US3 (foundation complete)**: Prepare `scenarios/incomplete-evidence.md` (T018) while extending valid/invalid uncertainty test cases (T019). Refinement of shared skill files waits for their earlier owners to finish; semantic evaluation follows refinement.

**Documentation**: T023 and T024 can proceed concurrently, then compare guidance for consistency during T026.

These are execution opportunities for a later implementation run, not delegation or execution initiated by this task-generation command.

## Requirements coverage

| Requirement / outcome | Primary tasks |
| --- | --- |
| FR-001 two entry paths | T001, T011, T014–T017 |
| FR-002 scope selection | T008, T010–T012 |
| FR-003 scope/version/gaps | T003–T005, T010, T020 |
| FR-004 decision reconstruction | T003, T010–T012, T018–T022 |
| FR-005 execution/comparability order | T011, T013, T018, T020–T022 |
| FR-006 six dimensions / no forced criticism | T011, T013, T021–T022 |
| FR-007 evidence attribution | T003–T005, T010, T018–T022 |
| FR-008 alternative causes / separate certainty | T003, T011, T018–T022 |
| FR-009 bounded actionable reflections | T003–T005, T011–T012, T021 |
| FR-010 next-step choices / stop / supplement | T003–T005, T011–T013, T017, T019–T022 |
| FR-011 goals, budgets, decision impact | T011, T013–T017, T020–T022 |
| FR-012 consistent preserved report pair | T003–T009, T026 |
| FR-013 optional round integration / Lead control | T014–T017 |
| FR-014 source read-only / no execution side effects | T006–T007, T009–T012, T014–T017, T025–T026 |
| FR-015 usage guidance | T001, T010–T011, T014–T016, T023–T024, T027 |
| SC-001 fresh-Agent evidence variants | T012, T017, T022 |
| SC-002 traceability / honest unknowns | T004–T005, T012, T017, T019, T022 |
| SC-003 actionable independent handoff | T012, T017, T022 |
| SC-004 four reasoning traps | T018, T020–T022 |
| SC-005 limits / sound process / stop | T004–T005, T012–T013, T017, T019, T022 |
| SC-006 source and prior-report preservation | T006–T007, T009, T012, T017, T022, T025–T026 |
| SC-007 output agreement | T006–T007, T009, T026 |

## Implementation strategy

1. **MVP:** T001–T012 establish a safe report foundation and independently useful standalone retrospective. Demonstrate with no original chat or checkpoint. Do not omit evidence safeguards from this increment.
2. **Round integration:** T013–T017 add optional Lead use and its decision-focused evaluation without changing loop contracts.
3. **Evidence-depth increment:** T018–T022 challenge incomplete/conflicting evidence and refine the common protocol, then rerun affected prior cases if behavior changes.
4. **Release readiness:** T023–T027 finish bilingual discovery, real repository compatibility evidence and all regression/semantic gates. A blocked semantic evaluation remains an explicit incomplete task; fabricated or self-authored “Agent results” do not count.

No task authorizes new experiment runs, automatic research updates, skill evolution, dependency changes, commits or pushes. Invocation-generated report artifacts are the only feature write surface. Commit hooks remain optional and separate from this checklist.
