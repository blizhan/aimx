# Phase 0 research: Aimx Retro

Research completed from local repository sources by the lead and two Luna workers (max). No external product/library facts or new dependencies are required. All planning unknowns are resolved below.

## 1. Runtime and packaging

**Decision:** Ship `skills/aimx-retro` with a short ordered SKILL.md, `agents/openai.yaml`, disclosed evidence/report references and one standard-library report helper. External Agents perform semantic analysis. Keep `src/aimx` unchanged.

**Rationale:** Existing skills are independently installed from this repository; there is no central registry. README.md:33–54 and README.zh-CN.md:31–45 list available skills. The autoresearch helper is standalone (`skills/aimx-autoresearch/scripts/control.py:429`) and already establishes a stdlib JSON command convention.

**Alternatives considered:** Core retro command, an embedded LLM runtime, a mandatory checkpoint collector, and a new state subsystem each broaden ownership without helping the first-version decision-review goal.

## 2. Evidence access and historical limits

**Decision:** Use supported current-state and run-read commands. Historical decisions use persisted task files, results, saved updates, receipts and any available dated snapshots; otherwise mark the historical context unknown. Do not implement historical replay or access SQLite internals in this feature.

**Rationale:** `src/aimx/commands/research.py:129,156,202` exposes current state/context with no as-of revision argument. `src/aimx/research/models.py:274` omits commit lists from serialized state. Internal replay exists (`src/aimx/research/replay.py:166`) but is not a public historical-read contract. `research next` is globally selected rather than objective scoped (`src/aimx/research/agenda.py:30`), so it must not determine the review objective automatically.

**Alternatives considered:** Replaying internal commits would add compatibility coupling and still could not reconstruct the actual Agent context. A mandatory historical API would delay a skill whose spec already supports explicit evidence gaps.

## 3. Evidence strength

**Decision:** Distinguish source availability, source provenance, statement basis, causal assessment and validation status. Inspect actual run parameters/metrics before relying on a Finding's claim. Represent concrete artifact selectors and full run hashes in an independent evidence catalog.

**Rationale:** `src/aimx/aim_bridge/research_evidence.py:55` checks run availability, not claim correctness. ResearchUpdate evidence currently allows only `aim_run` (`src/aimx/research/models.py:79`, `validation.py:443`). Retro needs file and user-input references without modifying that research contract.

**Alternatives considered:** One confidence score obscures the difference between an observed event, its cause and an untested recommendation. Extending Finding evidence solely for retro is unnecessary.

## 4. Optional round integration

**Decision:** Add a conditional pointer after round reconciliation and state update, before planning the next round. Read checkpoints only when provided or clearly associated with the selected scope. Preserve terminal/paused status and Lead-only updates.

**Rationale:** `skills/aimx-autoresearch/references/research-handoff.md` defines Lead ownership. `control.py:190,437` exposes read-only `status`; round records carry task/result/artifact references and update receipts, but no complete checkpoint-version history. References alone are not verified evidence (`references/loop-contract.md:82`).

**Alternatives considered:** Mandatory per-round execution or checkpoint state changes would add orchestration and artificial budget consumption. The skill should be callable from any host with the required read tools.

## 5. Report contract and storage

**Decision:** Agent writes a versioned `retro.json` draft; `scripts/report.py` validates it and renders `RETRO.md` from that same object. Publish a fresh directory under `<repo>/.aimx/retro/<report-id>/`, defaulting report IDs to a UTC timestamp plus random suffix chosen before publication. Offer an explicit separate output root for read-only repositories. Preserve the canonical JSON and full content in Markdown. Source locators remain references; relevant read responses may be saved as optional evidence artifacts.

**Rationale:** A single authored representation prevents divergent structured and human summaries. Fresh directories preserve prior retrospectives. Separate output ownership matches `.aimx/research` and `.aimx/autoresearch` without touching either. No new dependency is needed.

**Alternatives considered:** Two separately authored reports drift. Timestamp-only overwrite permits collisions. A universal evidence collector or custom database adds machinery unrelated to semantic review.

## 6. Verification

**Decision:** Combine deterministic report contract tests and isolated-package execution with a small semantic review evaluation suite. Checks verify source immutability, references, limits, output consistency and refusal to overwrite; semantic cases verify supported reasoning rather than matching exact prose.

**Rationale:** Existing unit and integration examples are `tests/unit/test_autoresearch_control.py` and `tests/integration/test_autoresearch_skill_loop.py:247–270` (copy skill, run under `python -I`). Existing pytest and Aim dev dependencies suffice. Structural validation cannot prove entailment or decision usefulness.

**Alternatives considered:** Static keyword checks alone do not test reasoning. Requiring live model calls in ordinary pytest would make deterministic verification costly and unreliable. Reasoning evaluations use manually reviewed saved outputs from fresh Agent sessions.

## 7. Spec Kit compatibility

**Decision:** Use the per-command environment override `SPECIFY_FEATURE=008-aimx-retro` for setup/context scripts while retaining actual branch `codex/008-aimx-retro` and `.specify/feature.json` pointing to this feature directory.

**Rationale:** The installed branch validator expects a leading number; the supported feature override resolves the mismatch without changing shared tooling or branch policy.

**Alternatives considered:** Renaming the branch or editing common.sh is unrelated to this feature.
