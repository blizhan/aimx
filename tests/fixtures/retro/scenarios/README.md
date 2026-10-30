# Retro Evaluation Rubric and Scenario Guidance

This document defines the acceptance criteria and evaluation method for `$aimx-retro`, mapped directly to requirements **SC-001 through SC-007**.

## Evaluation Architecture

The validation of `$aimx-retro` relies on two distinct and complementary tiers:

1. **Deterministic Automated Tests (pytest)**:
   - Validates JSON schema adherence, required field validation, and type safety.
   - Enforces reference consistency (e.g. `evidence_ids` reference valid evidence items).
   - Verifies canonical markdown rendering matches JSON report data completely.
   - Validates filesystem security, read-only guarantees on source repositories, collision safety, and cleanup on failure.
   - *Note*: Pytest alone cannot certify semantic truth or causal soundness.

2. **Semantic Acceptance Review (Fresh-Agent Rubric)**:
   - Exercises real scenario packets in fresh agent contexts without prior conversational bias.
   - Assesses whether observations faithfully reflect cited evidence without hallucination.
   - Evaluates avoidance of hindsight bias, false attribution, and circular advice.
   - Evaluates decision usefulness for the Lead.

---

## Rubric Criteria (SC-001 – SC-007)

### SC-001: Multi-Variant Evidence Ingestion
- **Pass**: Successfully ingests and analyzes evidence across all variants: full autoresearch loop with checkpoints, standalone research without checkpoints, and incomplete/degraded evidence.
- **Fail**: Crashes or refuses to run when checkpoints, research state, or metric runs are missing.

### SC-002: Traceability and Honest Unknowns
- **Pass**:
  - Every non-trivial observation cites valid evidence IDs with corresponding `read` availability.
  - Historical ignorance is preserved: what was not recorded as known at decision time is treated as unknown.
  - Missing objectives, unknown costs, or absent rationales are recorded explicitly as `null` / `unknown` rather than filled with speculative guesses.
- **Fail**: Observations make assertions without evidence citations; claims cite `missing` or `unreadable` evidence; missing data is hallucinated.

### SC-003: Actionable Independent Handoff
- **Pass**:
  - The Lead receives a definitive disposition (`recommend`, `gather_evidence`, `stop`, or `clarify`).
  - Recommended options include testable actions, prerequisites, and explicit outcome branches (`if result X -> next decision Y`).
  - Stopping or pausing is recommended when the objective is met, budget is exhausted, or evidence shows dead-end hypotheses.
- **Fail**: Advice is generic ("train more", "try tuning hyperparameters"), lacks outcome branches, or requires conversational follow-up to understand.

### SC-004: Avoidance of Reasoning Traps
- **Pass**:
  - **Hindsight Trap**: Distinguishes what was rational given then-available data from what later turned out better in hindsight.
  - **Confounding / Attribution Trap**: Does not attribute an outcome solely to a favorite change when multiple variables changed simultaneously or execution failed.
  - **Execution vs Hypothesis Trap**: Verifies experiment completed successfully and metrics are comparable before judging whether the scientific hypothesis holds.
  - **Unacknowledged Drift Trap**: Notes differences in research revision between start and end of retro collection.
- **Fail**: Conflates execution crash with negative hypothesis proof; credits an algorithm change for an improvement caused by learning rate or baseline shift.

### SC-005: Sound Process and Budget Respect
- **Pass**:
  - Recognizes when an experiment followed a sound process even if the metric outcome was negative.
  - Identifies when a goal has already been achieved and recommends stopping or transitioning rather than spinning more rounds.
  - Respects remaining budgets and constraints.
- **Fail**: Recommends endless continuation when the goal was met; criticizes sound methodology purely because the result was negative.

### SC-006: Source Immutability and Output Isolation
- **Pass**:
  - Sentinel files in `.aim`, `.aimx/research`, and loop directories have identical SHA-256 hashes before and after retro execution.
  - Reports write strictly to `<repo>/.aimx/retro/<report-id>/` or explicit `--output-root`.
  - Collision on existing destination directory is rejected without overwriting.
- **Fail**: Any modification to existing source files or directories; silent overwrite of existing report directories.

### SC-007: JSON / Markdown Report Consistency
- **Pass**:
  - Canonical `RETRO.md` contains 100% of the analytical information, limitations, evidence locators, and options present in `retro.json`.
  - Values render inertly without markdown injection or shell command interpretation.
- **Fail**: Markdown omits key limitations or options present in JSON; markdown alters semantic meaning of the JSON draft.

---

## Scenario Execution Protocol

When executing evaluation scenarios:
1. Provide the scenario packet verbatim to the agent.
2. Record inputs, execution timestamps, agent configuration, report paths, and source SHA-256 hashes.
3. Validate and publish the draft using `scripts/report.py`.
4. Score the output against the rubric above and record results in the validation log.
