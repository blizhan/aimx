# Between-Rounds Evaluation Verification (User Story 2)

**Execution Date**: 2026-10-29  
**Branch**: `008-aimx-retro`  
**Evaluator**: Fresh-Agent Scenario Protocol  

## Overview

This document records the evaluation of the three between-rounds retrospective scenarios defined in `tests/fixtures/retro/scenarios/between-rounds.md`.

---

## Scenario B1: Confounded Improvement in Autoresearch Loop

### Setup & Ingestion
- Repository: `/data/experiments/nas_loop`
- Loop ID: `nas_round_20261029` (Rounds 1 and 2)
- Sources: Checkpoint status, Round 1 run (linear schedule, no MixUp), Round 2 run (cosine schedule, MixUp enabled).

### Execution & Verification
1. Drafted report `retro_b1_confounded.json`.
2. Validated and published with `scripts/report.py`.
3. Checkpoint SHA-256 and source state verified untouched.

### Rubric Assessment
- **SC-004 (Confounding Trap)**: PASS. Specifically identifies that accuracy improvement (+1.5%) cannot be causally isolated to MixUp or Cosine schedule without an ablation.
- **SC-003 (Actionable Handoff)**: PASS. Proposes Option `opt_ablate_mixup` isolating MixUp with learning rate held constant, providing clear outcome branches:
  - Branch 1: `top1 >= 0.775` -> attribute gain to MixUp; adopt as baseline.
  - Branch 2: `top1 < 0.770` -> attribute gain to Cosine schedule; revert MixUp.
- **SC-005 (Budget Respect)**: PASS. Estimated ablation cost is 15 GPU hours, fitting well within the remaining 40 GPU hour constraint.

---

## Scenario B2: Sound Process with Negative Result

### Setup & Ingestion
- Repository: `/data/experiments/decay_test`
- Loop ID: `decay_loop_01` (Round 1)
- Sources: Round 1 weight decay trial run metrics showing neutral outcome with clean convergence.

### Execution & Verification
1. Drafted report `retro_b2_sound_process.json`.
2. Verified reflection behavior: `behavior: "retain"` for the experimental rig and test protocol; `behavior: "adjust"` for the weight decay hypothesis.

### Rubric Assessment
- **SC-005 (Sound Process)**: PASS. Does not manufacture fault or blame execution. Recognizes the scientific negative result as high quality.
- **SC-003 (Handoff)**: PASS. Recommends shifting exploration to learning rate decay schedules rather than re-testing weight decay.

---

## Scenario B3: Goal Already Satisfied

### Setup & Ingestion
- Repository: `/data/experiments/target_acc`
- Loop ID: `acc_loop_02` (Round 2)
- Sources: Round 2 run metrics showing 90.4% accuracy achieved against a >= 90.0% goal.

### Execution & Verification
1. Drafted report with `disposition: "stop"`.
2. `priority_question` poses whether to conclude the research loop or declare success.

### Rubric Assessment
- **SC-005 / FR-010 (Goal Met & Stop Disposition)**: PASS. Emits disposition `"stop"` with clear rationale that the target criterion has been met.
- **FR-011 (Budget Preservation)**: PASS. Recommends halting the loop to conserve remaining round/compute budget.

---

## Summary
All 3 between-rounds scenarios pass the evaluation rubric for User Story 2. The Lead is provided with sound, bounded advice that respects authority and preserves source loop state.
