# Between-Rounds Evaluation Scenarios (User Story 2)

This document contains scenario packets for validating User Story 2:
A Lead invokes `$aimx-retro` between autoresearch rounds to inform next-round allocation while preserving Lead authority and source immutability.

---

## Scenario B1: Confounded Improvement in Autoresearch Round

### Context
In Round 2 of a neural architecture search loop, Agent 1 changed both the learning rate schedule (cosine vs linear) AND enabled data augmentation (MixUp).
Accuracy improved by 1.5%, but the cause is confounded.
The Lead needs retro advice before planning Round 3.

### Input Packet
```yaml
invocation_mode: between_rounds
repo: "/data/experiments/nas_loop"
loop_id: "nas_round_20261029"
rounds: [1, 2]
objective: "Identify top architectural modification for CIFAR-100"
constraints:
  - "Remaining budget: 40 GPU hours"
evidence_sources:
  - id: ev_b1_checkpoint
    locator: "python $AUTORESEARCH_SKILL_DIR/scripts/control.py status --loop-dir /data/experiments/nas_loop/.aimx/autoresearch/nas_round_20261029"
    data:
      round: 2
      status: "round_completed"
  - id: ev_b1_run_r1
    locator: "aimx query metrics 'run.hash == \"run_nas_r1\"' --repo /data/experiments/nas_loop"
    data:
      top1: 0.762
      params: { lr_sched: "linear", mixup: false }
  - id: ev_b1_run_r2
    locator: "aimx query metrics 'run.hash == \"run_nas_r2\"' --repo /data/experiments/nas_loop"
    data:
      top1: 0.777
      params: { lr_sched: "cosine", mixup: true }
```

### Expected Evaluation Rubric
- **SC-004**: Refuses to attribute the 1.5% gain exclusively to cosine schedule or MixUp; highlights the confounding factor.
- **SC-003**: Recommends an ablation run isolating either MixUp or Cosine schedule with explicit outcome branches.
- **SC-005**: Respects the 40 GPU hour constraint (ablation cost estimated within budget).

---

## Scenario B2: Sound Process with Negative Result

### Context
In Round 1, an agent rigorously tested weight decay scaling. Metrics showed no improvement (neutral/slight regression).
The process was technically flawless (converged, well-controlled, verified seed).
Lead needs an honest retro that does not manufacture false blame.

### Input Packet
```yaml
invocation_mode: between_rounds
repo: "/data/experiments/decay_test"
loop_id: "decay_loop_01"
rounds: [1]
objective: "Determine if higher weight decay mitigates test drift"
evidence_sources:
  - id: ev_b2_run
    locator: "aimx query metrics 'run.hash == \"run_wd_01\"' --repo /data/experiments/decay_test"
    data:
      loss: 0.35
      drift_metric: 0.12
      clean_exit: true
```

### Expected Evaluation Rubric
- **SC-005**: Acknowledges the process was sound; does not invent flaws or force negative criticism where none existed.
- **SC-003**: Recommends retaining baseline weight decay and shifting frontier exploration to learning rate or model capacity.

---

## Scenario B3: Goal Already Satisfied

### Context
The original objective was to reach 90% validation accuracy within 5 rounds.
In Round 2, the run achieves 90.4% accuracy with stable training.

### Input Packet
```yaml
invocation_mode: between_rounds
repo: "/data/experiments/target_acc"
loop_id: "acc_loop_02"
rounds: [1, 2]
objective: "Achieve >= 90.0% validation accuracy"
evidence_sources:
  - id: ev_b3_run
    locator: "aimx query metrics 'run.hash == \"run_acc_90\"' --repo /data/experiments/target_acc"
    data:
      accuracy: 0.904
      completed: true
```

### Expected Evaluation Rubric
- **SC-005**: Recognizes the objective is satisfied.
- **SC-003**: Issues disposition `"stop"` (or recommends stopping), avoiding unnecessary GPU burn on additional rounds.
- **FR-011**: Confirms budget preservation and Lead handoff.
