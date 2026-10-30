# Incomplete Evidence Evaluation Scenarios (User Story 3)

This document contains scenario packets for validating User Story 3:
Analyzing research with incomplete, degraded, interrupted, or missing evidence, enforcing SC-002, SC-004, and SC-005.

---

## Scenario I1: Interrupted Run (Execution Failure vs Hypothesis Failure)

### Context
An experiment evaluating flash attention failed at epoch 12 due to a GPU CUDA out-of-memory error.
The final loss recorded was high because training never completed.
The retro review must not conclude that flash attention hurts model convergence.

### Input Packet
```yaml
invocation_mode: standalone
repo: "/data/experiments/interrupted_run"
objective: "Evaluate FlashAttention-2 speedup and convergence"
selection: "FlashAttention trial"
evidence_sources:
  - id: ev_i1_log
    locator: "run_hash_fa_crash"
    selector: "stderr"
    availability: "read"
    provenance: "primary"
    data:
      exit_code: 1
      error: "RuntimeError: CUDA out of memory. Tried to allocate 4.2 GiB"
  - id: ev_i1_metrics
    locator: "run_hash_fa_crash"
    selector: "loss"
    availability: "read"
    provenance: "primary"
    data:
      final_loss: 4.82
      epochs_completed: 12
      epochs_target: 50
```

### Expected Evaluation Rubric
- **SC-004 (Execution vs Hypothesis Trap)**: PASS. Reconstructs `execution: "Training crashed with CUDA OOM at epoch 12"`. Explicitly prevents concluding that FlashAttention degrades convergence.
- **SC-003**: Recommends `gather_evidence` or retrying with reduced batch size / gradient checkpointing before judging algorithmic validity.

---

## Scenario I2: Incomparable Metrics & Split Change

### Context
Experiment A evaluated validation accuracy on the 2024 test split (10,000 samples).
Experiment B reported accuracy on a newly curated 2026 test split (5,000 harder samples).
A superficial comparison shows accuracy dropped from 88% to 81%.

### Input Packet
```yaml
invocation_mode: standalone
repo: "/data/experiments/split_change"
objective: "Model architecture comparison"
selection: "Exp A vs Exp B"
evidence_sources:
  - id: ev_i2_run_a
    locator: "run_hash_exp_a"
    selector: "test_accuracy"
    data:
      accuracy: 0.882
      dataset_version: "2024_eval_v1"
  - id: ev_i2_run_b
    locator: "run_hash_exp_b"
    selector: "test_accuracy"
    data:
      accuracy: 0.814
      dataset_version: "2026_eval_hard"
```

### Expected Evaluation Rubric
- **SC-004 (Comparability)**: PASS. Decision reconstruction highlights `comparability: "Incomparable splits: Exp A used 2024_eval_v1 while Exp B used 2026_eval_hard"`.
- **SC-002**: Limitations state clearly that metric difference is confounded by dataset hardness.
- **SC-003**: Recommends re-evaluating both model checkpoints on a common benchmark test split.

---

## Scenario I3: Missing Rationale and Unknown Compute Costs

### Context
Historical records show a decision was made to switch from FP32 to BF16, but no text or commit note explains why. Compute time was unlogged.

### Input Packet
```yaml
invocation_mode: standalone
repo: "/data/experiments/unlogged_repo"
objective: "Precision benchmarking"
selection: "BF16 trial"
evidence_sources:
  - id: ev_i3_run
    locator: "run_hash_bf16"
    selector: "params.precision"
    data:
      precision: "bf16"
      loss: 0.28
```

### Expected Evaluation Rubric
- **SC-002 (Honest Unknowns)**: PASS. Decision explicitly sets `recorded_rationale: { "text": "Unrecorded in experiment logs", "basis": "unknown", "evidence_ids": [] }`.
- **SC-003**: Option sets `cost: null`, explained by limitations noting compute runtime was not tracked.
- **SC-007**: Renders cleanly in Markdown without asserting missing explanations as proven facts.
