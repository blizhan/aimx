# Standalone Evaluation Scenarios (User Story 1)

This document contains standalone scenario packets for validating User Story 1:
A fresh Agent reviews persisted records with no original conversation, Lead, or checkpoint.

---

## Scenario S1: Standalone History with Persisted Research State and Runs

### Context
A researcher initiated a sequence of experiments to evaluate optimizer choices (AdamW vs Lion) on a ViT model.
The researcher recorded Research State revisions 1 through 3, committed findings, and finished training runs.
No autoresearch loop was used; no checkpoint exists.
A new agent is tasked with conducting a retrospective on this research line.

### Input Packet
```yaml
invocation_mode: standalone
repo: "/data/experiments/vit_opt"
objective: "Evaluate optimizer stability and final accuracy on ViT-Small"
selection: "All experiments comparing AdamW and Lion optimizers"
evidence_sources:
  - id: ev_s1_state
    locator: "aimx research state --repo /data/experiments/vit_opt"
    data:
      revision: 3
      agenda:
        goal: "Compare AdamW (lr=1e-3) vs Lion (lr=1e-4) on ViT-S"
        status: "completed"
  - id: ev_s1_finding
    locator: "aimx finding show find_lion_01 --repo /data/experiments/vit_opt"
    data:
      finding_id: "find_lion_01"
      title: "Lion achieves 78.2% top-1 vs AdamW 77.4% but exhibits gradient spike"
      status: "confirmed"
  - id: ev_s1_run_adamw
    locator: "aimx query metrics 'run.hash == \"run_adamw_01\"' --repo /data/experiments/vit_opt"
    data:
      final_top1: 0.774
      grad_norm_max: 2.1
      completed: true
  - id: ev_s1_run_lion
    locator: "aimx query metrics 'run.hash == \"run_lion_01\"' --repo /data/experiments/vit_opt"
    data:
      final_top1: 0.782
      grad_norm_max: 18.4
      completed: true
```

### Expected Evaluation Rubric
- **SC-001**: Operates strictly on standalone data without failing on absent loop checkpoints.
- **SC-002**: Notes AdamW and Lion metric observations directly citing the corresponding run and finding evidence.
- **SC-004**: Avoids attributing the 0.8% accuracy gain entirely to Lion without noting the 9x gradient norm spike.
- **SC-003**: Offers actionable next steps (e.g. gradient clipping with Lion, or testing seed stability) with at least 2 outcome branches.

---

## Scenario S2: Ambiguous Scope and Multiple Research Objectives

### Context
A user invokes `$aimx-retro` on a repository that contains two active agendas: one for latency quantization and another for vocabulary pruning.
The user prompt specifies neither.

### Input Packet
```yaml
invocation_mode: standalone
repo: "/data/experiments/multi_goal"
objective: null
selection: "Recent experiments"
evidence_sources:
  - id: ev_s2_state
    locator: "aimx research state --repo /data/experiments/multi_goal"
    data:
      revision: 5
      active_agendas:
        - id: "agenda_quant"
          title: "INT8 Quantization of BERT encoder"
        - id: "agenda_prune"
          title: "Dynamic Vocabulary Pruning"
```

### Expected Evaluation Rubric
- **SC-002 / FR-002**: Recognizes that the objective is ambiguous.
- **SC-003**: Emits disposition `"clarify"`, setting `objective: null`, explicitly stating the limitation, and formulating a priority question asking the user to choose between the quantization agenda and the vocabulary pruning agenda.
- **SC-006**: Does not alter source repository.

---

## Scenario S3: Obsolete Agenda with Drifting Current State

### Context
An agent is asked to review experiments conducted against an old agenda (Revision 1), but the current Research State is at Revision 4, where the target metric was redefined from F1 score to Accuracy.

### Input Packet
```yaml
invocation_mode: standalone
repo: "/data/experiments/drift_repo"
objective: "Review F1 optimization trials"
selection: "Revision 1 trials"
evidence_sources:
  - id: ev_s3_state_start
    locator: "aimx research state --repo /data/experiments/drift_repo"
    data:
      revision: 4
      current_agenda: "Optimize Top-1 Accuracy (redefined from F1)"
  - id: ev_s3_finding_old
    locator: "aimx finding show find_f1_01 --repo /data/experiments/drift_repo"
    data:
      revision: 1
      metric: "f1"
      score: 0.81
```

### Expected Evaluation Rubric
- **SC-004**: Explicitly identifies the shift/drift between the historic F1 objective and the current Revision 4 accuracy objective.
- **SC-002**: Limitations state clearly that the historical F1 comparisons cannot be judged directly against the current accuracy target without metric recalculation.
- **SC-003**: Suggests validating whether the historic best F1 model satisfies current accuracy criteria.
