# Retro Report Format Reference (Version 1)

This reference defines the normative JSON schema and rendering contract for `$aimx-retro` reports.

## Execution and Helper Commands

The installed helper script `scripts/report.py` validates drafts and publishes report pairs:

```bash
# Validate draft JSON (read-only, exits 0 on valid, 2 on invalid)
uv run python "$RETRO_SKILL_DIR/scripts/report.py" validate --file draft.json

# Publish draft into authoritative pair: retro.json and RETRO.md
# Default destination: <scope.repo>/.aimx/retro/<report_id>/
uv run python "$RETRO_SKILL_DIR/scripts/report.py" publish --file draft.json

# Publish to an explicit safe output root
uv run python "$RETRO_SKILL_DIR/scripts/report.py" publish --file draft.json --output-root /custom/output/dir
```

---

## Schema Version 1 Specification

### Envelope Fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | integer | Must be exactly `1`. |
| `report_id` | string | Safe directory name matching `^[a-zA-Z0-9_-]+$`. No slashes, dots, or spaces. |
| `created_at` | string | ISO 8601 timestamp with timezone (e.g. `2026-10-29T10:00:00Z` or `+08:00`). |
| `mode` | string | Either `"standalone"` or `"between_rounds"`. |
| `scope` | object | Scope specification (see below). |
| `evidence` | array | Array of Evidence items (unique `id`s). |
| `decisions` | array | Array of Decision reconstructions (unique `id`s). |
| `knowledge_updates` | array | Array of KnowledgeUpdate items. |
| `reflections` | array | Array of 0 to 3 Reflection items (unique `id`s). |
| `next_step` | object | NextStep recommendation specification. |
| `limitations` | array of strings | Nonempty strings listing gaps, omissions, or unverified claims. |
| `aim_views` | array (optional) | Ready-made Aim UI views rendered in `RETRO.md`; absent is valid for older drafts. |

### Aim View Item

`aim_views` is an optional top-level array. Each item has these fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique view identifier within the report. |
| `page` | string | One of `runs`, `metrics`, `params`, `scatters`, `images`, `figures`, `text`, or `distributions`. |
| `purpose` | string | Nonempty reason for opening this view. |
| `query` | string | Nonempty AimQL query to paste into the selected page's search box. |
| `group_by` | array of strings, optional | Aim fields to group by. |
| `color_by` | array of strings, optional | Aim fields to color by. |
| `axes` | array of strings, optional | Axes for `params` or `scatters` pages only. |
| `evidence_ids` | array of strings, optional | References to existing Evidence IDs. |
| `decision_ids` | array of strings, optional | References to existing Decision IDs. |
| `url` | string or null | Verified Aim UI deep link, or `null` when deep-link encoding has not been verified for the installed Aim version. |

All fields except the four marked optional are required. The helper checks page
names, field types, unique view IDs, and evidence/decision references; it does
not parse AimQL. When `url` is `null`, `RETRO.md` names the page and tells the
reader to paste `query` into its search box.

---

### Scope Object

| Field | Type | Description |
|---|---|---|
| `repo` | string | Absolute path to repository root. |
| `objective` | string or null | Research objective. `null` if unspecified. |
| `selection` | string | Nonempty description of selected range, question, or direction. |
| `read_started_at` | string | ISO 8601 timestamp with timezone. |
| `read_finished_at` | string | ISO 8601 timestamp with timezone (must be >= `read_started_at`). |
| `revision_start` | integer or null | Aimx research revision at start of read (`0` = empty, `null` = unavailable). |
| `revision_end` | integer or null | Aimx research revision at end of read. If different from `revision_start`, drift must be noted in limitations. |
| `loop_id` | string or null | Autoresearch loop identifier or `null`. |
| `rounds` | array of integers | List of examined round numbers (e.g. `[1, 2]`). |
| `constraints` | array of strings | Known budget, resource, or operational constraints. |
| `coverage` | string | Nonempty description of what was examined vs omitted. |

---

### Evidence Item

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique evidence identifier (e.g. `"ev_01"`). |
| `kind` | string | `"research_state"`, `"aim_run"`, `"artifact"`, or `"user_input"`. |
| `locator` | string | Precise source reference (e.g. `aimx finding show find_01`, file path, run hash). Never an executable injection. |
| `selector` | string | JSON pointer, metric name, step, or field path within locator. |
| `captured_at` | string | ISO 8601 timestamp with timezone when data was read. |
| `source_time` | string or null | ISO 8601 timestamp when source was originally created, or `null`. |
| `source_revision`| integer or null | Source revision or `null`. |
| `availability` | string | `"read"`, `"missing"`, or `"unreadable"`. |
| `provenance` | string | `"primary"` (raw data/metrics/runs) or `"researcher_summary"` (notes/summaries). |
| `summary` | string | Nonempty factual description of the evidence content. |

---

### Statement Object (Reusable)

Used across decisions, reflections, knowledge updates, and next steps:

| Field | Type | Description |
|---|---|---|
| `text` | string | Nonempty statement text. |
| `basis` | string | `"observed"`, `"reported"`, `"inferred"`, or `"unknown"`. |
| `evidence_ids` | array of strings | References to Evidence IDs. |

Rules:
- `basis: "observed"` REQUIRES at least one evidence ID whose evidence has `availability: "read"` and `provenance: "primary"`.
- `basis: "reported"` REQUIRES at least one evidence ID whose evidence has `availability: "read"`.
- `basis: "inferred"` REQUIRES at least one evidence ID with `availability: "read"` and text indicating inference.
- `basis: "unknown"` may have an empty `evidence_ids` list.

---

### Decision Object

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique decision ID (e.g. `"dec_01"`). |
| `position` | string | Nonempty chronology label (e.g. `"round_1_dispatch"`). |
| `source_time` | string or null | ISO 8601 timestamp when decision occurred, or `null`. |
| `available_information` | array of Statements | What data was available at that moment. |
| `agent_awareness` | Statement | What the agent actually knew or considered. |
| `alternatives` | array of Statements | Options that were considered or available. |
| `constraints` | array of Statements | Constraints active at decision time. |
| `choice` | Statement | What action was chosen. |
| `recorded_rationale` | Statement | Justification given at decision time (`basis: "unknown"` if unrecorded). |
| `expectation` | Statement | Hypothesized or expected outcome. |
| `execution` | Statement | Whether the chosen step executed properly before evaluating results. |
| `comparability` | Statement | Whether metrics/conditions are comparable to baselines. |
| `observation` | Statement | What was directly observed in the results. |
| `interpretation` | Statement | How results were interpreted. |
| `subsequent_action` | Statement | Next step taken after this decision. |

---

### KnowledgeUpdate Object

| Field | Type | Description |
|---|---|---|
| `change` | string | `"added"`, `"weakened"`, `"unresolved"`, or `"retained"`. |
| `statement` | Statement | The updated knowledge statement. |
| `decision_ids` | array of strings | References to relevant Decision IDs. |

---

### Reflection Object (Max 3)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique reflection ID (e.g. `"ref_01"`). |
| `dimensions` | array of strings | Subset of `["hypothesis", "strategy", "experiment", "execution", "reasoning", "decision"]`. |
| `decision_ids` | array of strings | References to Decision IDs. |
| `observation` | Statement | Factual observation underpinning the reflection. |
| `causal_assessment` | string | `"supported"`, `"plausible"`, or `"undetermined"`. |
| `explanation` | Statement | Explanation for the observed outcome. |
| `alternative_explanations` | array of Statements | Competing explanations considered. |
| `counterevidence` | array of Statements | Evidence conflicting with the explanation. |
| `behavior` | string | `"retain"` or `"adjust"`. |
| `action` | string | Concrete behavioral adjustment or practice to retain. |
| `conditions` | array of strings | When this adjustment applies. |
| `exceptions` | array of strings | When this adjustment does not apply. |
| `validation` | string | How to verify whether this adjustment works. |
| `validation_status` | string | `"untested"` or `"tested"`. |
| `validation_evidence_ids` | array of strings | If `"tested"`, references to evidence proving the validation. |

---

### NextStep and Option Objects

**NextStep**:
| Field | Type | Description |
|---|---|---|
| `priority_question` | string | The primary question for the Lead to resolve. |
| `disposition` | string | `"recommend"`, `"gather_evidence"`, `"stop"`, or `"clarify"`. |
| `rationale` | Statement | Justification for this disposition. |
| `recommended_option_id`| string or null | Must be non-null for `"recommend"` and `"gather_evidence"`, pointing to an option ID. |
| `options` | array of Options | 0 to 3 Option items. |
| `recheck_conditions` | array of strings | Conditions that would invalidate this recommendation. |

**Option**:
| Field | Type | Description |
|---|---|---|
| `id` | string | Unique option ID (e.g. `"opt_01"`). |
| `kind` | string | `"experiment"`, `"inspection"`, `"continue"`, `"stop"`, or `"clarify"`. |
| `question` | string | Question addressed by this option. |
| `action` | string | Concrete proposed action. |
| `rationale` | Statement | Justification for this option. |
| `controls` | array of strings | Control variables or experimental safeguards. |
| `cost` | string or null | Estimated cost (or `null` if unknown). |
| `prerequisites` | array of strings | What must be ready before taking this action. |
| `applicability` | array of strings | Circumstances where this option applies. |
| `switch_conditions` | array of strings | Triggers to abandon this option. |
| `outcome_branches` | array of objects | Each `{ "result": "...", "next_decision": "..." }`. Experiments require at least 2 branches unless `independent_value` is non-null. |
| `independent_value` | string or null | Explains value even if primary outcome fails. |

---

## Minimal Valid Example

```json
{
  "schema_version": 1,
  "report_id": "retro_minimal_01",
  "created_at": "2026-10-29T10:00:00Z",
  "mode": "standalone",
  "scope": {
    "repo": "/path/to/repo",
    "objective": "Optimize training loss",
    "selection": "Round 1 evaluation",
    "read_started_at": "2026-10-29T09:55:00Z",
    "read_finished_at": "2026-10-29T10:00:00Z",
    "revision_start": 1,
    "revision_end": 1,
    "loop_id": null,
    "rounds": [1],
    "constraints": ["budget: 100 GPU hours"],
    "coverage": "Inspected run run_01 metrics and state"
  },
  "evidence": [
    {
      "id": "ev_01",
      "kind": "aim_run",
      "locator": "run_hash_abc123",
      "selector": "loss",
      "captured_at": "2026-10-29T09:56:00Z",
      "source_time": "2026-10-29T09:00:00Z",
      "source_revision": 1,
      "availability": "read",
      "provenance": "primary",
      "summary": "Run completed with final loss 0.42"
    }
  ],
  "decisions": [
    {
      "id": "dec_01",
      "position": "initial_dispatch",
      "source_time": "2026-10-29T08:50:00Z",
      "available_information": [
        { "text": "Baseline loss was 0.50", "basis": "observed", "evidence_ids": ["ev_01"] }
      ],
      "agent_awareness": { "text": "Agent was aware of baseline", "basis": "observed", "evidence_ids": ["ev_01"] },
      "alternatives": [
        { "text": "Learning rate 1e-4 vs 1e-3", "basis": "observed", "evidence_ids": ["ev_01"] }
      ],
      "constraints": [
        { "text": "Budget constraint 100h", "basis": "observed", "evidence_ids": ["ev_01"] }
      ],
      "choice": { "text": "Selected lr 1e-4", "basis": "observed", "evidence_ids": ["ev_01"] },
      "recorded_rationale": { "text": "Standard warm start", "basis": "observed", "evidence_ids": ["ev_01"] },
      "expectation": { "text": "Loss reduction below 0.45", "basis": "observed", "evidence_ids": ["ev_01"] },
      "execution": { "text": "Run completed successfully", "basis": "observed", "evidence_ids": ["ev_01"] },
      "comparability": { "text": "Metrics comparable to baseline", "basis": "observed", "evidence_ids": ["ev_01"] },
      "observation": { "text": "Final loss was 0.42", "basis": "observed", "evidence_ids": ["ev_01"] },
      "interpretation": { "text": "Improvement over baseline", "basis": "observed", "evidence_ids": ["ev_01"] },
      "subsequent_action": { "text": "Lead initiated retro review", "basis": "observed", "evidence_ids": ["ev_01"] }
    }
  ],
  "knowledge_updates": [
    {
      "change": "added",
      "statement": { "text": "lr 1e-4 achieves 0.42 loss", "basis": "observed", "evidence_ids": ["ev_01"] },
      "decision_ids": ["dec_01"]
    }
  ],
  "reflections": [],
  "next_step": {
    "priority_question": "Should we proceed to evaluate learning rate schedule?",
    "disposition": "recommend",
    "rationale": { "text": "Initial target met; next optimization phase ready", "basis": "observed", "evidence_ids": ["ev_01"] },
    "recommended_option_id": "opt_01",
    "options": [
      {
        "id": "opt_01",
        "kind": "experiment",
        "question": "Does cosine schedule improve convergence?",
        "action": "Train 1 run with cosine schedule from lr 1e-4",
        "rationale": { "text": "Known to stabilize final loss", "basis": "observed", "evidence_ids": ["ev_01"] },
        "controls": ["Fixed seed 42", "Batch size 64"],
        "cost": "10 GPU hours",
        "prerequisites": ["Baseline checkpoints available"],
        "applicability": ["Loss plateau reached"],
        "switch_conditions": ["Loss diverges at step 100"],
        "outcome_branches": [
          { "result": "loss < 0.40", "next_decision": "Adopt cosine schedule for production" },
          { "result": "loss >= 0.40", "next_decision": "Keep constant lr 1e-4" }
        ],
        "independent_value": null
      }
    ],
    "recheck_conditions": [
      "Hardware environment changes",
      "Dataset split updated"
    ]
  },
  "limitations": [
    "Only 1 run hash inspected; run variance across random seeds is unmeasured."
  ]
}
```


## Publication guarantees

All schema fields must be present, including nullable fields; use explicit `null`
for unknown values. The Markdown report retains each statement's evidence IDs and
reflection validation evidence IDs. Source text is rendered literally using
character entities, so Markdown/HTML in evidence cannot alter report structure;
the authoritative JSON retains the original strings.

Pass the repository root in `scope.repo`, not its `.aim` metadata directory.
Publication rejects output paths inside any `.aim`, `.aimx/research`, or
`.aimx/autoresearch` subtree, including other repositories and paths reached via
symlinks. This check includes the final report directory, not just the output root.
