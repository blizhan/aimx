---
name: aimx-retro
description: Review Aimx research state and experiment evidence, reconstruct decisions with honest awareness boundaries, challenge conclusions, and recommend the next research step in a structured report.
---

# Aimx Retro

Use this skill to conduct evidence-based retrospectives on research recorded in an
Aim repository and its optional Aimx research state / autoresearch loop artifacts.
This skill works both standalone (invoked by a user or fresh Agent) and between
rounds of an autoresearch loop (invoked by a Lead).

## Core Principles

1. **Evidence-first**: Reconstruct what was known and observable at decision time, separating historical awareness from subsequent outcomes and hindsight.
2. **Honest unknowns**: When data, rationale, or costs are unavailable, mark them explicitly as unknown; never fabricate explanations.
3. **Read-only on research sources**: Only create report artifacts; never modify `.aim`, `.aimx/research`, loop checkpoints, or source repositories.
4. **Actionable handoff**: Propose clear next steps with concrete outcome branches, or explicitly recommend stopping, clarifying, or gathering evidence.

## Ordered Retrospective Workflow

Follow these six ordered phases during every retrospective:

### Phase 1: Scope & Repository Resolution
- Resolve the absolute repository root (if path ends with `.aim`, use its parent directory).
- Determine the research objective. If multiple candidate objectives exist and none was specified, do not guess; prepare a `clarify` report.
- Record `read_started_at` and `revision_start` using `aimx research state --repo "$AIM_REPO" --json`.
- Consult [references/evidence-guide.md](references/evidence-guide.md) for supported read commands.

### Phase 2: Evidence Collection
- Collect primary metric data and run metadata via `aimx query params` and `aimx query metrics`.
- Read state summaries, findings, and lineage records.
- For each piece of evidence, assign a unique `id` (e.g. `ev_01`), classify its `kind`, record exact `locator` and `selector`, and distinguish `provenance: "primary"` from `"researcher_summary"`.
- If an expected source is missing or unreadable, mark `availability: "missing"` or `"unreadable"`.
- For autoresearch runs, inspect `run.experiment`, `run.aimx.*`, and tags when collecting metadata. Treat missing convention fields as unknown; do not infer them from task names or nearby runs.

### Phase 3: Chronological Decision Reconstruction
- Identify key decision points chronologically.
- For each decision, reconstruct:
  - `available_information`: What was observable in data at that point.
  - `agent_awareness`: What the researcher or agent actually knew or considered.
  - `alternatives` & `constraints`: What options and boundaries existed.
  - `choice` & `recorded_rationale`: What was picked and why (`basis: "unknown"` if unrecorded).
  - `execution`: Did the run finish cleanly without crash or divergence? (Verify before analyzing metrics).
  - `comparability`: Are the metrics comparable (identical splits, baselines, seeds)?
  - `observation` & `interpretation`: Direct metric outcomes and then-current interpretation.
  - `subsequent_action`: What was dispatched next.

### Phase 4: Six-Dimension Analysis & Reflection
Examine decisions across six core dimensions:
1. **Hypothesis**: Was the underlying premise sound?
2. **Strategy**: Was the exploration/exploitation balance appropriate?
3. **Experiment**: Were controls and baselines properly isolated?
4. **Execution**: Did scripts, hardware, or numerical stability fail?
5. **Reasoning**: Were interpretations supported by data or clouded by hindsight?
6. **Decision**: Was the choice logical given the constraints?

Author 0 to 3 actionable Reflections:
- Differentiate `causal_assessment`: `"supported"`, `"plausible"`, or `"undetermined"`.
- Offer competing `alternative_explanations` and document any `counterevidence`.
- Distinguish between execution failure (crashes, OOM, divergence) and hypothesis failure.
- Never force negative criticism: if an experiment followed a sound process and produced a negative scientific result, record the sound process and retain the methodology.
- Clearly specify `behavior` (`adjust` vs `retain`), `conditions`, `exceptions`, and how to validate.

### Phase 5: Recommendation & Next Steps
- Formulate a clear `priority_question` for the Lead.
- Choose a single `disposition`:
  - `recommend`: Advance an experiment or action with testable outcome branches.
  - `gather_evidence`: When critical data is missing before a sound decision can be made.
  - `stop`: When the research goal is reached or budget/diminishing returns make continuation inadvisable.
  - `clarify`: When scope or objective is ambiguous.
- Options must specify controls, prerequisites, switch conditions, and for experiments, at least two outcome branches (`if result A -> decision B`).

### Phase 6: Validation & Publication
- Check `revision_end`. If state revision drifted during the review, log it in `limitations`.
- Ensure all statements cite valid evidence items and follow basis rules (e.g., `observed` requires primary read evidence).

When the reviewed autoresearch runs follow the run convention, add ready-made
`aim_views` for the `runs`, `metrics`, `params`, and `scatters` pages where they
help compare the evidence. Scope each AimQL query by both experiment and loop:
`run.experiment == 'ar-{loop_id}' and run.aimx.loop_id == '{loop_id}'`, then
include the exact selected round or rounds, for example
`and (run.aimx.round == 1 or run.aimx.round == 2)`. Use `group_by` or `color_by`
such as `run.aimx.round` and `run.aimx.role` when useful. Set `axes` only for
`params` or `scatters` views and use axes supported by the inspected run data.
Include `evidence_ids` and `decision_ids` when a view supports specific report
claims. Other supported pages (`images`, `figures`, `text`, `distributions`)
may be included when the evidence calls for them. Do not create a loop-scoped
view when the run convention is absent or incomplete.

Use the host and port printed by `aimx up --repo "$AIM_REPO"` as the Aim UI base
URL. Set a view's `url` only after verifying the deep-link encoding against the
installed Aim version: enter the same page and query in that UI, use its
resulting URL, then reopen it and confirm the page and query match. Do not guess
routes or URL encoding. Until that check succeeds, set `url: null`; the rendered
report gives the page and query for pasting into Aim's search box.

- Reference the normative schema in [references/report-format.md](references/report-format.md).
- Validate the draft JSON with the standalone script:
  ```bash
  uv run python "$RETRO_SKILL_DIR/scripts/report.py" validate --file draft.json
  ```
- Publish the canonical report pair (`retro.json` and `RETRO.md`):
  ```bash
  uv run python "$RETRO_SKILL_DIR/scripts/report.py" publish --file draft.json
  ```
