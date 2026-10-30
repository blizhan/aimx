# Evidence Collection Guide for Aimx Retro

This guide provides instructions and supported CLI commands for collecting research evidence when running `$aimx-retro`.

## Safe Evidence Collection Principles

1. **Read-Only**: NEVER execute commands that modify research state, record findings, commit updates, or start training runs during a retrospective.
2. **Explicit Provenance**: Record the exact command, run hash, finding ID, or file path in `locator`, and the field or metric selector in `selector`.
3. **Primary vs. Researcher Summary**:
   - `primary`: Raw metric logs, hyperparameter dumps from Aim runs, actual standard output / error streams.
   - `researcher_summary`: Human or agent-written summaries in research updates, commit notes, or finding descriptions.
   - *Rule*: Observations requiring high certainty (`basis: "observed"`) MUST reference primary evidence.
4. **Historical Knowledge vs. Hindsight**: What is recorded in state today was not necessarily known at the time of an earlier decision. When reconstructing a past decision point, only attribute information that was available at that specific time.

---

## Supported Read Commands

Execute read commands via the target environment's CLI (`aimx` or `uv run aimx` in this repo):

### 1. Research State and Lineage
```bash
# Read current overall research state
aimx research state --repo "$AIM_REPO" --json

# Read research agenda items
aimx research agenda --repo "$AIM_REPO" --json

# Read bounded research context for a specific objective
aimx research context --repo "$AIM_REPO" --objective "$OBJECTIVE" --budget 12000 --json

# Inspect a specific finding
aimx finding show "$FINDING_ID" --repo "$AIM_REPO" --json

# Inspect lineage of findings
aimx lineage show "$FINDING_ID" --repo "$AIM_REPO" --json

# Read current frontier
aimx frontier show --repo "$AIM_REPO" --json
```

### 2. Aim Run Metadata and Metrics
```bash
# Query run parameters and configuration
aimx query params "run.hash == '$RUN_HASH'" --repo "$AIM_REPO" --json

# Query recorded scalar metrics
aimx query metrics "(run.hash == '$RUN_HASH') and metric.name == '$METRIC'" --repo "$AIM_REPO" --json

# Query metric series trace (bounded tail)
aimx trace "(run.hash == '$RUN_HASH') and metric.name == '$METRIC'" --repo "$AIM_REPO" --json --tail 100
```

### 3. Convention-scoped Aim UI views

When autoresearch runs follow the unified run convention, scope every ready-made
Aim UI view by experiment, loop ID, and the examined round or rounds. For
example:

```text
run.experiment == 'ar-study-a' and run.aimx.loop_id == 'study-a' and run.aimx.round == 2
```

Reuse the selected scope on the `runs`, `metrics`, `params`, and `scatters`
pages where it supports the report. Confirm the params and tags from collected
run metadata; do not infer missing convention fields. Retro is read-only and
must not backfill Aim run metadata. Use [the Aim Views workflow](../SKILL.md)
for the UI base URL and installed-version deep-link verification rules.

---

## Scope Selection and Revision Tracking

1. **Repository Normalization**: Resolve `$AIM_REPO` to an absolute path. If a path ending with `.aim` is passed, normalize it to its parent directory.
2. **Start and End Revisions**:
   - Read `revision_start` at the beginning of evidence collection from `aimx research state --repo "$AIM_REPO" --json`.
   - If Research State is empty/initialized without commits, record `revision_start: 0`.
   - If Research State sidecar does not exist or cannot be read, record `revision_start: null`.
   - Before drafting the report, check `revision_end`. If `revision_end != revision_start`, note state drift in `limitations`.
3. **Objective Ambiguity**:
   - If no objective is supplied and multiple active agendas exist in research state, do not guess. Emit a `clarify` disposition and record the missing goal in `limitations`.

---

## Absence of Checkpoints or AutoResearch Loop

When reviewing standalone experiments (no `loop_id`, no `control.py status`):
- Set `loop_id: null` and `rounds: []` (or list examined round numbers if standalone rounds were identified).
- Do not attempt to invoke `control.py`. Missing sibling skills or absent loop directories must not impede retrospective analysis.

---

## Optional Checkpoint Evidence Branch (Autoresearch Loops)

When reviewing an autoresearch loop where checkpoints are available:

```bash
# Check loop status using autoresearch control script (read-only)
uv run python "$AUTORESEARCH_SKILL_DIR/scripts/control.py" status --loop-dir "$LOOP_DIR"
```

Rules for checkpoint evidence:
1. **Read-Only**: Use `status` ONLY. Never invoke `init`, `begin-round`, `checkpoint`, or state mutation commands.
2. **Checkpoints Are Secondary Summaries**:
   - Checkpoint files (`status.json`, round manifests) represent the Lead's recorded understanding at that moment (`provenance: "researcher_summary"`).
   - They do NOT replace primary metric data or verifiable task artifacts. Claims of task success or metric improvements must still be corroborated against primary logs or metric queries (`provenance: "primary"`).
3. **Direct File Read Fallback**:
   - If `$AUTORESEARCH_SKILL_DIR` is not installed or available, read the checkpoint artifact directly from the filesystem (e.g. `$LOOP_DIR/status.json`) with `kind: "artifact"`.
   - Clearly mark `locator: "/path/to/status.json"` and note in summary that helper status validation was omitted.
4. **Loop Preservation**:
   - Retrospective execution must NEVER change the loop status, unreserve agent slots, or alter remaining round budgets.

---

## Degraded and Incomplete Evidence Decision Rules

When evidence is missing, unreadable, interrupted, or inconsistent:

1. **Available vs Known**:
   - Available data in the repository does not prove that an agent or researcher was aware of it at decision time.
   - Keep `available_information` separate from `agent_awareness`.
2. **Reported vs Observed**:
   - Claims made in summary notes without primary logs have `basis: "reported"`.
   - Never upgrade a reported claim to `basis: "observed"` unless raw run metrics or stderr/stdout are directly read.
3. **Execution Before Hypothesis**:
   - Always check if the experiment finished cleanly before assessing whether the underlying hypothesis holds.
   - An execution crash (e.g. OOM, syntax error, divergence) is an execution failure, NOT scientific evidence disproving an algorithm.
4. **Unknown Costs and Rationale**:
   - If GPU time or rationale was unrecorded, set `cost: null` and `recorded_rationale.basis: "unknown"`.
   - Disclose the omission in `limitations`. Do not guess plausible rationales.
5. **Incomparable Metrics**:
   - If test splits, seeds, or evaluation scripts differed, note the comparison failure under `comparability`.
   - Treat apparent metric gains or losses as confounded until re-evaluated under identical conditions.
