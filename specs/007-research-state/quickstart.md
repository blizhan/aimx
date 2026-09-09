# Quickstart: Agent-First AutoResearch Control Plane

This quickstart describes the intended user/agent workflow after implementation. Aim remains the evidence store; Aimx Research State is a separate local sidecar.

## 1. Inspect an empty project

```bash
aimx research state --repo data --json
```

Expected: revision `0`, empty findings/frontier/agenda, and **no** `.aimx` directory created.

## 2. Inspect Aim evidence normally

Use existing read-only Aimx evidence commands:

```bash
aimx query params --repo data --json
aimx query metrics "metric.name == 'acc'" --repo data --json
aimx trace "metric.name == 'acc'" --repo data --json --tail 100
```

Or collect the existing compact evidence bundle:

```bash
uv run python skills/aimx/scripts/collect_experiment_snapshot.py \
  --repo data \
  --metric acc \
  --trace-metric acc \
  --pretty
```

## 3. Commit the first research knowledge

Create `update.json` using the revision you inspected:

```json
{
  "schema_version": 1,
  "base_revision": 0,
  "client_update_id": "round-001-analysis",
  "author": {"kind": "agent", "name": "codex"},
  "message": "Record first comparison and next experiment.",
  "operations": [
    {
      "op": "finding.create",
      "local_id": "f1",
      "claim": "The candidate configuration improves validation accuracy over the baseline.",
      "evidence": [
        {"kind": "aim_run", "run_hash": "<baseline-hash>", "role": "baseline"},
        {"kind": "aim_run", "run_hash": "<candidate-hash>", "role": "candidate"}
      ],
      "epistemic_status": "candidate",
      "confidence": "medium",
      "governance_status": "proposed"
    },
    {
      "op": "frontier.lane.create",
      "local_id": "lane1",
      "name": "promising",
      "description": "Directions worth another controlled experiment."
    },
    {
      "op": "frontier.item.create",
      "local_id": "front1",
      "lane_id": "$lane1",
      "finding_id": "$f1",
      "rationale": "The observed gain should be tested under another seed.",
      "priority": 80,
      "frontier_status": "active"
    },
    {
      "op": "agenda.item.create",
      "local_id": "a1",
      "objective": "Test whether the gain survives a second seed.",
      "hypothesis": "The candidate configuration remains better than baseline under seed 43.",
      "motivating_finding_ids": ["$f1"],
      "frontier_item_ids": ["$front1"],
      "controls": {"keep_fixed": ["dataset_split", "training_budget"]},
      "factors": {"seed": 43},
      "success_criteria": {"metric": "val/acc", "comparison": "candidate > baseline"},
      "priority": 80,
      "agenda_status": "proposed"
    }
  ]
}
```

Validate without writing:

```bash
aimx research update --repo data --file update.json --dry-run --json
```

Then commit explicitly:

```bash
aimx research update --repo data --file update.json --json
```

Short run hashes are accepted only when unambiguous; committed Evidence References contain full run hashes.

## 4. Resume from bounded context

```bash
aimx research context \
  --repo data \
  --objective "improve low-data accuracy" \
  --budget 12000 \
  --json
```

`--budget` is measured in UTF-8 bytes of the selected context items, not model-specific tokens. The output includes the exact Research State revision and evidence references for deeper Aim inspection.

## 5. Ask for the next durable experiment contract

```bash
aimx research agenda --repo data --json
aimx research next --repo data --json
```

Aimx selects from already proposed/active Agenda Items; it does not invent a new hypothesis. The external agent decides what to propose and executes the experiment.

## 6. Human intervention persists across agents

```bash
aimx finding ls --repo data
aimx finding accept <finding-id> --reason "Worth carrying forward" --repo data

aimx frontier show --repo data
aimx frontier move <frontier-item-id> --lane <lane-id> --repo data
```

These convenience commands create the same durable commit/event history as an agent ResearchUpdate.

## 7. Complete the experiment externally

The external agent modifies code/config and launches training using the project's normal workflow. Aim records the new run. Aimx does not host or schedule the agent/training process.

After completion, inspect the new run with existing `$aimx`/query/trace evidence commands.

## 8. Record the result and close the agenda item

Submit one ResearchUpdate against the latest revision that can atomically:

- attach the new Aim run to the Agenda Item,
- transition the Agenda Item to `completed` or `failed`,
- create one or more Findings,
- connect them to earlier Findings with `supports`, `challenges`, `refines`, or `supersedes`,
- update Frontier policy when appropriate.

Then begin the next round:

```text
research context
    -> research next / agenda
    -> external experiment
    -> Aim evidence
    -> research update
    -> research context ...
```

The next agent session should see the previous round through Research State without requiring the prior conversation transcript.

## Conflict handling

If Research State changed after an agent compiled context, submitting an update with the old `base_revision` fails with exit status `3` and `revision_conflict`. The agent must fetch new state/context and reconsider; Aimx does not silently merge or retry semantic decisions.

## Safety checks

During implementation/acceptance testing, verify:

```bash
# Reads must not create research storage in a fresh repo
test ! -e data/.aimx/research/state.sqlite3

aimx research state --repo data --json >/tmp/state.json

test ! -e data/.aimx/research/state.sqlite3

# Existing evidence inspection remains usable
uv run aimx query params --repo data --json >/tmp/params.json
uv run pytest tests/integration/test_passthrough_behavior.py
```
