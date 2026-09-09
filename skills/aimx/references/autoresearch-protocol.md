# Aimx AutoResearch Protocol

Aimx separates three responsibilities:

- The external agent or researcher reasons about the objective, changes code,
  and executes experiments.
- Aim stores raw experiment evidence such as runs, parameters, metrics,
  traces, images, and distributions.
- Aimx stores durable research meaning and policy in
  `<repo>/.aimx/research/state.sqlite3`.

The Aim repository remains read-only during this protocol. Research reads do
not create the sidecar. An explicit `research update` is the only operation in
the protocol that creates or changes Aimx Research State.

## Round handoff

Each round follows this sequence:

```text
research state/context
    -> research next/agenda
    -> external agent executes the Experiment Contract
    -> query/trace or collect_experiment_snapshot.py observes Aim
    -> research update records evidence and interpretation
    -> repeat
```

Start with a read of the current revision:

```bash
aimx research state --repo <repo> --json
aimx research context \
  --repo <repo> \
  --objective "improve low-data accuracy" \
  --budget 12000 \
  --json
aimx research next --repo <repo> --json
```

`research context` is deterministic for an unchanged state, objective, and
budget. Its `items` budget is exact UTF-8 serialized JSON bytes. Its
`evidence_refs` list contains canonical full Aim run hashes for deeper
inspection.

`research next` selects an already persisted Agenda Item. It selects the
highest-priority active item first, then the highest-priority proposed item;
ties use creation revision and stable ID. If no item is actionable, it returns
exit status `0` with `status: "no_actionable_agenda"` and `item: null`.

## Observe and update

Use the existing read-only evidence commands after the external experiment:

```bash
aimx query params --repo <repo> --json
aimx query metrics "metric.name == 'accuracy'" --repo <repo> --json
aimx trace "metric.name == 'accuracy'" --repo <repo> --json --tail 100
uv run python skills/aimx/scripts/collect_experiment_snapshot.py \
  --repo <repo> --metric accuracy --trace-metric accuracy --pretty
```

Then author a `ResearchUpdate` against the revision that was observed. A
single update can atomically attach evidence, transition an Agenda Item,
create Findings, connect them with lineage, and update Frontier policy. Use
`--dry-run` before the explicit commit:

```bash
aimx research update --repo <repo> --file update.json --dry-run --json
aimx research update --repo <repo> --file update.json --json
```

The update envelope is agent-product-neutral:

```json
{
  "schema_version": 1,
  "base_revision": 13,
  "client_update_id": "round-002-analysis",
  "author": {"kind": "agent", "name": "external-adapter"},
  "operations": [
    {
      "op": "finding.create",
      "local_id": "result",
      "claim": "The candidate remains better under seed 43.",
      "evidence": [
        {"kind": "aim_run", "run_hash": "<full-or-unambiguous-hash>", "role": "candidate"}
      ],
      "epistemic_status": "candidate",
      "confidence": "medium",
      "governance_status": "proposed"
    }
  ]
}
```

Aimx resolves short Aim hashes before persistence. An unresolved or ambiguous
reference rejects the complete update, leaving the previous state unchanged.

## Conflicts and handoff

Research State uses optimistic revision checks. If the state has advanced since
the context was compiled, the update exits with status `3` and a JSON error
whose code is `revision_conflict`. The caller should read a new state/context,
reconsider the proposed change, and submit a new `base_revision`; it should not
silently replay a decision made from stale context.

Human commands such as `finding comment`, `finding accept`, `lineage link`,
and `frontier pause` use the same ResearchUpdate path. Their changes therefore
remain visible to a later agent. A handoff from Codex to another agent product
requires no product-specific database: the next agent reads the same state,
context, and agenda JSON and submits the same update schema with its own
`author.name`.

## Empty and interrupted rounds

An empty state is a valid revision-zero response. No agenda item means no
experiment should be invented by Aimx. If an external experiment fails or is
interrupted, leave the Agenda Item active or transition it to `failed` or
`abandoned` with a reason; record only evidence that actually exists. Accepted
Findings and prior commits remain durable throughout the interruption.
