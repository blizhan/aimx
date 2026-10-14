# Lead-only Aimx research handoff

All commands below target the same explicit, absolute Aim repository. Only
the lead authors and submits ResearchUpdate. Subagents receive bounded context
and return results; their normal training workflow may log new runs to Aim,
but evidence inspection and the helper never mutate Aim metadata.

## Read and bootstrap

```bash
aimx research state --repo "$AIM_REPO" --json
aimx research context --repo "$AIM_REPO" --objective "$OBJECTIVE" --budget 12000 --json
aimx research agenda --repo "$AIM_REPO" --json
aimx research next --repo "$AIM_REPO" --json
```

Use the configured `context_budget_bytes`, not round budget, for `--budget`.
It counts the complete UTF-8 JSON response, including metadata, evidence
references, selection notes, and the terminating LF. A fixed limit may contain
fewer knowledge items than the older items-only behavior. If the required
correction closure cannot fit, context returns a budget error with complete
query commands and no one-sided partial result.
Refresh at round boundaries and before aggregate writes. Scope selection to
the current goal; a globally selected Agenda item may belong to other work.
Do not reactivate human-paused directions or override rejected Findings.

Empty state is revision 0. With no actionable Agenda, the **lead** may create
a Frontier lane/direction from the user's goal and motivated Agenda tasks in
one update. Agenda needs a Finding or Frontier motivation; a direction avoids
inventing a baseline research claim:

```json
{
  "schema_version": 1,
  "base_revision": 0,
  "client_update_id": "LOOP_ID-initial-agenda",
  "author": {"kind": "agent", "name": "LEAD_ID"},
  "operations": [
    {"op": "frontier.lane.create", "local_id": "lane", "name": "LOOP_ID"},
    {
      "op": "frontier.item.create",
      "local_id": "direction",
      "lane_id": "$lane",
      "direction": "The user's research goal",
      "rationale": "Explicit user-requested research direction."
    },
    {
      "op": "agenda.item.create",
      "local_id": "task",
      "objective": "This agent's bounded experiment objective",
      "question": "The question being tested",
      "frontier_item_ids": ["$direction"],
      "constraints": {"loop_id": "LOOP_ID", "round": 1, "agent_id": "optimizer"},
      "success_criteria": ["Task-specific evidence required"],
      "agenda_status": "active"
    }
  ]
}
```

Save returned `created_ids` with task artifacts. Each configured slot should
have its own Agenda contract. Future rounds reuse relevant Findings/Frontier
motivation and the newly integrated baseline. Preserve data controls,
experiment resource constraints, and human steering when generating new work.

## Observe and aggregate

```bash
aimx query params "run.hash == 'FULL_RUN_HASH'" --repo "$AIM_REPO" --json
aimx query metrics "(run.hash == 'FULL_RUN_HASH') and metric.name == 'accuracy'" --repo "$AIM_REPO" --json
aimx trace "(run.hash == 'FULL_RUN_HASH') and metric.name == 'accuracy'" --repo "$AIM_REPO" --json --tail 100
```

Discover actual metric names/contexts and inspect task evidence before drawing
conclusions. Run refs must resolve in this repository; preserve local output
artifacts outside disposable worktrees. Do not initialize/repair/migrate Aim
as part of observation or import invented runs to make an update pass.

The lead's aggregate update uses the existing envelope:

```json
{
  "schema_version": 1,
  "base_revision": 1,
  "client_update_id": "LOOP_ID-round-1-aggregate",
  "author": {"kind": "agent", "name": "LEAD_ID"},
  "operations": [
    {
      "op": "finding.create",
      "local_id": "candidate",
      "claim": "The supported research conclusion",
      "evidence": [{"kind": "aim_run", "run_hash": "FULL_RUN_HASH", "role": "candidate"}],
      "epistemic_status": "candidate",
      "confidence": "medium",
      "governance_status": "proposed"
    },
    {
      "op": "agenda.item.transition",
      "agenda_item_id": "RETURNED_TASK_ID",
      "agenda_status": "completed"
    }
  ]
}
```

Include appropriate `relation.create` operations between real Findings;
supported types include `supports`, `challenges`, `refines`, and `derived_from`.
For failed/interrupted tasks use Agenda `failed`/`abandoned` plus a reason,
or leave active when execution status remains unknown. If there is no reliable
empirical claim, write an `annotation.create` with the honest failure or
artifact-backed analysis instead of manufacturing a Finding. Non-Aim source
URLs/test logs can be referenced in annotation text and durable loop artifacts;
the ResearchUpdate evidence kind remains `aim_run`.

## Dry-run, commit, and reconcile

```bash
aimx research update --repo "$AIM_REPO" --file "$UPDATE_FILE" --dry-run --json
aimx research update --repo "$AIM_REPO" --file "$UPDATE_FILE" --json
```

Use the freshly read revision as `base_revision`; do not make it a round number.
Before committing, checkpoint the exact file and `client_update_id` in the
saved update artifact. After committing, checkpoint the returned `commit_id`,
`revision`, and that same ID. The lead then assesses the overall goal, which
is separate from task completion or human governance acceptance.

- Exit 3 / `revision_conflict`: refresh context and human steering, reconsider
  the aggregate, and save a new update file/key. Never just overwrite the old
  revision and blindly resubmit a stale conclusion.
- Uncertain commit after interruption: replay the **exact same saved update**
  and key through the commit command to retrieve the idempotent receipt. Do not
  regenerate content under that key. An already committed update may fail
  dry-run against its old revision, so reconciliation uses the idempotent
  commit path rather than treating that dry-run as evidence of no commit.
- An unresolved/ambiguous Aim reference rejects the complete update. Preserve
  the failure and existing state, then fix the reference from actual evidence.

The helper tracks the lead's receipts but never writes this database itself.
Human or external writes remain possible and must be reconciled throughout
the loop.

## Terminal follow-up

After a loop ends, create a separate `follow_up` Activity that references the
registered loop. Add a new scoped Finding with real file evidence and link it
to the earlier Finding with `supports`, `challenges`, or `refines`. Save the
exact update and stable key, dry-run, commit, and inspect its receipt. Do not
call `begin-round`, change the old checkpoint, alter its original criteria, or
claim that later historical coverage is future independent confirmation.

A `research receipt` result of `not_found` only means the read snapshot did not
see the key; another writer may still be committing. Reconcile uncertain
writes by querying the same key/fingerprint or replaying the exact saved update.
A fingerprint conflict stops for refresh and review with a new key; never
replace content under an existing key or replay a reserved host slot.
