# CLI and Machine-Readable Contract: Research Control Plane

## Command ownership

The following roots become Aimx-owned:

```text
aimx research ...
aimx finding ...
aimx lineage ...
aimx frontier ...
```

Existing Aimx-owned roots remain unchanged. Every other root remains native Aim passthrough.

## Repository targeting

All commands accept `--repo <path>` where relevant. Paths may point to the Aim repository root or its `.aim` directory. Research State belongs to the normalized repository root under `.aimx/research/`; `.aim` is never written.

## Core agent commands

### `aimx research state`

```text
aimx research state [--repo <path>] [--json]
```

Read-only. Missing research storage returns revision `0` and an empty state without creating `.aimx`.

JSON success envelope:

```json
{
  "schema_version": 1,
  "revision": 12,
  "repo": "/project",
  "findings": [],
  "relations": [],
  "annotations": [],
  "frontier": {"lanes": [], "items": []},
  "agenda": {"items": []}
}
```

### `aimx research update`

```text
aimx research update --repo <path> --stdin [--dry-run] [--json]
aimx research update --repo <path> --file <update.json> [--dry-run] [--json]
```

`--stdin` and `--file` are mutually exclusive. Input follows `research-update.schema.json`.

Success:

```json
{
  "schema_version": 1,
  "status": "committed",
  "revision": 13,
  "commit_id": "commit_...",
  "client_update_id": "agent-round-42",
  "created_ids": {
    "f1": "f_...",
    "r1": "rel_..."
  }
}
```

Dry-run success:

```json
{
  "schema_version": 1,
  "status": "valid",
  "base_revision": 12,
  "would_create": ["finding", "relation"]
}
```

Dry-run MUST NOT create `.aimx`, allocate a visible revision, or persist entity IDs.
Its `would_create` list contains entity kinds only; IDs generated while
validating the in-memory update are discarded.

### `aimx research context`

```text
aimx research context --objective <text> --budget <bytes> [--repo <path>] [--json]
```

`--budget` is the maximum UTF-8 byte size of the serialized `items` payload.

JSON success envelope:

```json
{
  "schema_version": 1,
  "compiler_version": "context-v1",
  "revision": 13,
  "objective": "improve low-data accuracy",
  "budget_unit": "utf8_bytes",
  "budget_limit": 12000,
  "budget_used": 6842,
  "items": [],
  "evidence_refs": [],
  "selection_notes": {}
}
```

If mandatory contradiction closure cannot fit, the command fails rather than returning a misleading partial context.

### `aimx research agenda`

```text
aimx research agenda [--repo <path>] [--status <status>] [--json]
```

Read-only. Returns durable Agenda Items sorted by status/priority/creation order according to the documented deterministic policy.
The optional `--status` filter accepts `proposed`, `active`, `completed`,
`failed`, or `abandoned`; another value is an exit-status `2` validation error.

### `aimx research next`

```text
aimx research next [--repo <path>] [--json]
```

Selection policy:

1. Highest-priority `active` Agenda Item.
2. If none, highest-priority `proposed` Agenda Item.
3. Tie: lower creation revision, then stable ID.
4. If none exists, return a successful no-work envelope; never invent an experiment.

No-work JSON:

```json
{
  "schema_version": 1,
  "revision": 13,
  "status": "no_actionable_agenda",
  "item": null
}
```

## Human intervention commands

Read commands:

```text
aimx finding ls [--repo <path>] [--json]
aimx finding show <finding-id> [--repo <path>] [--json]
aimx lineage show <finding-id> [--repo <path>] [--json]
aimx frontier show [--repo <path>] [--json]
```

Explicit write commands:

```text
aimx finding comment <finding-id> <text> [--repo <path>]
aimx finding accept <finding-id> [--reason <text>] [--repo <path>]
aimx finding reject <finding-id> [--reason <text>] [--repo <path>]
aimx finding assess <finding-id> --status <status> --confidence <level> [--reason <text>] [--repo <path>]

aimx lineage link <source-id> <relation-type> <target-id> [--reason <text>] [--repo <path>]
aimx lineage retract <relation-id> --reason <text> [--repo <path>]

aimx frontier lane-add <name> [--description <text>] [--repo <path>]
aimx frontier add --lane <lane-id-or-name> (--finding <id> | --direction <text>) --rationale <text> [--priority N] [--repo <path>]
aimx frontier move <item-id> --lane <lane-id-or-name> [--repo <path>]
aimx frontier pause <item-id> [--reason <text>] [--repo <path>]
aimx frontier retire <item-id> [--reason <text>] [--repo <path>]
```

Human write commands compile to one internal ResearchUpdate against the revision they read. A concurrent change returns revision conflict; the command does not silently retry against unseen state.

## Exit status and errors

| Exit | Meaning |
| ---: | --- |
| `0` | Success, including valid empty/no-work results |
| `1` | Unexpected/internal execution failure |
| `2` | Invalid command input, schema/reference/state-transition/evidence validation failure |
| `3` | Research State revision conflict; caller must refresh and reconsider/retry |

In `--json` mode, expected errors are written to stderr as:

```json
{
  "schema_version": 1,
  "error": {
    "code": "revision_conflict",
    "message": "Research state advanced from revision 12 to 13.",
    "details": {
      "base_revision": 12,
      "current_revision": 13
    }
  }
}
```

Stable error codes include at least:

- `invalid_input`
- `invalid_schema`
- `unknown_entity`
- `invalid_transition`
- `unresolved_evidence`
- `ambiguous_evidence`
- `idempotency_conflict`
- `revision_conflict`
- `budget_too_small_for_required_context`
- `research_store_unreadable`

## Deterministic output rules

- All JSON envelopes include `schema_version` and the Research State `revision` when state was read.
- Entity lists use deterministic ordering; no command relies on database row-return order.
- Timestamps are UTC ISO-8601 values and are not part of context selection tie-breaking.
- Evidence references expose canonical full Aim run hashes.
- Read commands never initialize or mutate Research State.
