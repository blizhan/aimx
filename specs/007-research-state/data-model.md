# Data Model: Agent-First AutoResearch Control Plane

## Identity and versioning conventions

All durable records use opaque IDs with human-recognizable prefixes:

- `commit_<uuid>` — Research Commit
- `event_<uuid>` — Research Event
- `f_<uuid>` — Finding
- `rel_<uuid>` — Relation
- `ann_<uuid>` — Annotation
- `lane_<uuid>` — Frontier Lane
- `front_<uuid>` — Frontier Item
- `agenda_<uuid>` — Agenda Item

UUID generation is local and collision-resistant. IDs are stable and never reused. All external JSON envelopes carry `schema_version: 1`. Context output also carries a `compiler_version` so deterministic selection changes can be versioned independently.

`revision` is a repository-local monotonically increasing integer. Revision `0` represents an empty Research State before the first explicit write.

## Storage records

### Metadata

| Field | Type | Rules |
| --- | --- | --- |
| `key` | string | Primary key |
| `value` | string | Includes `schema_version` and `current_revision` |

### ResearchCommit

One accepted ResearchUpdate produces exactly one commit.

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Commit ID | Immutable, unique |
| `revision` | integer | Positive, unique, contiguous for committed writes |
| `base_revision` | integer | Must equal revision immediately before commit |
| `client_update_id` | string/null | Optional unique idempotency key |
| `author_kind` | `agent \| human \| system` | Required |
| `author_name` | string | Required, non-empty |
| `created_at` | UTC timestamp | Assigned at commit |
| `message` | string/null | Optional concise rationale |
| `schema_version` | integer | `1` for V1 |

### ResearchEvent

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Event ID | Immutable, unique |
| `commit_id` | Commit ID | Existing commit |
| `sequence` | integer | Zero-based order within commit; unique per commit |
| `event_type` | string | One supported operation/event type |
| `entity_id` | entity ID/null | Primary affected entity when applicable |
| `payload` | JSON object | Schema-versioned event payload |

Events are the source of truth. Current state is derived by replay in `(revision, sequence)` order.

## ResearchUpdate

External write envelope:

| Field | Type | Rules |
| --- | --- | --- |
| `schema_version` | integer | Required; V1 is `1` |
| `base_revision` | integer | Required and must equal current revision |
| `client_update_id` | string/null | Optional idempotency key |
| `author.kind` | enum | `agent`, `human`, or `system` |
| `author.name` | string | Required |
| `message` | string/null | Optional summary/rationale |
| `operations` | array | Non-empty ordered list; all validate before commit |

An operation may define `local_id`, unique inside the update. Later operations reference it with `$<local_id>`. Existing entities are referenced by durable IDs. All local and durable references resolve before the commit is accepted.

## Finding

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Finding ID | Stable |
| `claim` | string | Required, non-empty; immutable after creation |
| `evidence` | EvidenceReference[] | At least one for evidence-grounded claims unless explicitly marked as an untested hypothesis |
| `epistemic_status` | enum | `candidate`, `validated`, `contradicted` |
| `confidence` | enum | `low`, `medium`, `high` |
| `governance_status` | enum | `proposed`, `accepted`, `rejected` |
| `created_commit_id` | Commit ID | Provenance |
| `created_revision` | integer | Provenance |
| `author` | Author | Inherited from creating commit |

### Finding transitions

Claim text is not editable. Allowed state events:

- `finding.assess`: change `epistemic_status` and/or `confidence`, with reason.
- `finding.governance`: change governance status, with reason.
- Meaning change: create a new Finding and a `refines` or `supersedes` Relation.

Scientific and governance states are independent; e.g. `candidate + accepted` is valid.

## EvidenceReference

V1 supports Aim run evidence:

| Field | Type | Rules |
| --- | --- | --- |
| `kind` | string | `aim_run` |
| `run_hash` | string | Persisted as canonical full Aim run hash |
| `role` | string/null | Optional e.g. `baseline`, `candidate`, `counterexample` |
| `note` | string/null | Optional concise explanation of relevance |
| `availability` | string | Read-time projection only: `available` or `unavailable`; not persisted as evidence identity |

Input may use an unambiguous short prefix; validation persists the full hash. Unresolvable evidence blocks commit. If the Aim run later disappears, the historical reference remains but reads mark resolution unavailable.

## Relation

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Relation ID | Stable |
| `source_finding_id` | Finding ID | Existing or same-update local reference |
| `type` | enum | `derived_from`, `supports`, `challenges`, `refines`, `supersedes`, `tests`, `related_to` |
| `target_finding_id` | Finding ID | Existing or same-update local reference |
| `active` | boolean | True at creation; false after retraction |
| `reason` | string/null | Optional relation rationale |
| `created_commit_id` | Commit ID | Provenance |
| `retracted_commit_id` | Commit ID/null | History when retracted |

Self-relations are invalid. Duplicate active identical `(source,type,target)` relations are rejected unless the operation explicitly references/retracts the existing relation.

## Annotation

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Annotation ID | Stable |
| `target_kind` | enum | `research_state`, `finding`, `relation`, `frontier_item`, `agenda_item` |
| `target_id` | ID/null | Null only for research-state-level annotation |
| `text` | string | Required, non-empty |
| `author` | Author | From commit |
| `created_commit_id` | Commit ID | Provenance |
| `created_revision` | integer | Provenance |

Annotations are append-only; corrections are new annotations rather than destructive edits.

## FrontierLane

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Lane ID | Stable |
| `name` | string | Required; unique among active lanes |
| `description` | string/null | Project-specific meaning |
| `active` | boolean | Retired lanes remain in history |
| `created_commit_id` | Commit ID | Provenance |

Renaming is represented as a lane metadata event; stable ID preserves historical references.

## FrontierItem

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Frontier Item ID | Stable |
| `lane_id` | Lane ID | Must be active when item is created/moved |
| `subject.kind` | enum | `finding` or `direction` |
| `subject.finding_id` | Finding ID/null | Required for finding subject |
| `subject.title` | string/null | Required for direction subject |
| `rationale` | string | Required |
| `priority` | integer | `0..100`, higher means more important |
| `status` | enum | `active`, `paused`, `retired` |
| `supporting_finding_ids` | Finding ID[] | Optional |
| `challenging_finding_ids` | Finding ID[] | Optional |
| `created_commit_id` | Commit ID | Provenance |

Allowed events: create, move lane, set priority, set status, update rationale/knowledge links. History remains visible through events.

## AgendaItem / Experiment Contract

| Field | Type | Rules |
| --- | --- | --- |
| `id` | Agenda ID | Stable |
| `objective` | string | Required, non-empty |
| `hypothesis` | string/null | Optional when investigation is exploratory |
| `question` | string/null | Optional; at least hypothesis or question recommended |
| `motivating_finding_ids` | Finding ID[] | At least one motivation source overall |
| `frontier_item_ids` | Frontier Item ID[] | At least one motivation source overall |
| `controls` | JSON-compatible object/array | Optional explicit controls |
| `factors` | JSON-compatible object/array | Optional variables/factors to change |
| `constraints` | JSON-compatible object/array | Optional execution constraints |
| `success_criteria` | array/string/object | Required when completion cannot be judged from objective alone |
| `priority` | integer | `0..100`, higher means more important |
| `status` | enum | `proposed`, `active`, `completed`, `failed`, `abandoned` |
| `evidence` | EvidenceReference[] | Attached when experiments produce Aim runs |
| `created_commit_id` | Commit ID | Provenance |
| `result_commit_ids` | Commit ID[] | Commits that complete/interpret the experiment |

### Agenda transitions

- `proposed -> active`
- `proposed -> abandoned`
- `active -> completed | failed | abandoned`
- `failed | abandoned -> proposed` only through explicit reopen event with reason
- `completed` is terminal; a follow-up experiment is a new Agenda Item

When an Agenda Item transitions to `completed`, the same ResearchUpdate may create Findings from its evidence. The commit ID is recorded as a result commit automatically.

## ResearchState projection

A replayed state contains:

- `schema_version`
- `revision`
- Findings with current assessment/governance plus history references
- active and retracted Relations
- Annotations
- Frontier Lanes and Items
- Agenda Items and lifecycle/evidence links
- commit summary metadata

Output ordering is deterministic: entities sort primarily by creation revision and secondarily by stable ID unless a command contract specifies priority order.

## ResearchContext

| Field | Type | Rules |
| --- | --- | --- |
| `schema_version` | integer | `1` |
| `compiler_version` | string | Deterministic compiler algorithm version |
| `revision` | integer | Research State revision compiled |
| `objective` | string | Required |
| `budget_unit` | string | `utf8_bytes` |
| `budget_limit` | integer | Positive |
| `budget_used` | integer | Serialized byte size of selected items |
| `items` | array | Compact selected findings/relations/annotations/frontier/agenda artifacts |
| `evidence_refs` | EvidenceReference[] | Deduplicated canonical evidence refs |
| `selection_notes` | object | Machine-readable reasons/closures, no LLM prose required |

The compiler selects deterministic seeds, expands mandatory structural closure, then packs groups without exceeding the budget. A required contradiction group that cannot fit produces a structured budget error rather than a partial misleading context.

## ResearchUpdate operation catalog (V1)

- `finding.create`
- `finding.assess`
- `finding.governance`
- `relation.create`
- `relation.retract`
- `annotation.create`
- `frontier.lane.create`
- `frontier.lane.rename`
- `frontier.lane.retire`
- `frontier.item.create`
- `frontier.item.move`
- `frontier.item.set_priority`
- `frontier.item.set_status`
- `frontier.item.update`
- `agenda.item.create`
- `agenda.item.transition`
- `agenda.item.attach_evidence`
- `agenda.item.set_priority`

No V1 operation deletes accepted history.
