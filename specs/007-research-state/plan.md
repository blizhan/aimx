# Implementation Plan: Agent-First AutoResearch Control Plane

**Branch**: `008-research-state` | **Date**: 2026-09-08 | **Spec**: [spec.md](/Users/blizhan/data/code/github/aimx/specs/007-research-state/spec.md)
**Input**: Feature specification from `/Users/blizhan/data/code/github/aimx/specs/007-research-state/spec.md`

## Summary

Extend Aimx from a read-only experiment-evidence companion into an Agent-first research control plane while preserving Aim as the evidence source of truth and keeping external agents responsible for reasoning and experiment execution. The feature adds an Aimx-owned local sidecar research store with immutable commit/event history, atomic structured ResearchUpdates, durable Findings/Lineage/Annotations, deterministic bounded Research Context, project-defined Frontier, durable Agenda/Experiment Contracts, and a tool-neutral AutoResearch protocol that composes with the existing `$aimx` evidence-observation skill.

The design uses Python's standard-library SQLite support for the sidecar store. Each explicit write is one transaction that appends a commit and immutable events; current state is replayed from those events. Research reads do not create the sidecar and open an existing database read-only. Aim evidence references are canonicalized to full run hashes through a narrow Aim bridge and `.aim` is never mutated.

## Technical Context

**Language/Version**: Python 3.12 for development, runtime support `>=3.10,<3.13`  
**Primary Dependencies**: Python standard library (`sqlite3`, `json`, `pathlib`, `uuid`, `datetime`), existing Aim SDK usage for read-only run-reference validation, existing `rich` rendering support; no new runtime dependency planned  
**Storage**: Aimx-owned local SQLite sidecar at `<repo>/.aimx/research/state.sqlite3` using append-only commit/event records; associated Aim repository remains read-only  
**Testing**: pytest unit, contract, and integration suites; local Aim fixture repository at `data/.aim`; JSON contract validation with standard-library parsing plus focused schema/shape tests  
**Target Platform**: Terminal-first local CLI for developer workstations, SSH sessions, scripts, and CI on supported Python platforms  
**Project Type**: Single-project Python CLI with a local research-state subsystem and agent-facing JSON contracts  
**Performance Goals**: Research-state replay and bounded-context compilation should remain interactive (target under 2 seconds excluding Aim evidence lookup) for roughly 10,000 commits / 50,000 research events on a local project; write commits should touch only the submitted update plus required validation reads  
**Constraints**: Never mutate `.aim`; never create `.aimx` during read-only commands; explicit writes only; atomic all-or-nothing ResearchUpdates; optimistic revision conflict detection; deterministic context selection; no embedded LLM, agent runtime, scheduler, vector store, graph database, remote service, or new runtime package  
**Scale/Scope**: One local research project associated with one local Aim repository; thousands of findings/frontier/agenda entities and tens of thousands of historical events; multi-user sync and remote collaboration are out of scope

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Safe coexistence: all raw evidence reads remain against the existing Aim repository; the design does not modify the installed `aim` package, replace the native `aim` executable, monkey patch Aim, or write `.aim` data.
- [x] Ownership boundary: new top-level `research`, `finding`, `lineage`, and `frontier` command families are explicitly Aimx-owned. Existing query/trace/help/version/doctor ownership remains unchanged and every other command continues to native Aim passthrough.
- [x] Read-only default: state/context/frontier/agenda inspection does not initialize storage or write metadata. Only explicit update/steering commands create or mutate the Aimx-owned sidecar.
- [x] CLI-first contract: every core workflow has non-interactive JSON output/input contracts in addition to concise terminal views; stdin/file driven updates support SSH, scripting, and CI.
- [x] Compatibility plan: the only new Aim integration is full-run-hash resolution/existence validation through the existing public repository access pattern already used by Aimx tests. Existing passthrough and query/trace suites remain regression gates.
- [x] Focused expansion: the subsystem remains local, evidence-grounded, removable with `.aimx`, and does not host agents, schedule experiments, provide a server, or duplicate Aim run storage; this keeps the feature within the companion-CLI model rather than turning Aimx into a general MLOps platform.

## Project Structure

### Documentation (this feature)

```text
specs/007-research-state/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli-output.md
│   └── research-update.schema.json
├── checklists/
│   └── requirements.md
└── tasks.md                       # generated later by /speckit.tasks
```

### Source Code (repository root)

```text
src/aimx/
├── router.py                      # own research/finding/lineage/frontier families
├── cli.py                         # dispatch new owned command handlers
├── commands/
│   ├── research.py                # state/update/context/agenda/next
│   ├── finding.py                 # human finding inspection/steering
│   ├── lineage.py                 # human lineage inspection/steering
│   ├── frontier.py                # frontier inspection/steering
│   └── help.py                    # document the new owned surface
├── research/
│   ├── models.py                  # immutable domain records and enums
│   ├── ids.py                     # stable prefixed entity/commit IDs
│   ├── store.py                   # read-only open, schema init on write, transactions
│   ├── events.py                  # event types and commit serialization
│   ├── replay.py                  # commits/events -> current ResearchState
│   ├── validation.py              # ResearchUpdate/ref/state-transition validation
│   ├── context.py                 # deterministic bounded context compiler
│   └── agenda.py                  # deterministic agenda/next selection helpers
├── aim_bridge/
│   └── research_evidence.py       # canonical full Aim run hash validation
└── rendering/
    └── research_views.py          # human and stable JSON envelopes

skills/aimx/
├── SKILL.md                       # context -> agenda -> execute -> observe -> update loop
└── references/
    └── autoresearch-protocol.md   # agent-neutral protocol and examples

tests/
├── unit/
│   ├── test_research_store.py
│   ├── test_research_replay.py
│   ├── test_research_validation.py
│   ├── test_research_context.py
│   ├── test_research_agenda.py
│   └── test_research_evidence.py
├── contract/
│   └── test_research_contract.py
└── integration/
    ├── test_research_commands.py
    └── test_autoresearch_loop.py
```

**Structure Decision**: Keep Aimx as one Python CLI. Isolate the new write-capable research subsystem under `src/aimx/research/`, keep Aim-specific evidence access in `aim_bridge`, and keep command parsing/terminal rendering in the existing command/rendering layers. This preserves current boundaries while making the mutation surface easy to audit.

## Phase 0: Research Summary

Detailed decisions are recorded in [research.md](/Users/blizhan/data/code/github/aimx/specs/007-research-state/research.md). Key outcomes:

- Use a local SQLite sidecar rather than loose JSON commit files because combining state, context, frontier, agenda, and concurrent agent writes requires transaction ordering and conflict detection; `sqlite3` is already in Python and adds no package dependency.
- Persist only immutable commit/event history as the source of truth. Current Research State is derived by replay so claim history, status changes, relation retractions, human steering, and agenda lifecycle stay explainable.
- Require `base_revision` on structured ResearchUpdates and reject stale writes with a distinct revision-conflict result. An optional `client_update_id` makes safe retries idempotent.
- Canonicalize every Aim run evidence reference to a full run hash before commit; missing or ambiguous references reject the entire update.
- Compile context deterministically from lexical relevance, explicit graph links, governance/epistemic priority, human steering, active frontier/agenda state, and one-hop knowledge closure. No LLM or embeddings participate.
- Measure context budget as the UTF-8 byte size of the selected `items` payload. Contradiction closure is mandatory; if a selected finding plus a structurally material challenge cannot fit, return a budget error instead of silently dropping the challenge.
- Treat Frontier and Agenda proposals as research decisions produced by agents/humans, not intelligence generated by Aimx. Aimx validates, persists, ranks, and exposes them deterministically.

## Phase 1: Design Summary

- Add an Aimx-owned database at `<normalized-repo>/.aimx/research/state.sqlite3`. Missing storage is a valid empty state for reads. Writes initialize schema only after explicit user intent.
- Store `metadata`, `commits`, and `events`; one `BEGIN IMMEDIATE` transaction validates `base_revision`, validates all operations/references, inserts one commit plus ordered events, advances the revision, and commits atomically.
- Define prefixed immutable identities (`f_`, `rel_`, `ann_`, `lane_`, `front_`, `agenda_`, `commit_`). ResearchUpdate-local references use `$<local_id>` and resolve before the transaction becomes visible.
- Keep Finding claim/evidence identity immutable. Assessments and governance changes append events; changed research meaning creates a new Finding plus `refines`/`supersedes` relation.
- Expose the primary agent workflow through `aimx research state|context|agenda|next|update`; expose human intervention through `finding`, `lineage`, and `frontier` commands that internally produce the same ResearchUpdate/event history.
- Make `research update` accept schema-versioned JSON from stdin or a file, with `--dry-run` performing the same validation/evidence resolution without committing.
- Make every state/context/agenda JSON envelope include `schema_version`, `revision`, and deterministic ordering. Context additionally includes `compiler_version`, budget unit/used/limit, objective, selected artifacts, and evidence refs.
- `research next` never invents an experiment. It returns the highest-priority active Agenda Item; if none is active, the highest-priority proposed item; ties are deterministic by creation revision then ID. If no actionable item exists, it returns a valid no-work result.
- Update `$aimx` guidance so external agents follow `context -> agenda/next -> execute -> observe Aim evidence -> research update -> repeat`. Existing snapshot collection remains the evidence collection step rather than becoming the research-state store.

## Post-Design Constitution Check

- [x] `.aim` remains read-only; write transactions target only `.aimx/research/state.sqlite3`.
- [x] Read commands check for storage existence before opening and use read-only database access, so inspection has no initialization side effect.
- [x] All new command roots are explicitly reserved by the router; unrecognized roots preserve existing native Aim passthrough behavior.
- [x] Agent contracts are JSON/stdin/stdout based and do not depend on a specific agent product or interactive UI.
- [x] Research evidence validation is a narrow, read-only Aim bridge; the rest of Research State is Aim-independent and therefore does not widen coupling to Aim internals.
- [x] The feature remains removable and local: deleting `.aimx` removes Aimx research state without altering Aim experiments.

## Complexity Tracking

No constitution violation requires an exception. The only persistent write surface is the explicitly requested Aimx-owned sidecar; the use of SQLite reduces, rather than increases, custom concurrency and crash-consistency complexity and introduces no third-party runtime dependency.
