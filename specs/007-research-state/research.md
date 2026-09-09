# Research: Agent-First AutoResearch Control Plane

## Decision 1: Use an Aimx-owned SQLite sidecar for durable research state

**Decision**: Store research history in `<repo>/.aimx/research/state.sqlite3` using Python's standard-library `sqlite3`. The database is created only by an explicit write operation. Read commands treat a missing database as empty state and open an existing database in read-only mode.

**Rationale**: The combined feature needs atomic multi-artifact updates, a single monotonic revision, safe concurrent writers, idempotent retries, and durable history across Findings, Lineage, Frontier, and Agenda. SQLite supplies transactional ordering and crash consistency without a new runtime dependency. Keeping the database under `.aimx` preserves the ownership boundary and makes the feature removable without touching `.aim`.

**Alternatives considered**:
- One JSON file per commit: attractive for inspectability, but requires custom locking, revision allocation, partial-write recovery, and deterministic ordering under concurrent writers.
- JSONL append log: simpler append path but difficult to make a multi-record update atomic and portable under process concurrency.
- Graph/vector databases: unnecessary for the first local control-plane scope and would add operational/runtime dependencies.

## Decision 2: Make immutable commit/events the source of truth

**Decision**: Persist only append-only `commits` and ordered `events` plus minimal metadata such as schema/current revision. Build the current Research State by replaying events in revision/sequence order.

**Rationale**: Research history is itself product data. Replaying immutable events naturally preserves assessment changes, accept/reject decisions, relation retractions, frontier steering, and agenda lifecycle without destructive row overwrites. It also makes each visible state explainable by the commit that introduced it.

**Alternatives considered**:
- Mutable current-state tables only: easier reads but loses knowledge evolution unless a second audit system is added.
- Full relational current state plus event log: faster reads but duplicates state and requires transactional consistency between two representations before scale proves it necessary.

## Decision 3: Use optimistic base revisions plus transactional write serialization

**Decision**: Every structured ResearchUpdate carries `base_revision`. A write starts an immediate transaction, reads the current revision, and rejects the update if it differs. The optional `client_update_id` is unique and makes exact retries return the original committed result instead of duplicating events.

**Rationale**: External agents routinely reason from a context snapshot. Requiring its revision makes stale decisions visible rather than silently appending contradictory governance/frontier changes. SQLite serializes the actual commit, while the revision check gives the caller a clear semantic conflict boundary.

**Alternatives considered**:
- Last-write-wins: violates the requirement not to silently erase or override another writer's contribution.
- Automatic merge of stale updates: unsafe because semantic conflicts in governance/frontier/agenda cannot be resolved mechanically.
- Repository-wide lock with no revision: prevents simultaneous writes but cannot detect a stale agent acting on old context after the lock is released.

## Decision 4: Canonicalize Aim evidence before any research commit

**Decision**: V1 `EvidenceReference` supports Aim run references. Accept a full run hash or unambiguous short prefix at input; resolve it through a narrow read-only Aim bridge and persist only the canonical full run hash. Any missing or ambiguous required evidence rejects the entire ResearchUpdate.

**Rationale**: Full hashes are stable research anchors and the repository already has short-hash resolution behavior. Canonicalization at the write boundary prevents later ambiguity and avoids copying metrics/traces/artifacts into Research State.

**Alternatives considered**:
- Persist short prefixes: concise but can become ambiguous as new runs appear.
- Copy evidence summaries into Research State as the source of truth: creates a second experiment database and risks drift from Aim.
- Support arbitrary trace selectors in V1: useful later, but run-level anchoring is sufficient to establish the control-plane contract first.

## Decision 5: Use operation-based ResearchUpdate envelopes

**Decision**: ResearchUpdate V1 contains metadata (`schema_version`, `base_revision`, author, optional `client_update_id`) plus an ordered `operations` array. Operations use names such as `finding.create`, `finding.assess`, `relation.create`, `annotation.create`, `frontier.item.create`, and `agenda.item.transition`. New entities may declare a `local_id`; later operations in the same update refer to them as `$<local_id>`.

**Rationale**: One operation stream maps directly to immutable events, supports mixed atomic updates, preserves operation order, and is easier to extend than a growing set of parallel arrays. Local references allow an agent to create a finding and immediately link it without pre-allocating server IDs.

**Alternatives considered**:
- Separate arrays for findings/relations/annotations/frontier/agenda: readable for simple cases but awkward for ordered state transitions and future entity families.
- One CLI command per mutation: acceptable for human convenience, but poor as the primary agent contract because multi-step writes can partially succeed.

## Decision 6: Keep claim meaning immutable but allow append-only assessment/governance changes

**Decision**: A Finding's claim and original evidence identity are immutable. Scientific assessment and governance may evolve through explicit events. If the research meaning changes, create a new Finding and connect it with `refines`, `supersedes`, or another lineage relation.

**Rationale**: This preserves the distinction between “what was claimed” and “what we now think about that claim.” It also gives humans a durable intervention point without erasing agent history.

**Alternatives considered**:
- Editable Finding documents: convenient but makes lineage unreliable because earlier agent decisions can no longer be reconstructed.
- Fully immutable Findings including assessment/governance: historically clean but forces a new claim node for routine validation or acceptance decisions that do not change meaning.

## Decision 7: Compile context deterministically without an LLM or embeddings

**Decision**: Research Context V1 uses deterministic lexical objective matching plus explicit structural signals. Candidate artifacts are ranked from normalized text overlap, governance/epistemic priority, human annotations, active Frontier/Agenda relevance, and direct graph references. Selected findings expand one hop through `challenges`, `refines`, `supersedes`, `supports`, and `tests` as needed. Stable IDs break all ties.

**Rationale**: Aimx is the control plane, not the reasoning plane. Deterministic selection is testable, agent-neutral, offline, and reproducible. Explicit graph links and durable frontier/agenda state carry semantic intent that lexical matching alone cannot infer.

**Alternatives considered**:
- LLM summarization/ranking: non-deterministic, product-specific, and turns Aimx into an intelligence runtime.
- Embedding/vector retrieval: improves fuzzy similarity but introduces models/dependencies and a second index lifecycle before the core protocol is proven.
- Pure recency: easy but loses older decisive evidence and human steering.

## Decision 8: Define context budget in exact UTF-8 payload bytes

**Decision**: `--budget N` limits the UTF-8 byte size of the serialized `items` portion of Research Context. The envelope reports `budget_unit: "utf8_bytes"`, `budget_limit`, and `budget_used`.

**Rationale**: Agent tokenizers differ. Bytes are exact, deterministic, dependency-free, and enforceable. Callers can translate their own model-token budget into a conservative byte budget.

**Alternatives considered**:
- Model tokens: no single tokenizer is valid for Codex, Claude, OpenCode, and future agents.
- Artifact count: predictable but does not bound actual prompt size.
- Approximate tokens (`chars/4`): easy but communicates precision the system does not have.

## Decision 9: Treat structurally linked contradictions as mandatory context closure

**Decision**: If a selected finding has an active `challenges` relation to/from another finding, or is directly superseded/refined in a way that changes interpretation, the linked counter/evolution finding is part of the same packing group. If the minimum group cannot fit the requested budget, context compilation returns `budget_too_small_for_required_context` with the required minimum instead of omitting the contradiction.

**Rationale**: “Do not silently drop material contradiction” must be testable. Structural lineage gives a deterministic definition of materiality without asking Aimx to judge scientific truth.

**Alternatives considered**:
- Always include every contradicted finding: quickly overwhelms bounded context.
- Let ranking drop low-scoring challenges: can produce a misleading one-sided context.
- Generate summaries to squeeze both sides: requires an embedded reasoning component.

## Decision 10: Frontier is durable policy, with project-defined lanes

**Decision**: Frontier lanes are explicit project state with stable IDs and user-defined names. Frontier Items reference either a Finding or a named research direction, carry rationale/priority/status/provenance, and may be moved, paused, or retired through events.

**Rationale**: Different domains organize research differently. Stable lane IDs keep history valid if names change, while human edits become durable search-policy changes consumed by later context and agenda decisions.

**Alternatives considered**:
- Fixed global lane enum: too restrictive and conflicts with the feature requirement.
- Derive lanes automatically from findings: makes policy opaque and requires intelligence Aimx should not own.

## Decision 11: Aimx persists Agenda; agents propose experiments

**Decision**: Aimx does not synthesize novel hypotheses. External agents or humans propose `AgendaItem` / Experiment Contracts through ResearchUpdate, referencing motivating Findings/Frontier Items. Aimx validates and persists them. `research agenda` lists them; `research next` deterministically returns the highest-priority active item, otherwise the highest-priority proposed item, with creation revision then ID as tie breakers.

**Rationale**: Creating a useful experiment is a reasoning task. Keeping that intelligence outside Aimx preserves the Agent/Control Plane boundary while still making next work durable, queryable, steerable, and tool-neutral.

**Alternatives considered**:
- Aimx-generated agendas via built-in LLM: violates the architecture boundary and adds provider/runtime coupling.
- Frontier only, no Agenda persistence: agents would still have to reconstruct executable experiment contracts every session.

## Decision 12: Human convenience commands compile to the same ResearchUpdate path

**Decision**: Commands such as `finding comment`, `finding accept`, `finding assess`, `lineage link/retract`, and `frontier add/move/retire` load the latest revision, build one ResearchUpdate, and submit it through the same validator/store used by agents. A concurrent revision change returns a conflict instead of retrying silently.

**Rationale**: One mutation path avoids two semantics for “human state” versus “agent state” and guarantees history/provenance is preserved consistently.

**Alternatives considered**:
- Direct table updates for human CLI: simpler handlers but bypasses atomic/event/history guarantees.
- Automatic retry after conflict: risks applying a human decision to a state the user did not inspect.

## Decision 13: Use stable JSON envelopes and a small exit-status contract

**Decision**: Agent-facing commands support `--json`. Success envelopes contain `schema_version` and `revision`. Expected validation/input failures use exit status `2`; stale revision conflicts use `3`; unexpected internal failures use `1`. JSON mode emits a stable error object to stderr. Human mode emits concise actionable text.

**Rationale**: Existing Aimx uses conventional non-zero command results, while agents need machine-readable error categories—especially to distinguish “fix payload” from “refresh context and re-reason.”

**Alternatives considered**:
- Always exit `2`: simpler but forces agents to parse messages to identify concurrency conflicts.
- Put errors on stdout: complicates pipelines that assume stdout is a successful JSON document.

## Decision 14: Extend the existing Aimx skill rather than create an agent framework

**Decision**: Update `skills/aimx/SKILL.md` and add `references/autoresearch-protocol.md` so agents follow `research context -> research next/agenda -> execute externally -> collect Aim evidence -> research update -> repeat`. Keep `collect_experiment_snapshot.py` as the read-only evidence collector.

**Rationale**: The repo already defines `$aimx` as the Observe subsystem. Extending that skill turns the existing one-shot `log_experiment` result into a durable ResearchUpdate while keeping agent execution in Codex/Claude/OpenCode/etc.

**Alternatives considered**:
- Add an Aimx daemon/orchestrator: outside scope and constitution boundary.
- Create one integration per agent product: duplicates state/protocol logic and undermines tool-neutral handoff.
