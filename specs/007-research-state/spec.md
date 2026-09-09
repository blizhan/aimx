# Feature Specification: Agent-First AutoResearch Control Plane

**Feature Branch**: `008-research-state`  
**Created**: 2026-09-08  
**Status**: Draft  
**Input**: User description: "Extend Aimx toward AutoResearch by combining durable research state, bounded research context, research frontier, research agenda/experiment contracts, and external agent integration into one feature. Aim remains the evidence plane; Aimx becomes the shared research state/control plane; external agents remain responsible for reasoning and experiment execution."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preserve Research Knowledge Across Agent Sessions (Priority: P1)

As a researcher using an external coding or research agent, I want conclusions from Aim experiment evidence to become durable research knowledge, so a later agent session can understand what was learned, why it is believed, how reliable it is, and how that knowledge evolved without reconstructing the entire history from raw runs.

**Why this priority**: Durable shared knowledge is the foundation for every later AutoResearch capability. Without it, context, frontier, and agenda generation collapse back into session-local memory.

**Independent Test**: Record findings and relationships from existing Aim runs, end the producing agent session, start a separate session, and verify that the second session can inspect the same findings, evidence references, assessments, governance decisions, annotations, and knowledge relationships.

**Acceptance Scenarios**:

1. **Given** one or more Aim runs contain experimental evidence, **When** an agent submits a valid research update grounded in those runs, **Then** the resulting findings and relationships become durable Aimx-owned research state and remain available to later sessions.
2. **Given** one research update contains multiple new findings and relationships between them, **When** every element is valid, **Then** the complete update becomes visible as one consistent change to research state.
3. **Given** any element of a multi-part research update is invalid, **When** the update is submitted, **Then** none of its research-state changes become visible and the user receives an actionable explanation.
4. **Given** a finding is scientifically tentative but a human considers it worth retaining, **When** its scientific and collaboration states are recorded, **Then** a combination such as `candidate` and `accepted` is preserved without conflating the two meanings.
5. **Given** a later conclusion changes the research meaning of an earlier finding, **When** the new conclusion is recorded, **Then** the earlier finding remains historically visible and the new finding expresses the refinement or supersession relationship.

---

### User Story 2 - Resume Research From Bounded Shared Context (Priority: P1)

As an external research agent, I want Aimx to compile the research state relevant to my current objective into a bounded context, so I can resume a long-running investigation without loading every historical run, finding, annotation, and relationship.

**Why this priority**: AutoResearch requires reliable continuity across many experiments. A durable state is useful only if an agent can recover the relevant portion of it within a bounded working context.

**Independent Test**: Populate a research state containing relevant, irrelevant, supporting, contradictory, and human-annotated findings; request context for a specific objective and budget; verify that the result is bounded, repeatable, preserves decisive contradictions and human steering, and points back to evidence that can be inspected separately.

**Acceptance Scenarios**:

1. **Given** a research state contains more knowledge than fits in the requested context budget, **When** context is requested for a specific objective, **Then** Aimx returns a bounded selection centered on that objective rather than the full history.
2. **Given** relevant human annotations or accepted/rejected governance decisions exist, **When** context is compiled, **Then** those interventions are represented when they materially affect the current objective.
3. **Given** a finding relevant to the objective is contradicted or challenged by other retained knowledge, **When** context is compiled, **Then** the contradictory evidence is represented rather than silently presenting only the preferred conclusion.
4. **Given** research state and objective have not changed, **When** the same bounded context is requested again with the same constraints, **Then** the resulting selection is stable enough for an agent to make repeatable decisions.
5. **Given** the context references Aim evidence, **When** an agent needs more detail, **Then** it can identify the referenced runs without requiring Aimx research state to duplicate their full metrics, traces, or artifacts.

---

### User Story 3 - Steer The Research Frontier (Priority: P2)

As a researcher collaborating with an agent, I want a durable frontier of research directions worth carrying forward, so I can correct the agent's search priorities directly instead of relying on transient prompt instructions.

**Why this priority**: Findings describe what has been learned; the frontier adds the policy layer that says which branches of the research space deserve continued attention.

**Independent Test**: Define project-specific frontier lanes, let an agent propose frontier membership, apply a human change to that frontier, and verify that later context and agenda outputs reflect the human steering.

**Acceptance Scenarios**:

1. **Given** a project has defined frontier lanes appropriate to its research goals, **When** findings or research directions are added to those lanes, **Then** the frontier preserves their rationale, provenance, and supporting or challenging knowledge.
2. **Given** different research projects use different definitions of a useful frontier, **When** each project defines its lanes, **Then** Aimx does not require a fixed global lane taxonomy.
3. **Given** an agent has proposed a frontier item, **When** a human removes, moves, qualifies, or otherwise corrects it, **Then** the intervention is durable and affects subsequent research context and agenda generation.
4. **Given** a frontier item is based on a finding that is later challenged or superseded, **When** the frontier is inspected, **Then** users can still trace the item back to the knowledge state that motivated it and determine whether it should remain active.

---

### User Story 4 - Produce A Durable Research Agenda (Priority: P2)

As an external research agent, I want Aimx to expose concrete next-experiment contracts derived from the current research state and frontier, so I can execute the next investigation without needing to understand Aimx's internal representation.

**Why this priority**: A research control plane becomes actionable when retained knowledge can be transformed into explicit, traceable next steps while leaving experiment execution to the external agent.

**Independent Test**: Starting from findings and frontier items, obtain an agenda containing at least one next experiment with objective, hypothesis, relevant parent knowledge, controls or constraints when needed, and success criteria; then verify that its lifecycle can be tracked as evidence and findings are produced.

**Acceptance Scenarios**:

1. **Given** the research state contains an unresolved question or promising frontier item, **When** an agenda is produced, **Then** at least one agenda item expresses a concrete research objective and the knowledge that motivated it.
2. **Given** an agenda item requires controls, experimental factors, constraints, or success criteria to be meaningful, **When** it is presented to an agent, **Then** those requirements are available in a stable machine-readable form.
3. **Given** an agenda item has been selected for execution, **When** the external agent completes or abandons the experiment, **Then** the agenda state can record its lifecycle and remain traceable to subsequent Aim evidence and research updates.
4. **Given** a human changes the frontier or adds a steering annotation, **When** a new agenda is produced, **Then** the resulting priorities reflect the updated shared research state.

---

### User Story 5 - Complete A Multi-Round AutoResearch Loop (Priority: P1)

As a researcher, I want Codex, Claude Code, OpenCode, or another external agent to use the same Aimx research protocol for repeated experiment cycles, so AutoResearch can continue across tools and sessions without Aimx becoming an agent framework.

**Why this priority**: The feature succeeds only if the individual state, context, frontier, and agenda capabilities compose into a real multi-round research workflow.

**Independent Test**: Use one supported external agent workflow to complete two consecutive research rounds: read context and agenda, execute an experiment, inspect Aim evidence, submit a research update, then begin the second round and verify that the first round's knowledge and steering affect the next decision.

**Acceptance Scenarios**:

1. **Given** an external agent can inspect Aim evidence, **When** it participates in the AutoResearch workflow, **Then** it can consume current research context and agenda and submit structured research updates without Aimx starting or hosting that agent.
2. **Given** the first research round produces new evidence and findings, **When** a second round begins, **Then** the second round's context and agenda can incorporate knowledge created by the first round.
3. **Given** a human intervention is recorded between two research rounds, **When** the next agent resumes, **Then** that intervention is visible through the shared research state and can affect its next decision.
4. **Given** two different external agent products follow the same Aimx research contracts, **When** responsibility moves from one to the other, **Then** the second agent can resume the research without requiring a product-specific Aimx state format.

### Edge Cases

- A research-state read is performed before any research knowledge has been recorded; it returns a valid empty state and does not create or modify Aim data.
- An evidence reference uses an unknown or ambiguous Aim run identity; the research update is rejected before any partial state change becomes visible.
- A previously valid Aim evidence reference later becomes unavailable; the finding remains historically visible and clearly identifies that its referenced evidence cannot currently be resolved.
- A research update refers to both existing findings and findings created within the same update; all references must resolve unambiguously before the update is accepted.
- A finding is simultaneously `candidate` scientifically and `accepted` for collaboration purposes; the combination remains valid because the two states answer different questions.
- A human disagrees with an agent-generated claim; the human can annotate, reject, challenge, refine, or supersede it without erasing the original research history.
- Relevant supporting and contradicting findings compete for a very small context budget; the result must not silently remove a contradiction that materially changes the interpretation of the selected claim.
- No current frontier item or actionable agenda item exists; the system reports that state clearly instead of inventing an experiment.
- An agenda item references a finding that is later superseded; the agenda remains historically traceable while future agenda generation can use the updated knowledge.
- Two writers attempt conflicting changes to the same research state; the system must not silently erase either writer's historical contribution and must surface any unresolved conflict clearly.
- An external agent fails during experiment execution; Aimx research state remains consistent and the agenda item can remain open, be marked failed/abandoned, or be revisited without fabricating evidence.

## Constitution Alignment *(mandatory)*

- **CA-001 Safety & Mutability**: The feature introduces explicit writes only to Aimx-owned research state. Aim experiment repositories remain evidence sources and MUST remain read-only through research-state, context, frontier, agenda, and agent-integration flows. Read-only research inspection MUST NOT create research state as a side effect.
- **CA-002 Ownership Boundary**: Aimx owns the new `research`, `finding`, `lineage`, and `frontier` user-facing capabilities and their structured machine-readable contracts. Existing Aimx query/trace commands remain owned as today, and command paths outside Aimx's explicit surface continue to be delegated to native Aim.
- **CA-003 CLI & Output Contract**: All core capabilities MUST be usable non-interactively in local shells, SSH sessions, scripts, and CI. Human inspection surfaces MUST be readable in terminals, and agent-facing state, context, frontier, agenda, and update flows MUST have stable machine-readable representations suitable for tool-neutral automation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Aimx MUST maintain a durable research state that is independent of any single agent session and can be inspected by both humans and external agents.
- **FR-002**: Research-state write operations MUST modify only Aimx-owned state and MUST NOT modify Aim-managed experiment data.
- **FR-003**: Research-state read operations MUST be free of write side effects, including when no research state exists yet.
- **FR-004**: A Finding MUST represent a research claim, its evidence references, its scientific assessment, its collaboration/governance state, its provenance, and its creation history.
- **FR-005**: Finding scientific status MUST support at least `candidate`, `validated`, and `contradicted`, and scientific confidence MUST support at least `low`, `medium`, and `high`.
- **FR-006**: Finding governance status MUST support at least `proposed`, `accepted`, and `rejected`, independently of scientific status and confidence.
- **FR-007**: A Finding MUST be able to reference one or more Aim runs as evidence without requiring full run metrics, traces, images, distributions, params, or artifacts to be copied into the research state.
- **FR-008**: Before accepting a research update, Aimx MUST verify that every required evidence reference and research-state reference resolves unambiguously.
- **FR-009**: Aimx MUST support a structured ResearchUpdate that can propose multiple findings, relationships, annotations, and state changes as one logical update.
- **FR-010**: A ResearchUpdate MUST be all-or-nothing from the user's perspective: if any required element is invalid, none of that update's state changes may become visible.
- **FR-011**: A ResearchUpdate MUST be able to refer unambiguously to findings created within that same update as well as findings already present in research state.
- **FR-012**: Aimx MUST preserve provenance for agent-generated and human-generated research artifacts so later users can determine who or what introduced a claim, relationship, intervention, or agenda decision.
- **FR-013**: Aimx MUST support durable human annotations on research knowledge, and those annotations MUST remain available to later agent sessions.
- **FR-014**: Aimx MUST NOT destructively replace a finding when its research meaning changes; a changed claim MUST be representable as a new finding connected to the prior knowledge.
- **FR-015**: Finding relationships MUST support at least `derived_from`, `supports`, `challenges`, `refines`, `supersedes`, `tests`, and `related_to` so the system can represent knowledge evolution rather than only experiment ancestry.
- **FR-016**: Relationships MUST retain provenance and history, including when a human later corrects or retracts an earlier relationship.
- **FR-017**: Aimx MUST compile a bounded Research Context for a stated objective from the current research state.
- **FR-018**: Research Context MUST be able to include relevant findings, relationships, human annotations, governance decisions, contradictory or negative knowledge, frontier state, agenda state, and references to supporting Aim evidence.
- **FR-019**: Research Context MUST honor a caller-supplied size or budget constraint and MUST NOT require loading the complete research history when the history exceeds that bound.
- **FR-020**: Research Context selection MUST be repeatable for unchanged research state, objective, and context constraints, apart from explicitly documented non-semantic metadata such as generation time.
- **FR-021**: Research Context MUST preserve a materially relevant contradiction or challenge when omitting it would change the interpretation of a selected conclusion.
- **FR-022**: Aimx MUST maintain a durable Research Frontier representing research directions currently considered worth carrying forward.
- **FR-023**: Frontier lanes or categories MUST be project-defined rather than limited to one fixed global taxonomy.
- **FR-024**: Frontier items MUST retain their rationale, provenance, and links to relevant supporting or challenging research knowledge.
- **FR-025**: Both agents and humans MUST be able to propose or modify frontier state, and human frontier interventions MUST be available to later context and agenda generation.
- **FR-026**: Aimx MUST maintain a durable Research Agenda that expresses actionable next investigations derived from current research state and frontier.
- **FR-027**: An Agenda Item or Experiment Contract MUST identify its research objective, motivating knowledge, and hypothesis or question when one is applicable.
- **FR-028**: An Agenda Item MUST be able to express controls, experimental factors, constraints, and success criteria when these are required to make the intended experiment unambiguous.
- **FR-029**: Agenda Items MUST have a durable lifecycle sufficient to distinguish at least proposed/open work, selected or active work, completed work, and work that was abandoned or failed.
- **FR-030**: Completed or attempted Agenda Items MUST be traceable to resulting Aim evidence and subsequent research updates when those results exist.
- **FR-031**: Aimx MUST provide a stable machine-readable way for an external agent to obtain the current research state, bounded context, frontier, agenda, and next experiment information and to submit a ResearchUpdate.
- **FR-032**: The research protocol MUST be agent-product-neutral so multiple external agents can consume and update the same shared state without separate product-specific research databases.
- **FR-033**: Aimx MUST NOT require an embedded agent runtime to complete the AutoResearch workflow; reasoning, code changes, training, and experiment execution remain responsibilities of the external agent or user environment.
- **FR-034**: Existing Aimx evidence-observation capabilities MUST remain usable as the evidence-inspection step of the AutoResearch workflow.
- **FR-035**: The complete feature MUST support at least two consecutive AutoResearch rounds in which knowledge produced from the first round can affect context or agenda in the second round.
- **FR-036**: Existing Aimx query, trace, diagnostic, help/version, and native Aim passthrough behavior MUST remain available unless explicitly extended by this feature.
- **FR-037**: Expected empty-state, no-frontier, no-agenda, unresolved-evidence, invalid-update, and external-agent-failure cases MUST fail or complete clearly without traceback-style user output or silent research-state corruption.

### Key Entities *(include if feature involves data)*

- **Research State**: The durable, shared representation of what the project currently knows, how that knowledge evolved, how humans and agents have steered it, and what work is currently being considered or pursued.
- **ResearchUpdate**: One proposed logical change to Research State containing new or changed research artifacts that must be validated as a whole before becoming visible.
- **Finding**: The smallest durable research-knowledge unit, consisting of a claim, evidence references, scientific assessment, governance state, provenance, and history.
- **Evidence Reference**: A stable pointer from research knowledge to underlying Aim evidence, primarily one or more experiment runs, without duplicating the full evidence payload.
- **Relation**: A typed, provenance-carrying connection between findings that explains support, challenge, derivation, refinement, supersession, testing, or other research relationships.
- **Annotation**: Durable human or agent steering attached to research knowledge, used to preserve comments, cautions, requested validation, and other interventions across sessions.
- **Research Context**: A bounded, objective-specific projection of Research State prepared for an external agent's current decision.
- **Research Frontier**: The durable set of research directions considered worth carrying forward, organized according to project-defined lanes or categories.
- **Frontier Item**: One frontier membership or research direction with rationale, provenance, and links to knowledge that supports or challenges keeping it active.
- **Research Agenda**: The durable collection of proposed, active, completed, failed, or abandoned next investigations derived from Research State and Frontier.
- **Agenda Item / Experiment Contract**: A concrete next investigation containing the objective, motivating knowledge, hypothesis or question, and any controls, factors, constraints, or success criteria required for execution.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In acceptance testing, 100% of valid multi-artifact ResearchUpdates become visible as complete consistent updates, and 100% of updates containing a required invalid reference leave no partial visible changes.
- **SC-002**: In acceptance testing, a new agent session can recover findings, relationships, scientific assessments, governance decisions, and human annotations created in an earlier session without reading that earlier session's conversation transcript.
- **SC-003**: For an unchanged objective, research state, and context budget, repeated Research Context requests select the same substantive research artifacts in 100% of deterministic acceptance cases.
- **SC-004**: In context-budget acceptance tests where relevant supporting and contradictory knowledge both exist, 100% of materially interpretation-changing contradictions remain represented in the bounded context.
- **SC-005**: A researcher can change frontier membership or steering once and observe that change reflected in the next generated context or agenda without re-entering the same instruction into the agent conversation.
- **SC-006**: In agenda acceptance testing, 100% of actionable Agenda Items expose enough objective, motivation, and applicable controls/constraints/success criteria for an external agent to determine what experiment is being requested without reading Aimx internal implementation details.
- **SC-007**: A two-round AutoResearch acceptance scenario completes end to end, and the second round demonstrably consumes at least one finding, intervention, frontier change, or agenda outcome produced by the first round.
- **SC-008**: At least two different external agent products or adapters can consume the same machine-readable Research State/Context/Agenda contracts and submit compatible ResearchUpdates without maintaining separate Aimx research-state formats.
- **SC-009**: Across all feature acceptance tests, no research-state read or AutoResearch inspection flow modifies Aim experiment data, and existing Aimx evidence-query and native passthrough workflows remain usable.
- **SC-010**: Expected empty-state, no-work, invalid-update, unresolved-evidence, and interrupted-experiment scenarios complete with clear user-visible outcomes and no silent loss of previously accepted research history.

## Assumptions

- The first release targets one local research project associated with one local Aim repository; remote synchronization and multi-user collaboration services are outside this feature.
- Aim remains the source of truth for raw experiment evidence. Aimx research state stores research meaning, provenance, policy, and evidence references rather than becoming a second experiment database.
- External agents are the primary producers of findings, relationships, frontier proposals, and agenda updates during AutoResearch; human-facing commands exist primarily for inspection, correction, governance, and steering.
- Human intervention is part of Research State and must survive agent/session changes.
- The existing Aimx evidence-observation workflow remains the preferred way for agents to inspect params, metrics, traces, images, and distributions when deeper evidence is needed.
- This feature does not add an embedded reasoning service, autonomous agent host, experiment scheduler, remote service, or multi-user synchronization system, and it does not prescribe a particular storage or retrieval technology.
- Frontier policy is project-specific; no single set of lane names is assumed to be correct for all research domains.
- When Research State contains no justified next experiment, reporting that no actionable agenda item exists is preferable to inventing one.
