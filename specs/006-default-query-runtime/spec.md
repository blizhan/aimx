# Feature Specification: Default Query Expression And Run Duration

**Feature Branch**: `007-default-query-runtime`  
**Created**: 2026-06-14  
**Status**: Draft  
**Input**: User description: "feat - 新增默认在没有 Query language的cli输入的情况下，默认使用\"run.hash != ''\"，需要支持query metrics / query images/ query params / trace / trace distribution - query params 增加 运行时间"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Query Runs Without Typing The Default Expression (Priority: P1)

As an Aim user exploring a local repository from the terminal, I want the owned
`aimx query` commands to work when I omit the query-language expression, so I can
quickly inspect all recorded runs without repeatedly typing `run.hash != ''`.

**Why this priority**: This is the central requested behavior for the query
surface. It removes boilerplate from the most common "show me what is in this
repo" workflow while keeping the same default match set users already type
manually.

**Independent Test**: Run query metrics, query images, and query params against
the local test repository without a query-language expression and confirm that
each command behaves as though `run.hash != ''` had been supplied.

**Acceptance Scenarios**:

1. **Given** a local repository contains at least one run with metrics, **When**
   the user runs `aimx query metrics --repo data`, **Then** the command uses
   `run.hash != ''` as the effective expression and returns matching metric
   results.
2. **Given** a local repository contains at least one run with images, **When**
   the user runs `aimx query images --repo data`, **Then** the command uses
   `run.hash != ''` as the effective expression and returns matching image
   results.
3. **Given** a local repository contains runs with params, **When** the user runs
   `aimx query params --repo data`, **Then** the command uses `run.hash != ''` as
   the effective expression and returns matching params results.
4. **Given** the user supplies an explicit non-empty expression, **When** any of
   the query commands run, **Then** that explicit expression is used instead of
   the default.

---

### User Story 2 - Trace Runs Without Typing The Default Expression (Priority: P1)

As an Aim user inspecting time series or distributions, I want the owned trace
commands to use the same default expression when I omit the query-language
input, so trace exploration starts from all runs consistently with query
exploration.

**Why this priority**: The feature request explicitly includes both trace
surfaces. Applying the same default across query and trace avoids a split mental
model where some commands require boilerplate and others do not.

**Independent Test**: Run the metric trace command and distribution trace
command against a local repository without a query-language expression and
confirm both behave as though `run.hash != ''` had been supplied.

**Acceptance Scenarios**:

1. **Given** a local repository contains metric trace data, **When** the user
   runs `aimx trace --repo data`, **Then** the command uses `run.hash != ''` as
   the effective expression and returns matching trace output.
2. **Given** a local repository contains distribution trace data, **When** the
   user runs `aimx trace distribution --repo data`, **Then** the command uses
   `run.hash != ''` as the effective expression and returns matching
   distribution trace output.
3. **Given** the user supplies an explicit non-empty expression, **When** either
   trace command runs, **Then** that explicit expression is used instead of the
   default.

---

### User Story 3 - Compare Params With Run Duration Visible (Priority: P2)

As a user comparing experiment parameters across runs, I want `aimx query
params` results to include each run's recorded run duration, so I can compare
configuration choices together with how long each run took.

**Why this priority**: Duration adds important experiment-comparison context but
depends on the params query result surface already being available. It is
valuable as an independent enhancement once params rows are returned.

**Independent Test**: Run a params query against a repository containing runs
with recorded start/end timing and confirm that each returned run shows a run
duration in human-readable and machine-readable output modes.

**Acceptance Scenarios**:

1. **Given** a matching run has recorded timing information, **When** the user
   runs `aimx query params --repo data`, **Then** the run's duration is visible
   alongside its run identity, experiment label, and parameter values.
2. **Given** the user requests machine-readable params output, **When** the
   command succeeds, **Then** each returned run includes a stable duration field
   that automation can distinguish from parameter values.
3. **Given** a matching run lacks enough timing information to calculate a
   duration, **When** params results are rendered, **Then** that run remains in
   the results and its duration is clearly marked as unavailable rather than
   failing the command.

### Edge Cases

- The query-language expression is omitted entirely after the command target.
- The query-language expression is provided as an empty or whitespace-only
  argument.
- The user supplies a non-empty explicit expression that matches zero runs.
- The user supplies a syntactically invalid non-empty expression.
- Repository paths may point to either a repository root or an Aim metadata
  directory.
- The default expression matches many runs, metrics, images, params, or trace
  series, so existing output controls and limits must continue to apply.
- A run is still in progress or has incomplete timing metadata.
- A run has duration metadata but no recorded params, or has params but no
  duration metadata.
- Machine-readable output is consumed by scripts expecting stable top-level
  expression and result metadata.

## Constitution Alignment *(mandatory)*

- **CA-001 Safety & Mutability**: This feature is read-only. It changes default
  command interpretation and params output only; it MUST NOT modify the
  installed Aim package, `.aim` repository data, run records, images, metrics,
  params, or distribution data.
- **CA-002 Ownership Boundary**: `aimx` owns the default-expression behavior for
  `aimx query metrics`, `aimx query images`, `aimx query params`, `aimx trace`,
  and `aimx trace distribution`. Native Aim passthrough remains unchanged for
  every command path not explicitly owned by `aimx`.
- **CA-003 CLI & Output Contract**: The affected commands remain terminal-first
  and scriptable in local shells, SSH sessions, captured logs, and CI. Human
  output must show useful defaults clearly, and machine-readable params output
  must expose run duration without breaking existing result interpretation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: When the user omits the query-language expression, the system MUST
  use `run.hash != ''` as the effective expression.
- **FR-002**: The default expression behavior MUST apply to `aimx query metrics`,
  `aimx query images`, and `aimx query params`.
- **FR-003**: The default expression behavior MUST apply to `aimx trace` and
  `aimx trace distribution`.
- **FR-004**: Empty or whitespace-only query-language input MUST be treated the
  same as omitted input for the command surfaces listed in FR-002 and FR-003.
- **FR-005**: A non-empty explicit expression supplied by the user MUST take
  precedence over the default expression and preserve the command's existing
  expression semantics.
- **FR-006**: Any human-readable or machine-readable output that reports the
  effective expression MUST report `run.hash != ''` when that default was used.
- **FR-007**: Expected no-result cases under the default expression MUST
  complete as clear, non-destructive outcomes without traceback-style failures.
- **FR-008**: Invalid repository paths, invalid non-empty expressions, and
  unsupported option combinations MUST continue to fail clearly with actionable
  messages.
- **FR-009**: `aimx query params` MUST include a run-duration value for each
  returned run when recorded timing metadata is sufficient to determine it.
- **FR-010**: `aimx query params` MUST keep runs in the result set when duration
  cannot be determined and MUST mark duration as unavailable or still-running in
  a way users can distinguish from a zero-length run.
- **FR-011**: `aimx query params` human-readable and plain-text outputs MUST
  display run duration alongside existing run identity, experiment label, run
  name, and parameter information without removing any existing comparison
  fields.
- **FR-012**: `aimx query params` machine-readable output MUST include a stable
  duration field separate from the params map for every returned run.
- **FR-013**: Adding default expressions and run duration MUST NOT change image
  rendering limits, params selection behavior, trace visual/export modes, or
  native Aim passthrough behavior except where explicitly described above.
- **FR-014**: User-facing help or adjacent project documentation MUST show that
  the listed query and trace commands can be run without an expression and will
  default to `run.hash != ''`.

### Key Entities

- **Default Query Expression**: The literal expression `run.hash != ''`, used
  only when the user does not provide meaningful query-language input.
- **Query/Trace Invocation**: A single command request, including command target,
  repository path, optional user expression, effective expression, output mode,
  and any target-specific filters.
- **Run Identity**: The stable run hash and optional display metadata that
  identify a run in query and trace results.
- **Run Duration**: The elapsed runtime associated with a run, derived from
  recorded run timing metadata when available and represented distinctly from
  user-recorded params.
- **Params Result Row**: One returned run in `query params`, including run
  identity, experiment label, run name, duration status, and selected or default
  parameter values.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In acceptance testing, all five affected command forms can be run
  without a query-language expression and use `run.hash != ''` as the effective
  expression.
- **SC-002**: For each affected command, an explicit non-empty expression
  produces the same matched result set as before this feature for the same
  repository and output mode.
- **SC-003**: In params-query acceptance testing, 100% of returned runs show
  either a calculated run duration or an explicit unavailable/still-running
  duration status.
- **SC-004**: Machine-readable params output remains parseable and includes the
  effective expression, match count, run identity, params data, and run-duration
  field for 100% of returned rows.
- **SC-005**: Omitted-expression, zero-match, invalid-expression, missing-repo,
  and missing-duration cases complete without repository mutation and without
  traceback-style output for expected user-facing conditions.
- **SC-006**: Existing documented image-rendering, params-selection, trace
  visual/export, and passthrough workflows remain available after this feature.

## Assumptions

- "No Query language CLI input" means the query-language expression is omitted
  or is provided only as blank whitespace; it does not include non-empty invalid
  expressions.
- The default expression is exactly `run.hash != ''` across all affected
  command surfaces.
- "query params 增加运行时间" refers to each returned Aim run's recorded duration,
  not the wall-clock time spent executing the `aimx` command.
- If a run is still active or lacks enough timing metadata, showing an explicit
  unavailable/still-running status is preferable to hiding the run.
- This feature does not add duration filtering, duration sorting, repository
  mutation, data repair, or any new write-capable behavior.
