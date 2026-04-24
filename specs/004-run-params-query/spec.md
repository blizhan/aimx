# Feature Specification: Run Params Query And Experiment Comparison

**Feature Branch**: `004-run-params-query`  
**Created**: 2026-04-24  
**Status**: Draft  
**Input**: User description: "支持Run Params的query，可以查到Params方便对比多个Runs的参数，以及实验名相关对比"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Query Run Parameters From CLI (Priority: P1)

As an Aim user working in the terminal, I want `aimx query` to return run
parameter values for matching runs so I can inspect training configuration
differences without opening the Aim UI or writing Python.

**Why this priority**: The central value of this feature is making run params
queryable from the CLI, especially when comparing multiple runs.

**Independent Test**: Run a params query against the local test repository rooted
at `data` and confirm that matching runs include their run identity,
experiment name, and parameter values in the output.

**Acceptance Scenarios**:

1. **Given** a local Aim repository contains runs with recorded parameters,
   **When** the user runs a params query for matching runs, **Then** the output
   lists each matching run with its available run parameters.
2. **Given** the query matches multiple runs, **When** the command completes,
   **Then** the user can compare parameter values across those runs in a single
   terminal result.
3. **Given** a matching run has no recorded parameters, **When** it appears in
   the result set, **Then** the output clearly marks that run as having no
   parameter values instead of failing the whole query.

---

### User Story 2 - Compare Selected Parameters Across Runs (Priority: P1)

As a user comparing experiments, I want to focus the params output on selected
parameter names so I can quickly see which hyperparameters or configuration
values differ across runs.

**Why this priority**: Large runs can contain many parameter fields. Comparison
is only useful if the user can narrow the view to the parameter keys that matter
for the decision at hand.

**Independent Test**: Run a params query that requests a small set of parameter
names and verify that each matching run is displayed with comparable values for
those names, including clear placeholders for missing values.

**Acceptance Scenarios**:

1. **Given** multiple matching runs have overlapping parameter names, **When**
   the user requests specific parameter names, **Then** the output aligns those
   names across all matching runs for side-by-side comparison.
2. **Given** one run is missing a requested parameter, **When** results are
   rendered, **Then** that missing value is shown explicitly without hiding the
   run.
3. **Given** the user requests machine-readable output, **When** the params
   query succeeds, **Then** the output includes a stable structured
   representation of each run and its parameter values for downstream scripts.

---

### User Story 3 - Compare Runs By Experiment Name (Priority: P2)

As a user tracking multiple experiments, I want params query results to include
and support filtering by experiment name so I can compare runs within one
experiment or across related experiments.

**Why this priority**: Experiment names are often the natural grouping for
analysis. This story extends the core params query into the experiment-level
comparison requested by the user.

**Independent Test**: Run params queries that target one experiment name and
multiple experiment names, then verify that results expose experiment labels and
make cross-experiment comparison possible.

**Acceptance Scenarios**:

1. **Given** runs belong to different experiments, **When** the user queries
   params, **Then** each result includes the experiment name associated with the
   run.
2. **Given** the user filters for a specific experiment name, **When** the query
   runs, **Then** only runs from the requested experiment are returned.
3. **Given** the user compares related experiment names, **When** the command
   completes, **Then** the output groups or sorts results so experiment-level
   differences are easy to scan.

### Edge Cases

- The query expression is valid but matches zero runs.
- Some matching runs do not have any params recorded.
- Some requested parameter names exist on only a subset of matching runs.
- Parameter values include nested structures, long strings, numbers, booleans,
  or null-like values.
- Two or more runs share the same display name but have different run hashes or
  experiment names.
- Experiment names are missing, empty, duplicated, or differ only by case.
- The result set is large enough that terminal output must remain concise while
  still supporting complete machine-readable output.
- A repository path may point to either a repository root such as `data` or an
  Aim metadata directory such as `data/.aim`.

## Constitution Alignment *(mandatory)*

- **CA-001 Safety & Mutability**: This feature is read-only. It inspects run
  parameters and experiment metadata from existing Aim repositories and MUST NOT
  modify `.aim` data, run records, artifacts, or the installed native `aim`
  package.
- **CA-002 Ownership Boundary**: `aimx` owns only the new params query behavior
  under the existing `query` surface. Existing metrics and images query behavior
  remains unchanged, and all commands outside the explicitly owned `aimx`
  command surfaces continue to pass through to native Aim.
- **CA-003 CLI & Output Contract**: The feature must work in local shells, SSH
  sessions, scripts, and CI. Human-readable output must support quick terminal
  comparison, and machine-readable output must remain stable enough for scripts
  and tests.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a read-only params query workflow under
  `aimx query` for inspecting run parameter values in local Aim repositories.
- **FR-002**: The params query workflow MUST accept a user-provided run query
  expression and evaluate it against the selected local repository.
- **FR-003**: Each returned run MUST include a stable run identifier,
  experiment name when available, run name when available, and the matching run
  parameter values.
- **FR-004**: Users MUST be able to request specific parameter names for focused
  comparison across matching runs.
- **FR-005**: When no specific parameter names are requested, the system MUST
  provide a useful default params view that identifies available parameter keys
  and exposes enough values for users to understand the matched runs.
- **FR-006**: The system MUST represent missing parameter values explicitly when
  a requested parameter does not exist on a matching run.
- **FR-007**: The system MUST include experiment names in params query output
  and MUST support queries that filter matching runs by experiment name.
- **FR-008**: The system MUST make comparison across experiments easy to scan by
  grouping, sorting, or otherwise clearly labeling results by experiment name.
- **FR-009**: The system MUST provide a default human-readable output mode for
  terminal comparison of run params.
- **FR-010**: The system MUST provide a machine-readable output mode with a
  stable top-level shape containing the query target, match count, run metadata,
  and parameter values.
- **FR-011**: Zero-match params queries MUST complete as successful,
  non-destructive outcomes with an explicit no-results message.
- **FR-012**: Invalid query expressions, invalid repository paths, and
  unsupported params query options MUST fail clearly with actionable messages.
- **FR-013**: The params query workflow MUST NOT alter existing metrics query,
  images query, or native Aim passthrough behavior.
- **FR-014**: The system MUST document params query usage in the user-facing CLI
  help or adjacent project documentation so users can discover the workflow.

### Key Entities *(include if feature involves data)*

- **Params Query Invocation**: A single params query request, including the
  repository path, run query expression, optional requested parameter names, and
  output mode.
- **Run Parameter Set**: The parameter values associated with one Aim run,
  including scalar and structured values that users want to compare.
- **Run Identity**: The run identifier and optional run name that distinguish a
  run from other results.
- **Experiment Label**: The experiment name associated with a run, used for
  filtering, grouping, sorting, and comparison.
- **Params Query Result**: The collection of matched runs and parameter values
  returned to the user in human-readable or machine-readable form.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can retrieve run parameter values from the local test
  repository rooted at `data` in one params query invocation without writing
  Python or opening a GUI.
- **SC-002**: In acceptance testing, 100% of matching runs in a params query
  result show run identity and experiment information when those fields exist in
  the repository.
- **SC-003**: In acceptance testing, users can compare at least three selected
  parameter names across at least three matching runs in a single terminal
  result.
- **SC-004**: In acceptance testing, zero-match queries, runs with no params,
  and missing requested parameter values all produce clear non-destructive
  output without traceback-style failures.
- **SC-005**: Machine-readable params query output is parseable by automation
  and includes match count, run identity, experiment label, and parameter values
  for 100% of returned runs.
- **SC-006**: Existing metrics query, images query, and passthrough contract
  tests continue to pass after the params query workflow is added.

## Assumptions

- "Run Params" refers to run-level parameter data recorded in an Aim
  repository, including hyperparameter-style fields users commonly compare
  across training runs.
- "Experiment name related comparison" means params results must expose
  experiment names and support filtering or scanning by experiment name; it does
  not require a full statistical experiment dashboard in this feature.
- The params query should follow the existing `aimx query` command conventions
  for repository selection, query expressions, and output modes.
- Human-readable output may summarize very large or deeply nested values for
  terminal readability, while machine-readable output should preserve values in
  a script-friendly representation.
- This feature remains limited to read-only local repository inspection and does
  not add write, sync, migration, or server-side behavior.
