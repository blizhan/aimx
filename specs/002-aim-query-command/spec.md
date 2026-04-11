# Feature Specification: Aim Query Command

**Feature Branch**: `002-aim-query-command`  
**Created**: 2026-04-11  
**Status**: Draft  
**Input**: User description: "参考 aim 的文档增加 query 的命令行支持 https://aimstack.readthedocs.io/en/latest/using/query_runs.html ，用data下面的数据做测试"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Query Metric Runs From CLI (Priority: P1)

As an Aim user working in the terminal, I want `aimx` to query metric runs from
an Aim repository with an Aim-style query expression so I can inspect matching
runs without dropping into Python.

**Why this priority**: This is the core user value of the feature and the most
direct CLI gap described by the request.

**Independent Test**: Run an `aimx` metric-query command against the local test
repository rooted at `data` and confirm that it returns matching metric run
results for a known query expression.

**Acceptance Scenarios**:

1. **Given** the local Aim repository rooted at `data` is available, **When**
   the user runs `aimx query metrics "<query expression>" --repo data`,
   **Then** `aimx` evaluates the expression against that repository and prints
   the matching metric-run results.
2. **Given** the query matches multiple runs, **When** the command completes,
   **Then** the output makes it clear how many matching results were found and
   which runs they belong to.
3. **Given** the query matches no runs, **When** the user executes the command,
   **Then** `aimx` exits cleanly and reports that no matching metric results
   were found.

---

### User Story 2 - Query Image Sequences And Structured Output (Priority: P2)

As a user scripting around Aim data, I want `aimx` to query image sequences and
emit structured output so I can automate downstream inspection and filtering.

**Why this priority**: The Aim query documentation covers more than metrics,
and structured output is important for the repository's CLI-first and
scriptable goals.

**Independent Test**: Run image-query and structured-output invocations against
the local test repository rooted at `data` and verify that the command returns
stable machine-readable results without requiring interactive behavior.

**Acceptance Scenarios**:

1. **Given** the local Aim repository rooted at `data` is available, **When**
   the user runs `aimx query images "<query expression>" --repo data`,
   **Then** `aimx` returns the matching image-sequence results from that
   repository.
2. **Given** the user requests machine-readable output, **When** the query
   succeeds, **Then** `aimx` emits structured output with a stable top-level
   shape that scripts can parse.
3. **Given** the query returns results, **When** the user requests
   human-readable output, **Then** `aimx` presents a concise summary without
   hiding key result details.

---

### User Story 3 - Get Clear Guidance For Invalid Query Inputs (Priority: P3)

As a user learning the query feature, I want invalid queries and repository
problems to fail with actionable guidance so I can correct the command safely.

**Why this priority**: Query features are only adoptable if parse errors,
missing repositories, and unsupported invocations are easy to understand.

**Independent Test**: Execute invalid query expressions and invalid repository
paths through the new command and verify that failures are explicit,
non-destructive, and actionable.

**Acceptance Scenarios**:

1. **Given** the user provides an invalid query expression, **When** the query
   command runs, **Then** `aimx` fails with a clear message indicating the
   expression could not be evaluated.
2. **Given** the user provides a repository path that is not an Aim repository,
   **When** the query command runs, **Then** `aimx` reports that the repository
   could not be opened and suggests how to correct the path.
3. **Given** the user requests an unsupported query target or output mode,
   **When** the command is parsed, **Then** `aimx` rejects the invocation before
   running the query and explains the supported choices.

### Edge Cases

- The repository root provided by the user is the parent directory such as
  `data`, while the repository metadata lives under `data/.aim`.
- The query expression is syntactically valid but returns zero results.
- The query expression references fields that do not exist for the chosen query
  target.
- The repository contains enough matches that output may need concise summaries
  for terminal use while still supporting machine-readable output for scripts.
- A user runs `aimx query ...` on a machine where native `aim` is missing;
  `query` remains usable because it is an `aimx`-owned command, while unrelated
  unowned commands still follow passthrough rules.

## Constitution Alignment *(mandatory)*

- **CA-001 Safety & Mutability**: This feature is read-only. It inspects data
  from an existing local Aim repository and MUST NOT mutate repository data,
  rewrite metadata, or modify the installed native `aim` executable.
- **CA-002 Ownership Boundary**: `aimx` newly owns the `query` command surface,
  including the subcommands needed to select supported query targets and output
  modes. Commands outside the explicitly owned `query`, `help`, `version`, and
  `doctor` surfaces remain delegated to native `aim`.
- **CA-003 CLI & Output Contract**: The query command must work in local
  shells, SSH sessions, and CI. It must provide human-readable output for
  terminal inspection and machine-readable output for scripting use cases.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide an `aimx`-owned `query` command for
  read-only inspection of local Aim repository data.
- **FR-002**: The system MUST support a metric-query subcommand that accepts an
  Aim-style query expression and evaluates it against a specified local Aim
  repository.
- **FR-003**: The system MUST support an image-query subcommand that accepts an
  Aim-style query expression and evaluates it against a specified local Aim
  repository.
- **FR-004**: The system MUST allow the user to choose the Aim repository root
  path to query and MUST accept the local test repository rooted at `data` as a
  valid repository during development validation.
- **FR-005**: The system MUST provide a default human-readable output mode that
  summarizes matches and includes enough per-result detail for terminal use.
- **FR-006**: The system MUST provide a machine-readable output mode with a
  stable top-level structure suitable for scripts and tests.
- **FR-007**: The system MUST report zero-match queries as successful,
  non-destructive outcomes rather than treating them as command errors.
- **FR-008**: The system MUST fail clearly when the repository path cannot be
  opened as an Aim repository.
- **FR-009**: The system MUST fail clearly when the query expression cannot be
  evaluated for the selected query target.
- **FR-010**: The system MUST describe the owned `query` command in help output
  so users can distinguish it from passthrough behavior.
- **FR-011**: The system MUST preserve the existing passthrough behavior for
  all commands and flags that are not explicitly owned by `aimx`.
- **FR-012**: The system MUST NOT require the user to invoke Python APIs
  directly in order to perform the supported query workflows.
- **FR-013**: The system MUST NOT modify repository contents, run metadata, or
  image artifacts during query execution.

### Key Entities *(include if feature involves data)*

- **Query Invocation**: A single `aimx query` command, including the selected
  query target, repository path, query expression, and output mode.
- **Repository Root**: The user-supplied local filesystem path that identifies
  the Aim repository to inspect, such as `data`.
- **Metric Query Result**: A query result representing one or more matching
  metric sequences grouped by their associated runs.
- **Image Query Result**: A query result representing one or more matching
  image sequences and their associated run context.
- **Structured Query Output**: The machine-readable response envelope returned
  when the user requests script-friendly output.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can retrieve matching metric-query results from the local
  test repository rooted at `data` in a single `aimx query metrics ...`
  invocation without using Python.
- **SC-002**: A user can retrieve matching image-query results from the local
  test repository rooted at `data` in a single `aimx query images ...`
  invocation without using Python.
- **SC-003**: In acceptance testing, 100% of zero-match queries complete
  without mutating repository data and return an explicit zero-results message.
- **SC-004**: In acceptance testing, 100% of invalid repository path failures
  and invalid query-expression failures return actionable error messages that
  identify what the user needs to fix.
- **SC-005**: The new query feature leaves all pre-existing passthrough command
  behavior unchanged in the same test suite run.

## Assumptions

- The supported query syntax should follow the documented Aim query style closely
  enough that users can adapt examples from the Aim documentation.
- The initial owned query surface is intentionally limited to metrics and image
  queries rather than every possible Aim SDK query target.
- The repository path passed by users refers to the repository root such as
  `data`, not the internal metadata directory such as `data/.aim`.
- The local repository under `data` contains enough real data to validate at
  least one metric query during automated and manual testing.
- Machine-readable output is needed for tests and scripts, but interactive or
  paginated terminal UX is out of scope for this feature iteration.
