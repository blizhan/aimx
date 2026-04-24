# Phase 1 Data Model: Run Params Query And Experiment Comparison

**Feature**: `004-run-params-query`

## Params Query Invocation

Represents one CLI request to inspect run parameters.

**Fields**:

- `target`: literal `params`
- `expression`: Aim run query expression supplied by the user
- `repo_path`: local repository root or `.aim` path
- `output_mode`: one of rich, plain/oneline, or JSON
- `param_keys`: ordered tuple of user-requested flattened parameter keys

**Validation Rules**:

- `expression` must not be empty.
- `repo_path` must exist and is normalized so a `.aim` path resolves to its
  parent repository root.
- `param_keys` may be empty; when present, each key must be non-empty after
  trimming whitespace.
- Duplicate `param_keys` are rejected after trimming to avoid duplicate output
  columns.
- `--param` is valid only for the `params` target.

## Run Identity

Identifies one matched Aim run in all output modes.

**Fields**:

- `hash`: full Aim run hash
- `short_hash`: display helper derived from the full hash
- `experiment`: experiment name, nullable
- `name`: run display name, nullable
- `creation_time`: run creation timestamp, nullable

**Relationships**:

- One Run Identity belongs to one Run Parameter Set.
- Run Identity is used for sorting and grouping display output.

**Validation Rules**:

- `hash` must be present for every returned run.
- Missing `experiment`, `name`, or `creation_time` values are allowed and must
  not hide the run.

## Run Parameter Set

The parameter data associated with one matched run.

**Fields**:

- `run`: Run Identity
- `params`: flattened dictionary of parameter key to value
- `selected_keys`: ordered tuple of keys requested for comparison
- `missing_keys`: ordered tuple of requested keys absent from this run

**Relationships**:

- Produced by collecting metadata attributes from one Aim run.
- Included as one row in a Params Query Result.

**Validation Rules**:

- Nested metadata dictionaries are flattened with dot-separated paths.
- Scalar values are preserved where possible.
- Lists or nested non-scalar values are preserved in JSON and shortened for
  human-readable output.
- Missing requested keys are represented explicitly rather than dropping the
  run.

## Experiment Label

The experiment grouping value used in params comparison output.

**Fields**:

- `value`: experiment name from the run, nullable
- `display_value`: non-empty display fallback for missing names

**Relationships**:

- Derived from Run Identity.
- Used to sort or group Params Query Result rows.

**Validation Rules**:

- Empty or missing experiment names must remain distinguishable from real
  experiment names.
- Sorting must remain deterministic when experiment labels collide.

## Params Query Result

The complete response returned by the params query workflow.

**Fields**:

- `target`: literal `params`
- `repo`: normalized repository path
- `expression`: original query expression
- `runs_count`: number of matched runs
- `param_keys`: keys selected for comparison or displayed by default
- `runs`: ordered list of Run Parameter Sets
- `omitted_param_keys`: keys omitted from the human-readable default view, if
  any

**Relationships**:

- Contains zero or more Run Parameter Sets.
- Rendered into rich, plain/oneline, or JSON output.

**Validation Rules**:

- Zero-result queries produce an empty `runs` list and a successful command
  result.
- JSON output must include full selected parameter data for every returned run.
- Human-readable output may shorten long values but must not hide run identity.
