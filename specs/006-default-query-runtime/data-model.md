# Phase 1 Data Model: Default Query Expression And Run Duration

## Default Query Expression

Represents the query-language expression used when the user does not provide a
meaningful expression.

### Fields

- `value`: Always `run.hash != ''`.
- `source`: `default` when supplied by `aimx`, `explicit` when provided by the
  user.

### Validation Rules

- The default is used only when expression input is absent, empty, or
  whitespace-only.
- A non-empty explicit expression always takes precedence.
- Option tokens must remain options; they must not be consumed as expressions
  when expression input is omitted.

## Query Or Trace Invocation

Represents one parsed owned command request.

### Fields

- `target`: Query target (`metrics`, `images`, `params`) or trace target
  (`metrics`, `distribution`).
- `repo_path`: User-provided local repository path, normalized later to a repo
  root when `.aim` is supplied.
- `expression`: Effective expression after applying omitted-expression default.
- `expression_source`: Whether the effective expression came from user input or
  the default.
- `output_mode`: Rich/default, plain, table, CSV, or JSON depending on command
  surface.
- `filters`: Existing target-specific filters such as step range, epoch range,
  sampling, image cap, selected params, dimensions, or selected distribution
  step.

### Validation Rules

- Query target remains required.
- Metric trace target may be implicit; distribution trace target remains
  explicit through `trace distribution`.
- Existing option validation remains unchanged for missing values, unsupported
  options, mutually exclusive filters, and invalid numeric values.
- Invalid non-empty expressions are not repaired during parsing; they fail
  through the existing evaluation path.

## Run Identity

Represents the run identity shown in query and params results.

### Fields

- `hash`: Stable full run hash.
- `experiment`: Optional experiment label.
- `name`: Optional run display name.
- `creation_time`: Optional creation timestamp already used by existing run
  metadata helpers.

### Relationships

- Query metrics, query images, trace metrics, trace distributions, and query
  params all attach results to run identity.
- Params rows add run duration metadata next to the same identity.

## Run Duration

Represents elapsed runtime for a returned params run.

### Fields

- `seconds`: Numeric elapsed seconds when available; `null` when not
  calculated.
- `status`: `available`, `running`, or `unavailable`.
- `source`: `duration`, `end_time_minus_creation_time`, or `missing_metadata`.

### Validation Rules

- If Aim exposes a valid non-negative `duration`, use it.
- If `duration` is missing but `end_time` and `creation_time` are valid, use
  `end_time - creation_time`.
- Negative or non-numeric values are treated as unavailable.
- Runs with unavailable duration remain in output.
- A missing duration is never rendered as zero.

## Params Result Row

Represents one returned run in `aimx query params`.

### Fields

- `run`: Run identity.
- `duration`: Run duration metadata.
- `params`: Flattened selected or default parameter values.
- `selected_keys`: User-requested flattened parameter keys, if any.
- `missing_keys`: User-requested keys absent from this run.

### Relationships

- A params query result contains zero or more params result rows.
- Each row belongs to exactly one run identity.
- Duration is independent from the `params` map and does not affect parameter
  selection or missing-key tracking.

### Validation Rules

- Rows with no params remain visible.
- Rows with missing selected params remain visible and list missing keys.
- JSON output includes duration for every row.
- Rich and plain outputs display duration for every row as a value or explicit
  unavailable/running marker.
