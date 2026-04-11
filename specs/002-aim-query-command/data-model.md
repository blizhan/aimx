# Data Model: Aim Query Command

## QueryInvocation

- Purpose: Represents one `aimx query` CLI request after parsing.
- Fields:
  - `target`: Which query surface to use, initially `metrics` or `images`
  - `expression`: User-provided Aim query expression
  - `repo_path`: Original path supplied by the user
  - `normalized_repo_path`: Effective repository root path used for SDK access
  - `output_mode`: Human-readable or machine-readable rendering choice
- Validation rules:
  - `target` must be a supported query target
  - `expression` must be non-empty
  - `repo_path` must resolve to an accessible local path
  - `normalized_repo_path` must identify a valid Aim repository root

## RepositoryPathResolution

- Purpose: Captures the path-normalization result before querying.
- Fields:
  - `input_path`: Raw user input
  - `effective_path`: Repository root used for querying
  - `input_kind`: Repo root, `.aim` metadata directory, or invalid path
- Validation rules:
  - If `input_kind` is `.aim` metadata directory, `effective_path` must be the
    parent directory
  - Invalid paths must produce actionable errors and stop query execution

## QueryRow

- Purpose: Represents one machine-consumable result row emitted by the command.
- Fields:
  - `run_id`: Aim run identifier associated with the match
  - `target`: Query target that produced the row
  - `name`: Metric or image sequence name
  - `context`: Optional contextual attributes relevant to the match
  - `summary`: Human-readable compact description of the row
- Validation rules:
  - `run_id` and `target` must always be present
  - `name` should be present whenever the underlying Aim object exposes one

## StructuredQueryOutput

- Purpose: Stable machine-readable envelope for scripts and tests.
- Fields:
  - `target`: Query target used
  - `expression`: Query expression used
  - `repo_path`: Effective repository path used for querying
  - `count`: Number of returned rows
  - `rows`: Ordered list of `QueryRow`
- Validation rules:
  - `count` must equal the number of rows in `rows`
  - Zero-result queries must still return a valid envelope with `count = 0`
