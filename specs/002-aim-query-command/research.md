# Research: Aim Query Command

## Decision: Own an explicit `query` command surface with target subcommands

### Rationale

The current router model is binary: a command is either explicitly owned by
`aimx` or passed through to native Aim. Adding `query` as a top-level owned
command with explicit target selection through `metrics` and `images`
subcommands keeps that boundary clear and fits the existing command model.

### Alternatives considered

- A single `aimx query <expr> --type ...` command was rejected because it moves
  required command meaning into flags and complicates help and validation.
- Extending passthrough-oriented `runs` command paths was rejected because it
  blurs the owned-versus-delegated contract that the project is built around.

## Decision: Normalize repository paths before opening the Aim repository

### Rationale

Native Aim CLI accepts both repo roots like `data` and metadata-directory paths
like `data/.aim`, while the Python `Repo(...)` API expects the repository root.
To preserve user-facing compatibility, `aimx query` should accept both forms and
normalize `.aim` directory inputs to their parent directory before opening the
repository.

### Alternatives considered

- Accepting only repo roots was rejected because it would diverge from native
  Aim CLI behavior and break existing user habits.
- Passing repo paths straight through to the SDK was rejected because `.aim`
  directory inputs fail under the SDK.

## Decision: Query command should not depend on native `aim` availability

### Rationale

`query` is an `aimx`-owned command implemented in Python. Requiring native
`aim` for this path would create an unnecessary dependency and weaken the value
of the owned command surface.

### Alternatives considered

- Delegating query execution to the native `aim` executable was rejected
  because the feature request is explicitly about adding missing CLI support in
  `aimx`, not shelling out to another command when an internal implementation
  is available.

## Decision: Provide one shared structured-output envelope

### Rationale

Tests and scripts benefit from a stable response envelope that can identify the
query target, repo path, expression, result count, and rows. This keeps machine
consumers decoupled from minor human-readable formatting changes.

### Alternatives considered

- Returning target-specific unwrapped JSON arrays was rejected because it makes
  scripts harder to write and evolve.
- Returning only text output was rejected because the constitution explicitly
  values scriptable interfaces where they are realistic.

## Decision: Validate against real sample data under `data`

### Rationale

The repository already includes a local Aim test repository. Real-data
validation will catch SDK path semantics, zero-result handling, and formatting
assumptions that mocks would miss.

### Alternatives considered

- Mock-only tests were rejected because they would not validate real Aim
  repository behavior.
