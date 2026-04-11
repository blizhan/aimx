# Research: Native Aim Passthrough

## Decision 1: Use a command-router architecture with passthrough as the default

- **Decision**: `aimx` will implement a command router that claims only a small
  owned command surface and delegates all other invocations to native `aim`.
- **Rationale**: This gives `aimx` a stable boundary for future owned commands
  without prematurely rebuilding the native Aim command tree. It matches the
  constitution's requirements for explicit ownership, additive behavior, and
  bounded compatibility work.
- **Alternatives considered**:
  - Thin proxy with no explicit routing layer: simpler to start, but it makes
    future owned-command expansion less clean.
  - Full compatibility layer that mirrors Aim's command tree: more control, but
    too much early coupling and maintenance risk for v1.

## Decision 2: Resolve native Aim through executable discovery in the environment

- **Decision**: v1 will discover native `aim` from the user's executable
  environment and will not import or patch the Aim Python package as a fallback
  path.
- **Rationale**: This keeps runtime behavior honest and safe. Users continue to
  own their native Aim installation, and `aimx` remains a companion entrypoint
  instead of a replacement runtime.
- **Alternatives considered**:
  - Importing the Aim Python package directly when the executable is missing:
    risks tighter coupling and blurs the runtime contract.
  - Bundling an internal Aim implementation: conflicts with the companion CLI
    model and increases maintenance surface immediately.

## Decision 3: Keep runtime dependencies minimal and treat Aim as a dev-only validation aid

- **Decision**: The `aim` dependency remains in the repository's development
  dependency group for local testing and examples, while `aimx` runtime logic
  uses the Python standard library plus an externally available native `aim`
  executable.
- **Rationale**: This lets maintainers validate passthrough behavior locally
  without forcing `aimx` users onto a repo-defined Aim version. It aligns with
  the requirement that end users are not required to install a pinned Aim
  package as part of `aimx`.
- **Alternatives considered**:
  - Declaring `aim` as a runtime dependency of `aimx`: simpler for local
    development, but it changes the product contract and risks version
    conflicts.
  - Avoiding any Aim dependency in development: possible, but it makes local
    validation of real passthrough behavior harder.

## Decision 4: Preserve delegated process behavior as closely as practical

- **Decision**: Delegated commands will preserve user-supplied argument order,
  native `stdout`, native `stderr`, and native exit status whenever a native
  `aim` process is successfully launched.
- **Rationale**: Passthrough fidelity is the primary value of v1. Users need to
  feel that existing Aim commands continue to behave like Aim commands even when
  they are entered through `aimx`.
- **Alternatives considered**:
  - Parsing and reconstructing delegated command output: too risky and
    unnecessary for v1.
  - Wrapping delegated failures in custom Aimx-only summaries: reduces
    fidelity and can confuse scripts.

## Decision 5: Use a mixed help experience and explicit missing-Aim diagnostics

- **Decision**: `aimx --help` will describe owned commands and explain
  passthrough behavior. `aimx version` and `aimx doctor` remain available even
  when native `aim` is missing. Delegated commands fail fast with actionable
  guidance when native `aim` cannot be used.
- **Rationale**: Users need a clear mental model of what `aimx` owns and what
  it delegates. Diagnostics should still help users recover from a missing
  native Aim installation rather than hiding the problem.
- **Alternatives considered**:
  - Forwarding `aimx --help` directly to native Aim: clearer continuity with
    Aim, but it hides the `aimx` ownership boundary.
  - Disabling all commands when native Aim is missing: simple, but unhelpful
    and less scriptable.

## Decision 6: Validate behavior with unit, integration, and contract tests

- **Decision**: The feature will be validated with unit tests for routing and
  detection logic, integration tests for passthrough and missing-Aim behavior,
  and contract tests for the user-facing CLI interface.
- **Rationale**: This split keeps fidelity checks close to the user-facing
  contract while still letting the implementation evolve internally.
- **Alternatives considered**:
  - Only unit tests: faster to write, but insufficient for process and stream
    behavior.
  - Only end-to-end tests against a real Aim install: valuable, but too brittle
    as the sole verification strategy.
