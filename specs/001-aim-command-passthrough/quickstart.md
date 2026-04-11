# Quickstart: Native Aim Passthrough

## Prerequisites

- Python 3.10 to 3.12
- `uv`
- A native `aim` executable available in the environment for end-user style
  passthrough validation

## Local Setup

The repository may install Aim in the development dependency group for local
validation. This is a maintainer convenience and does not change the end-user
runtime contract of `aimx`.

```bash
uv sync --group dev
```

## Validate Owned Commands

After the feature is implemented, confirm that the owned CLI surface works:

```bash
uv run aimx --help
uv run aimx version
uv run aimx doctor
```

Expected outcomes:

- Help explains that `aimx` owns a small command surface and delegates the rest
  to native `aim`.
- Version shows the `aimx` version and native Aim version when discoverable.
- Doctor reports whether passthrough is ready and which native executable will
  be used.

## Validate Passthrough Behavior

Run a representative set of native Aim commands through `aimx`:

```bash
uv run aimx up
uv run aimx init --help
uv run aimx runs --help
```

Expected outcomes:

- Delegated commands behave like native Aim commands.
- Argument order is preserved.
- `stdout`, `stderr`, and exit status remain aligned with native Aim behavior.

## Validate Missing-Aim Failure Behavior

Run the same checks in an environment where native `aim` is not discoverable.

Expected outcomes:

- `aimx --help` and `aimx version` still succeed.
- `aimx doctor` reports that passthrough is not ready.
- Passthrough commands fail fast with actionable guidance instead of silent
  fallback behavior.

## Run the Automated Checks

```bash
uv run pytest
```

Focus validation on routing logic, native Aim discovery, passthrough fidelity,
and missing-Aim diagnostics.
