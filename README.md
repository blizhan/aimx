# aimx

`aimx` is a safe, additive, CLI-first companion for native Aim.

It keeps a small owned command surface for diagnostics and guidance, and
delegates everything else to the native `aim` executable already available in
the user's environment.

## What aimx owns

- `aimx`
- `aimx --help`
- `aimx help`
- `aimx version`
- `aimx doctor`

These commands explain how `aimx` works, show the `aimx` version, and report
whether native Aim is available for passthrough.

## What aimx delegates

Any unowned command path is passed through to native `aim`.

Examples:

```bash
aimx up
aimx init --help
aimx runs --help
aimx runs ls
```

## Runtime contract

- `aimx` does not replace the `aim` executable.
- `aimx` does not modify the installed `aim` package.
- `aimx` does not mutate `.aim` data during help, version, doctor, or
  passthrough flows.
- Native Aim remains an external runtime prerequisite for delegated commands.
- The repo's development dependency on Aim is only for local development and
  testing convenience.

## Local development

```bash
uv sync --group dev
uv run pytest
```

## Quick checks

```bash
uv run aimx --help
uv run aimx version
uv run aimx doctor
```
