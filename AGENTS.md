# AGENTS

This repository uses `AGENTS.md` as the shared working agreement for coding
agents such as Codex, Claude, and Gemini.

## Repo Context

- Read [CONSTITUTION.md](./CONSTITUTION.md) before making substantial changes.
- Treat `aimx` as a companion CLI to native Aim, not a replacement for it.
- Prefer small, reversible changes that preserve passthrough behavior and
  read-only defaults unless the task explicitly requires otherwise.

## Environment Management

- Use `uv` for environment and dependency management.
- Prefer `uv sync`, `uv run`, `uv add`, and `uv remove`.
- Do not introduce a parallel workflow with `pip install`, Poetry, Pipenv, or
  ad hoc virtualenv management unless the user explicitly asks for it.
- Keep project-local environments in `.venv`.

Recommended setup:

```bash
uv python install 3.12
uv venv --python 3.12
uv sync
```

When running project commands, prefer:

```bash
uv run <command>
```

## Python Version Policy

- The repository target is Python `3.12`.
- `pyproject.toml` currently declares `requires-python = ">=3.10,<3.13"`, but
  agents should prefer `3.12` for local development and CI-aligned work unless
  the repo is updated intentionally.

Why `3.12`:

- The dependency chain for `aim` currently pulls `aimrocks==0.5.2`.
- `aimrocks==0.5.2` does not provide installable artifacts for Python `3.13`.
- Using Python `3.13` can therefore fail during `uv add --dev aim` or related
  dependency resolution flows.
- Python `3.12` is the safest default that avoids this compatibility gap while
  keeping the project on a modern supported interpreter.

## Dependency Changes

- When adding dependencies, use `uv add` so changes are reflected in
  `pyproject.toml` and lockfile-managed workflows.
- Prefer the configured package indexes in `pyproject.toml`.
- If a dependency fails on the current interpreter or platform, diagnose the
  package compatibility first instead of assuming the index configuration is
  wrong.

## Working Style

- Make the smallest change that satisfies the task.
- Preserve user changes you did not make.
- Verify with targeted commands before claiming success.
- When a task affects behavior, document the new workflow in `README.md` or
  adjacent docs if that guidance would help future contributors or agents.

## Local Test Repo

- A local Aim test repository is available at `data/.aim`.
- Use `aim runs --repo data/.aim ls` as a quick validation command when you
  need a real repository to inspect or exercise CLI behavior.
- Prefer repo-scoped commands like `--repo data/.aim` during development so
  tests and manual checks stay isolated from any user-level Aim state.

## Active Technologies
- Python 3.12 for development, runtime support `>=3.10,<3.13` + Python standard library, native Aim CLI (external runtime prerequisite for delegated commands), pytest for test automation (001-aim-command-passthrough)

## Recent Changes
- 001-aim-command-passthrough: Added Python 3.12 for development, runtime support `>=3.10,<3.13` + Python standard library, native Aim CLI (external runtime prerequisite for delegated commands), pytest for test automation
