# Implementation Plan: Aim Query Command

**Branch**: `001-aim-command-passthrough` | **Date**: 2026-04-11 | **Spec**: [spec.md](/Users/blizhan/data/code/github/aimx/specs/002-aim-query-command/spec.md)
**Input**: Feature specification from `/specs/002-aim-query-command/spec.md`

**Note**: The expected feature branch `002-aim-query-command` could not be created in this session because git branch creation was blocked by sandbox restrictions. Planning proceeds against the checked-out branch while keeping feature artifacts under `specs/002-aim-query-command/`.

## Summary

Add an `aimx`-owned `query` command that exposes read-only Aim metric and image queries from the CLI, supports both terminal-friendly and machine-readable output, and preserves existing passthrough behavior for every non-owned command. The implementation should normalize repository paths so both repo roots such as `data` and metadata-directory paths such as `data/.aim` behave consistently for the user even though the underlying SDK expects the repository root.

## Technical Context

**Language/Version**: Python 3.12 for development, runtime support `>=3.10,<3.13`  
**Primary Dependencies**: Python standard library, Aim SDK from the dev dependency group for local development and tests  
**Storage**: Existing local Aim repositories on disk, including repo roots that contain a `.aim` metadata directory  
**Testing**: pytest with unit, integration, and contract coverage  
**Target Platform**: Local shell environments, SSH sessions, and CI on machines that can run Python CLI commands  
**Project Type**: Single-project CLI application  
**Performance Goals**: Query startup and output should remain acceptable for local interactive use on the sample repository under `data`  
**Constraints**: Read-only by default, preserve passthrough fidelity, support human-readable and machine-readable output, avoid requiring native `aim` for the owned query command  
**Scale/Scope**: One new owned command surface with metric and image query modes, shared path normalization, output formatting, and tests against the local sample repository

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Safe coexistence: no normal-path change modifies the installed `aim`
      package, replaces the native `aim` executable, or mutates `.aim` repo
      data.
- [x] Ownership boundary: each command or flag change is explicitly classified
      as `aimx`-owned behavior or native `aim` passthrough.
- [x] Read-only default: query, inspection, and diagnostic behavior stays
      read-only unless an explicit write path is justified and documented.
- [x] CLI-first contract: shell, SSH, automation, and CI usage are supported,
      with human-readable and machine-readable outputs defined where relevant.
- [x] Compatibility plan: affected Aim versions, passthrough fidelity
      expectations, and the validation approach are identified.

## Project Structure

### Documentation (this feature)

```text
specs/002-aim-query-command/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── query-command.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
└── aimx/
    ├── __main__.py
    ├── cli.py
    ├── router.py
    ├── commands/
    │   ├── __init__.py
    │   ├── doctor.py
    │   ├── help.py
    │   ├── version.py
    │   └── query.py
    └── native_aim/
        ├── __init__.py
        ├── locator.py
        └── passthrough.py

tests/
├── contract/
│   ├── test_cli_contract.py
│   └── test_query_contract.py
├── integration/
│   ├── test_missing_native_aim.py
│   ├── test_passthrough_behavior.py
│   └── test_query_command.py
└── unit/
    ├── test_locator.py
    ├── test_owned_commands.py
    ├── test_query_helpers.py
    └── test_router.py
```

**Structure Decision**: Keep the existing single-project CLI layout. Add query-specific logic under `src/aimx/commands/` with minimal shared helpers nearby, and extend the existing unit, integration, and contract test suites rather than introducing new top-level packages.

## Phase 0: Research Summary

Phase 0 decisions are captured in [research.md](/Users/blizhan/data/code/github/aimx/specs/002-aim-query-command/research.md). The key outcomes are:

- Use an explicit subcommand shape of `aimx query metrics ...` and `aimx query images ...`.
- Normalize `--repo` values so both repo roots and `.aim` metadata-directory paths are accepted.
- Treat native `aim` as irrelevant for the owned query command while preserving passthrough behavior elsewhere.
- Provide one stable structured-output envelope instead of target-specific ad hoc JSON shapes.

## Phase 1: Design Summary

- Define a `QueryInvocation` model that captures query target, expression, repo path, and output mode.
- Add a repo-path normalization step before opening the Aim repository.
- Separate result loading from rendering so human-readable and machine-readable outputs can share the same loaded result set.
- Cover command routing, query validation, repo normalization, result rendering, and passthrough non-regression with tests against the sample repository.

## Post-Design Constitution Check

- [x] Safe coexistence remains intact after design: the plan uses read-only SDK access and does not modify repository data.
- [x] Ownership boundary remains explicit: `query` is owned, all unrelated commands still route through passthrough.
- [x] Read-only default remains intact: no write-capable flows are introduced.
- [x] CLI-first contract remains intact: the design includes both terminal-friendly and structured outputs.
- [x] Compatibility validation remains intact: tests include both `data` and `data/.aim` repo-path forms and passthrough regression checks.

## Complexity Tracking

No constitution violations or exceptional complexity require justification.
