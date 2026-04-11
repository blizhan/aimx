# Implementation Plan: Native Aim Passthrough

**Branch**: `001-aim-command-passthrough` | **Date**: 2026-04-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-aim-command-passthrough/spec.md`

## Summary

Build the first `aimx` companion CLI as a Python command-router that owns a
minimal help/version/doctor surface and delegates all other command paths to
the installed native `aim` executable. The implementation keeps runtime
dependencies minimal, treats the repo's development-only Aim dependency as a
local validation aid rather than an end-user requirement, and optimizes for
passthrough fidelity of arguments, output streams, and exit status.

## Technical Context

**Language/Version**: Python 3.12 for development, runtime support `>=3.10,<3.13`  
**Primary Dependencies**: Python standard library, native Aim CLI (external runtime prerequisite for delegated commands), pytest for test automation  
**Storage**: N/A  
**Testing**: pytest  
**Target Platform**: macOS/Linux terminal environments, SSH sessions, and CI runners  
**Project Type**: CLI companion tool  
**Performance Goals**: Owned commands return quickly on local machines; routing adds negligible overhead before native Aim handoff; delegated commands preserve native command completion characteristics  
**Constraints**: No mutation of the installed Aim package or `.aim` data; no runtime dependency on a repo-pinned Aim package; preserve `argv`, `stdout`, `stderr`, and exit status for delegated commands; owned commands stay usable even when native `aim` is unavailable  
**Scale/Scope**: One CLI entrypoint, three owned commands (`help`, `version`, `doctor`), default passthrough for existing Aim commands, and initial validation against representative `up`, `init`, and `runs` workflows

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Safe coexistence: the plan only probes for and launches native `aim`; it
      does not modify the installed package or `.aim` data.
- [x] Ownership boundary: `help`, `version`, and `doctor` are explicitly owned
      by `aimx`; all other command paths remain passthrough.
- [x] Read-only default: owned commands are diagnostic/read-only, and
      passthrough behavior adds no implicit repair, migration, or mutation.
- [x] CLI-first contract: the interface is designed for shells, SSH sessions,
      automation, and CI, with human-readable diagnostics and preserved native
      output for delegated commands.
- [x] Compatibility plan: v1 validates PATH-based native Aim discovery,
      passthrough fidelity, and missing-Aim failure semantics without forcing a
      repo-defined Aim version onto end users.

## Project Structure

### Documentation (this feature)

```text
specs/001-aim-command-passthrough/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── cli-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
└── aimx/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── router.py
    ├── commands/
    │   ├── __init__.py
    │   ├── help.py
    │   ├── version.py
    │   └── doctor.py
    └── native_aim/
        ├── __init__.py
        ├── locator.py
        └── passthrough.py

tests/
├── contract/
│   └── test_cli_contract.py
├── integration/
│   ├── test_missing_native_aim.py
│   └── test_passthrough_behavior.py
└── unit/
    ├── test_locator.py
    ├── test_owned_commands.py
    └── test_router.py
```

**Structure Decision**: Use a single-project Python CLI package under
`src/aimx`, with isolated modules for command routing, owned-command handling,
and native Aim interaction. Keep unit, integration, and contract tests
separate so user-visible CLI behavior can be validated independently from
internal routing logic.

## Complexity Tracking

No constitution violations or deliberate complexity exceptions were identified
during planning.
