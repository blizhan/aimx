# aimx Constitution

This file mirrors the canonical Spec Kit constitution in
`.specify/memory/constitution.md`.

## Core Principles

### I. Safe Coexistence With Native Aim

`aimx` MUST coexist safely with a user's existing Aim installation and Aim data.

- `aimx` MUST NOT modify the installed `aim` package.
- `aimx` MUST NOT replace or shadow the native `aim` executable as a prerequisite
  for normal use.
- `aimx` MUST NOT persistently monkey patch Aim internals.
- `aimx` MUST NOT mutate `.aim` repository data during normal query, inspection,
  diagnostic, export, or passthrough flows.
- Any future write-capable feature that touches Aim-managed state MUST be
  explicit, narrowly scoped, documented with risks, and justified by a concrete
  user need.

Rationale: user trust depends on `aimx` being additive and easy to remove.

### II. Explicit Ownership And Passthrough

`aimx` MUST be explicit about what it owns and what it delegates to native Aim.

- `aimx` MUST own only the commands and flags it explicitly implements.
- Any command path not owned by `aimx` MUST be delegated to the installed
  `aim` CLI.
- Passthrough behavior MUST preserve `argv`, `stdout`, `stderr`, and exit codes
  as closely as practical.
- `aimx` MUST remain a companion CLI, not a fork, replacement runtime, or
  universal compatibility layer for Aim.

Rationale: bounded ownership keeps compatibility work maintainable and honest.

### III. CLI-First, Scriptable Interfaces

`aimx` MUST optimize for terminal-first workflows instead of GUI-only behavior.

- Core features MUST work in local shells, SSH sessions, remote machines,
  scripts, and CI environments.
- Core commands MUST provide predictable non-interactive behavior.
- Query, exploration, export, and diagnostic features SHOULD provide both
  human-readable output and machine-readable output when structured automation
  is a realistic use case.
- Interactive UX MAY be added, but it MUST NOT be the only interface for a core
  capability.

Rationale: the product exists to close CLI gaps, so scriptability is part of
the value proposition.

### IV. Read-Only By Default

`aimx` MUST help users inspect and understand Aim state without surprising
mutation.

- Query, inspection, and diagnostic commands MUST be read-only by default.
- Failures MUST be safe, explicit, and non-destructive.
- `aimx` MUST NOT silently repair, migrate, rewrite, or mutate user data during
  normal operations.
- Any write-capable behavior MUST require explicit user intent and clear
  documentation.

Rationale: safe defaults reduce the cost of adoption and failure recovery.

### V. Focused Expansion Driven By Real User Need

`aimx` MAY grow beyond run querying, but only where that growth stays aligned
with the companion CLI model.

- New `aimx`-owned features MUST solve a concrete gap for Aim users, fit the
  companion CLI model, and avoid destructive risk.
- Features that mainly duplicate native Aim, require deep ongoing coupling to
  unstable Aim internals, or blur `aimx` into a separate platform SHOULD NOT be
  added.
- When the value or safety case is unclear, the project MUST prefer not adding
  the feature yet.

Rationale: disciplined scope is what keeps `aimx` useful instead of sprawling.

## Operational Boundaries

- The user-facing entrypoint is `aimx`; the project MUST NOT depend on replacing
  the user's `aim` command.
- The initial owned surface area centers on terminal-first run querying. Future
  owned categories MAY include richer run exploration, export utilities,
  diagnostics, and safe repo inspection tools.
- The project is not a fork of Aim, a replacement for Aim server, a persistent
  patch layer for Aim internals, a general MLOps platform, an unrestricted repo
  mutation tool, a sync daemon, or a host for arbitrary third-party plugins.
- Architectural boundaries SHOULD stay explicit between the dispatch layer, the
  native Aim passthrough layer, `aimx` command handlers, and any
  parser/evaluator/output components for enhanced commands.
- Compatibility work SHOULD remain bounded to `aimx`-owned features and be
  validated against real Aim versions in active use.

## Delivery Workflow & Quality Gates

- `uv` is the standard for environment and dependency management. Contributors
  SHOULD use `uv sync`, `uv run`, and lockfile-backed workflows instead of ad
  hoc global installs or mixed toolchains.
- Every implementation plan MUST include a Constitution Check covering safety,
  ownership boundary, read-only behavior, CLI/scriptability, and compatibility
  validation.
- Every feature specification MUST state whether it introduces any
  write-capable behavior, what remains passthrough to native Aim, and what
  human-readable or machine-readable outputs are expected.
- Tasks for owned commands SHOULD include validation for passthrough behavior,
  exit-code fidelity, output contracts, and safe failure modes whenever those
  concerns are relevant.
- Changes that increase mutation risk, widen coupling to Aim internals, or
  bypass the `uv` workflow require explicit justification in the plan.

## Governance

This constitution is the canonical project policy for `aimx` and supersedes
conflicting ad hoc process notes.

- Amendments MUST update this document together with any affected templates or
  workflow guidance.
- Compliance MUST be checked during planning, specification, task generation,
  implementation, and review.
- Versioning follows semantic versioning for governance:
  - MAJOR for incompatible principle removals or redefinitions.
  - MINOR for new principles or materially expanded policy.
  - PATCH for clarifications and wording-only refinements.
- If a change requires violating a principle temporarily, the plan MUST
  document the exception, the reason, the safer alternative rejected, and the
  rollback path.
- The canonical Spec Kit copy lives at `.specify/memory/constitution.md`; any
  mirrored repo-level constitution SHOULD remain consistent with it.

**Version**: 1.0.0 | **Ratified**: 2026-04-11 | **Last Amended**: 2026-04-11
