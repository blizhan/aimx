# Feature Specification: Native Aim Passthrough

**Feature Branch**: `001-aim-command-passthrough`  
**Created**: 2026-04-11  
**Status**: Draft  
**Input**: User description: "先考虑架构设计，先支持aimx 跑原有的aim支持的命令"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Existing Aim Commands (Priority: P1)

As an existing Aim user, I want to invoke familiar Aim commands through `aimx`
so I can adopt `aimx` without breaking my current terminal workflow.

**Why this priority**: This is the core promise of the product. If delegated
commands do not behave like native Aim, `aimx` does not earn adoption.

**Independent Test**: With native `aim` available on the machine, run a small
set of representative existing Aim commands through `aimx` and confirm that the
observable behavior matches native Aim closely enough to preserve the workflow.

**Acceptance Scenarios**:

1. **Given** native `aim` is installed, **When** the user runs `aimx up`,
   **Then** the command is delegated to native `aim` and returns the native
   command output and result.
2. **Given** native `aim` is installed, **When** the user runs a delegated
   command with additional flags or subcommands, **Then** `aimx` forwards the
   full argument list in the same order provided by the user.
3. **Given** native `aim` returns a failure for a delegated command, **When**
   the user runs that command through `aimx`, **Then** `aimx` surfaces that
   failure clearly instead of masking or rewriting it.

---

### User Story 2 - Understand aimx Behavior Quickly (Priority: P2)

As a user trying `aimx` for the first time, I want the CLI to explain what
`aimx` owns and what it delegates so I know how to use it with confidence.

**Why this priority**: Users need a clear mental model of the product before
they can trust it as a companion CLI.

**Independent Test**: Run `aimx`, `aimx --help`, `aimx version`, and
`aimx doctor` on a machine with native `aim` available and confirm that the
user can understand the owned-command surface and environment status from the
CLI alone.

**Acceptance Scenarios**:

1. **Given** the user runs `aimx --help`, **When** the help text is displayed,
   **Then** it lists `aimx`-owned commands and clearly states that other
   commands are delegated to native `aim`.
2. **Given** native `aim` is available, **When** the user runs `aimx version`,
   **Then** the output includes the `aimx` version and the detected native Aim
   version.
3. **Given** native `aim` is available, **When** the user runs `aimx doctor`,
   **Then** the output explains whether native `aim` is available, which
   executable will be used, and whether delegation is ready.

---

### User Story 3 - Get Actionable Errors When Aim Is Missing (Priority: P3)

As a user whose machine does not currently have native `aim` available, I want
delegated commands to fail with clear guidance so I understand what to fix
without risking my environment.

**Why this priority**: Safe, explicit failure is part of the companion CLI
promise and prevents confusing adoption dead ends.

**Independent Test**: Run `aimx` on a machine where native `aim` is not
available and verify that owned commands still work while delegated commands
return actionable guidance.

**Acceptance Scenarios**:

1. **Given** native `aim` is unavailable, **When** the user runs `aimx help`,
   `aimx version`, or `aimx doctor`, **Then** those owned commands still work.
2. **Given** native `aim` is unavailable, **When** the user runs a delegated
   command such as `aimx up`, **Then** `aimx` fails clearly and tells the user
   that native `aim` must be installed or discoverable.
3. **Given** native `aim` becomes available later, **When** the user retries
   the delegated command, **Then** the workflow proceeds without cleanup or
   migration steps inside `aimx`.

### Edge Cases

- Native `aim` is present in the environment but cannot be executed.
- Native `aim` returns usage errors or non-zero exit status for a delegated
  command.
- The user passes an arbitrary combination of subcommands and flags that `aimx`
  does not own.
- Multiple `aim` executables exist on the machine and the user needs to know
  which one `aimx` will use.
- A future `aimx`-owned command name must not accidentally be delegated once it
  is claimed by `aimx`.

## Constitution Alignment *(mandatory)*

- **CA-001 Safety & Mutability**: This feature does not modify the installed
  `aim` package and does not mutate `.aim` data during normal use. The existing
  Aim dependency in the repo's development dependency group is treated as a
  development and testing convenience, not as an end-user runtime requirement.
- **CA-002 Ownership Boundary**: `aimx` owns a minimal command surface for
  help, version reporting, environment diagnosis, and command routing. All
  other command paths remain native `aim` passthrough in this feature.
- **CA-003 CLI & Output Contract**: The feature is designed for local shells,
  SSH sessions, scripts, and CI. Owned commands provide clear human-readable
  guidance, while delegated commands preserve native Aim output behavior as
  closely as practical.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide `aimx` as a user-facing CLI entrypoint
  for Aim-compatible terminal workflows.
- **FR-002**: The system MUST treat a defined minimal set of commands and flags
  for help, version reporting, and diagnosis as `aimx`-owned behavior.
- **FR-003**: The system MUST route any command path not owned by `aimx` to the
  installed native `aim` CLI rather than reinterpreting or reimplementing that
  command in v1.
- **FR-004**: The system MUST forward delegated command arguments in the same
  order supplied by the user.
- **FR-005**: The system MUST preserve delegated command `stdout`, `stderr`,
  and exit status as closely as practical.
- **FR-006**: The system MUST present a mixed help experience that explains the
  `aimx`-owned commands and states that unowned commands are delegated to
  native `aim`.
- **FR-007**: The system MUST allow `aimx version` to report the `aimx`
  version and, when native `aim` is available, the detected native Aim version.
- **FR-008**: The system MUST allow `aimx doctor` to report whether native
  `aim` is available, which executable will be used, and whether basic
  delegation is ready.
- **FR-009**: When native `aim` is unavailable, the system MUST keep
  `aimx`-owned commands usable and MUST return a clear actionable error for
  delegated commands.
- **FR-010**: The system MUST NOT require users to replace the existing `aim`
  command in order to use `aimx`.
- **FR-011**: The system MUST NOT require end users to install or accept a
  repo-defined native Aim version as part of `aimx` runtime installation.
- **FR-012**: The system MUST NOT modify the installed `aim` package or `.aim`
  repository data during help, version, doctor, or delegated-command execution.
- **FR-013**: The system MUST make the owned-versus-delegated command boundary
  understandable from the CLI experience.

### Key Entities *(include if feature involves data)*

- **Command Invocation**: A single user-entered `aimx` CLI invocation,
  including the raw argument list and intent to run a command.
- **Command Route**: The classification of a command invocation as either an
  `aimx`-owned command or a native `aim` passthrough command.
- **Native Aim Availability**: The current status of whether native `aim` can
  be found and used, including the executable selected for delegation.
- **Diagnostic Report**: The user-visible status information returned by owned
  commands such as version reporting and environment diagnosis.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In acceptance testing, representative existing Aim workflows such
  as `up`, `init`, and one `runs` subcommand can be invoked through `aimx` with
  matching success or failure status to native `aim`.
- **SC-002**: A first-time user can determine from `aimx --help` in one CLI
  session which commands are owned by `aimx` and that all other commands are
  delegated to native `aim`.
- **SC-003**: When native `aim` is unavailable, 100% of delegated-command test
  attempts fail with an actionable message that explains the missing dependency
  and the next step.
- **SC-004**: Users can install `aimx` without replacing their existing `aim`
  command or being forced onto a repo-defined native Aim version.

## Assumptions

- Users who want delegated Aim behavior already have, or are willing to install,
  native `aim` separately from `aimx`.
- `aimx` v1 is focused on terminal-first environments; GUI-specific behavior is
  out of scope.
- The initial `aimx`-owned command surface is intentionally small and centers
  on help, version reporting, diagnosis, and routing.
- Native Aim remains the source of truth for the behavior of delegated
  commands.
- The Aim dependency in the development dependency group is only for local
  development, testing, and examples, and does not redefine the runtime
  contract for end users.
