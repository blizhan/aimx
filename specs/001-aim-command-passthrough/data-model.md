# Data Model: Native Aim Passthrough

## Overview

This feature does not introduce persistent storage. The data model is the
runtime information that flows through `aimx` while it decides whether to own a
command or delegate it to native Aim.

## Entity: CommandInvocation

**Purpose**: Represents a single CLI invocation received by `aimx`.

| Field | Type | Description |
|-------|------|-------------|
| `raw_args` | ordered list of strings | All command-line arguments provided after `aimx` |
| `requested_path` | string or empty | The command path inferred from the arguments, such as `version`, `doctor`, or `up` |
| `working_directory` | path | The directory from which the command was launched |
| `route_kind` | enum | The chosen route: `owned`, `passthrough`, or `error` |

**Validation rules**:

- Argument order is preserved exactly as provided by the user.
- An empty invocation is treated as an owned help-style request.
- Route selection must happen before any native Aim process is launched.

## Entity: CommandRoute

**Purpose**: Captures the routing decision made for a `CommandInvocation`.

| Field | Type | Description |
|-------|------|-------------|
| `route_kind` | enum | `owned`, `passthrough`, or `error` |
| `owned_command` | enum or empty | `help`, `version`, `doctor`, or empty when delegated |
| `delegated_args` | ordered list of strings | The exact argument list to send to native `aim` when passthrough is chosen |
| `reason` | string | Human-readable explanation for why this route was selected |

**Validation rules**:

- Owned commands are claimed explicitly and never silently delegated once
  reserved by `aimx`.
- Passthrough routes must carry the full delegated argument list without
  reordering.

## Entity: NativeAimResolution

**Purpose**: Describes whether native `aim` is available for passthrough.

| Field | Type | Description |
|-------|------|-------------|
| `status` | enum | `available`, `missing`, or `unusable` |
| `executable_path` | path or empty | The resolved native `aim` executable path when available |
| `version_text` | string or empty | Native Aim version text if it can be determined |
| `diagnostic_message` | string | Explanation suitable for `doctor` output or delegated failure messages |

**Validation rules**:

- `available` requires a launchable executable path.
- `missing` and `unusable` must include actionable diagnostics.

## Entity: DiagnosticReport

**Purpose**: The user-visible status returned by owned diagnostic commands.

| Field | Type | Description |
|-------|------|-------------|
| `aimx_version` | string | The version reported by `aimx` |
| `native_aim_status` | enum | Mirrors `NativeAimResolution.status` |
| `native_aim_path` | path or empty | The executable that would be used for passthrough |
| `native_aim_version` | string or empty | Version information when native Aim is available |
| `passthrough_ready` | boolean | Whether delegated commands can run successfully |

**Validation rules**:

- `passthrough_ready` is true only when native Aim is available.
- The report must be human-readable and safe to emit in CI logs.

## Entity: DelegatedExecutionResult

**Purpose**: Represents the observed outcome of a delegated native Aim command.

| Field | Type | Description |
|-------|------|-------------|
| `delegated_args` | ordered list of strings | The arguments sent to native `aim` |
| `process_started` | boolean | Whether native Aim was launched successfully |
| `exit_status` | integer | The resulting process exit status or Aimx-defined failure status |
| `error_message` | string or empty | Actionable message when passthrough could not be started |

**Validation rules**:

- If `process_started` is true, the exit status reflects the native process.
- If `process_started` is false, the result includes an actionable error
  message.

## Relationships

- A `CommandInvocation` produces exactly one `CommandRoute`.
- A passthrough `CommandRoute` depends on a `NativeAimResolution`.
- `aimx doctor` and `aimx version` may expose part of `NativeAimResolution`
  through a `DiagnosticReport`.
- A passthrough `CommandRoute` plus a successful `NativeAimResolution` can
  produce a `DelegatedExecutionResult`.

## State Transitions

### Invocation Lifecycle

1. `received` → user runs `aimx ...`
2. `routed` → `aimx` classifies the invocation as owned, passthrough, or error
3. `owned-complete` → owned command returns output directly
4. `delegated` → native Aim is launched for passthrough
5. `completed` or `failed` → command exits with native or Aimx-defined status

### Native Aim Availability Lifecycle

1. `unknown` → no probe has been run yet
2. `available` → native Aim executable is resolved and launchable
3. `missing` → no native Aim executable is discoverable
4. `unusable` → an executable was found but cannot be launched correctly
