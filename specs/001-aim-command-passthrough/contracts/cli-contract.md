# CLI Contract: aimx Native Aim Passthrough

## Entry Point

`aimx [ARGS...]`

`aimx` is the user-facing companion CLI. It owns a small diagnostic/help
surface and delegates every other command path to native `aim`.

## Owned Command Contract

| Invocation | Behavior | Success Exit Status | Notes |
|------------|----------|---------------------|-------|
| `aimx` | Show the mixed help experience | `0` | Equivalent to a help-style invocation |
| `aimx --help` | Show the mixed help experience | `0` | Lists owned commands and explains passthrough |
| `aimx help` | Show the mixed help experience | `0` | Reserved by `aimx` and never delegated |
| `aimx version` | Show the `aimx` version and native Aim version when available | `0` | Still succeeds when native Aim is missing |
| `aimx doctor` | Show native Aim availability, resolved executable path, and passthrough readiness | `0` when ready, `1` when not ready | Must remain readable in shell and CI logs |

## Passthrough Contract

Any command path not reserved by `aimx` is treated as a passthrough invocation.

Examples:

- `aimx up`
- `aimx init`
- `aimx runs --help`
- `aimx runs ls`

Passthrough guarantees:

- `aimx` forwards delegated arguments in the same order provided by the user.
- `aimx` preserves native `stdout` and `stderr` behavior as closely as
  practical.
- If native Aim starts successfully, `aimx` exits with the native process exit
  status.
- `aimx` does not rewrite delegated command semantics in v1.

## Missing-Or-Unusable Native Aim Contract

| Condition | Invocation Type | Behavior | Exit Status |
|-----------|-----------------|----------|-------------|
| Native `aim` is not discoverable | Owned command | Owned command still runs and reports the missing dependency where relevant | `0` for `help` and `version`, `1` for `doctor` |
| Native `aim` is not discoverable | Passthrough command | Fail fast with actionable guidance telling the user to install native Aim or make it discoverable | `127` |
| Native `aim` is found but cannot be executed | Passthrough command | Fail fast with actionable guidance describing the unusable executable | `126` |

## Reserved-Command Boundary

- `help`, `version`, and `doctor` are reserved for `aimx`.
- Once a command name is claimed by `aimx`, it must not silently fall back to
  passthrough.
- Future owned commands must be added explicitly to the routing contract before
  implementation.

## Compatibility Notes

- The development dependency group may include Aim for local validation, but
  the CLI contract does not require end users to accept that repo-defined Aim
  version.
- The contract applies to terminal-first environments such as local shells, SSH
  sessions, and CI jobs.
