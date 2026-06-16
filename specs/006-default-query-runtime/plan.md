# Implementation Plan: Default Query Expression And Run Duration

**Branch**: `007-default-query-runtime` | **Date**: 2026-06-14 | **Spec**: [spec.md](/Users/blizhan/data/code/github/aimx/specs/006-default-query-runtime/spec.md)
**Input**: Feature specification from `/Users/blizhan/data/code/github/aimx/specs/006-default-query-runtime/spec.md`

## Summary

Make the existing `aimx`-owned query and trace commands easier to use by
defaulting omitted or blank query-language input to `run.hash != ''`. Apply the
same effective-expression behavior to `aimx query metrics`, `aimx query images`,
`aimx query params`, `aimx trace`, and `aimx trace distribution`, while
preserving explicit expressions and existing output modes. Extend the params
query result with read-only run duration metadata so users can compare run
configuration alongside elapsed runtime. The implementation stays within
current parser, bridge, and renderer boundaries; it adds no new runtime
dependency and does not mutate Aim repositories or native Aim passthrough.

## Technical Context

**Language/Version**: Python 3.12 for development, runtime support `>=3.10,<3.13`  
**Primary Dependencies**: Python standard library, `numpy>=1.24`, `rich>=13.7`, `plotext>=5.3`, `textual-image>=0.12.0`, existing Aim SDK usage for owned query and trace commands; no new dependency planned  
**Storage**: Existing local Aim repositories on disk, read-only; query/trace data and run timing metadata are read from `.aim` repositories without modification  
**Testing**: pytest unit, integration, and contract suites; sample Aim repository rooted at `/Users/blizhan/data/code/github/aimx/data` for end-to-end validation  
**Target Platform**: Terminal-first CLI for local shells, SSH sessions, scripts, and CI on Python-supported platforms  
**Project Type**: Single-project Python CLI application  
**Performance Goals**: Omitted-expression commands should add no extra repository scans beyond the same commands run explicitly with `run.hash != ''`; params duration extraction should be O(1) per returned run and avoid metric/image/blob loading  
**Constraints**: Read-only; preserve native Aim passthrough behavior; preserve explicit-expression semantics; preserve existing image rendering caps, params selection, trace export modes, and stable machine-readable output parseability  
**Scale/Scope**: Five owned command forms, one shared default expression, params duration extraction/rendering, help/README updates, and focused unit/integration/contract tests

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Safe coexistence: default-expression handling and params duration display
      only read existing local Aim data; no normal-path change modifies the
      installed `aim` package, replaces the native `aim` executable, or mutates
      `.aim` repo data.
- [x] Ownership boundary: all behavior changes are limited to `aimx`-owned
      command paths: `query metrics`, `query images`, `query params`, `trace`,
      and `trace distribution`. Native Aim passthrough remains unchanged.
- [x] Read-only default: the feature inspects query results and run metadata
      only and does not introduce write, repair, migration, or sync behavior.
- [x] CLI-first contract: default rich/plain/JSON/export workflows remain
      non-interactive and scriptable; params JSON gains a stable duration field
      while existing explicit-expression workflows remain test-covered.
- [x] Compatibility plan: design reuses current repo normalization, short-hash
      expansion, parser error paths, query/trace collection helpers, and pytest
      suites; tests compare omitted-expression behavior against the explicit
      `run.hash != ''` baseline.

## Project Structure

### Documentation (this feature)

```text
/Users/blizhan/data/code/github/aimx/specs/006-default-query-runtime/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── cli-output.md
├── checklists/
│   └── requirements.md
└── tasks.md            # created later by /speckit.tasks
```

### Source Code (repository root)

```text
/Users/blizhan/data/code/github/aimx/
├── README.md
├── src/aimx/
│   ├── commands/
│   │   ├── help.py                    # document omitted-expression defaults
│   │   ├── query.py                   # default expression parsing for query targets
│   │   └── trace.py                   # default expression parsing for trace targets
│   ├── aim_bridge/
│   │   └── run_params.py              # add run duration metadata to params rows
│   └── rendering/
│       └── params_views.py            # show duration in rich/plain/JSON params output
└── tests/
    ├── contract/
    │   ├── test_query_contract.py     # omitted-expression and params-duration contracts
    │   └── test_trace_contract.py     # omitted-expression trace contracts
    ├── integration/
    │   ├── test_query_command.py      # sample-repo omitted-expression query coverage
    │   └── test_trace_command.py      # sample-repo omitted-expression trace coverage
    └── unit/
        ├── test_query_helpers.py      # parser defaults and option ordering
        ├── test_trace_helpers.py      # parser defaults for trace/distribution
        └── test_run_params.py         # duration extraction and unavailable status
```

**Structure Decision**: Keep the existing single-project CLI layout. Put
default-expression parsing in the command modules because that is where user
arguments become invocations. Keep duration extraction inside the params bridge
and duration formatting inside params renderers so metrics, images, trace data,
and native passthrough stay untouched.

## Phase 0: Research Summary

Phase 0 decisions are captured in [research.md](/Users/blizhan/data/code/github/aimx/specs/006-default-query-runtime/research.md). Key outcomes:

- Use the literal shared default expression `run.hash != ''` when query-language
  input is omitted or blank.
- Treat option-looking tokens such as `--repo` or `--json` as the start of
  options, not as expressions, so `aimx query params --repo data` and
  `aimx trace --repo data` are valid default-expression forms.
- Preserve all non-empty explicit expressions exactly as today, including
  invalid expressions that should still fail through the existing Aim evaluator.
- Use Aim run timing metadata already exposed on local run objects. Prefer
  `run.duration` when available, fall back to `run.end_time - run.creation_time`,
  and mark duration as unavailable/still-running when timing metadata is
  incomplete.
- Keep duration as first-class run metadata in params output, not as a synthetic
  parameter key, so selected param comparisons remain stable.

## Phase 1: Design Summary

- Add a shared constant such as `DEFAULT_QUERY_EXPRESSION = "run.hash != ''"`
  in the command parsing layer or a small shared helper, then use it from query
  and trace parsers.
- Update `parse_query_invocation` so the target remains required, but the
  expression becomes optional. If the next token is absent, blank, or starts
  with `-`, set the expression to the default and parse remaining tokens as
  options. If the next token is non-empty and does not start with `-`, preserve
  it as the explicit expression.
- Update `parse_trace_invocation` for both metric and distribution forms:
  `aimx trace --repo data`, `aimx trace --json --repo data`, and
  `aimx trace distribution --repo data` all use the default expression, while
  non-empty explicit expressions continue to win.
- Keep current error handling for unsupported options, missing option values,
  invalid repositories, and invalid non-empty expressions.
- Extend params data with a duration model that carries `seconds: float | None`
  and `status` (`available`, `unavailable`, or `running`). Derive it during
  `collect_run_params` while the raw Aim run object is still available.
- Update params renderers:
  - Rich table adds a `DURATION` column after run identity columns.
  - Plain output adds a duration cell after run name.
  - JSON output adds a stable `duration` object per run, separate from `params`.
- Update README/help examples to show omitted-expression forms and explain that
  `run.hash != ''` is the default when no expression is supplied.
- Validate with parser unit tests, params duration unit tests, query/trace
  contract tests, integration tests against `data`, and existing passthrough
  tests.

## Post-Design Constitution Check

- [x] Safe coexistence: design reads existing query results and run timing
      attributes only; it does not modify the installed Aim package,
      executable, or repository data.
- [x] Ownership boundary: behavior is contained to existing `aimx`-owned query
      and trace command paths plus their docs/tests; native Aim passthrough is
      not intercepted or extended.
- [x] Read-only default: default expressions broaden inspection convenience but
      still invoke read-only Aim query APIs and in-memory renderers.
- [x] CLI-first contract: [contracts/cli-output.md](/Users/blizhan/data/code/github/aimx/specs/006-default-query-runtime/contracts/cli-output.md)
      defines command shapes, output expectations, JSON duration shape, and
      exit statuses for automation.
- [x] Compatibility: explicit-expression behavior, output parseability, image
      rendering controls, trace modes, params selection, and passthrough tests
      remain part of the validation set.

## Complexity Tracking

No constitution violations; no exceptional complexity requires justification.
The feature uses existing command, bridge, renderer, and test boundaries.
