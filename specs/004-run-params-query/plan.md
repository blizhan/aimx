# Implementation Plan: Run Params Query And Experiment Comparison

**Branch**: `004-run-params-query` | **Date**: 2026-04-24 | **Spec**: [spec.md](/Users/blizhan/data/code/github/aimx/specs/004-run-params-query/spec.md)
**Input**: Feature specification from `/Users/blizhan/data/code/github/aimx/specs/004-run-params-query/spec.md`

## Summary

Extend the existing `aimx`-owned `query` command with a read-only `params`
target for run-level parameter comparison. The implementation will route
`aimx query params <expression>` through Aim's run-query surface, collect each
matching run's metadata attributes from the local repository, flatten nested
parameter dictionaries into stable dotted keys, and render the result as a
terminal comparison table, tab-separated plain output, or a stable JSON
envelope. The feature stays within the current companion CLI model: no
repository mutation, no native `aim` replacement, no new runtime dependency,
and no behavior change for existing `metrics`, `images`, `trace`, or passthrough
commands.

## Technical Context

**Language/Version**: Python 3.12 for development, runtime support `>=3.10,<3.13`  
**Primary Dependencies**: Python standard library, `numpy>=1.24`, `rich>=13.7`, `textual-image>=0.12.0`, existing Aim SDK usage for owned query commands via the local/dev environment; no new dependency planned  
**Storage**: Existing local Aim repositories on disk, read-only; run params are read from Aim run metadata attributes under `.aim`  
**Testing**: pytest unit, integration, and contract suites; sample Aim
repository rooted at `/Users/blizhan/data/code/github/aimx/data` for end-to-end
validation  
**Target Platform**: Terminal-first CLI for local shells, SSH sessions, scripts,
and CI on Python-supported platforms  
**Project Type**: Single-project Python CLI application  
**Performance Goals**: Params queries over the sample repository complete in a
single command invocation; comparison of at least 3 selected params across at
least 3 runs remains readable in terminal output; machine-readable output
includes all returned run rows without truncating selected values  
**Constraints**: Read-only; preserve native Aim passthrough behavior; keep
existing `query metrics` and `query images` output contracts stable; avoid
loading metric/image blobs for params queries; support repo root and `.aim`
paths consistently  
**Scale/Scope**: One new `query` target (`params`), repeatable `--param KEY`
selection, run-param collection helper, params renderers, help/README updates,
and focused tests

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Safe coexistence: params queries only read local Aim run metadata; no
      normal-path change modifies the installed `aim` package, replaces the
      native `aim` executable, or mutates `.aim` repo data.
- [x] Ownership boundary: `aimx` newly owns only the `query params` target and
      the `--param KEY` option for that target; existing `metrics`, `images`,
      `trace`, help, doctor, version, and native passthrough boundaries remain
      unchanged.
- [x] Read-only default: all params query behavior is inspection-only and uses
      no Aim mutation APIs.
- [x] CLI-first contract: the plan defines rich terminal output, tab-separated
      plain output, and JSON output so shell, SSH, automation, and CI users can
      consume the feature.
- [x] Compatibility plan: design reuses the same repo normalization, short-hash
      expansion, Aim query error handling, and pytest suites used by existing
      owned query commands.

## Project Structure

### Documentation (this feature)

```text
/Users/blizhan/data/code/github/aimx/specs/004-run-params-query/
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
├── README.md                              # document query params examples
├── src/aimx/
│   ├── commands/
│   │   ├── help.py                        # add params usage to owned-command help
│   │   └── query.py                       # add params target, --param parsing, dispatch
│   ├── aim_bridge/
│   │   └── run_params.py                  # NEW: collect and normalize run params
│   └── rendering/
│       └── params_views.py                # NEW: rich/plain/JSON params renderers
└── tests/
    ├── contract/
    │   └── test_query_contract.py         # add params output contract coverage
    ├── integration/
    │   └── test_query_command.py          # add sample-repo params query coverage
    └── unit/
        ├── test_query_helpers.py          # add params target and --param parsing tests
        └── test_run_params.py             # NEW: flattening, selection, missing values
```

**Structure Decision**: Keep the existing single-project CLI layout. Add
`aim_bridge/run_params.py` so params extraction is not buried in the existing
metric/image bridge, and add `rendering/params_views.py` so query renderers stay
focused by target. `commands/query.py` remains the thin orchestration point that
parses CLI arguments, normalizes repository paths, invokes the bridge, and
chooses the output renderer.

## Phase 0: Research Summary

Phase 0 decisions are captured in [research.md](/Users/blizhan/data/code/github/aimx/specs/004-run-params-query/research.md). Key outcomes:

- Use Aim `Repo.query_runs(expression, report_mode=QueryReportMode.DISABLED)`
  for params queries because it returns run-level matches without iterating
  metric or image sequences.
- Extract user-visible params from each matched run's
  `run.meta_run_tree.collect()["attrs"]`, then flatten nested dictionaries into
  dotted keys such as `hparam.lr` and `hparam.optimizer`.
- Support a repeatable `--param KEY` option for focused comparison. Keys are
  matched after flattening, so users can request nested params by dotted path.
- Preserve existing query expression behavior by resolving short `run.hash`
  literals before forwarding the expression to Aim. Experiment-name filtering
  uses Aim's existing run expression fields, for example
  `run.experiment == 'cloud-segmentation'`.
- Sort/group params output by experiment label, then run name, then run hash to
  make experiment comparison stable and easy to scan.

## Phase 1: Design Summary

- Extend `SUPPORTED_TARGETS` with `params` and add `param_keys: tuple[str, ...]`
  to `QueryInvocation`. Parse `--param KEY` as a repeatable params-only option;
  reject missing values, empty keys, duplicate keys after trimming, and usage on
  `metrics` or `images`.
- Introduce `RunParams` in `aim_bridge/run_params.py` with `run: RunMeta`,
  `params: dict[str, Any]`, `selected_keys: tuple[str, ...]`, and
  `missing_keys: tuple[str, ...]`.
- Implement `collect_run_params(expression, repo_path, selected_keys)`:
  normalize short hashes, run `Repo.query_runs`, iterate `collection.run`,
  extract run metadata with the same semantics as existing query collectors,
  flatten metadata attrs, apply selected-key filtering, and never call Aim write
  APIs.
- Implement `params_views.py` renderers:
  - Rich table: one row per run, columns for run hash, experiment, run name, and
    selected/default params; missing values render as `-`; long values are
    shortened for terminal fit.
  - Plain output: tab-separated rows with repo, short hash, experiment, run
    name, and `key=value` cells.
  - JSON output: stable envelope with `target`, `repo`, `expression`,
    `runs_count`, `param_keys`, and `runs`.
- Default key behavior: when no `--param` is provided, use the sorted union of
  flattened param keys for the human/plain view, capped to a readable default
  column budget with an omitted-count note; JSON includes the complete params
  object for every run.
- Update help and README with examples for all three output modes and for
  experiment-name filtering.

## Post-Design Constitution Check

- [x] Safe coexistence: design reads run metadata through Aim's public run query
      surface and does not modify the installed package, executable, or repo.
- [x] Ownership boundary: all new behavior is inside `aimx query params`; no
      unowned native Aim commands are intercepted.
- [x] Read-only default: bridge code extracts metadata attributes only and
      avoids `run.set`, tracking, artifact logging, or migration APIs.
- [x] CLI-first contract: rich, plain, and JSON outputs are defined in
      `/Users/blizhan/data/code/github/aimx/specs/004-run-params-query/contracts/cli-output.md`.
- [x] Compatibility: existing metrics/images query contracts and passthrough
      tests remain part of the validation set; params query uses the same Aim
      package path already required by owned query commands.

## Complexity Tracking

No constitution violations; no exceptional complexity requires justification.
The main design choice is adding two small focused modules instead of extending
the already-mixed metric/image files further.
