# Phase 0 Research: Run Params Query And Experiment Comparison

**Feature**: `004-run-params-query`  
**Date**: 2026-04-24

## Decision: Add `aimx query params` As A New Owned Query Target

Use `aimx query params <expression>` for run parameter inspection and
comparison.

**Rationale**: The existing command already uses target names (`metrics`,
`images`) to select a data shape. Params are run-level data rather than metric
or image sequences, so a third target keeps behavior discoverable and avoids
overloading the existing outputs.

**Alternatives considered**:

- Add params columns to `query metrics`: rejected because metrics queries return
  metric series, not runs, and would mix two result granularities.
- Add a separate top-level `aimx params` command: rejected because repository
  selection, query expressions, output modes, and error handling already exist
  under `aimx query`.

## Decision: Use Aim `Repo.query_runs()` For Matching Runs

Collect params by calling `Repo.query_runs(expression,
report_mode=QueryReportMode.DISABLED)` and iterating the returned run
collections.

**Rationale**: Local probing against the sample repository showed Aim exposes
`query_runs` alongside `query_metrics` and `query_images`. It returns run-level
matches without requiring traversal of metric/image sequences, which keeps this
feature read-only and avoids unnecessary blob loading.

**Alternatives considered**:

- Derive runs from metric query results: rejected because runs without matching
  metrics would be excluded and query expressions would require metric fields.
- Walk repository internals directly: rejected because it would increase
  coupling to Aim storage internals beyond the existing public query surface.

## Decision: Extract Params From Run Metadata Attributes

Read params from `run.meta_run_tree.collect()["attrs"]` and flatten nested
dictionaries into dotted keys such as `hparam.lr`, `hparam.optimizer`, and
`model`.

**Rationale**: The sample repository stores user-facing run attributes under
`attrs`; for example, one run has `attrs.hparam.lr`, `attrs.hparam.optimizer`,
and `attrs.model`. Flattening produces stable CLI column names while preserving
nested values for JSON.

**Alternatives considered**:

- Use `run.hparams` or `run.params` attributes: rejected because local probing
  showed those attributes are absent for the current Aim SDK/sample repository.
- Output raw nested dictionaries only: rejected because side-by-side terminal
  comparison needs stable scalar-ish columns.

## Decision: Select Params With Repeatable `--param KEY`

Support focused comparison via one or more `--param KEY` options, where `KEY`
is a flattened dotted parameter path.

**Rationale**: A repeatable option avoids comma parsing ambiguities for param
names, matches common CLI convention, and stays simple in the current manual
argument parser. It also lets users compare nested Aim params directly:
`--param hparam.lr --param hparam.optimizer`.

**Alternatives considered**:

- `--params a,b,c`: rejected for v1 because it introduces escaping questions
  for commas and nested values.
- Positional param names after the expression: rejected because the existing
  parser treats tokens after the expression as flags, and unflagged values would
  make errors less clear.

## Decision: Preserve Aim Query Expression Semantics

Forward the user expression to Aim after the existing short-hash expansion. For
experiment comparison, users can write expressions such as
`run.experiment == 'cloud-segmentation'`; for param filtering, Aim supports
expressions such as `run.hparam.lr == 0.0001` in the sample repository.

**Rationale**: Reusing AimQL-style expressions keeps `aimx` a companion CLI and
avoids inventing a second filtering language. Existing short-hash expansion is
already part of the owned query contract and should apply consistently.

**Alternatives considered**:

- Add custom `--experiment` filtering: rejected for v1 because Aim expressions
  already support experiment fields and custom filters would duplicate query
  language behavior.
- Post-filter runs in `aimx`: rejected because it would split filtering logic
  between Aim and `aimx`, increasing edge cases.

## Decision: Use Three Output Modes Matching Existing Query Commands

Render params as rich table output by default, tab-separated rows for
`--plain`/`--oneline`, and a stable JSON envelope for `--json`.

**Rationale**: This mirrors existing `metrics` and `images` query behavior,
keeps terminal comparison ergonomic, and provides scriptable output without
introducing a separate export feature.

**Alternatives considered**:

- JSON only: rejected because the feature is explicitly about convenient
  comparison in terminal workflows.
- Rich table only: rejected because the constitution requires scriptable
  interfaces where structured automation is realistic.

## Decision: Sort Params Results For Experiment Comparison

Order human-readable and plain params results by experiment label, then run
name, then full run hash.

**Rationale**: The requested experiment-name comparison is easier to scan when
related runs stay together. The run hash remains present so duplicate run names
or empty experiment names are still distinguishable.

**Alternatives considered**:

- Preserve Aim iterator order only: rejected because it is less useful for
  cross-experiment comparison and may vary with repository internals.
- Add a sorting flag in v1: rejected as unnecessary scope until real users need
  alternate ordering.
