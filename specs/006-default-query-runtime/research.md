# Phase 0 Research: Default Query Expression And Run Duration

## Decision: Use A Single Shared Default Expression

Use the literal expression `run.hash != ''` whenever query-language input is
omitted or blank for the owned query and trace surfaces.

**Rationale**: This is the expression requested in the feature description and
matches existing examples already used for "all runs" params queries. Sharing
one value across command parsers keeps behavior consistent and makes output
contracts easy to verify.

**Alternatives considered**:

- Target-specific defaults such as `metric.name != ''`, `images`, or
  `distribution.name != ''`: rejected because the request explicitly asks for
  `run.hash != ''` and consistency matters more than target-specific narrowing.
- Leaving trace commands expression-required: rejected because the feature
  explicitly includes `trace` and `trace distribution`.
- Empty string forwarded to Aim: rejected because Aim expression behavior would
  be less explicit and less stable for users and tests.

## Decision: Treat Option-Looking Tokens As Omitted Expression

When the parser sees an option token where an expression could appear, treat the
expression as omitted and parse the token as the first option.

**Rationale**: Users should be able to run concise commands like
`aimx query params --repo data`, `aimx trace --repo data`, and
`aimx trace distribution --repo data`. Treating `--repo` as an expression would
be surprising and would break normal command-line conventions.

**Alternatives considered**:

- Require a `--` delimiter before options when omitting expressions: rejected
  as too awkward for the main usability feature.
- Require a new `--all-runs` flag: rejected because the request asks for a
  default when no query language input is present, not another explicit option.
- Accept only fully omitted expressions but not blank strings: rejected because
  whitespace-only input is semantically empty and the spec requires the same
  default behavior.

## Decision: Preserve Explicit Expression Semantics Exactly

Any non-empty, non-option expression token supplied by the user remains the
effective expression, even if the expression is syntactically invalid.

**Rationale**: Existing explicit-expression behavior is a compatibility
contract. Invalid expressions should still be forwarded to the existing
evaluation path and fail with the same actionable error pattern, while valid
expressions should produce the same result set as before.

**Alternatives considered**:

- Validate expression syntax in the parser: rejected because Aim's evaluator is
  already the source of truth and the project currently keeps parser validation
  limited to command shape and options.
- Auto-repair invalid expressions to the default: rejected because that would
  hide user mistakes and change failure semantics.

## Decision: Keep Duration As Params Run Metadata

Add run duration to `query params` as first-class run metadata, separate from
the parameter map and missing-parameter tracking.

**Rationale**: Duration describes the run, not a user-recorded hyperparameter.
Keeping it separate prevents accidental collision with user param keys,
preserves selected-parameter behavior, and gives JSON consumers a stable place
to read duration for every run.

**Alternatives considered**:

- Add duration as a synthetic param key such as `run.duration`: rejected because
  it would mix system metadata with user params and affect default/selected
  param key ordering.
- Show duration only in rich output: rejected because scripts also need a stable
  machine-readable field.
- Add duration filtering or sorting in this feature: rejected as additional
  scope not requested by the user.

## Decision: Derive Duration From Existing Aim Run Timing Metadata

Prefer Aim's exposed `duration` value when present. If it is missing, calculate
duration from `end_time - creation_time` when both values exist. If timing is
incomplete, keep the run visible and mark duration as unavailable or running.

**Rationale**: Local inspection of the sample Aim repository shows run objects
already expose `duration`, `end_time`, `creation_time`, and `created_at`.
Reading these attributes is cheap, read-only, and avoids deeper coupling to
stored Aim internals. The fallback keeps older or partially recorded runs
usable.

**Alternatives considered**:

- Calculate duration from metric timestamps: rejected because it would load
  unrelated sequence data and could be inaccurate for runs without metrics.
- Require both `creation_time` and `end_time`: rejected because `duration` may
  already be directly available and should be used when present.
- Drop runs with missing duration: rejected because params comparison should not
  lose runs due to incomplete metadata.

## Decision: No New Runtime Dependency

Use existing Python and Aim SDK access paths only.

**Rationale**: The repository already depends on the libraries needed for query,
trace, rendering, and tests. Default-expression parsing and duration extraction
do not require additional packages.

**Alternatives considered**:

- Add a duration-formatting dependency: rejected because simple elapsed-time
  formatting is small and should stay in the existing renderer layer.
- Add a CLI parser framework: rejected because the current parsers are small,
  tested, and project-local.
