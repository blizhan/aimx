# Quickstart: Default Query Expression And Run Duration

This quickstart validates the feature against the local Aim test repository at
`/Users/blizhan/data/code/github/aimx/data`.

## 1. Sync The Environment

```bash
uv sync --group dev
```

## 2. Query Without Expressions

Each command should behave as though `"run.hash != ''"` was supplied.

```bash
uv run aimx query metrics --repo data
uv run aimx query images --repo data --plain
uv run aimx query params --repo data
```

Expected:

- all commands exit with status `0`
- query output reports or behaves according to the effective expression
  `run.hash != ''`
- explicit output controls such as `--plain` keep their existing shape

## 3. Trace Without Expressions

```bash
uv run aimx trace --repo data --head 5
uv run aimx trace --repo data --json --head 1
uv run aimx trace distribution --repo data --head 2 --no-color
```

Expected:

- metric trace uses all runs as the default match set
- JSON export remains parseable
- distribution trace prints its default visual output when the repository has
  distribution data, or the existing no-matches message when it does not

## 4. Confirm Explicit Expressions Still Win

```bash
uv run aimx query metrics "metric.name == 'loss'" --repo data --json
uv run aimx trace "metric.name == 'loss'" --repo data --table --head 2
uv run aimx trace distribution "distribution.name != ''" --repo data --json --head 1
```

Expected:

- explicit expressions are preserved
- known invalid expressions still fail with exit status `2`
- structured outputs remain parseable

## 5. Check Params Duration

```bash
uv run aimx query params --repo data
uv run aimx query params --repo data --plain
uv run aimx query params --repo data --json
```

Expected:

- rich output includes a duration column
- plain output includes a duration field after run name
- JSON output includes `duration.seconds`, `duration.status`, and
  `duration.source` for every returned run
- runs without complete timing metadata remain visible with an explicit
  unavailable or running duration status

## 6. Focused Regression Checks

```bash
uv run pytest tests/unit/test_query_helpers.py tests/unit/test_trace_helpers.py tests/unit/test_run_params.py -q
uv run pytest tests/contract/test_query_contract.py tests/contract/test_trace_contract.py -q
uv run pytest tests/integration/test_query_command.py tests/integration/test_trace_command.py -q
uv run pytest tests/integration/test_passthrough_behavior.py -q
```

Expected:

- parser defaults are covered for query and trace
- params duration contract is covered in rich/plain/JSON paths
- omitted-expression behavior matches explicit `run.hash != ''`
- native Aim passthrough behavior remains unchanged

## 7. Implementation Verification Notes

Verified on 2026-06-15 against the local `data` Aim repository.

- Sections 2-5 manual CLI commands exited with status `0`.
- Distribution quickstart commands exited with status `0`; the local sample
  repository currently reports the existing no-matches path for distribution
  data in some pytest scenarios, so those tests keep their skip behavior.
- Focused unit checks passed: `144 passed`.
- Focused contract checks passed: `39 passed, 8 skipped`.
- Focused integration checks passed: `40 passed, 7 skipped`.
- Passthrough regression checks passed: `10 passed`.
- Full suite passed: `347 passed, 15 skipped`.
