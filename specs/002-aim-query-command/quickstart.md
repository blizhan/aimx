# Quickstart: Aim Query Command

## Prerequisites

1. Sync the development environment:

```bash
uv sync --group dev
```

2. Confirm the sample Aim repository is present:

```bash
ls data/.aim
```

## Development checks

1. Run the existing test suite before changes:

```bash
uv run pytest
```

2. Verify native Aim path behavior on the sample repository:

```bash
uv run aim runs --repo data ls
uv run aim runs --repo data/.aim ls
```

3. Verify SDK path behavior that the implementation must normalize:

```bash
uv run python - <<'PY'
from aim import Repo
Repo("data")
print("data OK")
PY
```

## Feature verification targets

After implementation, verify the new command with both repository-path forms.

### Human-readable metric query

```bash
uv run aimx query metrics "metric.name == 'loss'" --repo data
uv run aimx query metrics "metric.name == 'loss'" --repo data/.aim
```

Expected outcome:

- Both commands succeed
- Both commands report the same match count
- Output is readable in the terminal

### Structured metric query

```bash
uv run aimx query metrics "metric.name == 'loss'" --repo data --json
```

Expected outcome:

- Command succeeds
- Output is valid JSON
- Output contains target, expression, repo path, count, and rows

### Image query

```bash
uv run aimx query images "images" --repo data --json
```

Expected outcome:

- Command succeeds
- Output includes at least one image-sequence row from the sample repository

### Error handling

```bash
uv run aimx query metrics "metric.name ==" --repo data
uv run aimx query metrics "metric.name == 'loss'" --repo does-not-exist
```

Expected outcome:

- Invalid expression returns an actionable error
- Invalid repo path returns an actionable error
- Neither command mutates repository data

## Targeted tests

```bash
uv run pytest tests/unit/test_router.py tests/unit/test_query_helpers.py
uv run pytest tests/integration/test_query_command.py
uv run pytest tests/contract/test_query_contract.py tests/contract/test_cli_contract.py
```
