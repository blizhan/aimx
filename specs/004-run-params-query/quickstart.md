# Quickstart: Run Params Query And Experiment Comparison

**Feature**: `004-run-params-query`

## 1. Prepare The Environment

```bash
uv sync
```

The sample Aim repository should exist at:

```text
/Users/blizhan/data/code/github/aimx/data/.aim
```

## 2. Query Params For All Matching Runs

```bash
uv run aimx query params "run.experiment == 'cloud-segmentation'" --repo data
```

Expected result:

- exit code `0`
- output includes `Repo: data`
- output includes run hashes and experiment labels
- output includes visible parameter values or a no-params marker per run

## 3. Compare Selected Params

```bash
uv run aimx query params "run.experiment == 'cloud-segmentation'" --repo data \
  --param hparam.lr \
  --param hparam.optimizer \
  --param hparam.weight_decay
```

Expected result:

- exit code `0`
- the selected keys appear as comparable columns or cells
- missing values are shown as `-`

## 4. Use Params In The Query Expression

```bash
uv run aimx query params "run.hparam.lr == 0.0001" --repo data --param hparam.lr
```

Expected result:

- exit code `0`
- every returned row matches the Aim run-query expression

## 5. Machine-Readable Output

```bash
uv run aimx query params "run.experiment == 'cloud-segmentation'" --repo data \
  --param hparam.lr \
  --param hparam.optimizer \
  --json
```

Expected result:

- valid JSON
- top-level `target` is `params`
- `runs_count` equals the length of `runs`
- each run contains `hash`, `experiment`, `name`, `params`, and
  `missing_params`

## 6. Plain Output For Pipelines

```bash
uv run aimx query params "run.experiment == 'cloud-segmentation'" --repo data \
  --param hparam.lr \
  --plain
```

Expected result:

- one tab-separated row per matched run
- no ANSI styling
- cells include `hparam.lr=<value>` or `hparam.lr=-`

## 7. Error Checks

Invalid repository:

```bash
uv run aimx query params "run.hash != ''" --repo missing-repo
```

Expected: exit code `2`, actionable repository error on stderr.

Invalid params option usage:

```bash
uv run aimx query metrics "metric.name == 'loss'" --repo data --param hparam.lr
```

Expected: exit code `2`, actionable option error on stderr.

## 8. Test Commands

```bash
uv run pytest tests/unit/test_query_helpers.py tests/unit/test_run_params.py -q
uv run pytest tests/integration/test_query_command.py -q
uv run pytest tests/contract/test_query_contract.py -q
uv run pytest -q
```
