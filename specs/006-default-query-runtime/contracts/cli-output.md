# CLI Output Contract: Default Query Expression And Run Duration

**Feature**: `006-default-query-runtime`

This contract defines observable CLI behavior for omitted query-language input
and run duration in `aimx query params`.

## Affected Command Shapes

The following owned command forms accept an omitted expression and use
`run.hash != ''` as the effective expression:

```text
aimx query metrics [<expression>] [--repo <path>] [--json] [--oneline | --plain]
                   [--steps start:end | --epochs start:end]
                   [--head N] [--tail N] [--every K]

aimx query images [<expression>] [--repo <path>] [--json] [--oneline | --plain]
                  [--steps start:end | --epochs start:end]
                  [--head N] [--tail N] [--every K] [--max-images N]

aimx query params [<expression>] [--repo <path>] [--json] [--oneline | --plain]
                  [--param <key>]...

aimx trace [<expression>] [--repo <path>] [--table | --csv | --json]
           [--steps start:end] [--head N] [--tail N] [--every K]
           [--width W] [--height H] [--no-color]

aimx trace distribution [<expression>] [--repo <path>]
                        [--steps start:end]
                        [--head N] [--tail N] [--every K]
                        [--step N]
                        [--width W] [--height H] [--no-color]
                        [--table | --csv | --json]
```

The command target is still required for `aimx query`. The `distribution`
subtarget is still required for distribution trace.

## Effective Expression Rules

| Input Shape | Effective Expression |
|-------------|----------------------|
| Expression omitted | `run.hash != ''` |
| Expression token is empty or whitespace-only | `run.hash != ''` |
| First token after target is an option such as `--repo` | `run.hash != ''`; option parsing starts at that token |
| Non-empty explicit expression | The explicit expression exactly as supplied |

Examples:

```bash
aimx query metrics --repo data
aimx query images --repo data --plain
aimx query params --repo data --json
aimx trace --repo data
aimx trace --json --repo data
aimx trace distribution --repo data --head 2
```

Each example above is equivalent to the same command with
`"run.hash != ''"` supplied as the expression.

## Query Params Duration Output

`aimx query params` includes run duration in every output mode.

### Human-Readable Output

The default params table includes a duration column near the run identity
columns:

```text
RUN       EXPERIMENT           NAME                 DURATION  hparam.lr ...
eca37394  cloud-segmentation   ucloudnet-pre-0503   1h20m13s  0.0001
```

If duration cannot be determined, the cell is an explicit marker such as
`unavailable` or `running`; it must not be blank or `0s` unless duration is
actually zero.

### Plain Output

Plain output emits one tab-separated row per matched run:

```text
<repo>	<short_hash>	<experiment>	<run_name>	<duration>	<key=value>...
```

Missing duration uses the same explicit marker semantics as the rich output.

### JSON Output

JSON output keeps the existing top-level params envelope and adds a stable
`duration` object for every run:

```json
{
  "target": "params",
  "repo": "data",
  "expression": "run.hash != ''",
  "runs_count": 1,
  "param_keys": ["hparam.lr"],
  "runs": [
    {
      "hash": "eca37394eeb84f48a5d2d736",
      "experiment": "cloud-segmentation",
      "name": "ucloudnet-pre-0503",
      "duration": {
        "seconds": 4813.352312088013,
        "status": "available",
        "source": "duration"
      },
      "params": {
        "hparam.lr": 0.0001
      },
      "missing_params": []
    }
  ]
}
```

When duration cannot be calculated, `duration.seconds` is `null` and
`duration.status` is `unavailable` or `running`.

## Exit Status

| Condition | Exit Status | Output |
|-----------|-------------|--------|
| Omitted expression with valid repo and matches | `0` | Same mode-specific output as explicit `run.hash != ''` |
| Omitted expression with valid repo and zero matches | `0` | Existing no-results or empty structured output behavior |
| Explicit valid expression | `0` | Same output behavior as before this feature |
| Explicit invalid expression | `2` | Existing actionable evaluation error on stderr |
| Missing repository path | `2` | Existing actionable repository error on stderr |
| Missing option value | `2` | Existing actionable parser error on stderr |
| Missing duration metadata on a params run | `0` | Run remains visible with duration status marker |

## Non-Regression Requirements

- Explicit query and trace expressions keep existing match semantics.
- `query images` inline rendering and `--max-images` behavior remain unchanged.
- `query params --param` selection and missing-param behavior remain unchanged.
- Metric trace and distribution trace `--table`, `--csv`, and `--json` modes
  remain parseable with their existing data shapes, except where an existing
  output already reports an expression and must now report the effective
  default.
- Commands outside owned `aimx` surfaces continue to delegate to native `aim`.
