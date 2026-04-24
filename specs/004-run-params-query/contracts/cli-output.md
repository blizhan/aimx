# CLI Output Contract: `aimx query params`

**Feature**: `004-run-params-query`

This contract defines the observable CLI behavior for run parameter querying.

## Command Shape

```text
aimx query params <expression> [--repo <path>] [--json] [--oneline | --plain]
                  [--no-color] [--verbose] [--param <key>]...
```

## Owned Options

| Option | Applies To | Behavior |
|--------|------------|----------|
| `--repo <path>` | params | Uses a local Aim repository root or `.aim` directory. Defaults to the current directory. |
| `--json` | params | Emits a JSON document and no rich formatting. |
| `--oneline` / `--plain` | params | Emits tab-separated rows suitable for shell pipelines. |
| `--no-color` | params | Disables ANSI styling in human-readable output. |
| `--verbose` | params | Includes expanded header details when supported by the renderer. |
| `--param <key>` | params only | Adds one flattened parameter key to the comparison view. May be repeated. |

Using `--param` with `metrics` or `images` is an error with exit code `2`.

## Query Expression

The expression is passed to Aim's run-query evaluator after existing short
`run.hash` literals are expanded.

Examples:

```bash
aimx query params "run.experiment == 'cloud-segmentation'" --repo data
aimx query params "run.hparam.lr == 0.0001" --repo data --param hparam.lr
aimx query params "run.hash == 'eca37394'" --repo data --json
```

## Human-Readable Output

Default output is a comparison table with:

- repository and query summary
- run hash
- experiment label
- run name when available
- selected parameter columns, or a readable default set of discovered params

Missing requested params are displayed as `-`. Runs with no params remain in
the output and are marked clearly.

## Plain Output

Plain output emits one tab-separated row per matched run:

```text
<repo>	<short_hash>	<experiment>	<run_name>	<key=value>...
```

Missing requested params are emitted as `key=-`.

## JSON Output

JSON output uses this stable top-level envelope:

```json
{
  "target": "params",
  "repo": "data",
  "expression": "run.experiment == 'cloud-segmentation'",
  "runs_count": 1,
  "param_keys": ["hparam.lr", "hparam.optimizer"],
  "runs": [
    {
      "hash": "eca37394eeb84f48a5d2d736",
      "experiment": "cloud-segmentation",
      "name": "ucloudnet-pre-0503",
      "params": {
        "hparam.lr": 0.0001,
        "hparam.optimizer": "AdamW"
      },
      "missing_params": []
    }
  ]
}
```

## Exit Status

| Condition | Exit Status | Output |
|-----------|-------------|--------|
| Valid query with one or more matches | `0` | Rendered params result |
| Valid query with zero matches | `0` | Explicit no-results message or empty JSON envelope |
| Run has no params | `0` | Run remains visible with no-params marker |
| Missing requested param on a run | `0` | Missing value marker for that run |
| Missing repository path | `2` | Actionable error on stderr |
| Invalid query expression | `2` | Actionable error on stderr |
| Invalid `--param` usage | `2` | Actionable error on stderr |

## Non-Regression Requirements

- `aimx query metrics ...` JSON, plain, and rich output shapes remain unchanged.
- `aimx query images ...` JSON, plain, rich, and inline image behavior remain
  unchanged.
- Commands outside the owned `aimx` surfaces continue to delegate to native
  `aim`.
