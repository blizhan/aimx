# Query Command Contract

## Owned command surface

`aimx` owns the following new command surface:

```text
aimx query metrics <expression> --repo <path> [--json]
aimx query images <expression> --repo <path> [--json]
```

## Parsing contract

- `query` is a reserved top-level `aimx` command
- `metrics` and `images` are the only supported target selectors in this
  feature iteration
- `<expression>` is required and is treated as the full Aim query expression
- `--repo <path>` is required for explicit repository selection in this feature
- `--json` switches output from human-readable text to machine-readable JSON

## Repository-path contract

- Repo roots such as `data` must be accepted
- `.aim` metadata-directory paths such as `data/.aim` must also be accepted
- If a `.aim` directory path is supplied, `aimx` must normalize it to the
  parent repo root before opening the repository
- Invalid paths must fail before query execution with an actionable error

## Success output contract

### Human-readable output

- Must identify the query target
- Must identify the effective repository path
- Must report match count
- Must include concise per-row details sufficient for terminal inspection
- Zero-result queries must print an explicit zero-results message

### JSON output

Successful JSON responses must follow this shape:

```json
{
  "target": "metrics",
  "expression": "metric.name == 'loss'",
  "repo_path": "data",
  "count": 2,
  "rows": [
    {
      "run_id": "abc123",
      "target": "metrics",
      "name": "loss",
      "context": {},
      "summary": "run abc123 metric loss"
    }
  ]
}
```

Notes:

- `target`, `expression`, `repo_path`, `count`, and `rows` are mandatory
- `rows` may be empty when `count` is `0`
- Row ordering should be deterministic within a single run of the command

## Error contract

- Unsupported targets must fail during argument parsing
- Missing expressions must fail during argument parsing
- Invalid repo paths must fail before query evaluation
- Query-evaluation failures must fail with actionable messaging
- Error responses do not need JSON support in this feature iteration unless the
  implementation can provide it without complicating the CLI contract
