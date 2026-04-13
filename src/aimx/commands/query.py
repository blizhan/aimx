from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_TARGETS = {"metrics", "images"}


@dataclass(frozen=True)
class QueryInvocation:
    target: str
    expression: str
    repo_path: Path
    output_json: bool = False

    def __post_init__(self) -> None:
        if self.target not in SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported query target: {self.target}. Supported targets: metrics, images."
            )
        if not self.expression.strip():
            raise ValueError("Query expression must not be empty.")


@dataclass(frozen=True)
class QueryCommandResult:
    exit_status: int
    output: str | None = None
    error_message: str | None = None


def load_aim_query_support():
    try:
        from aim import Repo
        from aim.sdk.types import QueryReportMode
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "`aimx query` requires the Python `aim` package in the current environment."
        ) from error

    return Repo, QueryReportMode


def normalize_repo_path(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"Repository path does not exist: {path}")
    if path.name == ".aim":
        return path.parent
    return path


def parse_query_invocation(args: list[str]) -> QueryInvocation:
    if len(args) < 4:
        raise ValueError(
            "Usage: aimx query <metrics|images> <expression> --repo <path> [--json]"
        )

    target = args[0]
    expression = args[1]
    rest = args[2:]

    output_json = False
    repo_value: str | None = None
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--json":
            output_json = True
            index += 1
            continue
        if token == "--repo":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --repo.")
            repo_value = rest[index + 1]
            index += 2
            continue
        raise ValueError(f"Unsupported query option: {token}")

    if repo_value is None:
        raise ValueError("Missing required --repo <path> option.")

    return QueryInvocation(
        target=target,
        expression=expression,
        repo_path=Path(repo_value),
        output_json=output_json,
    )


def run_query_command(args: list[str]) -> QueryCommandResult:
    try:
        invocation = parse_query_invocation(args)
        normalized_repo_path = normalize_repo_path(invocation.repo_path)
        rows = collect_query_rows(invocation, normalized_repo_path)
    except ValueError as error:
        return QueryCommandResult(exit_status=2, error_message=str(error))
    except RuntimeError as error:
        return QueryCommandResult(exit_status=2, error_message=str(error))
    except Exception as error:
        return QueryCommandResult(
            exit_status=2, error_message=f"Failed to evaluate query: {error}"
        )

    payload = {
        "target": invocation.target,
        "expression": invocation.expression,
        "repo_path": str(normalized_repo_path),
        "count": len(rows),
        "rows": rows,
    }
    if invocation.output_json:
        return QueryCommandResult(exit_status=0, output=json.dumps(payload))
    return QueryCommandResult(exit_status=0, output=render_text_output(payload))


def collect_query_rows(invocation: QueryInvocation, repo_path: Path) -> list[dict[str, Any]]:
    Repo, QueryReportMode = load_aim_query_support()
    repo = Repo(str(repo_path))

    if invocation.target == "metrics":
        rows: list[dict[str, Any]] = []
        results = repo.query_metrics(
            invocation.expression, report_mode=QueryReportMode.DISABLED
        )
        for run_collection in results.iter_runs():
            for metric in run_collection:
                rows.append(
                    build_row(
                        run_id=metric.run.hash,
                        target="metrics",
                        name=metric.name,
                        context=metric.context.to_dict(),
                    )
                )
        return rows

    rows = []
    results = repo.query_images(invocation.expression, report_mode=QueryReportMode.DISABLED)
    for image in results.iter():
        rows.append(
            build_row(
                run_id=image.run.hash,
                target="images",
                name=image.name,
                context=image.context.to_dict(),
            )
        )
    return rows


def build_row(
    *, run_id: str, target: str, name: str, context: dict[str, Any]
) -> dict[str, Any]:
    summary = f"run {run_id} {target[:-1] if target.endswith('s') else target} {name}"
    return {
        "run_id": run_id,
        "target": target,
        "name": name,
        "context": context,
        "summary": summary,
    }


def render_text_output(payload: dict[str, Any]) -> str:
    lines = [
        f"target: {payload['target']}",
        f"repo: {payload['repo_path']}",
        f"expression: {payload['expression']}",
        f"matches: {payload['count']}",
    ]
    if payload["rows"]:
        for row in payload["rows"]:
            lines.append(
                f"- run={row['run_id']} name={row['name']} context={json.dumps(row['context'], sort_keys=True)}"
            )
    else:
        lines.append("No matching results found.")
    return "\n".join(lines)
