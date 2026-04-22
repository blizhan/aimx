from __future__ import annotations

import sys
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
    plain: bool = False
    no_color: bool = False
    verbose: bool = False
    step_slice: str | None = None

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


def normalize_repo_path(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"Repository path does not exist: {path}")
    if path.name == ".aim":
        return path.parent
    return path


def parse_query_invocation(args: list[str]) -> QueryInvocation:
    if len(args) < 2:
        raise ValueError(
            "Usage: aimx query <metrics|images> <expression> --repo <path> "
            "[--json] [--oneline] [--no-color] [--verbose] [--steps start:end]"
        )

    target = args[0]
    expression = args[1]
    rest = args[2:]

    output_json = False
    plain = False
    no_color = False
    verbose = False
    step_slice: str | None = None
    repo_value: str | None = None

    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--json":
            output_json = True
            index += 1
        elif token in ("--oneline", "--plain"):
            plain = True
            index += 1
        elif token == "--no-color":
            no_color = True
            index += 1
        elif token == "--verbose":
            verbose = True
            index += 1
        elif token == "--steps":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --steps.")
            step_slice = rest[index + 1]
            index += 2
        elif token == "--repo":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --repo.")
            repo_value = rest[index + 1]
            index += 2
        else:
            raise ValueError(f"Unsupported query option: {token}")

    if repo_value is None:
        raise ValueError("Missing required --repo <path> option.")

    return QueryInvocation(
        target=target,
        expression=expression,
        repo_path=Path(repo_value),
        output_json=output_json,
        plain=plain,
        no_color=no_color,
        verbose=verbose,
        step_slice=step_slice,
    )


def run_query_command(args: list[str]) -> QueryCommandResult:
    try:
        invocation = parse_query_invocation(args)
        normalized_repo_path = normalize_repo_path(invocation.repo_path)
    except ValueError as error:
        return QueryCommandResult(exit_status=2, error_message=str(error))

    is_tty = sys.stdout.isatty()
    effective_no_color = invocation.no_color or not is_tty

    header_info: dict[str, Any] = {
        "target": invocation.target,
        "repo": str(normalized_repo_path),
        "expression": invocation.expression,
        "verbose": invocation.verbose,
    }

    try:
        if invocation.target == "metrics":
            return _run_metrics_query(invocation, normalized_repo_path, header_info, effective_no_color)
        return _run_images_query(invocation, normalized_repo_path, header_info, effective_no_color)
    except RuntimeError as error:
        return QueryCommandResult(exit_status=2, error_message=str(error))
    except Exception as error:
        return QueryCommandResult(
            exit_status=2, error_message=f"Failed to evaluate query: {error}"
        )


def _run_metrics_query(
    invocation: QueryInvocation,
    repo_path: Path,
    header_info: dict[str, Any],
    no_color: bool,
) -> QueryCommandResult:
    from aimx.aim_bridge.metric_stats import (
        collect_metric_series,
        filter_by_step_range,
        group_by_run,
        parse_step_slice,
    )
    from aimx.rendering.query_views import (
        render_json,
        render_oneline,
        render_rich_table,
    )

    series_list = collect_metric_series(invocation.expression, repo_path)

    if invocation.step_slice is not None:
        step_start, step_end = parse_step_slice(invocation.step_slice)
        series_list = [filter_by_step_range(s, step_start, step_end) for s in series_list]

    groups = group_by_run(series_list)

    if invocation.output_json:
        return QueryCommandResult(exit_status=0, output=render_json(groups, header_info))
    if invocation.plain:
        return QueryCommandResult(exit_status=0, output=render_oneline(groups, header_info))
    return QueryCommandResult(
        exit_status=0,
        output=render_rich_table(groups, header_info, no_color=no_color),
    )


def _run_images_query(
    invocation: QueryInvocation,
    repo_path: Path,
    header_info: dict[str, Any],
    no_color: bool,
) -> QueryCommandResult:
    from aimx.aim_bridge.metric_stats import collect_image_series
    from aimx.rendering.query_views import (
        render_image_json,
        render_image_oneline,
        render_image_rich_table,
    )

    image_rows = collect_image_series(invocation.expression, repo_path)

    if invocation.output_json:
        return QueryCommandResult(exit_status=0, output=render_image_json(image_rows, header_info))
    if invocation.plain:
        return QueryCommandResult(exit_status=0, output=render_image_oneline(image_rows, header_info))
    return QueryCommandResult(
        exit_status=0,
        output=render_image_rich_table(image_rows, header_info, no_color=no_color),
    )
