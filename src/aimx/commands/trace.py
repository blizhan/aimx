from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from aimx.commands.query import (
    DEFAULT_QUERY_EXPRESSION,
    QueryCommandResult,
    normalize_repo_path,
)

_MODES = {"plot", "table", "csv", "json"}


@dataclass(frozen=True)
class _TracePipeline:
    collect: Callable[[str, Path], list[Any]]
    filter_by_step_range: Callable[[Any, int | None, int | None], Any]
    subsample: Callable[..., Any]
    render: Callable[[list[Any], "TraceInvocation", bool], str]
    no_matches_message: str


@dataclass(frozen=True)
class TraceInvocation:
    target: Literal["metrics", "distribution"]
    expression: str
    repo_path: Path
    mode: Literal["plot", "table", "csv", "json"] = "plot"
    head: int | None = None
    tail: int | None = None
    every: int | None = None
    width: int | None = None
    height: int | None = None
    no_color: bool = False
    step_slice: str | None = None
    selected_step: int | None = None


def parse_trace_invocation(args: list[str]) -> TraceInvocation:
    target: Literal["metrics", "distribution"] = "metrics"
    expression: str | None = None
    rest = args
    if args and args[0] == "distribution":
        target = "distribution"
        if len(args) >= 2 and not args[1].startswith("-"):
            expression = args[1].strip() or DEFAULT_QUERY_EXPRESSION
            rest = args[2:]
        else:
            expression = DEFAULT_QUERY_EXPRESSION
            rest = args[1:]
    elif args and not args[0].startswith("-"):
        expression = args[0].strip() or DEFAULT_QUERY_EXPRESSION
        rest = args[1:]
    else:
        expression = DEFAULT_QUERY_EXPRESSION
        rest = args

    mode: Literal["plot", "table", "csv", "json"] = "plot"
    repo_value = "."
    head: int | None = None
    tail: int | None = None
    every: int | None = None
    width: int | None = None
    height: int | None = None
    no_color = False
    step_slice: str | None = None
    selected_step: int | None = None

    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--table":
            mode = "table"
            index += 1
        elif token == "--csv":
            mode = "csv"
            index += 1
        elif token == "--json":
            mode = "json"
            index += 1
        elif token == "--no-color":
            no_color = True
            index += 1
        elif token == "--repo":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --repo.")
            repo_value = rest[index + 1]
            index += 2
        elif token == "--steps":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --steps.")
            step_slice = rest[index + 1]
            index += 2
        elif token == "--step":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --step.")
            try:
                selected_step = int(rest[index + 1])
            except ValueError:
                raise ValueError(f"--step requires an integer, got: {rest[index + 1]}")
            index += 2
        elif token == "--head":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --head.")
            try:
                head = int(rest[index + 1])
            except ValueError:
                raise ValueError(f"--head requires an integer, got: {rest[index + 1]}")
            index += 2
        elif token == "--tail":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --tail.")
            try:
                tail = int(rest[index + 1])
            except ValueError:
                raise ValueError(f"--tail requires an integer, got: {rest[index + 1]}")
            index += 2
        elif token == "--every":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --every.")
            try:
                every = int(rest[index + 1])
                if every < 1:
                    raise ValueError("--every must be >= 1.")
            except ValueError as exc:
                if "every" in str(exc):
                    raise
                raise ValueError(f"--every requires a positive integer, got: {rest[index + 1]}")
            index += 2
        elif token == "--width":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --width.")
            try:
                width = int(rest[index + 1])
            except ValueError:
                raise ValueError(f"--width requires an integer, got: {rest[index + 1]}")
            index += 2
        elif token == "--height":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --height.")
            try:
                height = int(rest[index + 1])
            except ValueError:
                raise ValueError(f"--height requires an integer, got: {rest[index + 1]}")
            index += 2
        else:
            raise ValueError(f"Unsupported trace option: {token}")

    return TraceInvocation(
        target=target,
        expression=expression,
        repo_path=Path(repo_value),
        mode=mode,
        head=head,
        tail=tail,
        every=every,
        width=width,
        height=height,
        no_color=no_color,
        step_slice=step_slice,
        selected_step=selected_step,
    )


def _render_distribution_trace(
    series_list: list[Any],
    invocation: TraceInvocation,
    no_color: bool,
) -> str:
    from aimx.rendering.trace_views import (
        render_distribution_csv,
        render_distribution_json,
        render_distribution_table,
        render_distribution_visual,
    )

    if invocation.mode == "json":
        return render_distribution_json(series_list)
    if invocation.mode == "csv":
        return render_distribution_csv(series_list)
    if invocation.mode == "table":
        return render_distribution_table(series_list, no_color=no_color)
    return render_distribution_visual(
        series_list,
        selected_step=invocation.selected_step,
        width=invocation.width,
        height=invocation.height,
        no_color=no_color,
    )


def _render_metric_trace(
    series_list: list[Any],
    invocation: TraceInvocation,
    no_color: bool,
) -> str:
    from aimx.rendering.trace_views import (
        render_csv,
        render_plot,
        render_trace_json,
        render_trace_table,
    )

    if invocation.mode == "json":
        return render_trace_json(series_list)
    if invocation.mode == "csv":
        return render_csv(series_list)
    if invocation.mode == "table":
        return render_trace_table(series_list, no_color=no_color)
    return render_plot(
        series_list,
        width=invocation.width,
        height=invocation.height,
    )


def _build_trace_pipeline(
    target: Literal["metrics", "distribution"],
) -> _TracePipeline:
    if target == "distribution":
        from aimx.aim_bridge.metric_stats import (
            collect_distribution_series,
            filter_distribution_by_step_range,
            subsample_distribution,
        )

        return _TracePipeline(
            collect=collect_distribution_series,
            filter_by_step_range=filter_distribution_by_step_range,
            subsample=subsample_distribution,
            render=_render_distribution_trace,
            no_matches_message="No matching distributions found.",
        )

    from aimx.aim_bridge.metric_stats import (
        collect_metric_series,
        filter_by_step_range,
        subsample,
    )

    return _TracePipeline(
        collect=collect_metric_series,
        filter_by_step_range=filter_by_step_range,
        subsample=subsample,
        render=_render_metric_trace,
        no_matches_message="No matching metrics found.",
    )


def _execute_trace_pipeline(
    invocation: TraceInvocation,
    normalized_repo_path: Path,
    pipeline: _TracePipeline,
    *,
    no_color: bool,
) -> QueryCommandResult:
    from aimx.aim_bridge.metric_stats import parse_step_slice

    series_list = pipeline.collect(invocation.expression, normalized_repo_path)
    if not series_list:
        return QueryCommandResult(exit_status=0, output=pipeline.no_matches_message)

    if invocation.step_slice is not None:
        step_start, step_end = parse_step_slice(invocation.step_slice)
        series_list = [
            pipeline.filter_by_step_range(series, step_start, step_end)
            for series in series_list
        ]
        series_list = [series for series in series_list if series.count > 0]

    if not series_list:
        return QueryCommandResult(exit_status=0, output="No data in the requested step range.")

    needs_sample = any(x is not None for x in (invocation.head, invocation.tail, invocation.every))
    if needs_sample:
        series_list = [
            pipeline.subsample(
                series,
                head=invocation.head,
                tail=invocation.tail,
                every=invocation.every,
            )
            for series in series_list
        ]

    return QueryCommandResult(
        exit_status=0,
        output=pipeline.render(series_list, invocation, no_color),
    )


def run_trace_command(args: list[str]) -> QueryCommandResult:
    try:
        invocation = parse_trace_invocation(args)
        normalized_repo_path = normalize_repo_path(invocation.repo_path)
    except ValueError as error:
        return QueryCommandResult(exit_status=2, error_message=str(error))

    is_tty = sys.stdout.isatty()
    effective_no_color = invocation.no_color or not is_tty

    try:
        return _execute_trace_pipeline(
            invocation,
            normalized_repo_path,
            _build_trace_pipeline(invocation.target),
            no_color=effective_no_color,
        )

    except RuntimeError as error:
        return QueryCommandResult(exit_status=2, error_message=str(error))
    except Exception as error:
        return QueryCommandResult(
            exit_status=2, error_message=f"Failed to evaluate trace: {error}"
        )
