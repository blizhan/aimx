from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aimx.commands.query import QueryCommandResult, normalize_repo_path

_MODES = {"plot", "table", "csv", "json"}


@dataclass(frozen=True)
class TraceInvocation:
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


def parse_trace_invocation(args: list[str]) -> TraceInvocation:
    if len(args) < 1:
        raise ValueError(
            "Usage: aimx trace <expression> --repo <path> [--table|--csv|--json]"
            " [--steps start:end] [--head N] [--tail N] [--every K]"
            " [--width W] [--height H] [--no-color]"
        )

    expression = args[0]
    rest = args[1:]

    mode: Literal["plot", "table", "csv", "json"] = "plot"
    repo_value: str | None = None
    head: int | None = None
    tail: int | None = None
    every: int | None = None
    width: int | None = None
    height: int | None = None
    no_color = False
    step_slice: str | None = None

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

    if repo_value is None:
        raise ValueError("Missing required --repo <path> option.")

    return TraceInvocation(
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
        from aimx.aim_bridge.metric_stats import (
            collect_metric_series,
            filter_by_step_range,
            parse_step_slice,
            subsample,
        )
        from aimx.rendering.trace_views import (
            render_csv,
            render_plot,
            render_trace_json,
            render_trace_table,
        )

        series_list = collect_metric_series(invocation.expression, normalized_repo_path)

        if not series_list:
            return QueryCommandResult(exit_status=0, output="No matching metrics found.")

        # Step range filter is a hard constraint applied before density subsampling
        if invocation.step_slice is not None:
            step_start, step_end = parse_step_slice(invocation.step_slice)
            series_list = [filter_by_step_range(s, step_start, step_end) for s in series_list]
            # Drop empty series so they don't clutter plots
            series_list = [s for s in series_list if s.count > 0]

        if not series_list:
            return QueryCommandResult(exit_status=0, output="No data in the requested step range.")

        # Density subsampling for visualisation
        needs_sample = any(
            x is not None for x in (invocation.head, invocation.tail, invocation.every)
        )
        if needs_sample:
            series_list = [
                subsample(s, head=invocation.head, tail=invocation.tail, every=invocation.every)
                for s in series_list
            ]

        if invocation.mode == "json":
            output = render_trace_json(series_list)
        elif invocation.mode == "csv":
            output = render_csv(series_list)
        elif invocation.mode == "table":
            output = render_trace_table(series_list, no_color=effective_no_color)
        else:
            output = render_plot(
                series_list,
                width=invocation.width,
                height=invocation.height,
            )

        return QueryCommandResult(exit_status=0, output=output)

    except RuntimeError as error:
        return QueryCommandResult(exit_status=2, error_message=str(error))
    except Exception as error:
        return QueryCommandResult(
            exit_status=2, error_message=f"Failed to evaluate trace: {error}"
        )
