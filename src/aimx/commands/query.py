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
    epoch_slice: str | None = None
    head: int | None = None
    tail: int | None = None
    every: int | None = None
    max_images: int = 6

    def __post_init__(self) -> None:
        if self.target not in SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported query target: {self.target}. Supported targets: metrics, images."
            )
        if not self.expression.strip():
            raise ValueError("Query expression must not be empty.")
        if self.max_images < 0:
            raise ValueError(
                f"Invalid --max-images value: {self.max_images!r}. Must be a non-negative integer."
            )
        if self.step_slice is not None and self.epoch_slice is not None:
            raise ValueError("--steps and --epochs are mutually exclusive.")
        if self.every is not None and self.every < 1:
            raise ValueError(f"--every must be >= 1, got: {self.every!r}.")


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


def _parse_positive_int(flag: str, raw: str) -> int:
    """Parse *raw* as a positive integer for *flag*; raise ``ValueError`` with a clear message."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{flag} requires a positive integer, got: {raw!r}")
    if value < 1:
        raise ValueError(f"{flag} must be >= 1, got: {raw!r}")
    return value


def _parse_non_negative_int(flag: str, raw: str) -> int:
    """Parse *raw* as a non-negative integer for *flag*; raise ``ValueError`` on bad input."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{flag} requires an integer, got: {raw!r}")
    if value < 0:
        raise ValueError(f"{flag} requires a non-negative integer, got: {raw!r}")
    return value


def parse_query_invocation(args: list[str]) -> QueryInvocation:
    if len(args) < 2:
        raise ValueError(
            "Usage: aimx query <metrics|images> <expression> [--repo <path>] "
            "[--json] [--oneline] [--no-color] [--verbose] "
            "[--steps start:end | --epochs start:end] "
            "[--head N] [--tail N] [--every K] "
            "[--max-images N]"
        )

    target = args[0]
    expression = args[1]
    rest = args[2:]

    output_json = False
    plain = False
    no_color = False
    verbose = False
    step_slice: str | None = None
    epoch_slice: str | None = None
    head: int | None = None
    tail: int | None = None
    every: int | None = None
    repo_value = "."
    max_images: int = 6

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
        elif token == "--epochs":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --epochs.")
            epoch_slice = rest[index + 1]
            index += 2
        elif token == "--head":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --head.")
            head = _parse_non_negative_int("--head", rest[index + 1])
            index += 2
        elif token == "--tail":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --tail.")
            tail = _parse_non_negative_int("--tail", rest[index + 1])
            index += 2
        elif token == "--every":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --every.")
            every = _parse_positive_int("--every", rest[index + 1])
            index += 2
        elif token == "--repo":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --repo.")
            repo_value = rest[index + 1]
            index += 2
        elif token == "--max-images":
            if index + 1 >= len(rest):
                raise ValueError("Missing value for --max-images.")
            raw = rest[index + 1]
            try:
                max_images = int(raw)
            except ValueError:
                raise ValueError(f"Invalid --max-images value: {raw!r}. Must be a non-negative integer.")
            if max_images < 0:
                raise ValueError(f"Invalid --max-images value: {raw!r}. Must be a non-negative integer.")
            index += 2
        else:
            raise ValueError(f"Unsupported query option: {token}")

    return QueryInvocation(
        target=target,
        expression=expression,
        repo_path=Path(repo_value),
        output_json=output_json,
        plain=plain,
        no_color=no_color,
        verbose=verbose,
        step_slice=step_slice,
        epoch_slice=epoch_slice,
        head=head,
        tail=tail,
        every=every,
        max_images=max_images,
    )


def _sort_image_value(value: Any) -> tuple[int, Any]:
    """Normalize values so image rows sort numerically when possible."""
    if value is None:
        return (3, "")
    if isinstance(value, bool):
        return (2, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    if isinstance(value, str):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)
    return (2, str(value))


def _image_context_sort_key(context: dict[str, Any]) -> tuple[tuple[str, tuple[int, Any]], ...]:
    """Build a deterministic context sort key, excluding explicit epoch/step fields."""
    return tuple(
        (key, _sort_image_value(value))
        for key, value in sorted(context.items())
        if key not in {"epoch", "step"}
    )


def _sort_image_rows(image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep run-group order stable, while sorting rows within a run by epoch/step."""
    run_order: dict[str, int] = {}
    for row in image_rows:
        run_hash = row["run"].hash
        if run_hash not in run_order:
            run_order[run_hash] = len(run_order)

    return sorted(
        image_rows,
        key=lambda row: (
            run_order[row["run"].hash],
            _sort_image_value(row.get("_sort_epoch", row["context"].get("epoch"))),
            _sort_image_value(row.get("_sort_step")),
            row["name"],
            _image_context_sort_key(row["context"]),
        ),
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
        filter_by_epoch_range,
        filter_by_step_range,
        group_by_run,
        parse_epoch_slice,
        parse_step_slice,
        subsample,
    )
    from aimx.rendering.query_views import (
        render_json,
        render_oneline,
        render_rich_table,
    )

    series_list = collect_metric_series(invocation.expression, repo_path)

    # Range filter (--steps or --epochs, mutually exclusive)
    if invocation.step_slice is not None:
        step_start, step_end = parse_step_slice(invocation.step_slice)
        series_list = [filter_by_step_range(s, step_start, step_end) for s in series_list]
        series_list = [s for s in series_list if s.count > 0]
    elif invocation.epoch_slice is not None:
        epoch_start, epoch_end = parse_epoch_slice(invocation.epoch_slice)
        series_list = [filter_by_epoch_range(s, epoch_start, epoch_end) for s in series_list]
        series_list = [s for s in series_list if s.count > 0]

    # Density subsampling (--head / --tail / --every)
    needs_sample = any(x is not None for x in (invocation.head, invocation.tail, invocation.every))
    if needs_sample:
        series_list = [
            subsample(s, head=invocation.head, tail=invocation.tail, every=invocation.every)
            for s in series_list
        ]

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
    from aimx.aim_bridge.metric_stats import (
        collect_image_series,
        filter_image_rows_by_epoch_range,
        filter_image_rows_by_step_range,
        parse_epoch_slice,
        parse_step_slice,
        subsample_image_rows,
    )
    from aimx.rendering.query_views import (
        render_image_json,
        render_image_oneline,
        render_image_rich_table,
    )

    image_rows = collect_image_series(invocation.expression, repo_path)
    image_rows = _sort_image_rows(image_rows)

    # Range filter (--steps or --epochs, mutually exclusive)
    if invocation.step_slice is not None:
        step_start, step_end = parse_step_slice(invocation.step_slice)
        image_rows = filter_image_rows_by_step_range(image_rows, step_start, step_end)
    elif invocation.epoch_slice is not None:
        epoch_start, epoch_end = parse_epoch_slice(invocation.epoch_slice)
        image_rows = filter_image_rows_by_epoch_range(image_rows, epoch_start, epoch_end)

    # Global subsampling across the sorted result list (--head / --tail / --every)
    needs_sample = any(x is not None for x in (invocation.head, invocation.tail, invocation.every))
    if needs_sample:
        image_rows = subsample_image_rows(
            image_rows, head=invocation.head, tail=invocation.tail, every=invocation.every
        )

    # Machine-readable paths — byte-identical to pre-003 output; MUST NOT
    # touch _image_accessor or produce any graphics escape sequences.
    if invocation.output_json:
        return QueryCommandResult(exit_status=0, output=render_image_json(image_rows, header_info))
    if invocation.plain:
        return QueryCommandResult(exit_status=0, output=render_image_oneline(image_rows, header_info))

    # TTY inline-image extension (US1/US3): only when stdout is a TTY and
    # the rendering path is "rich" (not --json / --plain).
    from aimx.rendering.image_render import detect_capability, plan_render, render_inline  # noqa: PLC0415

    capability = detect_capability()
    if capability.protocol != "disabled":
        plan = plan_render(image_rows, capability, max_images=invocation.max_images)
        inline = render_inline(plan)
        if inline:
            repo = header_info.get("repo", "")
            expr = header_info.get("expression", "")
            total = len(image_rows)
            shown = len(plan.rendered_rows)
            compact_header = (
                f"Repo: {repo}  ·  {total} matches  ·  images where {expr}"
            )
            if shown < total:
                compact_header += f"  ·  previewing {shown}"
            compact_header += "\n"

            return QueryCommandResult(exit_status=0, output=compact_header + inline)

    summary = render_image_rich_table(image_rows, header_info, no_color=no_color)
    return QueryCommandResult(exit_status=0, output=summary)
