from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from aimx.commands.trace import (
    TraceInvocation,
    _TracePipeline,
    _execute_trace_pipeline,
    parse_trace_invocation,
)
from aimx.commands.query import DEFAULT_QUERY_EXPRESSION


@dataclass
class _FakeSeries:
    points: list[int]

    @property
    def count(self) -> int:
        return len(self.points)


def test_parse_trace_defaults() -> None:
    inv = parse_trace_invocation(["metric.name=='loss'"])
    assert inv.target == "metrics"
    assert inv.expression == "metric.name=='loss'"
    assert inv.repo_path == Path(".")
    assert inv.mode == "plot"
    assert inv.head is None
    assert inv.tail is None
    assert inv.every is None
    assert inv.width is None
    assert inv.height is None
    assert not inv.no_color


def test_parse_trace_uses_default_expression_when_omitted() -> None:
    inv = parse_trace_invocation([])

    assert inv.target == "metrics"
    assert inv.expression == DEFAULT_QUERY_EXPRESSION


def test_parse_trace_treats_option_first_as_omitted_expression() -> None:
    inv = parse_trace_invocation(["--repo", "data", "--json", "--head", "5"])

    assert inv.expression == DEFAULT_QUERY_EXPRESSION
    assert inv.repo_path == Path("data")
    assert inv.mode == "json"
    assert inv.head == 5


def test_parse_trace_distribution_uses_default_expression_when_omitted() -> None:
    inv = parse_trace_invocation(["distribution"])

    assert inv.target == "distribution"
    assert inv.expression == DEFAULT_QUERY_EXPRESSION


def test_parse_trace_distribution_treats_option_first_as_omitted_expression() -> None:
    inv = parse_trace_invocation(["distribution", "--repo", "data", "--table"])

    assert inv.target == "distribution"
    assert inv.expression == DEFAULT_QUERY_EXPRESSION
    assert inv.repo_path == Path("data")
    assert inv.mode == "table"


@pytest.mark.parametrize(
    ("args", "attr", "expected"),
    [
        (["--repo", "data"], "repo_path", Path("data")),
        (["--json"], "mode", "json"),
        (["--table"], "mode", "table"),
        (["--csv"], "mode", "csv"),
        (["--steps", "1:5"], "step_slice", "1:5"),
        (["--head", "2"], "head", 2),
        (["--tail", "3"], "tail", 3),
        (["--every", "4"], "every", 4),
        (["--width", "100"], "width", 100),
        (["--height", "30"], "height", 30),
        (["--no-color"], "no_color", True),
    ],
)
def test_parse_trace_option_first_flags_preserve_option_parsing(
    args: list[str],
    attr: str,
    expected: object,
) -> None:
    inv = parse_trace_invocation(args)

    assert inv.expression == DEFAULT_QUERY_EXPRESSION
    assert getattr(inv, attr) == expected


def test_parse_trace_distribution_option_first_step_preserves_option_parsing() -> None:
    inv = parse_trace_invocation(["distribution", "--step", "123"])

    assert inv.target == "distribution"
    assert inv.expression == DEFAULT_QUERY_EXPRESSION
    assert inv.selected_step == 123


def test_parse_trace_uses_default_expression_when_blank() -> None:
    inv = parse_trace_invocation(["   ", "--repo", "data"])

    assert inv.expression == DEFAULT_QUERY_EXPRESSION
    assert inv.repo_path == Path("data")


def test_parse_trace_preserves_explicit_expression_precedence() -> None:
    inv = parse_trace_invocation(["metric.name=='loss'", "--repo", "data"])

    assert inv.expression == "metric.name=='loss'"
    assert inv.repo_path == Path("data")


def test_parse_trace_table_mode() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--table"])
    assert inv.mode == "table"


def test_parse_trace_csv_mode() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--csv"])
    assert inv.mode == "csv"


def test_parse_trace_json_mode() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--json"])
    assert inv.mode == "json"


def test_parse_trace_head_flag() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--head", "50"])
    assert inv.head == 50


def test_parse_trace_tail_flag() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--tail", "10"])
    assert inv.tail == 10


def test_parse_trace_every_flag() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--every", "5"])
    assert inv.every == 5


def test_parse_trace_width_and_height() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--width", "100", "--height", "30"])
    assert inv.width == 100
    assert inv.height == 30


def test_parse_trace_no_color_flag() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--no-color"])
    assert inv.no_color is True


def test_parse_trace_explicit_repo_overrides_default() -> None:
    inv = parse_trace_invocation(["metric.name=='loss'", "--repo", "data"])
    assert inv.repo_path == Path("data")


def test_parse_trace_distribution_target() -> None:
    inv = parse_trace_invocation(["distribution", "distribution.name=='weights'", "--repo", "data"])
    assert inv.target == "distribution"
    assert inv.expression == "distribution.name=='weights'"
    assert inv.repo_path == Path("data")


def test_parse_trace_rejects_unknown_flag() -> None:
    with pytest.raises(ValueError, match="Unsupported trace option"):
        parse_trace_invocation(["expr", "--repo", "data", "--bogus"])


def test_parse_trace_rejects_non_integer_head() -> None:
    with pytest.raises(ValueError, match="--head requires an integer"):
        parse_trace_invocation(["expr", "--repo", "data", "--head", "abc"])


def test_parse_trace_rejects_every_less_than_one() -> None:
    with pytest.raises(ValueError, match="--every must be >= 1"):
        parse_trace_invocation(["expr", "--repo", "data", "--every", "0"])


def test_parse_trace_steps_closed_range() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--steps", "50:100"])
    assert inv.step_slice == "50:100"


def test_parse_trace_steps_open_end() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--steps", "50:"])
    assert inv.step_slice == "50:"


def test_parse_trace_steps_open_start() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data", "--steps", ":100"])
    assert inv.step_slice == ":100"


def test_parse_trace_steps_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --steps"):
        parse_trace_invocation(["expr", "--repo", "data", "--steps"])


def test_parse_trace_steps_defaults_to_none() -> None:
    inv = parse_trace_invocation(["expr", "--repo", "data"])
    assert inv.step_slice is None


def test_parse_trace_step_flag() -> None:
    inv = parse_trace_invocation(["distribution", "expr", "--repo", "data", "--step", "12300"])
    assert inv.selected_step == 12300


def test_parse_trace_step_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --step"):
        parse_trace_invocation(["distribution", "expr", "--repo", "data", "--step"])


def test_parse_trace_step_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="--step requires an integer"):
        parse_trace_invocation(["distribution", "expr", "--repo", "data", "--step", "abc"])


def test_execute_trace_pipeline_filters_before_sampling_and_renders_remaining() -> None:
    calls: list[tuple[object, ...]] = []
    invocation = TraceInvocation(
        target="metrics",
        expression="expr",
        repo_path=Path("."),
        mode="json",
        head=2,
        every=2,
        step_slice="2:4",
    )

    def collect(expression: str, repo_path: Path) -> list[_FakeSeries]:
        calls.append(("collect", expression, repo_path))
        return [_FakeSeries([1, 2, 3, 4, 5]), _FakeSeries([10])]

    def filter_by_step_range(
        series: _FakeSeries,
        start: int | None,
        end: int | None,
    ) -> _FakeSeries:
        calls.append(("filter", tuple(series.points), start, end))
        return _FakeSeries(
            [
                point
                for point in series.points
                if (start is None or point >= start) and (end is None or point <= end)
            ]
        )

    def subsample(
        series: _FakeSeries,
        *,
        head: int | None,
        tail: int | None,
        every: int | None,
    ) -> _FakeSeries:
        calls.append(("subsample", tuple(series.points), head, tail, every))
        points = series.points
        if head is not None:
            points = points[:head]
        if tail is not None:
            points = points[-tail:]
        if every is not None and every > 1:
            points = points[::every]
        return _FakeSeries(points)

    def render(
        series_list: list[_FakeSeries],
        render_invocation: TraceInvocation,
        no_color: bool,
    ) -> str:
        calls.append(
            (
                "render",
                tuple(tuple(series.points) for series in series_list),
                render_invocation.mode,
                no_color,
            )
        )
        return "rendered"

    result = _execute_trace_pipeline(
        invocation,
        Path("repo"),
        _TracePipeline(
            collect=collect,
            filter_by_step_range=filter_by_step_range,
            subsample=subsample,
            render=render,
            no_matches_message="No matching fake series found.",
        ),
        no_color=True,
    )

    assert result.exit_status == 0
    assert result.output == "rendered"
    assert calls == [
        ("collect", "expr", Path("repo")),
        ("filter", (1, 2, 3, 4, 5), 2, 4),
        ("filter", (10,), 2, 4),
        ("subsample", (2, 3, 4), 2, None, 2),
        ("render", ((2,),), "json", True),
    ]


def test_execute_trace_pipeline_returns_step_range_message_when_filter_empties() -> None:
    invocation = TraceInvocation(
        target="metrics",
        expression="expr",
        repo_path=Path("."),
        step_slice="2:4",
    )

    def collect(expression: str, repo_path: Path) -> list[_FakeSeries]:
        return [_FakeSeries([1])]

    def filter_by_step_range(
        series: _FakeSeries,
        start: int | None,
        end: int | None,
    ) -> _FakeSeries:
        return _FakeSeries([])

    def subsample(
        series: _FakeSeries,
        *,
        head: int | None,
        tail: int | None,
        every: int | None,
    ) -> _FakeSeries:
        raise AssertionError("subsample should not run after the filter empties all series")

    def render(
        series_list: list[_FakeSeries],
        render_invocation: TraceInvocation,
        no_color: bool,
    ) -> str:
        raise AssertionError("render should not run after the filter empties all series")

    result = _execute_trace_pipeline(
        invocation,
        Path("repo"),
        _TracePipeline(
            collect=collect,
            filter_by_step_range=filter_by_step_range,
            subsample=subsample,
            render=render,
            no_matches_message="No matching fake series found.",
        ),
        no_color=False,
    )

    assert result.exit_status == 0
    assert result.output == "No data in the requested step range."
