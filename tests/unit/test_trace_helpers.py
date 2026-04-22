from __future__ import annotations

from pathlib import Path

import pytest

from aimx.commands.trace import TraceInvocation, parse_trace_invocation


def test_parse_trace_defaults() -> None:
    inv = parse_trace_invocation(["metric.name=='loss'", "--repo", "data"])
    assert inv.expression == "metric.name=='loss'"
    assert inv.repo_path == Path("data")
    assert inv.mode == "plot"
    assert inv.head is None
    assert inv.tail is None
    assert inv.every is None
    assert inv.width is None
    assert inv.height is None
    assert not inv.no_color


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


def test_parse_trace_rejects_missing_repo() -> None:
    with pytest.raises(ValueError, match="Missing required --repo"):
        parse_trace_invocation(["metric.name=='loss'"])


def test_parse_trace_rejects_unknown_flag() -> None:
    with pytest.raises(ValueError, match="Unsupported trace option"):
        parse_trace_invocation(["expr", "--repo", "data", "--bogus"])


def test_parse_trace_rejects_missing_expression() -> None:
    with pytest.raises(ValueError, match="Usage"):
        parse_trace_invocation([])


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
