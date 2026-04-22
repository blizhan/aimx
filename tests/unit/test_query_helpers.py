from __future__ import annotations

from pathlib import Path

import pytest

from aimx.commands.query import (
    QueryInvocation,
    normalize_repo_path,
    parse_query_invocation,
)


def test_normalize_repo_path_keeps_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    normalized = normalize_repo_path(repo_root)

    assert normalized == repo_root


def test_normalize_repo_path_converts_dot_aim_directory_to_parent(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dot_aim = repo_root / ".aim"
    dot_aim.mkdir(parents=True)

    normalized = normalize_repo_path(dot_aim)

    assert normalized == repo_root


def test_normalize_repo_path_rejects_missing_path() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        normalize_repo_path(Path("missing-repo"))


def test_query_invocation_rejects_unsupported_target() -> None:
    with pytest.raises(ValueError, match="Unsupported query target"):
        QueryInvocation(
            target="artifacts",
            expression="metric.name == 'loss'",
            repo_path=Path("data"),
        )


def test_parse_query_invocation_defaults() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data"])
    assert inv.target == "metrics"
    assert inv.expression == "metric.name == 'loss'"
    assert inv.repo_path == Path("data")
    assert not inv.output_json
    assert not inv.plain
    assert not inv.no_color
    assert not inv.verbose


def test_parse_query_invocation_json_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--json"])
    assert inv.output_json is True


def test_parse_query_invocation_oneline_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--oneline"])
    assert inv.plain is True


def test_parse_query_invocation_plain_flag_alias() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--plain"])
    assert inv.plain is True


def test_parse_query_invocation_no_color_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--no-color"])
    assert inv.no_color is True


def test_parse_query_invocation_verbose_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--verbose"])
    assert inv.verbose is True


def test_parse_query_invocation_rejects_missing_repo() -> None:
    with pytest.raises(ValueError, match="Missing required --repo"):
        parse_query_invocation(["metrics", "metric.name == 'loss'"])


def test_parse_query_invocation_rejects_unknown_flag() -> None:
    with pytest.raises(ValueError, match="Unsupported query option"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--bogus"])


def test_parse_query_invocation_steps_closed_range() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data", "--steps", "100:500"])
    assert inv.step_slice == "100:500"


def test_parse_query_invocation_steps_open_end() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data", "--steps", "100:"])
    assert inv.step_slice == "100:"


def test_parse_query_invocation_steps_open_start() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data", "--steps", ":500"])
    assert inv.step_slice == ":500"


def test_parse_query_invocation_steps_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --steps"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--steps"])


def test_parse_query_invocation_steps_defaults_to_none() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data"])
    assert inv.step_slice is None
