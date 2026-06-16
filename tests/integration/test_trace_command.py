from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from aimx.__main__ import main
from aimx.aim_bridge.metric_stats import DistributionSeries, collect_distribution_series
from aimx.commands.query import DEFAULT_QUERY_EXPRESSION, normalize_repo_path


def _require_distribution_series(sample_repo_root: Path) -> list[DistributionSeries]:
    """Return distribution series for ``data/``-style fixtures, or skip when absent."""
    normalized = normalize_repo_path(sample_repo_root)
    series_list = collect_distribution_series("distribution.name != ''", normalized)
    if not series_list:
        pytest.skip("sample Aim repository has no distribution sequences for integration tests")
    return series_list


def _first_non_empty_series(series_list: list[DistributionSeries]) -> DistributionSeries:
    for series in series_list:
        if series.count > 0:
            return series
    pytest.skip("sample Aim repository has no non-empty distribution series for integration tests")


def _sorted_unique_steps(series: DistributionSeries) -> list[int]:
    return sorted({point.step for point in series.points})


def _pick_exact_and_nearest_request(steps: list[int]) -> tuple[int, int]:
    """Return ``(requested_step, expected_resolved_step)`` for nearest-step coverage."""
    if len(steps) < 2:
        pytest.skip("sample distribution series needs at least two steps for step selection tests")

    lower, higher = steps[0], steps[1]
    if higher - lower <= 1:
        pytest.skip("sample distribution steps are too dense to construct a nearest-step gap test")

    requested = (lower + higher) // 2
    if abs(requested - lower) == abs(requested - higher):
        pytest.skip("sample distribution steps do not produce a unique nearest-step candidate")

    if abs(requested - lower) < abs(requested - higher):
        expected = lower
    else:
        expected = higher

    return requested, expected


def test_trace_plot_produces_output_containing_metric_name(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert captured.out.strip(), "Expected non-empty plotext output"


def test_trace_defaults_repo_to_current_directory(
    capfd, monkeypatch, sample_repo_root
) -> None:
    monkeypatch.chdir(sample_repo_root)

    exit_code = main(["trace", "metric.name == 'loss'", "--json"])

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload


def test_trace_table_mode_contains_step_and_value_columns(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--table"]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "STEP" in captured.out
    assert "VALUE" in captured.out


def test_trace_json_mode_returns_full_value_arrays(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--json"]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert isinstance(payload, list)
    for series in payload:
        assert series["metric"] == "loss"
        assert len(series["steps"]) == len(series["values"])
        assert len(series["values"]) > 0


def test_omitted_trace_query_matches_explicit_default_expression(
    capfd, sample_repo_root
) -> None:
    explicit_exit = main(
        ["trace", DEFAULT_QUERY_EXPRESSION, "--repo", str(sample_repo_root), "--json"]
    )
    explicit_payload = json.loads(capfd.readouterr().out)

    omitted_exit = main(["trace", "--repo", str(sample_repo_root), "--json"])
    omitted_payload = json.loads(capfd.readouterr().out)

    assert explicit_exit == 0
    assert omitted_exit == 0
    assert omitted_payload == explicit_payload


def test_trace_csv_mode_contains_correct_fields(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--csv"]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    reader = csv.DictReader(io.StringIO(captured.out))
    rows = list(reader)
    assert rows, "Expected at least one CSV data row"
    for row in rows:
        assert row["metric"] == "loss"
        assert row["step"].isdigit()


def test_trace_head_limits_to_n_points_per_series(capfd, sample_repo_root) -> None:
    exit_code = main(
        [
            "trace",
            "metric.name == 'loss'",
            "--repo",
            str(sample_repo_root),
            "--json",
            "--head",
            "5",
        ]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    for series in payload:
        assert len(series["values"]) <= 5


def test_trace_no_matching_expression_exits_cleanly(capfd, sample_repo_root) -> None:
    exit_code = main(
        [
            "trace",
            "metric.name == 'nonexistent_metric_xyz'",
            "--repo",
            str(sample_repo_root),
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "No matching" in captured.out


def test_trace_distribution_default_visual_contains_list_histogram_and_heatmap(
    capfd, sample_repo_root: Path
) -> None:
    series_list = _require_distribution_series(sample_repo_root)
    first_point = _first_non_empty_series(series_list).points[0]
    repo_root = sample_repo_root

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--head",
            "3",
            "--no-color",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "Distributions" in captured.out
    assert "▌ " in captured.out
    if _first_non_empty_series(series_list).context:
        assert "kind=" in captured.out
    assert "Histogram" in captured.out
    assert f"Step {first_point.step}" in captured.out
    assert "Heatmap (steps x bins)" in captured.out
    assert "Scale: low -> high" in captured.out


def test_trace_distribution_step_selects_requested_step(capfd, sample_repo_root: Path) -> None:
    series = _first_non_empty_series(_require_distribution_series(sample_repo_root))
    steps = _sorted_unique_steps(series)
    if len(steps) < 2:
        pytest.skip("sample distribution series needs at least two steps for exact step selection")
    exact_step = steps[1]
    repo_root = sample_repo_root

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--step",
            str(exact_step),
            "--no-color",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert f"Step {exact_step}" in captured.out


def test_trace_distribution_step_uses_nearest_tracked_step(capfd, sample_repo_root: Path) -> None:
    series = _first_non_empty_series(_require_distribution_series(sample_repo_root))
    steps = _sorted_unique_steps(series)
    requested, expected = _pick_exact_and_nearest_request(steps)
    repo_root = sample_repo_root

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--step",
            str(requested),
            "--no-color",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert (
        f"Requested step {requested}; showing nearest tracked step {expected}."
        in captured.out
    )


def test_trace_distribution_table_mode_preserves_tensor_output_with_step(
    capfd, sample_repo_root: Path
) -> None:
    series = _first_non_empty_series(_require_distribution_series(sample_repo_root))
    steps = _sorted_unique_steps(series)
    if len(steps) < 2:
        pytest.skip("sample distribution series needs at least two steps for table step carryover checks")
    table_step = steps[1]
    repo_root = sample_repo_root

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--table",
            "--step",
            str(table_step),
            "--head",
            "2",
            "--no-color",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "TENSOR" in captured.out
    assert "Histogram" not in captured.out
    assert "Heatmap" not in captured.out


def test_trace_distribution_json_mode_preserves_series_payload_with_step(
    capfd, sample_repo_root: Path
) -> None:
    series = _first_non_empty_series(_require_distribution_series(sample_repo_root))
    steps = _sorted_unique_steps(series)
    if len(steps) < 2:
        pytest.skip("sample distribution series needs at least two points for JSON head checks")
    json_step = steps[1]
    repo_root = sample_repo_root

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--json",
            "--step",
            str(json_step),
            "--head",
            "2",
        ]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload
    assert payload[0]["count"] == 2
    assert "points" in payload[0]


def test_omitted_distribution_trace_matches_explicit_default_expression(
    capfd, sample_repo_root: Path
) -> None:
    _require_distribution_series(sample_repo_root)

    explicit_exit = main(
        [
            "trace",
            "distribution",
            DEFAULT_QUERY_EXPRESSION,
            "--repo",
            str(sample_repo_root),
            "--json",
            "--head",
            "2",
        ]
    )
    explicit_payload = json.loads(capfd.readouterr().out)

    omitted_exit = main(
        ["trace", "distribution", "--repo", str(sample_repo_root), "--json", "--head", "2"]
    )
    omitted_payload = json.loads(capfd.readouterr().out)

    assert explicit_exit == 0
    assert omitted_exit == 0
    assert omitted_payload == explicit_payload


def test_trace_distribution_csv_mode_preserves_rows_with_step(
    capfd, sample_repo_root: Path
) -> None:
    series = _first_non_empty_series(_require_distribution_series(sample_repo_root))
    if series.count < 1:
        pytest.skip("sample distribution series needs points for CSV checks")
    csv_step = series.points[0].step
    repo_root = sample_repo_root

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--csv",
            "--step",
            str(csv_step),
            "--head",
            "1",
        ]
    )

    captured = capfd.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert exit_code == 0
    assert rows
    assert {"run_hash", "distribution", "step", "bin_edges", "weights"}.issubset(rows[0])
