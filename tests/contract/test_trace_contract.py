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
    normalized = normalize_repo_path(sample_repo_root)
    series_list = collect_distribution_series("distribution.name != ''", normalized)
    if not series_list:
        pytest.skip("sample Aim repository has no distribution sequences for contract tests")
    return series_list


def _first_non_empty_series(series_list: list[DistributionSeries]) -> DistributionSeries:
    for series in series_list:
        if series.count > 0:
            return series
    pytest.skip("sample Aim repository has no non-empty distribution series for contract tests")


def test_trace_plot_contract_produces_non_empty_output(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert captured.out.strip(), "Expected plotext chart output"


def test_trace_table_contract_contains_step_and_value_columns(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--table"]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "STEP" in captured.out
    assert "VALUE" in captured.out


def test_trace_json_contract_has_steps_and_values_arrays(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--json"]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert isinstance(payload, list)
    assert payload, "Expected at least one series"
    first = payload[0]
    assert first["metric"] == "loss"
    assert "steps" in first
    assert "values" in first
    assert isinstance(first["steps"], list)
    assert isinstance(first["values"], list)


def test_trace_json_contract_omitted_expression_matches_explicit_default(
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


def test_trace_plot_contract_omitted_expression_produces_output(
    capfd, sample_repo_root
) -> None:
    exit_code = main(["trace", "--repo", str(sample_repo_root), "--head", "5"])

    captured = capfd.readouterr()
    assert exit_code == 0
    assert captured.out.strip()


def test_trace_csv_contract_has_correct_headers(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--csv"]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    reader = csv.DictReader(io.StringIO(captured.out))
    assert reader.fieldnames is not None
    for field in ("run_hash", "metric", "step", "value"):
        assert field in reader.fieldnames


def test_trace_invalid_repo_reports_error(capfd) -> None:
    exit_code = main(["trace", "metric.name == 'loss'", "--repo", "missing-repo"])

    captured = capfd.readouterr()
    assert exit_code == 2
    assert "Repository path does not exist" in captured.err


def test_trace_invalid_expression_reports_error(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name ==", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 2
    assert "Failed to evaluate trace" in captured.err


def test_trace_distribution_default_visual_contract(capfd, sample_repo_root: Path) -> None:
    repo_root = sample_repo_root
    _require_distribution_series(repo_root)

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--head",
            "2",
            "--no-color",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "Distributions" in captured.out
    assert "Histogram" in captured.out
    assert "Heatmap (steps x bins)" in captured.out
    assert not captured.err


def test_trace_distribution_default_visual_omitted_expression_matches_explicit_default(
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
            "--head",
            "2",
            "--no-color",
        ]
    )
    explicit_output = capfd.readouterr().out

    omitted_exit = main(
        [
            "trace",
            "distribution",
            "--repo",
            str(sample_repo_root),
            "--head",
            "2",
            "--no-color",
        ]
    )
    omitted_output = capfd.readouterr().out

    assert explicit_exit == 0
    assert omitted_exit == 0
    assert omitted_output == explicit_output


def test_trace_distribution_step_missing_value_reports_error(
    capfd, sample_repo_root: Path
) -> None:
    repo_root = sample_repo_root
    _require_distribution_series(repo_root)

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--step",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 2
    assert "Missing value for --step" in captured.err


def test_trace_distribution_step_non_integer_reports_error(capfd, sample_repo_root: Path) -> None:
    repo_root = sample_repo_root
    _require_distribution_series(repo_root)

    exit_code = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--step",
            "abc",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 2
    assert "--step requires an integer" in captured.err


def test_trace_distribution_explicit_modes_exclude_visual_sections(
    capfd, sample_repo_root: Path
) -> None:
    repo_root = sample_repo_root
    _require_distribution_series(repo_root)

    table_exit = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--table",
            "--head",
            "1",
            "--no-color",
        ]
    )
    table_output = capfd.readouterr().out
    assert table_exit == 0
    assert "TENSOR" in table_output
    assert "Heatmap (steps x bins)" not in table_output

    csv_exit = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--csv",
            "--head",
            "1",
        ]
    )
    csv_output = capfd.readouterr().out
    assert csv_exit == 0
    reader = csv.DictReader(io.StringIO(csv_output))
    assert reader.fieldnames is not None
    assert "weights" in reader.fieldnames
    assert "Heatmap (steps x bins)" not in csv_output

    json_exit = main(
        [
            "trace",
            "distribution",
            "distribution.name != ''",
            "--repo",
            str(repo_root),
            "--json",
            "--head",
            "1",
        ]
    )
    json_output = capfd.readouterr().out
    payload = json.loads(json_output)
    assert json_exit == 0
    assert payload
    assert "points" in payload[0]
    assert "Heatmap (steps x bins)" not in json_output


@pytest.mark.parametrize("mode", ["--table", "--csv", "--json"])
def test_trace_distribution_explicit_modes_omitted_expression_match_default(
    capfd,
    sample_repo_root: Path,
    mode: str,
) -> None:
    _require_distribution_series(sample_repo_root)

    explicit_exit = main(
        [
            "trace",
            "distribution",
            DEFAULT_QUERY_EXPRESSION,
            "--repo",
            str(sample_repo_root),
            mode,
            "--head",
            "1",
            "--no-color",
        ]
    )
    explicit_output = capfd.readouterr().out

    omitted_exit = main(
        [
            "trace",
            "distribution",
            "--repo",
            str(sample_repo_root),
            mode,
            "--head",
            "1",
            "--no-color",
        ]
    )
    omitted_output = capfd.readouterr().out

    assert explicit_exit == 0
    assert omitted_exit == 0
    assert omitted_output == explicit_output
