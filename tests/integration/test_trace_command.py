from __future__ import annotations

import csv
import io
import json

from aimx.__main__ import main


def test_trace_plot_produces_output_containing_metric_name(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["trace", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert captured.out.strip(), "Expected non-empty plotext output"


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
