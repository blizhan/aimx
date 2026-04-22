from __future__ import annotations

import csv
import io
import json

from aimx.__main__ import main


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
