from __future__ import annotations

import json

from aimx.__main__ import main


def test_query_metrics_json_contract_uses_nested_runs_envelope(capfd, sample_repo_root) -> None:
    exit_code = main(
        [
            "query",
            "metrics",
            "metric.name == 'loss'",
            "--repo",
            str(sample_repo_root),
            "--json",
        ]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["target"] == "metrics"
    assert payload["expression"] == "metric.name == 'loss'"
    assert payload["repo"] == str(sample_repo_root)
    assert payload["runs_count"] > 0
    assert payload["metrics_count"] > 0
    assert payload["runs"]
    first_run = payload["runs"][0]
    assert "hash" in first_run
    assert "experiment" in first_run
    assert "metrics" in first_run
    first_metric = first_run["metrics"][0]
    assert first_metric["name"] == "loss"
    assert "steps" in first_metric
    assert "last" in first_metric
    assert "min" in first_metric
    assert "max" in first_metric


def test_query_metrics_text_contract_reports_repo_count_and_metric_name(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "Repo:" in captured.out
    assert "match" in captured.out
    assert "loss" in captured.out


def test_query_metrics_oneline_contract_is_tab_separated_and_contains_metric_name(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        [
            "query",
            "metrics",
            "metric.name == 'loss'",
            "--repo",
            str(sample_repo_root),
            "--oneline",
        ]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert lines, "Expected at least one output line"
    assert "loss" in captured.out
    assert "\t" in lines[0]


def test_query_images_json_contract_uses_stable_envelope(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--json"]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["target"] == "images"
    assert payload["expression"] == "images"
    assert payload["count"] > 0
    assert payload["rows"][0]["name"] == "example"


def test_query_invalid_expression_reports_actionable_error(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["query", "metrics", "metric.name ==", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 2
    assert "Failed to evaluate query" in captured.err


def test_query_invalid_repo_reports_actionable_error(capfd) -> None:
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", "missing-repo"]
    )

    captured = capfd.readouterr()
    assert exit_code == 2
    assert "Repository path does not exist" in captured.err
