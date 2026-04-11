from __future__ import annotations

import json

from aimx.__main__ import main


def test_query_metrics_json_contract_uses_stable_envelope(capfd, sample_repo_root) -> None:
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
    assert payload["repo_path"] == "data"
    assert payload["count"] > 0
    assert payload["rows"]
    assert payload["rows"][0]["target"] == "metrics"
    assert payload["rows"][0]["name"] == "loss"
    assert payload["rows"][0]["run_id"]
    assert "summary" in payload["rows"][0]


def test_query_metrics_text_contract_reports_target_repo_and_count(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "target: metrics" in captured.out
    assert f"repo: {sample_repo_root}" in captured.out
    assert "matches:" in captured.out
    assert "loss" in captured.out


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
    assert payload["rows"][0]["target"] == "images"
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
