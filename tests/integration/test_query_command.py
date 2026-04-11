from __future__ import annotations

import json

from aimx.__main__ import main


def test_metric_query_accepts_repo_root_and_dot_aim_paths(
    capfd, sample_repo_root, sample_repo_dot_aim
) -> None:
    root_exit_code = main(
        [
            "query",
            "metrics",
            "metric.name == 'loss'",
            "--repo",
            str(sample_repo_root),
            "--json",
        ]
    )
    root_captured = capfd.readouterr()
    root_payload = json.loads(root_captured.out)

    dot_aim_exit_code = main(
        [
            "query",
            "metrics",
            "metric.name == 'loss'",
            "--repo",
            str(sample_repo_dot_aim),
            "--json",
        ]
    )
    dot_aim_captured = capfd.readouterr()
    dot_aim_payload = json.loads(dot_aim_captured.out)

    assert root_exit_code == 0
    assert dot_aim_exit_code == 0
    assert root_payload["count"] == dot_aim_payload["count"]
    assert root_payload["rows"] == dot_aim_payload["rows"]


def test_metric_query_returns_matches_from_sample_repository(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "loss" in captured.out
    assert "matches:" in captured.out


def test_image_query_returns_matches_from_sample_repository(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--json"]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["count"] > 0
    assert payload["rows"][0]["name"] == "example"


def test_invalid_query_expression_fails_cleanly(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["query", "metrics", "metric.name ==", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 2
    assert "Failed to evaluate query" in captured.err
