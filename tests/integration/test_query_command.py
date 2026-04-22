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
    assert root_payload["metrics_count"] == dot_aim_payload["metrics_count"]
    assert root_payload["runs_count"] == dot_aim_payload["runs_count"]


def test_metric_query_returns_matches_from_sample_repository(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "loss" in captured.out
    assert "match" in captured.out


def test_metric_query_oneline_mode_returns_tab_separated_rows(
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
    assert lines
    assert "loss" in captured.out
    assert "\t" in lines[0]
    assert "steps=" in lines[0]
    assert "last=" in lines[0]


def test_metric_query_json_mode_returns_nested_structure(
    capfd, sample_repo_root
) -> None:
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
    assert payload["runs_count"] > 0
    assert payload["metrics_count"] > 0
    first_run = payload["runs"][0]
    first_metric = first_run["metrics"][0]
    assert first_metric["name"] == "loss"
    # last/min/max should have real numeric values (not null) since the data exists
    last_val = first_metric["last"]["value"]
    assert last_val is not None


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
