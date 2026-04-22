from __future__ import annotations

"""Integration tests for short-hash expansion and --steps filtering.

These tests require the sample Aim repository at data/.aim (same fixture as
other integration tests).  They are skipped automatically if the repo is
absent.
"""

import json

from aimx.__main__ import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_run_hash(sample_repo_root) -> str:
    """Return the full hash of one run known to have a 'loss' metric."""
    from aim import Repo

    repo = Repo(str(sample_repo_root))
    return repo.list_all_runs()[0]


# ---------------------------------------------------------------------------
# Short hash: query
# ---------------------------------------------------------------------------


def test_query_with_short_hash_returns_same_result_as_full_hash(
    capfd, sample_repo_root
) -> None:
    full_hash = _first_run_hash(sample_repo_root)
    short_hash = full_hash[:8]

    exit_full = main(
        [
            "query", "metrics",
            f"run.hash=='{full_hash}' and metric.name=='loss'",
            "--repo", str(sample_repo_root),
            "--json",
        ]
    )
    out_full = json.loads(capfd.readouterr().out)

    exit_short = main(
        [
            "query", "metrics",
            f"run.hash=='{short_hash}' and metric.name=='loss'",
            "--repo", str(sample_repo_root),
            "--json",
        ]
    )
    out_short = json.loads(capfd.readouterr().out)

    assert exit_full == 0
    assert exit_short == 0
    assert out_full["metrics_count"] == out_short["metrics_count"]
    assert out_full["runs_count"] == out_short["runs_count"]


def test_query_with_short_hash_exits_zero_and_contains_metric_name(
    capfd, sample_repo_root
) -> None:
    full_hash = _first_run_hash(sample_repo_root)
    short_hash = full_hash[:8]

    exit_code = main(
        [
            "query", "metrics",
            f"run.hash=='{short_hash}' and metric.name=='loss'",
            "--repo", str(sample_repo_root),
        ]
    )
    captured = capfd.readouterr()

    assert exit_code == 0
    assert "loss" in captured.out


# ---------------------------------------------------------------------------
# Short hash: trace
# ---------------------------------------------------------------------------


def test_trace_with_short_hash_produces_output(capfd, sample_repo_root) -> None:
    full_hash = _first_run_hash(sample_repo_root)
    short_hash = full_hash[:8]

    exit_code = main(
        [
            "trace",
            f"run.hash=='{short_hash}' and metric.name=='loss'",
            "--repo", str(sample_repo_root),
            "--json",
        ]
    )
    captured = capfd.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    for series in payload:
        assert series["run"]["hash"] == full_hash


# ---------------------------------------------------------------------------
# Short hash: error paths
# ---------------------------------------------------------------------------


def test_query_nonexistent_short_hash_reports_error(capfd, sample_repo_root) -> None:
    exit_code = main(
        [
            "query", "metrics",
            "run.hash=='0000000000000000' and metric.name=='loss'",
            "--repo", str(sample_repo_root),
        ]
    )
    captured = capfd.readouterr()

    assert exit_code == 2
    assert "did not match" in captured.err


def test_query_ambiguous_short_hash_reports_error(capfd, sample_repo_root) -> None:
    """A prefix of 1 char will almost certainly match multiple runs."""
    from aim import Repo

    repo = Repo(str(sample_repo_root))
    all_hashes = repo.list_all_runs()
    # Use the first character of the first hash; if more than one run shares
    # that character, the prefix is ambiguous.
    first_char = all_hashes[0][0]
    matches = [h for h in all_hashes if h.startswith(first_char)]
    if len(matches) < 2:
        import pytest
        pytest.skip("No ambiguous prefix available in this repository")

    exit_code = main(
        [
            "query", "metrics",
            f"run.hash=='{first_char}' and metric.name=='loss'",
            "--repo", str(sample_repo_root),
        ]
    )
    captured = capfd.readouterr()

    assert exit_code == 2
    assert "ambiguous" in captured.err


# ---------------------------------------------------------------------------
# --steps filter: query
# ---------------------------------------------------------------------------


def test_query_steps_filter_reduces_step_count(capfd, sample_repo_root) -> None:
    """Steps in the filtered result should all be <= the bound."""
    exit_code = main(
        [
            "query", "metrics",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--steps", ":50",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["metrics_count"] >= 0
    for run in payload["runs"]:
        for metric in run["metrics"]:
            assert metric["steps"] <= 50


def test_query_steps_filter_closed_range_bounds_last_value(
    capfd, sample_repo_root
) -> None:
    """last.step must fall within the requested window (or be -1 for empty)."""
    exit_code = main(
        [
            "query", "metrics",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--steps", "50:100",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    for run in payload["runs"]:
        for metric in run["metrics"]:
            last_step = metric["last"]["step"]
            if last_step != -1:
                assert 50 <= last_step <= 100


# ---------------------------------------------------------------------------
# --steps filter: trace
# ---------------------------------------------------------------------------


def test_trace_steps_filter_constrains_step_values(capfd, sample_repo_root) -> None:
    exit_code = main(
        [
            "trace",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--steps", "1:50",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    for series in payload:
        for step in series["steps"]:
            assert 1 <= step <= 50


def test_trace_steps_open_end_filter_keeps_steps_from_start(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        [
            "trace",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--steps", "50:",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    for series in payload:
        for step in series["steps"]:
            assert step >= 50


def test_trace_steps_filter_and_short_hash_work_together(
    capfd, sample_repo_root
) -> None:
    full_hash = _first_run_hash(sample_repo_root)
    short_hash = full_hash[:8]

    exit_code = main(
        [
            "trace",
            f"run.hash=='{short_hash}' and metric.name=='loss'",
            "--repo", str(sample_repo_root),
            "--steps", "1:200",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    for series in payload:
        assert series["run"]["hash"] == full_hash
        for step in series["steps"]:
            assert 1 <= step <= 200


# ---------------------------------------------------------------------------
# --head / --tail / --every for query metrics
# ---------------------------------------------------------------------------


def test_query_metrics_head_limits_step_count(capfd, sample_repo_root) -> None:
    """--head 5 should yield at most 5 steps per metric."""
    exit_code = main(
        [
            "query", "metrics",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--head", "5",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    for run in payload["runs"]:
        for metric in run["metrics"]:
            assert metric["steps"] <= 5


def test_query_metrics_tail_limits_step_count(capfd, sample_repo_root) -> None:
    """--tail 5 should yield at most 5 steps per metric."""
    exit_code = main(
        [
            "query", "metrics",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--tail", "5",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    for run in payload["runs"]:
        for metric in run["metrics"]:
            assert metric["steps"] <= 5


def test_query_metrics_every_reduces_step_count(capfd, sample_repo_root) -> None:
    """--every 2 should roughly halve the step count."""
    # baseline (no sampling)
    exit_code_base = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--json"]
    )
    payload_base = json.loads(capfd.readouterr().out)

    exit_code = main(
        [
            "query", "metrics",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--every", "2",
            "--json",
        ]
    )
    payload = json.loads(capfd.readouterr().out)

    assert exit_code_base == 0
    assert exit_code == 0
    # step count should be reduced
    base_total = sum(m["steps"] for r in payload_base["runs"] for m in r["metrics"])
    every_total = sum(m["steps"] for r in payload["runs"] for m in r["metrics"])
    assert every_total < base_total


def test_query_metrics_epochs_filter_works(capfd, sample_repo_root) -> None:
    """--epochs filter should restrict rows (exits 0 even with empty results)."""
    exit_code = main(
        [
            "query", "metrics",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--epochs", "1:5",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert "runs" in payload


def test_query_metrics_steps_and_epochs_mutually_exclusive(capfd, sample_repo_root) -> None:
    exit_code = main(
        [
            "query", "metrics",
            "metric.name == 'loss'",
            "--repo", str(sample_repo_root),
            "--steps", "1:50",
            "--epochs", "1:5",
        ]
    )
    assert exit_code == 2
    assert "mutually exclusive" in capfd.readouterr().err
