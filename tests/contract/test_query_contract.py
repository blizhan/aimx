from __future__ import annotations

import json
import sys
from unittest.mock import patch

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


def test_query_params_json_contract_uses_stable_envelope(capfd, sample_repo_root) -> None:
    exit_code = main(
        ["query", "params", "run.hash != ''", "--repo", str(sample_repo_root), "--json"]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["target"] == "params"
    assert payload["expression"] == "run.hash != ''"
    assert payload["repo"] == str(sample_repo_root)
    assert payload["runs_count"] > 0
    assert payload["param_keys"]
    assert payload["runs"]
    first_run = payload["runs"][0]
    assert "hash" in first_run
    assert "experiment" in first_run
    assert "name" in first_run
    assert "params" in first_run
    assert "missing_params" in first_run


def test_query_params_text_contract_reports_repo_count_and_params(
    capfd, sample_repo_root
) -> None:
    exit_code = main(["query", "params", "run.hash != ''", "--repo", str(sample_repo_root)])

    captured = capfd.readouterr()
    assert exit_code == 0
    assert "Repo:" in captured.out
    assert "match" in captured.out
    assert "hparam.lr" in captured.out


def test_query_params_json_contract_honors_selected_params(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        [
            "query",
            "params",
            "run.hash != ''",
            "--repo",
            str(sample_repo_root),
            "--param",
            "hparam.lr",
            "--param",
            "hparam.weight_decay",
            "--param",
            "missing.key",
            "--json",
        ]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["param_keys"] == ["hparam.lr", "hparam.weight_decay", "missing.key"]
    assert payload["runs"]
    first_run = payload["runs"][0]
    assert set(first_run["params"]) <= {"hparam.lr", "hparam.weight_decay"}
    assert "missing.key" in first_run["missing_params"]


def test_query_params_zero_match_json_preserves_selected_param_keys(
    capfd, sample_repo_root
) -> None:
    exit_code = main(
        [
            "query",
            "params",
            "run.name == 'definitely-missing-run'",
            "--repo",
            str(sample_repo_root),
            "--param",
            "hparam.lr",
            "--param",
            "missing.key",
            "--json",
        ]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["runs_count"] == 0
    assert payload["param_keys"] == ["hparam.lr", "missing.key"]
    assert payload["runs"] == []


def test_query_param_option_rejected_for_metrics_and_images(capfd, sample_repo_root) -> None:
    metrics_exit = main(
        [
            "query",
            "metrics",
            "metric.name == 'loss'",
            "--repo",
            str(sample_repo_root),
            "--param",
            "hparam.lr",
        ]
    )
    metrics_captured = capfd.readouterr()

    images_exit = main(
        [
            "query",
            "images",
            "images",
            "--repo",
            str(sample_repo_root),
            "--param",
            "hparam.lr",
        ]
    )
    images_captured = capfd.readouterr()

    assert metrics_exit == 2
    assert images_exit == 2
    assert "--param is only supported for query params" in metrics_captured.err
    assert "--param is only supported for query params" in images_captured.err


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


# ---------------------------------------------------------------------------
# T011–T014: SC-003 byte-stability — no image graphics bytes on non-TTY paths
# ---------------------------------------------------------------------------

_GRAPHICS_MARKERS = [
    "\x1b_G",       # Kitty graphics
    "\x1b]1337;File=",  # iTerm2 inline image
    "\x90",         # DCS / Sixel start
    "\x1bP",        # Sixel (DCS)
]


def _has_graphics_bytes(text: str) -> bool:
    return any(marker in text for marker in _GRAPHICS_MARKERS)


def test_images_json_output_contains_no_graphics_bytes(capfd, sample_repo_root) -> None:
    """T011: --json output MUST contain zero image-protocol escape sequences."""
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--json"]
    )

    captured = capfd.readouterr()
    payload = json.loads(captured.out)  # also validates it is legal JSON
    assert exit_code == 0
    assert not _has_graphics_bytes(captured.out), (
        "--json stdout contained graphics escape sequences"
    )
    assert payload["count"] > 0


def test_images_plain_output_contains_no_graphics_bytes(capfd, sample_repo_root) -> None:
    """T012: --plain output MUST be tab-separated text with no binary image bytes."""
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--plain"]
    )

    captured = capfd.readouterr()
    assert exit_code == 0
    assert not _has_graphics_bytes(captured.out), (
        "--plain stdout contained graphics escape sequences"
    )
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert lines, "Expected at least one output line"
    for line in lines:
        assert "\t" in line, f"Plain output line is not tab-separated: {line!r}"


def test_images_non_tty_stdout_contains_no_graphics_bytes(
    capfd, sample_repo_root
) -> None:
    """T013: when stdout is not a TTY, no image bytes MUST appear in output."""
    from aimx.commands.query import run_query_command

    # run_query_command returns the rendered string without actually writing to
    # stdout, so we can inspect it directly; the is_tty check inside
    # detect_capability() will see a non-TTY context in the test runner.
    result = run_query_command(
        ["images", "images", "--repo", str(sample_repo_root)]
    )

    assert result.exit_status == 0
    output = result.output or ""
    assert not _has_graphics_bytes(output), (
        "Non-TTY rich output contained graphics escape sequences"
    )


def test_max_images_has_no_effect_on_json_or_plain(capfd, sample_repo_root) -> None:
    """T014: --max-images must NOT change --json or --plain output bytes."""
    def _get_json(extra_args: list[str]) -> str:
        main(
            ["query", "images", "images", "--repo", str(sample_repo_root), "--json"]
            + extra_args
        )
        return capfd.readouterr().out

    baseline = _get_json([])
    with_three = _get_json(["--max-images", "3"])
    with_zero = _get_json(["--max-images", "0"])
    with_hundred = _get_json(["--max-images", "100"])

    assert with_three == baseline, "--max-images 3 changed --json output"
    assert with_zero == baseline, "--max-images 0 changed --json output"
    assert with_hundred == baseline, "--max-images 100 changed --json output"

    def _get_plain(extra_args: list[str]) -> str:
        main(
            ["query", "images", "images", "--repo", str(sample_repo_root), "--plain"]
            + extra_args
        )
        return capfd.readouterr().out

    plain_baseline = _get_plain([])
    plain_three = _get_plain(["--max-images", "3"])
    assert plain_three == plain_baseline, "--max-images 3 changed --plain output"


def test_missing_max_images_value_exits_with_code_2(capfd) -> None:
    """T018 (contract): --max-images with no value → exit 2."""
    exit_code = main(["query", "images", "images", "--max-images"])
    captured = capfd.readouterr()
    assert exit_code == 2


def test_negative_max_images_exits_with_code_2(capfd) -> None:
    """T018 (contract): negative --max-images → exit 2."""
    exit_code = main(["query", "images", "images", "--max-images", "-1"])
    captured = capfd.readouterr()
    assert exit_code == 2


def test_non_integer_max_images_exits_with_code_2(capfd) -> None:
    """T018 (contract): non-integer --max-images → exit 2."""
    exit_code = main(["query", "images", "images", "--max-images", "abc"])
    captured = capfd.readouterr()
    assert exit_code == 2


# ---------------------------------------------------------------------------
# SC-filter: new filter flags keep the JSON/plain envelope shape stable
# ---------------------------------------------------------------------------


def test_head_flag_preserves_json_envelope_shape_for_images(capfd, sample_repo_root) -> None:
    """--head on query images must preserve the JSON envelope structure."""
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--head", "3", "--json"]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["target"] == "images"
    assert "count" in payload
    assert "rows" in payload
    assert payload["count"] <= 3


def test_epochs_flag_preserves_json_envelope_shape_for_images(capfd, sample_repo_root) -> None:
    """--epochs on query images must preserve the JSON envelope structure."""
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--epochs", "10:30", "--json"]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["target"] == "images"
    assert "count" in payload
    assert "rows" in payload


def test_steps_and_epochs_mutually_exclusive_exits_2_for_images(capfd) -> None:
    """--steps and --epochs together → exit 2 with a clear error message."""
    exit_code = main(
        ["query", "images", "images", "--steps", "1:50", "--epochs", "1:10"]
    )
    captured = capfd.readouterr()
    assert exit_code == 2
    assert "mutually exclusive" in captured.err


def test_steps_and_epochs_mutually_exclusive_exits_2_for_metrics(capfd) -> None:
    """--steps and --epochs together → exit 2 for query metrics too."""
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--steps", "1:50", "--epochs", "1:5"]
    )
    captured = capfd.readouterr()
    assert exit_code == 2
    assert "mutually exclusive" in captured.err


def test_head_flag_preserves_json_envelope_shape_for_metrics(capfd, sample_repo_root) -> None:
    """--head on query metrics must preserve the JSON envelope structure."""
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(sample_repo_root), "--head", "5", "--json"]
    )
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["target"] == "metrics"
    assert "runs_count" in payload
    assert "metrics_count" in payload
    assert "runs" in payload
    for run in payload["runs"]:
        for metric in run["metrics"]:
            assert metric["steps"] <= 5


def test_every_flag_preserves_plain_format_for_images(capfd, sample_repo_root) -> None:
    """--every on query images --plain must remain tab-separated."""
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--every", "2", "--plain"]
    )
    captured = capfd.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    if lines:
        assert "\t" in lines[0], "Plain output with --every must remain tab-separated"


def test_filter_flags_do_not_produce_graphics_bytes(capfd, sample_repo_root) -> None:
    """--head / --epochs must never emit image-protocol escape bytes on --json."""
    exit_code = main(
        ["query", "images", "images", "--repo", str(sample_repo_root), "--head", "2", "--json"]
    )
    captured = capfd.readouterr()
    assert exit_code == 0
    assert not _has_graphics_bytes(captured.out)
