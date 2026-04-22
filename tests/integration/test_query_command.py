from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import patch

import pytest

from aimx.__main__ import main
from aimx.aim_bridge.metric_stats import RunMeta
from aimx.rendering.image_render import ImageRenderPlan, TerminalCapability, plan_render, render_inline


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


def test_metric_query_defaults_repo_to_current_directory(
    capfd, monkeypatch, sample_repo_root
) -> None:
    monkeypatch.chdir(sample_repo_root)

    exit_code = main(["query", "metrics", "metric.name == 'loss'"])

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


# ---------------------------------------------------------------------------
# T006 / T019 / T020: image rendering integration tests
# ---------------------------------------------------------------------------

def _make_tiny_pil_image() -> Any:
    """Return a tiny 4×4 RGB PIL image for test fixtures."""
    from PIL import Image as PILImage
    return PILImage.new("RGB", (4, 4), color=(100, 150, 200))


def _fake_image_rows(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(count):
        run = RunMeta(
            hash=f"deadbeef{i:08d}"[:16],
            experiment=f"exp{i}",
            name=None,
            creation_time=None,
        )
        rows.append({
            "run": run,
            "name": f"img{i}",
            "context": {},
            "_image_accessor": _make_tiny_pil_image,
        })
    return rows


def _fake_auto_capability(columns: int = 80, rows: int = 24) -> TerminalCapability:
    return TerminalCapability(
        is_tty=True,
        columns=columns,
        rows=rows,
        protocol="fallback_text",
        reason=None,
    )


def test_image_query_non_tty_output_has_no_image_bytes(capfd, sample_repo_root) -> None:
    """T006: with stdout not a TTY, run_query_command must NOT add image bytes."""
    from aimx.commands.query import run_query_command

    result = run_query_command(["images", "images", "--repo", str(sample_repo_root)])

    assert result.exit_status == 0
    output = result.output or ""
    # No Kitty/iTerm2/Sixel markers
    assert "\x1b_G" not in output
    assert "\x1b]1337;File=" not in output
    assert "\x90" not in output


def test_image_query_accessor_returns_real_pil_image(sample_repo_root) -> None:
    """Regression: Aim query_images() yields `Images`, but accessor must return PIL."""
    from aimx.aim_bridge.metric_stats import collect_image_series

    rows = collect_image_series("images", sample_repo_root)

    assert rows, "Expected at least one image row from sample repository"
    accessor = rows[0]["_image_accessor"]
    pil_img = accessor()

    assert pil_img.__class__.__module__.startswith("PIL.")
    assert hasattr(pil_img, "size")
    assert pil_img.size[0] > 0
    assert pil_img.size[1] > 0


def test_plan_render_default_cap_splits_correctly() -> None:
    """T019: plan_render with max_images=6 on 8 rows → 6 rendered, 2 skipped."""
    cap = _fake_auto_capability()
    rows = _fake_image_rows(8)
    plan = plan_render(rows, cap, max_images=6)
    assert len(plan.rendered_rows) == 6
    assert len(plan.skipped_rows) == 2


def test_plan_render_unlimited_renders_all() -> None:
    """T019: plan_render with max_images=0 renders all rows."""
    cap = _fake_auto_capability()
    rows = _fake_image_rows(8)
    plan = plan_render(rows, cap, max_images=0)
    assert len(plan.rendered_rows) == 8
    assert plan.skipped_rows == []


def test_plan_render_partial_cap() -> None:
    """T019: plan_render with max_images=3 on 8 rows → 3 rendered, 5 skipped."""
    cap = _fake_auto_capability()
    rows = _fake_image_rows(8)
    plan = plan_render(rows, cap, max_images=3)
    assert len(plan.rendered_rows) == 3
    assert len(plan.skipped_rows) == 5


def test_render_inline_truncation_footer_present_when_skipped() -> None:
    """T020: render_inline emits footer when skipped_rows is non-empty."""
    cap = _fake_auto_capability()
    rows = _fake_image_rows(8)
    plan = plan_render(rows, cap, max_images=6)
    output = render_inline(plan)
    assert "rendered 6 of 8 images" in output
    assert "--max-images=0" in output


def test_render_inline_no_footer_when_all_rendered() -> None:
    """T020: render_inline omits footer when nothing is skipped."""
    cap = _fake_auto_capability()
    rows = _fake_image_rows(4)
    plan = plan_render(rows, cap, max_images=6)
    output = render_inline(plan)
    assert "--max-images=0" not in output


def test_image_query_inline_preview_preserves_full_rich_table() -> None:
    """Interactive inline previews must not hide rows beyond the preview cap."""
    from aimx.commands.query import run_query_command

    rows = _fake_image_rows(8)

    with (
        patch("aimx.aim_bridge.metric_stats.collect_image_series", return_value=rows),
        patch("aimx.rendering.image_render.detect_capability", return_value=_fake_auto_capability()),
        patch("aimx.rendering.image_render.render_inline", return_value="<<INLINE PREVIEW>>\n"),
    ):
        result = run_query_command(["images", "images", "--repo", "data"])

    assert result.exit_status == 0
    assert result.output is not None
    assert "Repo: data" in result.output
    assert "img0" in result.output
    assert "img7" in result.output
    assert "<<INLINE PREVIEW>>" in result.output


def test_max_images_flag_default_is_six(sample_repo_root) -> None:
    """T006: --max-images default is 6 (reflected in QueryInvocation)."""
    from aimx.commands.query import parse_query_invocation

    inv = parse_query_invocation(["images", "images", "--repo", str(sample_repo_root)])
    assert inv.max_images == 6


def test_max_images_flag_zero_means_unlimited(sample_repo_root) -> None:
    """T006: --max-images 0 parses and triggers unlimited render."""
    from aimx.commands.query import parse_query_invocation

    inv = parse_query_invocation(
        ["images", "images", "--repo", str(sample_repo_root), "--max-images", "0"]
    )
    assert inv.max_images == 0


# ---------------------------------------------------------------------------
# query images: --head / --tail / --every
# ---------------------------------------------------------------------------

def test_query_images_head_reduces_row_count(sample_repo_root) -> None:
    """--head 3 should yield at most 3 rows in --json output."""
    from aimx.commands.query import run_query_command
    import json

    result_all = run_query_command(["images", "images", "--repo", str(sample_repo_root), "--json"])
    result_head = run_query_command(["images", "images", "--repo", str(sample_repo_root), "--head", "3", "--json"])

    assert result_all.exit_status == 0
    assert result_head.exit_status == 0

    all_rows = json.loads(result_all.output)["rows"]
    head_rows = json.loads(result_head.output)["rows"]

    assert len(head_rows) <= 3
    assert len(head_rows) <= len(all_rows)
    # First rows must match baseline (head preserves order)
    for i, row in enumerate(head_rows):
        assert row["name"] == all_rows[i]["name"]


def test_query_images_tail_reduces_row_count(sample_repo_root) -> None:
    """--tail 3 should yield the last 3 rows in --json output."""
    from aimx.commands.query import run_query_command
    import json

    result_all = run_query_command(["images", "images", "--repo", str(sample_repo_root), "--json"])
    result_tail = run_query_command(["images", "images", "--repo", str(sample_repo_root), "--tail", "3", "--json"])

    assert result_all.exit_status == 0
    assert result_tail.exit_status == 0

    all_rows = json.loads(result_all.output)["rows"]
    tail_rows = json.loads(result_tail.output)["rows"]

    assert len(tail_rows) <= 3
    # Last rows must match baseline
    for i, row in enumerate(tail_rows):
        assert row["name"] == all_rows[len(all_rows) - len(tail_rows) + i]["name"]


def test_query_images_every_reduces_row_count(sample_repo_root) -> None:
    """--every 2 should roughly halve the image row count."""
    from aimx.commands.query import run_query_command
    import json

    result_all = run_query_command(["images", "images", "--repo", str(sample_repo_root), "--json"])
    result_every = run_query_command(["images", "images", "--repo", str(sample_repo_root), "--every", "2", "--json"])

    assert result_all.exit_status == 0
    assert result_every.exit_status == 0

    total = json.loads(result_all.output)["count"]
    every_count = json.loads(result_every.output)["count"]

    assert every_count < total


def test_query_images_epochs_filter_reduces_row_count(sample_repo_root) -> None:
    """--epochs 10:30 should return fewer rows than the full query."""
    from aimx.commands.query import run_query_command
    import json

    result_all = run_query_command(["images", "images", "--repo", str(sample_repo_root), "--json"])
    result_filtered = run_query_command(
        ["images", "images", "--repo", str(sample_repo_root), "--epochs", "10:30", "--json"]
    )

    assert result_all.exit_status == 0
    assert result_filtered.exit_status == 0

    total = json.loads(result_all.output)["count"]
    filtered = json.loads(result_filtered.output)["count"]

    assert filtered < total


def test_query_images_head_and_epochs_compose(sample_repo_root) -> None:
    """--epochs filter followed by --head should further reduce result."""
    from aimx.commands.query import run_query_command
    import json

    result = run_query_command(
        ["images", "images", "--repo", str(sample_repo_root), "--epochs", "5:50", "--head", "2", "--json"]
    )

    assert result.exit_status == 0
    rows = json.loads(result.output)["rows"]
    assert len(rows) <= 2


def test_query_images_steps_and_epochs_exclusive_exits_2(sample_repo_root) -> None:
    """--steps and --epochs together must exit with code 2."""
    from aimx.commands.query import run_query_command

    result = run_query_command(
        ["images", "images", "--repo", str(sample_repo_root), "--steps", "1:10", "--epochs", "1:5"]
    )

    assert result.exit_status == 2
    assert result.error_message is not None
    assert "mutually exclusive" in result.error_message
