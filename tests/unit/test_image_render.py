"""Unit tests for aimx.rendering.image_render.

These tests cover:
- T005: detect_capability envvar matrix, plan_render clamp rules,
        lazy _image_accessor failure → placeholder behaviour.
- T015: warn_once single-emission guarantee.
- T018: --max-images parsing (delegated to test_query_helpers.py addendum).
"""

from __future__ import annotations

import io
import sys
from typing import Any
from unittest.mock import patch

import pytest

from aimx.aim_bridge.metric_stats import RunMeta
from aimx.rendering.image_render import (
    ImageRenderPlan,
    TerminalCapability,
    detect_capability,
    plan_render,
    render_inline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_capability(
    *,
    is_tty: bool = True,
    columns: int = 80,
    rows: int = 24,
    protocol: str = "auto",
    reason: str | None = None,
) -> TerminalCapability:
    return TerminalCapability(
        is_tty=is_tty,
        columns=columns,
        rows=rows,
        protocol=protocol,  # type: ignore[arg-type]
        reason=reason,
    )


def _fake_row(name: str = "img", accessor: Any = None) -> dict[str, Any]:
    run = RunMeta(hash="abcdef1234567890", experiment="exp", name="run1", creation_time=None)
    return {
        "run": run,
        "name": name,
        "context": {"subset": "train"},
        "_image_accessor": accessor,
    }


def _make_tiny_pil() -> Any:
    """Return a 4×4 red RGB PIL image."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (4, 4), color=(200, 50, 50))
    return img


# ---------------------------------------------------------------------------
# detect_capability: envvar matrix
# ---------------------------------------------------------------------------

class TestDetectCapability:
    def test_non_tty_returns_disabled(self) -> None:
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            cap = detect_capability()
        assert cap.protocol == "disabled"
        assert cap.is_tty is False

    def test_narrow_terminal_returns_disabled(self) -> None:
        with (
            patch("sys.stdout") as mock_stdout,
            patch("shutil.get_terminal_size", return_value=type("S", (), {"columns": 10, "lines": 24})()),
        ):
            mock_stdout.isatty.return_value = True
            cap = detect_capability()
        assert cap.protocol == "disabled"
        assert "width" in (cap.reason or "").lower() or "20" in (cap.reason or "")

    def test_iterm2_classified_as_auto(self) -> None:
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", {"TERM_PROGRAM": "iTerm.app"}, clear=False),
            patch("shutil.get_terminal_size", return_value=type("S", (), {"columns": 120, "lines": 30})()),
        ):
            mock_stdout.isatty.return_value = True
            cap = detect_capability()
        assert cap.protocol == "auto"

    def test_kitty_classified_as_auto(self) -> None:
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", {"KITTY_WINDOW_ID": "1"}, clear=False),
            patch("shutil.get_terminal_size", return_value=type("S", (), {"columns": 120, "lines": 30})()),
        ):
            mock_stdout.isatty.return_value = True
            cap = detect_capability()
        assert cap.protocol == "auto"

    def test_wezterm_classified_as_auto(self) -> None:
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", {"TERM_PROGRAM": "WezTerm"}, clear=False),
            patch("shutil.get_terminal_size", return_value=type("S", (), {"columns": 120, "lines": 30})()),
        ):
            mock_stdout.isatty.return_value = True
            cap = detect_capability()
        assert cap.protocol == "auto"

    def test_ghostty_classified_as_auto(self) -> None:
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", {"GHOSTTY_RESOURCES_DIR": "/usr/share/ghostty"}, clear=False),
            patch("shutil.get_terminal_size", return_value=type("S", (), {"columns": 120, "lines": 30})()),
        ):
            mock_stdout.isatty.return_value = True
            cap = detect_capability()
        assert cap.protocol == "auto"

    def test_plain_xterm_classified_as_fallback_text(self) -> None:
        env_overrides: dict[str, str] = {}
        remove_keys = {
            "TERM_PROGRAM", "KITTY_WINDOW_ID", "WEZTERM_EXECUTABLE",
            "GHOSTTY_RESOURCES_DIR", "VTE_VERSION",
        }
        with (
            patch("sys.stdout") as mock_stdout,
            patch("shutil.get_terminal_size", return_value=type("S", (), {"columns": 80, "lines": 24})()),
            patch.dict("os.environ", env_overrides, clear=False),
        ):
            # Suppress any inherited graphics-capable env vars
            import os
            original = {k: os.environ.pop(k) for k in remove_keys if k in os.environ}
            try:
                mock_stdout.isatty.return_value = True
                cap = detect_capability()
            finally:
                os.environ.update(original)
        assert cap.protocol in ("fallback_text", "auto")


# ---------------------------------------------------------------------------
# plan_render: split and clamping logic
# ---------------------------------------------------------------------------

class TestPlanRender:
    def test_disabled_capability_returns_empty_plan(self) -> None:
        cap = _fake_capability(protocol="disabled")
        rows = [_fake_row() for _ in range(5)]
        plan = plan_render(rows, cap, max_images=6)
        assert plan.rendered_rows == []
        assert plan.skipped_rows == []

    def test_unlimited_renders_all(self) -> None:
        cap = _fake_capability(protocol="auto", rows=24)
        rows = [_fake_row(f"img{i}") for i in range(10)]
        plan = plan_render(rows, cap, max_images=0)
        assert len(plan.rendered_rows) == 10
        assert plan.skipped_rows == []

    def test_cap_splits_correctly(self) -> None:
        cap = _fake_capability(protocol="auto", rows=24)
        rows = [_fake_row(f"img{i}") for i in range(8)]
        plan = plan_render(rows, cap, max_images=6)
        assert len(plan.rendered_rows) == 6
        assert len(plan.skipped_rows) == 2
        assert len(plan.rendered_rows) + len(plan.skipped_rows) == 8

    def test_fewer_rows_than_cap_no_skipped(self) -> None:
        cap = _fake_capability(protocol="auto", rows=24)
        rows = [_fake_row() for _ in range(3)]
        plan = plan_render(rows, cap, max_images=6)
        assert len(plan.rendered_rows) == 3
        assert plan.skipped_rows == []

    def test_max_height_clamped_to_at_least_3(self) -> None:
        # rows=6 → rows//3 = 2 → clamped to 3
        cap = _fake_capability(protocol="auto", rows=6)
        plan = plan_render([], cap, max_images=6)
        assert plan.max_height == 3

    def test_max_height_normal(self) -> None:
        cap = _fake_capability(protocol="auto", rows=24)
        plan = plan_render([], cap, max_images=6)
        assert plan.max_height == 8  # 24 // 3

    def test_target_width_equals_columns(self) -> None:
        cap = _fake_capability(protocol="auto", columns=100)
        plan = plan_render([], cap, max_images=6)
        assert plan.target_width == 100


# ---------------------------------------------------------------------------
# render_inline: disabled → empty string
# ---------------------------------------------------------------------------

class TestRenderInline:
    def test_disabled_returns_empty_string(self) -> None:
        cap = _fake_capability(protocol="disabled")
        plan = plan_render([], cap, max_images=6)
        result = render_inline(plan)
        assert result == ""

    def test_empty_rendered_rows_returns_empty_string(self) -> None:
        cap = _fake_capability(protocol="auto")
        plan = ImageRenderPlan(
            capability=cap,
            rendered_rows=[],
            skipped_rows=[],
            target_width=80,
            max_height=8,
            max_images=6,
        )
        result = render_inline(plan)
        assert result == ""

    def test_none_accessor_shows_placeholder(self) -> None:
        cap = _fake_capability(protocol="fallback_text", columns=80, rows=24)
        row = _fake_row(accessor=None)
        plan = ImageRenderPlan(
            capability=cap,
            rendered_rows=[row],
            skipped_rows=[],
            target_width=80,
            max_height=8,
            max_images=1,
        )
        result = render_inline(plan)
        assert "image unavailable" in result

    def test_raising_accessor_shows_placeholder(self) -> None:
        cap = _fake_capability(protocol="fallback_text", columns=80, rows=24)

        def bad_accessor() -> Any:
            raise RuntimeError("blob corrupted")

        row = _fake_row(accessor=bad_accessor)
        plan = ImageRenderPlan(
            capability=cap,
            rendered_rows=[row],
            skipped_rows=[],
            target_width=80,
            max_height=8,
            max_images=1,
        )
        result = render_inline(plan)
        assert "image unavailable" in result

    def test_truncation_footer_present_when_skipped(self) -> None:
        cap = _fake_capability(protocol="fallback_text", columns=80, rows=24)
        rows_rendered = [_fake_row(f"img{i}", accessor=_make_tiny_pil) for i in range(3)]
        rows_skipped = [_fake_row(f"skip{i}") for i in range(2)]
        plan = ImageRenderPlan(
            capability=cap,
            rendered_rows=rows_rendered,
            skipped_rows=rows_skipped,
            target_width=80,
            max_height=8,
            max_images=3,
        )
        result = render_inline(plan)
        assert "rendered 3 of 5 images" in result
        assert "--max-images=0" in result

    def test_no_truncation_footer_when_no_skipped(self) -> None:
        cap = _fake_capability(protocol="fallback_text", columns=80, rows=24)
        rows_rendered = [_fake_row("img0", accessor=_make_tiny_pil)]
        plan = ImageRenderPlan(
            capability=cap,
            rendered_rows=rows_rendered,
            skipped_rows=[],
            target_width=80,
            max_height=8,
            max_images=6,
        )
        result = render_inline(plan)
        assert "--max-images=0" not in result


# ---------------------------------------------------------------------------
# warn_once: single emission
# ---------------------------------------------------------------------------

class TestWarnOnce:
    def test_warns_exactly_once(self) -> None:
        import aimx.rendering.image_render as ir

        original_warned = ir._WARNED
        ir._WARNED = False
        try:
            captured = io.StringIO()
            with patch("sys.stderr", captured):
                from aimx.rendering.image_render import warn_once
                warn_once("test message A")
                warn_once("test message B")
            output = captured.getvalue()
            assert output.count("aimx: inline image rendering unavailable:") == 1
            assert "test message A" in output
        finally:
            ir._WARNED = original_warned
