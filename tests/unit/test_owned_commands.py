from __future__ import annotations

from aimx.commands.help import render_help
from aimx.commands.version import render_version


def test_render_help_lists_owned_commands_and_passthrough_boundary() -> None:
    help_text = render_help()

    assert "help" in help_text
    assert "version" in help_text
    assert "doctor" in help_text
    assert "query" in help_text
    assert "delegated to native `aim`" in help_text


def test_render_version_includes_native_version_when_available() -> None:
    version_text = render_version(native_aim_version="3.29.1")

    assert "aimx 0.1.0" in version_text
    assert "native aim 3.29.1" in version_text
