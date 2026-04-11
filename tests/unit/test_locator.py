from __future__ import annotations

from aimx.native_aim.locator import resolve_native_aim


def test_resolve_native_aim_reports_missing_when_executable_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PATH", "")

    result = resolve_native_aim()

    assert result.status == "missing"
    assert result.executable_path is None
    assert "install native Aim" in result.diagnostic_message


def test_resolve_native_aim_reports_unusable_when_probe_fails(
    monkeypatch, tmp_path
) -> None:
    script = tmp_path / "aim"
    script.write_text("#!/bin/sh\nexit 3\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    result = resolve_native_aim()

    assert result.status == "unusable"
    assert result.executable_path == str(script)
    assert "could not be executed" in result.diagnostic_message
