from __future__ import annotations

from aimx.__main__ import main


def test_owned_commands_still_work_when_native_aim_is_missing(
    capsys, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", "")

    exit_code = main(["version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aimx 0.1.0" in captured.out


def test_doctor_reports_missing_native_aim(capsys, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "passthrough: not ready" in captured.out


def test_passthrough_fails_fast_with_actionable_message_when_native_aim_is_missing(
    capsys, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", "")

    exit_code = main(["up"])

    captured = capsys.readouterr()
    assert exit_code == 127
    assert "install native Aim" in captured.err


def test_query_owned_command_still_works_when_native_aim_is_missing(
    capsys, monkeypatch, sample_repo_root
) -> None:
    monkeypatch.setenv("PATH", "")

    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(sample_repo_root)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "target: metrics" in captured.out
    assert "matches:" in captured.out
