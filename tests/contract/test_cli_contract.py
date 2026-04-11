from __future__ import annotations

import json
import os

from aimx.__main__ import main


def test_help_contract_describes_owned_and_passthrough_commands(capsys) -> None:
    exit_code = main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "help" in captured.out
    assert "version" in captured.out
    assert "doctor" in captured.out
    assert "delegated to native `aim`" in captured.out


def test_passthrough_contract_preserves_exit_status_and_output(
    capfd, monkeypatch, fake_aim_script
) -> None:
    monkeypatch.setenv("PATH", f"{fake_aim_script.parent}:{os.environ.get('PATH', '')}")
    exit_code = main(["up"])

    captured = capfd.readouterr()
    payload = json.loads(captured.out.strip())
    assert exit_code == 0
    assert payload["argv"] == ["up"]
    assert "fake-aim-stderr" in captured.err
