from __future__ import annotations

import json
import os

from aimx.__main__ import main


def test_passthrough_forwards_subcommands_and_flags_in_order(
    capfd, monkeypatch, fake_aim_script
) -> None:
    monkeypatch.setenv("PATH", f"{fake_aim_script.parent}:{os.environ.get('PATH', '')}")

    exit_code = main(["runs", "ls", "--json"])

    captured = capfd.readouterr()
    payload = json.loads(captured.out.strip())
    assert exit_code == 0
    assert payload["argv"] == ["runs", "ls", "--json"]


def test_passthrough_returns_native_exit_status(
    capfd, monkeypatch, fake_aim_script
) -> None:
    monkeypatch.setenv("PATH", f"{fake_aim_script.parent}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_AIM_EXIT", "7")

    exit_code = main(["up"])

    capfd.readouterr()
    assert exit_code == 7
