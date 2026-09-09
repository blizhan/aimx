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
    assert "query" in captured.out
    assert "research" in captured.out
    assert "finding" in captured.out
    assert "lineage" in captured.out
    assert "frontier" in captured.out
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


def test_new_research_roots_are_owned_and_unrelated_roots_remain_passthrough() -> None:
    from aimx.router import route_args

    for command in ("research", "finding", "lineage", "frontier"):
        route = route_args([command, "state"])
        assert route.route_kind == "owned"
        assert route.owned_command == command

    delegated = route_args(["runs", "ls", "--repo", "data"])
    assert delegated.route_kind == "passthrough"
    assert delegated.delegated_args == ["runs", "ls", "--repo", "data"]
