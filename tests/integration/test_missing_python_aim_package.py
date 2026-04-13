from __future__ import annotations

import importlib
import sys


def _reload_main_without_python_aim(monkeypatch):
    monkeypatch.setitem(sys.modules, "aim", None)
    monkeypatch.delitem(sys.modules, "aim.sdk", raising=False)
    monkeypatch.delitem(sys.modules, "aim.sdk.types", raising=False)
    sys.modules.pop("aimx.__main__", None)
    sys.modules.pop("aimx.cli", None)
    sys.modules.pop("aimx.commands.query", None)
    return importlib.import_module("aimx.__main__").main


def test_owned_commands_still_work_when_python_aim_package_is_missing(
    capsys, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", "")

    main = _reload_main_without_python_aim(monkeypatch)
    exit_code = main(["version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "aimx 0.1.0" in captured.out


def test_query_reports_actionable_error_when_python_aim_package_is_missing(
    capsys, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PATH", "")

    main = _reload_main_without_python_aim(monkeypatch)
    exit_code = main(
        ["query", "metrics", "metric.name == 'loss'", "--repo", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "requires the Python `aim` package" in captured.err
