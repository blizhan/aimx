from __future__ import annotations

import importlib
import sys

from aimx import __version__


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
    assert f"aimx {__version__}" in captured.out


def test_research_file_help_and_state_work_without_python_aim_package(
    capsys, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PATH", "")
    main = _reload_main_without_python_aim(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()

    assert main(["research", "evidence", "check", "--help", "--repo", "/missing/path"]) == 0
    help_output = capsys.readouterr()
    assert "Help does not read the repository" in help_output.out
    assert main(["research", "state", "--repo", str(repo), "--json"]) == 0
    state_output = capsys.readouterr()
    assert '"revision": 0' in state_output.out
    assert not (repo / ".aimx").exists()


def test_research_local_artifact_check_works_without_python_aim_package(
    capsys, monkeypatch, tmp_path
) -> None:
    import hashlib
    import json

    monkeypatch.setenv("PATH", "")
    main = _reload_main_without_python_aim(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = repo / "report.json"
    artifact.write_text('{"result": "fixture"}\\n', encoding="utf-8")
    content = artifact.read_bytes()
    update = {
        "schema_version": 1,
        "base_revision": 0,
        "client_update_id": "file-only-finding",
        "author": {"kind": "human", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "activity", "kind": "analysis", "title": "File analysis", "objective": "Check local evidence", "source": {"kind": "fixture", "location": str(artifact)}},
            {"op": "finding.create", "local_id": "finding", "claim": "File backed fixture result", "source_activity_id": "$activity", "topics": ["fixture"], "result_kind": "insufficient_evidence", "scope": {"description": "fixture window", "dataset_scope_ids": [], "conditions": ["local evaluator"]}, "limitations": ["no Aim run"], "evidence": [{"kind": "local_artifact", "location": str(artifact), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content), "evidence_type": "evaluation_report", "purpose": "local-only test evidence", "source_activity_id": "$activity"}]},
        ],
    }
    update_path = tmp_path / "update.json"
    update_path.write_text(json.dumps(update), encoding="utf-8")
    assert main(["research", "update", "--repo", str(repo), "--file", str(update_path), "--dry-run", "--json"]) == 0
    capsys.readouterr()
    assert not (repo / ".aimx").exists()
    assert main(["research", "update", "--repo", str(repo), "--file", str(update_path), "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    finding_id = receipt["created_ids"]["finding"]
    assert main(["research", "evidence", "check", "--finding", finding_id, "--repo", str(repo), "--json"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["references"][0]["identity_status"] == "matches"
    assert not any(path.name == ".aim" for path in repo.iterdir())


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
