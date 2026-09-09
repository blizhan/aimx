from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.research import run_research_command


def _write_update(path: Path, *, base_revision: int = 0, claim: str = "accuracy claim") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": base_revision,
                "client_update_id": f"update-{base_revision}-{claim}",
                "author": {"kind": "agent", "name": "contract-agent"},
                "operations": [
                    {
                        "op": "finding.create",
                        "local_id": "f1",
                        "claim": claim,
                        "epistemic_status": "candidate",
                        "confidence": "medium",
                        "governance_status": "proposed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_research_state_empty_and_update_contracts_are_json_stable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    empty = run_research_command(["state", "--repo", str(repo), "--json"])
    empty_payload = json.loads(empty.output)
    assert empty.exit_status == 0
    assert empty_payload["schema_version"] == 1
    assert empty_payload["revision"] == 0
    assert empty_payload["findings"] == []
    assert not (repo / ".aimx").exists()

    update_file = tmp_path / "update.json"
    _write_update(update_file)
    dry_run = run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--dry-run", "--json"]
    )
    dry_payload = json.loads(dry_run.output)
    assert dry_run.exit_status == 0
    assert dry_payload == {
        "base_revision": 0,
        "schema_version": 1,
        "status": "valid",
        "would_create": ["finding"],
    }
    assert not (repo / ".aimx").exists()

    committed = run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
    )
    committed_payload = json.loads(committed.output)
    assert committed.exit_status == 0
    assert committed_payload["status"] == "committed"
    assert committed_payload["revision"] == 1
    assert committed_payload["created_ids"]["f1"].startswith("f_")


def test_invalid_and_stale_updates_have_expected_exit_status_and_error_shape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update_file = tmp_path / "update.json"
    _write_update(update_file)
    assert run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
    ).exit_status == 0

    stale_file = tmp_path / "stale.json"
    _write_update(stale_file, base_revision=0, claim="stale claim")
    stale = run_research_command(
        ["update", "--repo", str(repo), "--file", str(stale_file), "--json"]
    )
    stale_payload = json.loads(stale.error_message)
    assert stale.exit_status == 3
    assert stale_payload["error"]["code"] == "revision_conflict"
    assert stale_payload["error"]["details"] == {"base_revision": 0, "current_revision": 1}

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{\"schema_version\": 1}", encoding="utf-8")
    invalid = run_research_command(
        ["update", "--repo", str(repo), "--file", str(invalid_file), "--json"]
    )
    invalid_payload = json.loads(invalid.error_message)
    assert invalid.exit_status == 2
    assert invalid_payload["error"]["code"] == "invalid_schema"
