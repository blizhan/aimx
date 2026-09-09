from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.research import run_research_command


def test_agenda_lifecycle_and_evidence_remain_traceable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aimx.commands.research.make_evidence_resolver",
        lambda _root: lambda value: value.lower(),
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    first_file = tmp_path / "first.json"
    first_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent-one"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy result"},
                    {
                        "op": "agenda.item.create",
                        "local_id": "a1",
                        "objective": "Repeat accuracy test",
                        "motivating_finding_ids": ["$f1"],
                        "success_criteria": {"metric": "accuracy"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    first = json.loads(
        run_research_command(["update", "--repo", str(repo), "--file", str(first_file), "--json"]).output
    )
    agenda_id = first["created_ids"]["a1"]

    transition_file = tmp_path / "transition.json"
    transition_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 1,
                "author": {"kind": "human", "name": "reviewer"},
                "operations": [
                    {
                        "op": "agenda.item.transition",
                        "agenda_item_id": agenda_id,
                        "agenda_status": "active",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert run_research_command(
        ["update", "--repo", str(repo), "--file", str(transition_file), "--json"]
    ).exit_status == 0

    complete_file = tmp_path / "complete.json"
    complete_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 2,
                "author": {"kind": "agent", "name": "agent-two"},
                "operations": [
                    {
                        "op": "agenda.item.transition",
                        "agenda_item_id": agenda_id,
                        "agenda_status": "completed",
                        "evidence": [{"kind": "aim_run", "run_hash": "b" * 32}],
                    },
                    {
                        "op": "finding.create",
                        "local_id": "result",
                        "claim": "The repeated accuracy test passed.",
                        "evidence": [{"kind": "aim_run", "run_hash": "b" * 32}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert run_research_command(
        ["update", "--repo", str(repo), "--file", str(complete_file), "--json"]
    ).exit_status == 0

    state = json.loads(run_research_command(["state", "--repo", str(repo), "--json"]).output)
    item = next(value for value in state["agenda"]["items"] if value["id"] == agenda_id)
    assert item["status"] == "completed"
    assert item["evidence"][0]["run_hash"] == "b" * 32
    assert len(item["result_commit_ids"]) == 1
    assert any(value["claim"].startswith("The repeated") for value in state["findings"])
