from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.research import run_research_command


def _json(result) -> dict:
    assert result.output is not None
    return json.loads(result.output)


def test_agenda_and_next_contracts_select_persisted_work(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update = tmp_path / "agenda.json"
    update.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy result"},
                    {
                        "op": "agenda.item.create",
                        "local_id": "proposed",
                        "objective": "Test seed 43",
                        "question": "Does the gain persist?",
                        "motivating_finding_ids": ["$f1"],
                        "success_criteria": {"metric": "accuracy"},
                        "priority": 100,
                        "agenda_status": "proposed",
                    },
                    {
                        "op": "agenda.item.create",
                        "local_id": "active",
                        "objective": "Validate the baseline",
                        "motivating_finding_ids": ["$f1"],
                        "success_criteria": "same metric",
                        "priority": 10,
                        "agenda_status": "active",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    committed = _json(
        run_research_command(["update", "--repo", str(repo), "--file", str(update), "--json"])
    )
    assert committed["revision"] == 1

    agenda = _json(run_research_command(["agenda", "--repo", str(repo), "--json"]))
    assert [item["status"] for item in agenda["items"]] == ["active", "proposed"]
    assert agenda["items"][0]["objective"] == "Validate the baseline"
    assert agenda["items"][1]["question"] == "Does the gain persist?"

    selected = _json(run_research_command(["next", "--repo", str(repo), "--json"]))
    assert selected["status"] == "selected"
    assert selected["item"]["objective"] == "Validate the baseline"

    filtered = _json(
        run_research_command(["agenda", "--repo", str(repo), "--status", "proposed", "--json"])
    )
    assert len(filtered["items"]) == 1
    assert filtered["items"][0]["status"] == "proposed"


def test_next_returns_successful_no_work_envelope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = run_research_command(["next", "--repo", str(repo), "--json"])
    payload = _json(result)
    assert result.exit_status == 0
    assert payload["status"] == "no_actionable_agenda"
    assert payload["item"] is None
    assert payload["revision"] == 0
