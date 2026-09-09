from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.frontier import run_frontier_command
from aimx.commands.research import run_research_command


def _payload(result) -> dict:
    assert result.output is not None
    return json.loads(result.output)


def _commit(repo: Path, tmp_path: Path, name: str, payload: dict) -> dict:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return _payload(
        run_research_command(["update", "--repo", str(repo), "--file", str(path), "--json"])
    )


def test_two_round_autoresearch_loop_reuses_shared_state_and_human_steering(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    first = _commit(
        repo,
        tmp_path,
        "round-one",
        {
            "schema_version": 1,
            "base_revision": 0,
            "client_update_id": "round-one",
            "author": {"kind": "agent", "name": "agent-one"},
            "operations": [
                {
                    "op": "finding.create",
                    "local_id": "f1",
                    "claim": "The candidate improves low-data accuracy.",
                },
                {"op": "frontier.lane.create", "local_id": "lane", "name": "promising"},
                {
                    "op": "frontier.item.create",
                    "local_id": "front",
                    "lane_id": "$lane",
                    "finding_id": "$f1",
                    "rationale": "Repeat the low-data comparison under another seed.",
                    "priority": 90,
                },
                {
                    "op": "agenda.item.create",
                    "local_id": "agenda",
                    "objective": "Repeat the low-data comparison",
                    "question": "Does the gain survive a new seed?",
                    "motivating_finding_ids": ["$f1"],
                    "frontier_item_ids": ["$front"],
                    "success_criteria": {"metric": "accuracy"},
                    "priority": 90,
                },
            ],
        },
    )
    assert first["revision"] == 1
    first_context = _payload(
        run_research_command(
            [
                "context",
                "--repo",
                str(repo),
                "--objective",
                "low-data accuracy",
                "--budget",
                "20000",
                "--json",
            ]
        )
    )
    next_item = _payload(run_research_command(["next", "--repo", str(repo), "--json"]))
    agenda_id = next_item["item"]["id"]
    frontier_id = first["created_ids"]["front"]
    assert first_context["revision"] == 1
    assert next_item["status"] == "selected"

    paused = run_frontier_command(
        ["pause", frontier_id, "--reason", "Human requests an additional baseline.", "--repo", str(repo), "--json"]
    )
    assert paused.exit_status == 0
    paused_state = _payload(run_research_command(["state", "--repo", str(repo), "--json"]))
    assert paused_state["revision"] == 2

    second = _commit(
        repo,
        tmp_path,
        "round-two",
        {
            "schema_version": 1,
            "base_revision": 2,
            "client_update_id": "round-two",
            "author": {"kind": "agent", "name": "agent-two"},
            "operations": [
                {
                    "op": "agenda.item.transition",
                    "agenda_item_id": agenda_id,
                    "agenda_status": "active",
                },
                {
                    "op": "agenda.item.transition",
                    "agenda_item_id": agenda_id,
                    "agenda_status": "completed",
                },
                {
                    "op": "finding.create",
                    "local_id": "f2",
                    "claim": "The low-data gain survives the additional baseline.",
                },
                    {
                        "op": "relation.create",
                        "source": "$f2",
                        "target": first["created_ids"]["f1"],
                    "relation_type": "supports",
                    "reason": "Round two reproduced the direction.",
                },
            ],
        },
    )
    assert second["revision"] == 3
    second_context = _payload(
        run_research_command(
            [
                "context",
                "--repo",
                str(repo),
                "--objective",
                "low-data accuracy",
                "--budget",
                "20000",
                "--json",
            ]
        )
    )
    ids = {item["id"] for item in second_context["items"]}
    assert second_context["revision"] == 3
    assert second["created_ids"]["f2"] in ids
    assert frontier_id in ids
    assert next(item for item in second_context["items"] if item["id"] == frontier_id)["data"]["status"] == "paused"
