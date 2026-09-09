from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.frontier import run_frontier_command
from aimx.commands.research import run_research_command


def _payload(result) -> dict:
    assert result.output is not None
    return json.loads(result.output)


def test_frontier_commands_have_stable_json_and_persist_human_steering(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update = tmp_path / "finding.json"
    update.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy claim"}
                ],
            }
        ),
        encoding="utf-8",
    )
    finding = _payload(
        run_research_command(["update", "--repo", str(repo), "--file", str(update), "--json"])
    )["created_ids"]["f1"]

    lane_one = _payload(
        run_frontier_command(["lane-add", "promising", "--repo", str(repo), "--json"])
    )
    lane_one_id = lane_one["created_ids"]["lane"]
    lane_two = _payload(
        run_frontier_command(["lane-add", "validate", "--repo", str(repo), "--json"])
    )
    lane_two_id = lane_two["created_ids"]["lane"]

    item_result = run_frontier_command(
        [
            "add",
            "--lane",
            lane_one_id,
            "--finding",
            finding,
            "--rationale",
            "Test the observed gain.",
            "--priority",
            "80",
            "--repo",
            str(repo),
            "--json",
        ]
    )
    item_id = _payload(item_result)["created_ids"]["frontier"]
    assert item_result.exit_status == 0

    moved = run_frontier_command(
        ["move", item_id, "--lane", lane_two_id, "--repo", str(repo), "--json"]
    )
    assert moved.exit_status == 0
    paused = run_frontier_command(
        ["pause", item_id, "--reason", "Need a baseline.", "--repo", str(repo), "--json"]
    )
    assert paused.exit_status == 0
    retired = run_frontier_command(
        ["retire", item_id, "--reason", "Direction is complete.", "--repo", str(repo), "--json"]
    )
    assert retired.exit_status == 0

    shown = _payload(run_frontier_command(["show", "--repo", str(repo), "--json"]))
    assert shown["revision"] == 7
    item = next(value for value in shown["frontier"]["items"] if value["id"] == item_id)
    assert item["lane_id"] == lane_two_id
    assert item["status"] == "retired"
    assert item["priority"] == 80


def test_frontier_invalid_lane_and_revision_conflicts_are_machine_readable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    invalid = run_frontier_command(
        [
            "add",
            "--lane",
            "missing",
            "--direction",
            "try something",
            "--rationale",
            "because",
            "--repo",
            str(repo),
            "--json",
        ]
    )
    assert invalid.exit_status == 2
    assert json.loads(invalid.error_message)["error"]["code"] == "unknown_entity"
