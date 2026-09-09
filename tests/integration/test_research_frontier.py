from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.frontier import run_frontier_command
from aimx.commands.research import run_research_command


def test_frontier_steering_changes_later_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update = tmp_path / "update.json"
    update.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy result"},
                    {"op": "frontier.lane.create", "local_id": "lane", "name": "promising"},
                    {
                        "op": "frontier.item.create",
                        "local_id": "front",
                        "lane_id": "$lane",
                        "finding_id": "$f1",
                        "rationale": "Repeat the accuracy result with another seed.",
                        "priority": 80,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_research_command(["update", "--repo", str(repo), "--file", str(update), "--json"])
    created = json.loads(result.output)
    item_id = created["created_ids"]["front"]

    first_context = json.loads(
        run_research_command(
            ["context", "--repo", str(repo), "--objective", "accuracy", "--budget", "10000", "--json"]
        ).output
    )
    assert item_id in {item["id"] for item in first_context["items"]}

    paused = run_frontier_command(
        ["pause", item_id, "--reason", "Human review required.", "--repo", str(repo), "--json"]
    )
    assert paused.exit_status == 0
    second_context = json.loads(
        run_research_command(
            ["context", "--repo", str(repo), "--objective", "accuracy", "--budget", "10000", "--json"]
        ).output
    )
    frontier_item = next(item for item in second_context["items"] if item["id"] == item_id)
    assert frontier_item["data"]["status"] == "paused"
