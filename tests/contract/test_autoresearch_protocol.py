from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.research import run_research_command


def test_two_agent_identities_share_product_neutral_json_contract(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    initial = tmp_path / "initial.json"
    initial.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "codex-adapter"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "shared accuracy result"}
                ],
            }
        ),
        encoding="utf-8",
    )
    committed = json.loads(
        run_research_command(["update", "--repo", str(repo), "--file", str(initial), "--json"]).output
    )
    state = json.loads(run_research_command(["state", "--repo", str(repo), "--json"]).output)
    context = json.loads(
        run_research_command(
            ["context", "--repo", str(repo), "--objective", "accuracy", "--budget", "10000", "--json"]
        ).output
    )
    agenda = json.loads(run_research_command(["agenda", "--repo", str(repo), "--json"]).output)

    for payload in (state, context, agenda):
        assert payload["schema_version"] == 1
        assert "product" not in json.dumps(payload).lower()
    assert committed["created_ids"]["f1"] in {item["id"] for item in context["items"]}

    follow_up = tmp_path / "follow-up.json"
    follow_up.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 1,
                "author": {"kind": "agent", "name": "second-adapter"},
                "operations": [
                    {
                        "op": "annotation.create",
                        "annotation_target_kind": "finding",
                        "annotation_target_id": committed["created_ids"]["f1"],
                        "text": "Second adapter can continue from this claim.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_research_command(
        ["update", "--repo", str(repo), "--file", str(follow_up), "--json"]
    )
    assert result.exit_status == 0
    assert json.loads(result.output)["revision"] == 2
