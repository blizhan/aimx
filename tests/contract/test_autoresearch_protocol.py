from __future__ import annotations

import json
from pathlib import Path

from aimx.commands.research import run_research_command


def test_protocol_examples_expose_required_failure_policy_and_preflight_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("preflight.example.json", "experiment-contract.example.json", "project-capabilities.example.json"):
        payload = json.loads((root / "skills/aimx-autoresearch/assets" / name).read_text(encoding="utf-8"))
        assert payload
    preflight = json.loads((root / "skills/aimx-autoresearch/assets/preflight.example.json").read_text(encoding="utf-8"))
    assert preflight["failure_policy"] in {"stop_pending", "continue_independent"}
    assert preflight["checks"]["failure_policy"]["status"] in {"available", "missing", "unverified"}
    contract = json.loads((root / "skills/aimx-autoresearch/assets/experiment-contract.example.json").read_text(encoding="utf-8"))
    assert contract["failure_policy"] in {"stop_pending", "continue_independent"}
    assert contract["evaluation"]["argv"] and contract["evaluation"]["preflight_evidence"]
    if contract["failure_policy"] == "continue_independent":
        assert contract["extensions"]["host_failure_policy_verified"] is True
        assert contract["extensions"]["host_failure_policy_preflight_sha256"] == contract["evaluation"]["preflight_evidence"][0]["sha256"]


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
