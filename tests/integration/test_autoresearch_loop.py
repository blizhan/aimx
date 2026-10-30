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


def test_terminal_follow_up_finding_preserves_original_checkpoint_and_is_visible(
    tmp_path: Path,
) -> None:
    import hashlib
    from aimx.commands.finding import run_finding_command
    from aimx.aim_bridge.research_evidence import make_evidence_resolver
    from aimx.research.models import ResearchUpdate
    from aimx.research.store import apply_update, read_state

    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = repo / ".aimx/autoresearch/terminal-loop/state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({
        "schema_version": 1, "loop_id": "terminal-loop", "status": "budget_exhausted",
        "rounds_started": 10, "config": {"max_rounds": 10, "goal": {"success_criteria": ["original criterion"]}},
        "goal_assessment": [{"criterion": "original criterion", "met": False, "evidence": [], "rationale": "not met"}],
        "rounds": [],
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    original_checkpoint = checkpoint.read_bytes()
    artifact = tmp_path / "later-evaluation.json"
    artifact.write_text('{"window":"extended","gain":0.0}\\n', encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    loop_update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "client_update_id": "register-terminal-loop",
        "author": {"kind": "agent", "name": "lead"},
        "operations": [{"op": "activity.create", "local_id": "loop", "kind": "loop", "loop_id": "terminal-loop", "title": "Terminal loop", "objective": "original criterion", "source": {"kind": "test", "location": str(checkpoint)}, "checkpoint": {"path": str(checkpoint), "repo": str(repo), "loop_id": "terminal-loop"}}],
    })
    registered = apply_update(repo, loop_update)
    loop_activity_id = registered.created_ids["loop"]
    finding_update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 1, "client_update_id": "terminal-follow-up",
        "author": {"kind": "agent", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "follow", "kind": "follow_up", "related_loop_id": "terminal-loop", "title": "Extended window", "objective": "Check generalization", "source": {"kind": "artifact", "location": str(artifact)}},
            {"op": "finding.create", "local_id": "original", "claim": "Candidate gained locally", "source_activity_id": loop_activity_id, "topics": ["candidate"], "result_kind": "improvement", "scope": {"description": "original window", "dataset_scope_ids": [], "conditions": ["original evaluator"]}, "limitations": ["local window only"], "evidence": [{"kind": "local_artifact", "location": str(artifact), "sha256": digest, "size_bytes": artifact.stat().st_size, "evidence_type": "report", "purpose": "original scope", "source_activity_id": loop_activity_id}]},
            {"op": "finding.create", "local_id": "later", "claim": "Extended history shows zero stable added gain", "source_activity_id": "$follow", "topics": ["candidate"], "result_kind": "scope_boundary", "scope": {"description": "extended historical window", "dataset_scope_ids": [], "conditions": ["same evaluator"]}, "limitations": ["does not establish future independent validation"], "evidence": [{"kind": "local_artifact", "location": str(artifact), "sha256": digest, "size_bytes": artifact.stat().st_size, "evidence_type": "report", "purpose": "later scope", "source_activity_id": "$follow"}]},
            {"op": "relation.create", "source": "$later", "target": "$original", "relation_type": "refines", "reason": "Later historical scope limits the local claim", "source_activity_id": "$follow"},
        ],
    })
    update_file = tmp_path / "follow-up.json"
    update_file.write_text(json.dumps(finding_update.as_dict()), encoding="utf-8")
    from aimx.commands.research import run_research_command
    dry_run = run_research_command(["update", "--repo", str(repo), "--file", str(update_file), "--dry-run", "--json"])
    assert dry_run.exit_status == 0
    committed = apply_update(repo, finding_update, make_evidence_resolver(repo))
    replayed = apply_update(repo, finding_update, make_evidence_resolver(repo))
    assert replayed.idempotent and replayed.commit_id == committed.commit_id
    result = run_finding_command(["ls", "--repo", str(repo), "--loop", "terminal-loop", "--topic", "candidate", "--json"])
    payload = json.loads(result.output)
    assert len(payload["findings"]) == 1
    assert payload["follow_up_findings"][0]["result_kind"] == "scope_boundary"
    assert payload["follow_up_findings"][0]["limitations"] == ["does not establish future independent validation"]
    assert read_state(repo).activities[committed.created_ids["follow"]]["related_loop_id"] == "terminal-loop"
    assert checkpoint.read_bytes() == original_checkpoint
