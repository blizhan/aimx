from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aimx.commands.finding import run_finding_command
from aimx.commands.lineage import run_lineage_command
from aimx.commands.research import run_research_command
from aimx.research.models import ResearchUpdate
from aimx.research.store import apply_update, read_state


def _json_output(result) -> dict:
    assert result.output is not None
    return json.loads(result.output)


def test_evidence_check_distinguishes_missing_changed_unreadable_and_sdk_missing(tmp_path: Path, monkeypatch) -> None:
    import aimx.aim_bridge.research_evidence as bridge
    import aimx.commands.research as research
    from aimx.aim_bridge.research_evidence import EvidenceResolver
    from aimx.research.models import ResearchState

    root = tmp_path / "repo"
    root.mkdir()
    artifact = root / "result.json"
    artifact.write_text("original", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    finding = {"id": "f1", "evidence": [{"kind": "local_artifact", "location": "result.json", "sha256": digest, "size_bytes": artifact.stat().st_size}, {"kind": "local_artifact", "location": "missing.json", "sha256": "0" * 64, "size_bytes": 1}]}
    state = ResearchState(findings={"f1": finding})
    monkeypatch.setattr(research, "read_state", lambda _root: state)
    rows = research.check_finding_evidence(finding, root, state)
    assert rows[0]["availability"] == "available" and rows[0]["identity_status"] == "matches"
    assert rows[1]["availability"] == "missing" and rows[1]["identity_status"] == "unknown"
    artifact.write_text("changed", encoding="utf-8")
    changed = EvidenceResolver(root).resolve_artifact("result.json")
    assert changed[1] != digest
    finding["evidence"] = [finding["evidence"][0]]
    state.findings["f1"] = finding
    diagnosis = research.check_finding_evidence(finding, root, state)
    assert diagnosis[0]["availability"] == "available"
    assert diagnosis[0]["identity_status"] == "changed"
    assert diagnosis[0]["scientific_review"] == "not_reviewed"

    monkeypatch.setattr(bridge, "Repo", None)
    aim_finding = {"id": "f2", "evidence": [{"kind": "aim_run", "run_hash": "a" * 32}]}
    result = research.check_finding_evidence(aim_finding, root, ResearchState())
    assert result[0]["availability"] == "unsupported"


def test_file_finding_topic_show_and_evidence_check_preserve_candidate_semantics(tmp_path: Path) -> None:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    artifact = repo / "evaluation.json"
    artifact.write_text('{"delta":0.01,"interval":[-0.02,0.04]}\\n', encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "client_update_id": "interval-crosses-zero",
        "author": {"kind": "human", "name": "lead"},
        "operations": [{"op": "activity.create", "local_id": "activity", "kind": "analysis", "title": "Interval review", "objective": "Assess gain", "source": {"kind": "report", "location": str(artifact)}},
            {"op": "finding.create", "local_id": "finding", "claim": "The interval crosses zero, so improvement is not established", "source_activity_id": "$activity", "topics": ["candidate", "accuracy"], "result_kind": "insufficient_evidence", "scope": {"description": "fixed candidate and sample window", "dataset_scope_ids": [], "conditions": ["same evaluator"]}, "limitations": ["does not establish no effect"], "evidence": [{"kind": "local_artifact", "location": str(artifact), "sha256": digest, "size_bytes": artifact.stat().st_size, "evidence_type": "evaluation_report", "purpose": "interval estimate", "source_activity_id": "$activity"}]}],
    })
    update_path = tmp_path / "structured-finding.json"
    update_path.write_text(json.dumps(update.as_dict()), encoding="utf-8")
    dry_run = run_research_command(["update", "--repo", str(repo), "--file", str(update_path), "--dry-run", "--json"])
    assert dry_run.exit_status == 0
    committed = run_research_command(["update", "--repo", str(repo), "--file", str(update_path), "--json"])
    ids = json.loads(committed.output)["created_ids"]
    topic = _json_output(run_finding_command(["ls", "--repo", str(repo), "--topic", "accuracy", "--json"]))
    assert [row["id"] for row in topic["findings"]] == [ids["finding"]]
    assert topic["match_reasons"][ids["finding"]] == ["accuracy"]
    show = _json_output(run_finding_command(["show", ids["finding"], "--repo", str(repo), "--json"]))
    assert show["finding"]["result_kind"] == "insufficient_evidence"
    assert show["finding"]["epistemic_status"] == "candidate"
    check = _json_output(run_research_command(["evidence", "check", "--finding", ids["finding"], "--repo", str(repo), "--json"]))
    assert check["references"][0]["identity_status"] == "matches"
    assert check["references"][0]["scientific_review"] == "not_reviewed"
    review_update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 1, "author": {"kind": "human", "name": "reviewer"},
        "operations": [{"op": "evidence.review", "finding_id": ids["finding"], "evidence_fingerprint": check["references"][0]["evidence_fingerprint"], "scientific_review": "inconclusive", "reason": "Interval crosses zero", "basis": [{"kind": "local_artifact", "location": str(artifact), "sha256": digest, "size_bytes": artifact.stat().st_size, "evidence_type": "evaluation_report", "purpose": "review basis", "source_activity_id": ids["activity"]}], "source_activity_id": ids["activity"]}],
    })
    from aimx.aim_bridge.research_evidence import make_evidence_resolver
    apply_update(repo, review_update, make_evidence_resolver(repo))
    reviewed = _json_output(run_research_command(["evidence", "check", "--finding", ids["finding"], "--repo", str(repo), "--json"]))
    assert reviewed["references"][0]["scientific_review"] == "inconclusive"
    artifact.unlink()
    deleted = _json_output(run_research_command(["evidence", "check", "--finding", ids["finding"], "--repo", str(repo), "--json"]))
    assert deleted["references"][0]["availability"] == "missing"
    assert deleted["references"][0]["scientific_review"] == "inconclusive"
    persisted = read_state(repo).findings[ids["finding"]]
    assert persisted["evidence"][0]["sha256"] == digest and read_state(repo).revision == 2
    assert len(read_state(repo).evidence_reviews[ids["finding"]]) == 1


@pytest.mark.parametrize(
    ("result_kind", "claim"),
    [
        ("improvement", "Candidate improves in the fixed evaluation window"),
        ("degradation", "Candidate degrades in the fixed evaluation window"),
        ("insufficient_evidence", "Interval crosses zero; improvement is not established"),
        ("scope_boundary", "The result applies only to the declared evaluation window"),
    ],
)
def test_all_file_backed_result_kinds_persist_and_retrieve(
    tmp_path: Path, result_kind: str, claim: str,
) -> None:
    repo = tmp_path / result_kind
    repo.mkdir()
    artifact = repo / "evaluation.json"
    artifact.write_text(json.dumps({"result_kind": result_kind, "claim": claim}) + "\\n", encoding="utf-8")
    content = artifact.read_bytes()
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "client_update_id": f"result-{result_kind}",
        "author": {"kind": "human", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "activity", "kind": "analysis", "title": "Evaluation", "objective": "Record result", "source": {"kind": "report", "location": str(artifact)}},
            {"op": "finding.create", "local_id": "finding", "claim": claim, "source_activity_id": "$activity", "topics": ["candidate"], "result_kind": result_kind, "scope": {"description": "fixed evaluation window", "dataset_scope_ids": [], "conditions": ["same evaluator"]}, "limitations": ["single fixture evaluation"], "evidence": [{"kind": "local_artifact", "location": str(artifact), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content), "evidence_type": "evaluation_report", "purpose": f"evidence for {result_kind}", "source_activity_id": "$activity"}]},
        ],
    })
    from aimx.aim_bridge.research_evidence import make_evidence_resolver

    receipt = apply_update(repo, update, make_evidence_resolver(repo))
    finding_id = receipt.created_ids["finding"]
    listed = _json_output(run_finding_command(["ls", "--repo", str(repo), "--topic", "candidate", "--json"]))
    assert [item["id"] for item in listed["findings"]] == [finding_id]
    shown = _json_output(run_finding_command(["show", finding_id, "--repo", str(repo), "--json"]))
    assert shown["finding"]["result_kind"] == result_kind
    assert shown["finding"]["scope"]["description"] == "fixed evaluation window"
    checked = _json_output(run_research_command(["evidence", "check", "--finding", finding_id, "--repo", str(repo), "--json"]))
    assert checked["references"][0]["identity_status"] == "matches"
    assert checked["references"][0]["scientific_review"] == "not_reviewed"


def test_findings_lineage_and_annotations_survive_a_second_process(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    aim_dir = repo / ".aim"
    aim_dir.mkdir()
    sentinel = aim_dir / "sentinel"
    sentinel.write_text("aim-data", encoding="utf-8")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    update = {
        "schema_version": 1,
        "base_revision": 0,
        "author": {"kind": "agent", "name": "round-one"},
        "operations": [
            {
                "op": "finding.create",
                "local_id": "f1",
                "claim": "The candidate improves accuracy on low data.",
            },
            {
                "op": "finding.create",
                "local_id": "f2",
                "claim": "The improvement needs a second seed.",
            },
            {
                "op": "relation.create",
                "local_id": "r1",
                "source": "$f2",
                "target": "$f1",
                "relation_type": "tests",
            },
        ],
    }
    update_file = tmp_path / "update.json"
    update_file.write_text(json.dumps(update), encoding="utf-8")
    first = run_research_command(
        ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
    )
    first_payload = _json_output(first)
    finding_id = first_payload["created_ids"]["f1"]
    second_finding_id = first_payload["created_ids"]["f2"]
    relation_id = first_payload["created_ids"]["r1"]

    assert run_finding_command(
        ["comment", finding_id, "review", "the", "seed", "--repo", str(repo), "--json"]
    ).exit_status == 0
    assert run_finding_command(
        ["accept", finding_id, "--reason", "Useful working result", "--repo", str(repo), "--json"]
    ).exit_status == 0
    assert run_finding_command(
        [
            "assess",
            finding_id,
            "--status",
            "validated",
            "--confidence",
            "high",
            "--repo",
            str(repo),
            "--json",
        ]
    ).exit_status == 0

    finding_list = _json_output(run_finding_command(["ls", "--repo", str(repo), "--json"]))
    assert {item["id"] for item in finding_list["findings"]} == {finding_id, second_finding_id}
    assert next(item for item in finding_list["findings"] if item["id"] == finding_id)[
        "governance_status"
    ] == "accepted"

    lineage = _json_output(
        run_lineage_command(["show", finding_id, "--repo", str(repo), "--json"])
    )
    assert relation_id in {item["id"] for item in lineage["relations"]}

    after = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    assert before == after


def test_finding_shorthand_rejects_unknown_options_without_committing(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    finding_id = _commit_finding_with_evidence(repo, "a" * 32)

    result = run_finding_command(
        ["accept", finding_id, "--reasn", "typo", "--repo", str(repo), "--json"]
    )

    assert result.exit_status == 2
    assert read_state(repo).revision == 1


def test_research_context_reads_durable_annotations(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update_file = tmp_path / "update.json"
    update_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy result"},
                    {
                        "op": "annotation.create",
                        "annotation_target_kind": "finding",
                        "annotation_target_id": "$f1",
                        "text": "accuracy needs a controlled follow-up",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    committed = _json_output(
        run_research_command(
            ["update", "--repo", str(repo), "--file", str(update_file), "--json"]
        )
    )
    assert committed["revision"] == 1
    context = _json_output(
        run_research_command(
            [
                "context",
                "--repo",
                str(repo),
                "--objective",
                "accuracy follow-up",
                "--budget",
                "10000",
                "--json",
            ]
        )
    )
    assert context["revision"] == 1
    assert any(item["kind"] == "annotation" for item in context["items"])


def test_research_help_overview_annotations_and_receipt_are_read_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    for argv in (
        ["overview", "--repo", str(repo), "--help"],
        ["overview", "--repo", str(repo), "--help", "--json"],
        ["evidence", "check", "--repo", "/missing/path", "--help"],
        ["receipt", "--help"],
    ):
        result = run_research_command(argv)
        assert result.exit_status == 0
        assert result.output
    assert not (repo / ".aimx").exists()

    overview = _json_output(run_research_command(["overview", "--repo", str(repo), "--json"]))
    assert overview["loops"] == []
    from aimx.commands.help import render_command_help
    help_targets = [
        "research", "research state", "research update", "research context", "research agenda", "research next", "research overview", "research annotations", "research receipt", "research evidence", "research evidence check",
        "finding", "finding ls", "finding show", "finding comment", "finding accept", "finding reject", "finding assess",
        "lineage", "lineage show", "lineage link", "lineage retract",
        "frontier", "frontier show", "frontier lane-add", "frontier add", "frontier move", "frontier pause", "frontier retire",
    ]
    for target in help_targets:
        rendered = render_command_help(target, output_json=True)
        payload = json.loads(rendered)
        assert payload["command"] == target
        assert payload["description"] and payload["usage"] and payload["examples"] and payload["exit_codes"]
        assert isinstance(payload["mutation"], bool)
        for option in payload["options"]:
            for flag in (token for token in option.split() if token.startswith("-")):
                assert flag in payload["usage"] or flag in {"-h", "--help"}, (target, option, flag)
    annotations = _json_output(run_research_command(["annotations", "--repo", str(repo), "--json"]))
    assert annotations["annotations"] == []
    missing = _json_output(run_research_command(["receipt", "--client-update-id", "not-here", "--repo", str(repo), "--json"]))
    assert missing["status"] == "not_found"
    assert not (repo / ".aimx").exists()


def test_receipt_fixture_scenarios_are_discoverable_without_cross_loop_mixing(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/research_knowledge_loop/loop-receipts.json"
    scenarios = json.loads(fixture.read_text(encoding="utf-8"))["scenarios"]

    for scenario in scenarios:
        repo = tmp_path / scenario["name"]
        repo.mkdir()
        loop_ids = [scenario["loop_id"]]
        if scenario["name"] == "multiple_loops":
            loop_ids = ["loop-one", scenario["loop_id"]]
        operations = []
        checkpoints = {}
        annotation_counts = {}
        for index, loop_id in enumerate(loop_ids):
            checkpoint = repo / ".aimx" / "autoresearch" / loop_id / "state.json"
            checkpoint.parent.mkdir(parents=True)
            report = repo / f"{loop_id}-report.json"
            report.write_text(json.dumps({"loop_id": loop_id, "status": scenario["loop_status"]}), encoding="utf-8")
            assessment = ([{"criterion": "fixture goal", "met": scenario["goal_status"] == "met"}]
                          if scenario["goal_status"] in {"met", "not_met"} else [])
            rounds = [{"round": number, "research_update": {"client_update_id": f"{loop_id}-r{number}", "commit_id": f"commit-{loop_id}-{number}", "revision": number}}
                      for number in range(1, 11)]
            checkpoint.write_text(json.dumps({
                "schema_version": 1, "loop_id": loop_id, "status": scenario["loop_status"],
                "rounds_started": 10, "stop_reason": "fixture acceptance stop",
                "config": {"max_rounds": 10, "goal": {"success_criteria": ["fixture goal"]}},
                "goal_assessment": assessment, "rounds": rounds,
            }), encoding="utf-8")
            checkpoints[loop_id] = checkpoint
            activity_local_id = f"activity_{index}"
            annotation_count = scenario["annotations"] if len(loop_ids) == 1 else (1 if index == 1 else 2)
            annotation_counts[loop_id] = annotation_count
            operations.extend([
                {"op": "activity.create", "local_id": activity_local_id, "kind": "loop",
                 "loop_id": loop_id, "title": loop_id, "objective": "fixture objective",
                 "source": {"kind": "fixture", "location": str(checkpoint)},
                 "report_locations": [str(report)], "artifact_locations": [str(report)],
                 "checkpoint": {"path": str(checkpoint), "repo": str(repo), "loop_id": loop_id}},
                *[{"op": "annotation.create", "source_activity_id": f"${activity_local_id}",
                   "text": f"{loop_id} persisted fixture annotation {annotation_index + 1}"}
                  for annotation_index in range(annotation_count)],
            ])
        update = ResearchUpdate.from_dict({
            "schema_version": 1, "base_revision": 0,
            "client_update_id": f"fixture-{scenario['name']}",
            "author": {"kind": "human", "name": "acceptance"}, "operations": operations,
        })
        apply_update(repo, update)
        store_path = repo / ".aimx" / "research" / "state.sqlite3"
        protected_bytes = {path: path.read_bytes() for path in [store_path, *checkpoints.values()]}
        started = __import__("time").monotonic()
        result = run_research_command(["overview", "--repo", str(repo), "--loop", scenario["loop_id"], "--json"])
        elapsed = __import__("time").monotonic() - started
        assert elapsed < 120
        assert result.exit_status == 0, result.error_message
        payload = json.loads(result.output)
        assert payload["repo"] == str(repo.resolve())
        assert payload["loop_id"] == scenario["loop_id"]
        assert payload["loop_status"] == scenario["loop_status"]
        assert payload["goal_status"] == scenario["goal_status"]
        assert payload["rounds_started"] == payload["max_rounds"] == 10
        assert payload["counts"]["annotations"] == annotation_counts[scenario["loop_id"]]
        assert payload["latest_persisted_receipt"]["client_update_id"] == f"{scenario['loop_id']}-r10"
        assert payload["report_locations"] and payload["external_artifacts"]
        assert payload["query_commands"]
        assert {path: path.read_bytes() for path in protected_bytes} == protected_bytes
        all_loops = json.loads(run_research_command(["overview", "--repo", str(repo), "--json"]).output)
        assert {item["loop_id"] for item in all_loops["loops"]} == set(loop_ids)
        by_loop = {item["loop_id"]: item for item in all_loops["loops"]}
        assert by_loop[scenario["loop_id"]]["annotation_count"] == annotation_counts[scenario["loop_id"]]
        assert by_loop[scenario["loop_id"]]["finding_count"] == 0
        assert all_loops["counts"]["findings"] == 0
        assert all_loops["counts"]["annotations"] == sum(annotation_counts.values())
        human = run_research_command(["overview", "--repo", str(repo)])
        assert "findings=0" in human.output
        assert f"annotations={annotation_counts[scenario['loop_id']]}" in human.output
        if scenario["name"] == "multiple_loops":
            assert by_loop["loop-one"]["annotation_count"] == annotation_counts["loop-one"]
            assert by_loop["loop-two"]["annotation_count"] == annotation_counts["loop-two"]


def test_unknown_repository_is_not_reported_as_empty_overview(tmp_path: Path) -> None:
    result = run_research_command(["overview", "--repo", str(tmp_path / "missing"), "--json"])
    assert result.exit_status == 2
    assert "does not exist" in result.error_message


def test_terminal_follow_up_is_visible_without_changing_checkpoint(tmp_path: Path) -> None:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    checkpoint = repo / ".aimx" / "autoresearch" / "loop-1" / "state.json"
    checkpoint.parent.mkdir(parents=True)
    original_checkpoint = {"schema_version": 1, "loop_id": "loop-1", "status": "budget_exhausted", "rounds_started": 10, "config": {"max_rounds": 10, "goal": {"success_criteria": ["original criterion"]}}, "goal_assessment": [{"criterion": "original criterion", "met": False, "evidence": [], "rationale": "not met"}], "rounds": []}
    checkpoint.write_text(json.dumps(original_checkpoint, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    checkpoint_bytes = checkpoint.read_bytes()
    seed = checkpoint.parent / "report.json"
    seed.write_text('{"result":"later window"}', encoding="utf-8")
    digest = hashlib.sha256(seed.read_bytes()).hexdigest()
    update = {
        "schema_version": 1, "base_revision": 0, "client_update_id": "follow-up-1",
        "author": {"kind": "human", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "loop", "kind": "loop", "loop_id": "loop-1", "title": "Original loop", "objective": "Original criterion", "source": {"kind": "checkpoint", "location": str(checkpoint)}, "checkpoint": {"path": str(checkpoint), "repo": str(repo), "loop_id": "loop-1"}},
            {"op": "activity.create", "local_id": "follow", "kind": "follow_up", "related_loop_id": "loop-1", "title": "Later evaluation", "objective": "Check generalization", "source": {"kind": "report", "location": str(seed)}},
            {"op": "finding.create", "local_id": "original", "claim": "Candidate improved the measured window", "source_activity_id": "$loop", "result_kind": "improvement", "topics": ["candidate"], "scope": {"description": "original measured window", "dataset_scope_ids": [], "conditions": ["original evaluator"]}, "limitations": ["limited horizon"], "evidence": [{"kind": "local_artifact", "location": str(seed), "sha256": digest, "size_bytes": seed.stat().st_size, "evidence_type": "evaluation_report", "purpose": "original result", "source_activity_id": "$loop"}]},
            {"op": "finding.create", "local_id": "later", "claim": "Longer history shows no stable added gain", "source_activity_id": "$follow", "result_kind": "scope_boundary", "topics": ["candidate"], "scope": {"description": "extended historical window", "dataset_scope_ids": [], "conditions": ["later evaluator"]}, "limitations": ["not future independent validation"], "evidence": [{"kind": "local_artifact", "location": str(seed), "sha256": digest, "size_bytes": seed.stat().st_size, "evidence_type": "evaluation_report", "purpose": "later scope", "source_activity_id": "$follow"}]},
            {"op": "relation.create", "source": "$later", "target": "$original", "relation_type": "refines", "reason": "Later evidence limits the scope of the original result", "source_activity_id": "$follow"},
        ],
    }
    update_path = tmp_path / "follow-up.json"
    update_path.write_text(json.dumps(update), encoding="utf-8")
    result = run_research_command(["update", "--repo", str(repo), "--file", str(update_path), "--json"])
    assert result.exit_status == 0, result.error_message
    loop_result = _json_output(run_finding_command(["ls", "--repo", str(repo), "--loop", "loop-1", "--topic", "candidate", "--json"]))
    assert {item["id"] for item in loop_result["findings"]} == {result and json.loads(result.output)["created_ids"]["original"]}
    assert loop_result["follow_up_findings"][0]["id"] == json.loads(result.output)["created_ids"]["later"]
    later_show = _json_output(run_finding_command(["show", json.loads(result.output)["created_ids"]["later"], "--repo", str(repo), "--json"]))
    assert later_show["related_findings"][0]["relation"]["type"] == "refines"
    context = _json_output(run_research_command(["context", "--repo", str(repo), "--loop", "loop-1", "--objective", "candidate", "--budget", "20000", "--json"]))
    assert result and {item["id"] for item in context["items"] if item["kind"] == "finding"} == {json.loads(result.output)["created_ids"]["original"], json.loads(result.output)["created_ids"]["later"]}
    receipt = _json_output(run_research_command(["overview", "--repo", str(repo), "--loop", "loop-1", "--json"]))
    assert receipt["loop_status"] == "budget_exhausted"
    assert receipt["follow_up_activities"][0]["findings"][0]["result_kind"] == "scope_boundary"
    assert receipt["follow_up_activities"][0]["artifact_locations"] == []
    assert receipt["rounds_started"] == receipt["max_rounds"] == 10
    assert receipt["goal_status"] == "not_met"
    assert receipt["original_criteria"] == ["original criterion"]
    assert checkpoint.read_bytes() == checkpoint_bytes


def test_overview_keeps_checkpoint_and_research_attempt_axes_separate(tmp_path: Path) -> None:
    from aimx.research.overview import compile_overview
    from aimx.research.models import ResearchState

    attempt = {"id": "attempt-1", "source_activity_id": "act-loop", "attempt_key": "native-a", "execution_status": "failed", "artifact_status": "complete", "evaluation_status": "passed", "audit_status": "not_run", "resources": [{"kind": "calls", "unit": "count", "planned": 1, "actual": 2, "verification": "verified", "enforcement": "declared_only", "source": "executor log"}], "requested_settings": {"model": "requested"}, "actual_settings": {"model": "actual"}, "round": 1, "slot_id": "slot-a"}
    state = ResearchState(activities={"act-loop": {"id": "act-loop", "kind": "loop", "loop_id": "loop-axes", "report_locations": [], "artifact_locations": []}}, outcomes={"attempt-1": attempt})
    checkpoint = {"loop_id": "loop-axes", "status": "budget_exhausted", "rounds_started": 1, "config": {"max_rounds": 1, "goal": {"success_criteria": ["criterion"]}}, "goal_assessment": [{"criterion": "criterion", "met": False}], "rounds": [{"round": 1, "agents": [{"id": "slot-a", "dispatch_status": "finished", "result": {"status": "succeeded", "summary": "Prediction complete"}}]}]}
    repo = tmp_path / "repo"
    checkpoint_path = repo / ".aimx/autoresearch/loop-axes/state.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    payload = compile_overview(repo, state, "loop-axes")
    assert payload["attempts"]["axes"]["execution_status"] == {"failed": 1}
    assert payload["attempts"]["axes"]["evaluation_status"] == {"passed": 1}
    assert payload["attempts"]["resource_summary"]["over_plan"][0]["actual"] == 2
    assert payload["attempts"]["resource_summary"]["not_enforced"][0]["enforcement"] == "declared_only"
    assert payload["attempts"]["actual_settings"][0]["settings"] == {"model": "actual"}
    assert payload["outcome_differences"][0]["checkpoint_status"] == "succeeded"
    assert payload["outcome_differences"][0]["research_axes"]["execution_status"] == "failed"
    assert payload["outcome_differences"][0]["research_axes"]["evaluation_status"] == "passed"
    assert payload["goal_status"] == "not_met"
    from aimx.rendering.research_views import render_overview_human
    human = render_overview_human(payload)
    assert "execution_status" in human and "evaluation_status" in human
    assert "requested settings" in human.casefold() and "actual settings" in human.casefold()
    assert "enforcement=declared_only" in human


def test_registered_corrupt_loop_still_returns_partial_overview(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = repo / ".aimx/autoresearch/loop-bad/state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"schema_version": 1, "loop_id": "loop-bad", "status": "interrupted"}), encoding="utf-8")
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "lead"},
        "operations": [{"op": "activity.create", "kind": "loop", "loop_id": "loop-bad", "title": "Bad checkpoint", "objective": "test", "source": {"kind": "test", "location": "fixture"}, "checkpoint": {"path": str(checkpoint), "repo": str(repo), "loop_id": "loop-bad"}}],
    })
    apply_update(repo, update)
    checkpoint.write_text("{bad json", encoding="utf-8")
    before = checkpoint.read_bytes()
    result = run_research_command(["overview", "--repo", str(repo), "--loop", "loop-bad", "--json"])
    payload = json.loads(result.error_message)
    assert result.exit_status == 2 and payload["status"] == "partial"
    assert payload["checkpoint_status"] == "unreadable"
    assert "checkpoint unavailable" in payload["pending_reconciliation"][0]
    assert checkpoint.read_bytes() == before


def test_overview_handles_malformed_checkpoint_artifact_sections_without_crashing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = repo / ".aimx/autoresearch/loop-malformed/state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"schema_version": 1, "loop_id": "loop-malformed", "status": "interrupted", "rounds": [{"artifacts": "bad", "agents": {"bad": True}}]}), encoding="utf-8")
    from aimx.research.overview import compile_overview
    payload = compile_overview(repo, None)
    assert payload["status"] == "complete"
    row = payload["loops"][0]
    assert row["checkpoint_status"] == "readable"
    assert row["loop_status"] == "interrupted"


def test_overview_handles_corrupt_checkpoint_partial_state_and_unregistered_files(tmp_path: Path) -> None:
    from aimx.commands.research import _run_overview
    from aimx.research.models import ResearchUpdate

    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = repo / ".aimx/autoresearch/loop-bad/state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"schema_version": 1, "loop_id": "loop-bad", "status": "interrupted"}), encoding="utf-8")
    unregistered = repo / ".aimx/autoresearch/unregistered/state.json"
    unregistered.parent.mkdir(parents=True)
    unregistered.write_text(json.dumps({"schema_version": 1, "loop_id": "unregistered", "status": "interrupted", "rounds_started": 1}), encoding="utf-8")
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "lead"},
        "operations": [{"op": "activity.create", "kind": "loop", "loop_id": "loop-bad", "title": "Bad checkpoint", "objective": "test", "source": {"kind": "test", "location": "fixture"}, "checkpoint": {"path": str(checkpoint), "repo": str(repo), "loop_id": "loop-bad"}}],
    })
    apply_update(repo, update)
    checkpoint.write_text("{bad json", encoding="utf-8")
    before = checkpoint.read_bytes()
    result = _run_overview(["--repo", str(repo), "--json"])
    payload = json.loads(result.error_message)
    assert result.exit_status == 2 and payload["status"] == "partial"
    assert any(row["loop_id"] == "loop-bad" and row["checkpoint_status"] == "unreadable" for row in payload["loops"])
    assert any(row["loop_id"] == "unregistered" and row["registered"] is False for row in payload["loops"])
    assert checkpoint.read_bytes() == before


def test_receipt_query_fingerprint_conflict_is_reported(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update = {"schema_version": 1, "base_revision": 0, "client_update_id": "key-1", "author": {"kind": "agent", "name": "agent"}, "operations": [{"op": "finding.create", "local_id": "f", "claim": "claim"}]}
    update_path = tmp_path / "update.json"
    update_path.write_text(json.dumps(update), encoding="utf-8")
    committed = _json_output(run_research_command(["update", "--repo", str(repo), "--file", str(update_path), "--json"]))
    from aimx.research.store import _request_fingerprint
    fingerprint = _request_fingerprint(ResearchUpdate.from_dict(update))
    found = _json_output(run_research_command(["receipt", "--repo", str(repo), "--client-update-id", "key-1", "--fingerprint", fingerprint, "--json"]))
    assert found["commit_id"] == committed["commit_id"]
    conflict = run_research_command(["receipt", "--repo", str(repo), "--client-update-id", "key-1", "--fingerprint", "0" * 64, "--json"])
    assert conflict.exit_status == 2
    assert json.loads(conflict.error_message)["error"]["code"] == "idempotency_conflict"


def test_finding_empty_list_reports_unattributed_annotations_and_other_records(tmp_path: Path) -> None:
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "human", "name": "lead"},
        "operations": [
            {"op": "annotation.create", "text": "historical conclusion remains useful"},
            {"op": "finding.create", "local_id": "other", "claim": "deployment notes"},
        ],
    })
    apply_update(repo, update)
    payload = _json_output(run_finding_command(["ls", "--repo", str(repo), "--topic", "unmatched-topic", "--json"]))
    assert payload["findings"] == []
    assert payload["other_record_counts"]["annotations"] == 1
    assert payload["unattributed"]["annotations"] == 1
    assert "annotations --repo" in payload["query_commands"][0]
    human = run_finding_command(["ls", "--repo", str(repo), "--topic", "unmatched-topic"])
    assert "1 other research records" in human.output
    assert str(repo) in human.output


def test_attempt_outcome_does_not_change_checkpoint_finished_result(tmp_path: Path) -> None:
    from aimx.research.models import ResearchState
    from aimx.research.overview import compile_overview
    from aimx.rendering.research_views import render_overview_human

    repo = tmp_path / "repo"
    checkpoint = repo / ".aimx/autoresearch/loop-1/state.json"
    checkpoint.parent.mkdir(parents=True)
    original_result = {"status": "failed", "summary": "Service failed", "evidence": ["prediction.json"]}
    checkpoint.write_text(json.dumps({"schema_version": 1, "loop_id": "loop-1", "status": "budget_exhausted", "rounds_started": 1, "config": {"max_rounds": 1}, "rounds": [{"round": 1, "agents": [{"id": "worker", "dispatch_status": "finished", "result": original_result}]}]}), encoding="utf-8")
    checkpoint_before = checkpoint.read_bytes()
    state = ResearchState(activities={"activity": {"id": "activity", "kind": "loop", "loop_id": "loop-1"}}, outcomes={"attempt": {"id": "attempt", "source_activity_id": "activity", "round": 1, "slot_id": "worker", "execution_status": "failed", "artifact_status": "complete", "evaluation_status": "passed", "audit_status": "not_run", "summary": "Model service failed after prediction was saved", "evidence": [], "resources": []}})
    payload = compile_overview(repo, state, "loop-1")
    assert payload["outcome_differences"][0]["checkpoint_status"] == "failed"
    assert payload["outcome_differences"][0]["research_axes"]["execution_status"] == "failed"
    assert payload["outcome_differences"][0]["research_axes"]["artifact_status"] == "complete"
    assert payload["outcome_differences"][0]["research_axes"]["evaluation_status"] == "passed"
    human = render_overview_human(payload)
    assert "Checkpoint result statuses (separate from research axes)" in human
    assert "evaluation_status" in human and "failed" in human
    assert checkpoint.read_bytes() == checkpoint_before
    assert json.loads(checkpoint.read_text())["rounds"][0]["agents"][0]["result"] == original_result


def test_loop_finding_query_keeps_unrelated_annotations_out_of_other_counts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "human", "name": "lead"},
        "operations": [
            {"op": "activity.create", "local_id": "loop", "kind": "loop", "loop_id": "loop-counts", "title": "Loop", "objective": "candidate", "source": {"kind": "test", "location": "loop"}, "checkpoint": {"path": ".aimx/autoresearch/loop-counts/state.json", "repo": str(repo), "loop_id": "loop-counts"}},
            {"op": "activity.create", "local_id": "other", "kind": "analysis", "title": "Other", "objective": "private", "source": {"kind": "test", "location": "other"}},
            {"op": "annotation.create", "source_activity_id": "$loop", "text": "loop record"},
            {"op": "annotation.create", "source_activity_id": "$other", "text": "unrelated record"},
        ],
    })
    checkpoint = repo / ".aimx/autoresearch/loop-counts/state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"loop_id": "loop-counts", "status": "running"}), encoding="utf-8")
    apply_update(repo, update)
    payload = _json_output(run_finding_command(["ls", "--repo", str(repo), "--loop", "loop-counts", "--topic", "missing", "--json"]))
    assert payload["findings"] == []
    assert payload["other_record_counts"]["annotations"] == 1
    assert payload["unattributed"]["annotations"] == 0


def test_research_context_json_stdout_matches_declared_byte_budget(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = run_research_command(["context", "--repo", str(repo), "--objective", "候選 accuracy", "--budget", "4096", "--json"])
    payload = _json_output(result)
    actual = (result.output + "\n").encode("utf-8")
    assert payload["budget_used"] == len(actual)
    assert len(actual) <= payload["budget_limit"]


def _commit_finding_with_evidence(repo: Path, run_hash: str) -> str:
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent"},
            "operations": [
                {
                    "op": "finding.create",
                    "local_id": "f1",
                    "claim": "Evidence-backed finding.",
                    "evidence": [{"kind": "aim_run", "run_hash": run_hash}],
                }
            ],
        }
    )
    result = apply_update(repo, update, evidence_resolver=lambda value: value)
    assert result.created_ids is not None
    return result.created_ids["f1"]


def test_cli_research_help_returns_before_repository_access(capsys, tmp_path: Path) -> None:
    from aimx.__main__ import main

    for argv in (
        ["research", "overview", "--help", "--repo", str(tmp_path / "missing")],
        ["research", "receipt", "--help"],
        ["finding", "show", "--help", "--repo", str(tmp_path / "missing")],
        ["lineage", "link", "--help"],
        ["frontier", "add", "--help", "--repo", str(tmp_path / "missing")],
    ):
        assert main(argv) == 0
        assert capsys.readouterr().out
    assert not (tmp_path / "missing").exists()


def test_finding_read_marks_missing_aim_evidence_unavailable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_hash = "a" * 32
    finding_id = _commit_finding_with_evidence(repo, run_hash)

    payload = _json_output(
        run_finding_command(["show", finding_id, "--repo", str(repo), "--json"])
    )

    assert payload["finding"]["evidence"] == [
        {"kind": "aim_run", "run_hash": run_hash, "availability": "unavailable"}
    ]
    human = run_finding_command(["show", finding_id, "--repo", str(repo)])
    assert human.output is not None
    assert f"{run_hash} (unavailable)" in human.output


def test_finding_read_marks_resolvable_aim_evidence_available(
    tmp_path: Path, monkeypatch
) -> None:
    import aimx.aim_bridge.research_evidence as research_evidence

    repo = tmp_path / "repo"
    repo.mkdir()
    run_hash = "b" * 32
    finding_id = _commit_finding_with_evidence(repo, run_hash)

    class _ReadableRepo:
        def __init__(self, path: str, *, read_only: bool | None = None) -> None:
            assert path == str(repo)
            assert read_only is True

        def list_all_runs(self):
            return [run_hash]

    monkeypatch.setattr(research_evidence, "Repo", _ReadableRepo)

    payload = _json_output(
        run_finding_command(["show", finding_id, "--repo", str(repo), "--json"])
    )

    assert payload["finding"]["evidence"] == [
        {"kind": "aim_run", "run_hash": run_hash, "availability": "available"}
    ]
