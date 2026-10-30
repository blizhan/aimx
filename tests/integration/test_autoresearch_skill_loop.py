"""Exercise the skill protocol with a deterministic host fixture, not live LLMs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aimx.commands.research import run_research_command
from aimx.research.context import compile_context
from aimx.research.errors import RevisionConflict, ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.store import apply_update, database_path, dry_run_update, read_commits, read_state


def _git(project: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Research Fixture", "-c", "user.email=fixture@example.invalid",
         "-C", str(project), *args], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _project(project: Path) -> str:
    _git(project, "init", "-q")
    (project / "model.py").write_text("def score():\n    return 0\n", encoding="utf-8")
    _git(project, "add", "model.py")
    _git(project, "commit", "-qm", "baseline")
    return _git(project, "rev-parse", "HEAD")


def _lead_update(repo: Path, update_id: str, operations: list[dict]) -> ResearchUpdate:
    return ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": read_state(repo).revision,
        "client_update_id": update_id, "author": {"kind": "agent", "name": "lead"},
        "operations": operations,
    })


def _fixture_dispatch(task_file: Path, **requested: str) -> dict:
    # A tiny subprocess stands in for the native task API and experiment runner.
    worker = '''
import json, pathlib, subprocess, sys
from aimx.research.store import read_state
task = json.loads(pathlib.Path(sys.argv[1]).read_text())
requested = json.loads(sys.argv[2])
worktree = pathlib.Path(task["worktree"])
state = read_state(pathlib.Path(task["aim_repo"]))
sys.path.insert(0, str(worktree))
from model import score
baseline_score = score()
evidence = pathlib.Path(task["evidence_file"])
evidence.write_text(json.dumps({
    "baseline_score": baseline_score,
    "score": task["candidate_score"],
    "finding_ids": list(state.findings),
    "requested": requested,
}))
(worktree / "model.py").write_text("def score():\\n    return " + str(task["candidate_score"]) + "\\n")
def git(*args):
    return subprocess.check_output([
        "git", "-c", "user.name=Research Fixture", "-c", "user.email=fixture@example.invalid",
        "-C", str(worktree), *args], text=True).strip()
git("add", "model.py")
git("commit", "-qm", "candidate")
result = {
    "status": "succeeded", "summary": "Candidate score " + str(task["candidate_score"]),
    "evidence": [str(evidence)], "candidate_ref": git("rev-parse", "HEAD"),
    "aim_runs": [],
    "actual_settings": {
        "model": requested.get("model", "fixture/native-default"),
        "reasoning_effort": requested.get("reasoning_effort", "fixture/native-default"),
    },
}
pathlib.Path(task["result_file"]).write_text(json.dumps(result))
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    process = subprocess.Popen(
        [sys.executable, "-c", worker, str(task_file), json.dumps(requested)],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stdout + stderr
    return {"task_id": f"fixture-{process.pid}",
            "result": json.loads(Path(json.loads(task_file.read_text())["result_file"]).read_text())}


def _validate_candidate(worktree: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", "-c", "from model import score; assert score() >= 0; print(score())"],
        cwd=worktree, capture_output=True, text=True, check=False,
    )


def test_two_round_loop_passes_settings_shares_findings_and_only_lead_writes_state(
    autoresearch_control, autoresearch_config: Path,
) -> None:
    control = autoresearch_control
    initialized = control.init(autoresearch_config, "fixture-loop")
    directory = Path(initialized["loop_dir"])
    config = initialized["state"]["config"]
    project, repo = Path(config["project_root"]), Path(config["aim_repo"])
    baseline = _project(project)
    sentinel = repo / ".aim/sentinel"
    sentinel.write_bytes(b"existing Aim metadata")
    bootstrap = _lead_update(repo, "bootstrap", [
        {"op": "frontier.lane.create", "local_id": "lane", "name": "score"},
        {"op": "frontier.item.create", "local_id": "direction", "lane_id": "$lane",
         "direction": config["goal"]["description"], "rationale": "User research goal"},
    ])
    receipt = apply_update(repo, bootstrap)
    direction_id = receipt.created_ids["direction"]
    previous_finding = None

    for number in (1, 2):
        context = compile_context(read_state(repo), config["goal"]["description"], config["context_budget_bytes"])
        if previous_finding is not None:
            assert previous_finding in {item["id"] for item in context["items"]}
        settings = control.effective_agents(config, number)
        agenda = _lead_update(repo, f"round-{number}-agenda", [
            {"op": "agenda.item.create", "local_id": agent["id"], "objective": "Improve score",
             "question": "Does the candidate improve this baseline?", "frontier_item_ids": [direction_id],
             "motivating_finding_ids": [previous_finding] if previous_finding else [], "agenda_status": "active"}
            for agent in settings
        ])
        agenda_receipt = apply_update(repo, agenda)
        control.begin_round(directory, number, baseline)
        research_before = database_path(repo).read_bytes()
        revision_before = read_state(repo).revision
        results = []
        for index, agent in enumerate(settings):
            worktree = autoresearch_config.parent / f"round-{number}-{agent['id']}"
            _git(project, "worktree", "add", "--detach", str(worktree), baseline)
            task_file = directory / f"round-{number}-{agent['id']}.json"
            evidence_file = directory / f"round-{number}-{agent['id']}-evidence.json"
            task_file.write_text(json.dumps({
                "worktree": str(worktree), "aim_repo": str(repo), "context": context,
                "candidate_score": number * 2 - index,
                "evidence_file": str(evidence_file),
                "result_file": str(directory / f"round-{number}-{agent['id']}-result.json"),
            }), encoding="utf-8")
            control.checkpoint(directory, {"round": number, "agents": [{
                "id": agent["id"], "task_file": str(task_file), "worktree": str(worktree),
                "dispatch_status": "reserved",
            }]})
            loop_before = (directory / "state.json").read_bytes()
            requested = {key: value for key, value in agent.items() if key != "id"}
            native = _fixture_dispatch(task_file, **requested)
            assert (directory / "state.json").read_bytes() == loop_before
            assert database_path(repo).read_bytes() == research_before
            assert read_state(repo).revision == revision_before
            evidence = json.loads(evidence_file.read_text())
            assert evidence["requested"] == requested
            assert evidence["baseline_score"] == (number - 1) * 2
            if number == 2:
                assert previous_finding in evidence["finding_ids"]
            control.checkpoint(directory, {"round": number, "agents": [{
                "id": agent["id"], "dispatch_status": "started", "task_id": native["task_id"],
            }]})
            control.checkpoint(directory, {"round": number, "agents": [{
                "id": agent["id"], "dispatch_status": "finished", "result": native["result"],
            }]})
            results.append(native["result"])

        integration = autoresearch_config.parent / f"integration-{number}"
        _git(project, "worktree", "add", "--detach", str(integration), baseline)
        _git(integration, "cherry-pick", results[0]["candidate_ref"])
        validation = _validate_candidate(integration)
        assert validation.returncode == 0, validation.stderr
        assert _git(project, "rev-parse", "HEAD") == baseline
        assert _git(project, "status", "--porcelain") == ""
        integration_ref = _git(integration, "rev-parse", "HEAD")
        control.checkpoint(directory, {"round": number, "artifacts": [integration_ref]})
        _git(project, "merge", "--ff-only", integration_ref)
        baseline = _git(project, "rev-parse", "HEAD")
        control.checkpoint(directory, {"round": number, "merged_ref": baseline})

        operations = [
            {"op": "finding.create", "local_id": "score", "claim": f"Validated score improved to {number * 2}."},
            {"op": "annotation.create", "annotation_target_kind": "finding", "annotation_target_id": "$score",
             "text": "Validated experiment evidence: " + results[0]["evidence"][0]},
            *[{"op": "agenda.item.transition", "agenda_item_id": agenda_receipt.created_ids[agent["id"]],
               "agenda_status": "completed"} for agent in settings],
        ]
        if previous_finding:
            operations.append({"op": "relation.create", "source": "$score", "target": previous_finding,
                               "relation_type": "supports", "reason": "Next round uses the validated baseline."})
        update = _lead_update(repo, f"round-{number}-aggregate", operations)
        if number == 2:
            apply_update(repo, _lead_update(repo, "human-steering", [{"op": "annotation.create",
                "text": "Keep the score validation command fixed."}]))
            with pytest.raises(RevisionConflict):
                dry_run_update(repo, update)
            refreshed = read_state(repo)
            assert any("fixed" in annotation["text"] for annotation in refreshed.annotations.values())
            update = _lead_update(repo, f"round-{number}-aggregate-refreshed", operations)
        update_file = directory / f"round-{number}-update.json"
        update_file.write_text(json.dumps(update.as_dict()), encoding="utf-8")
        control.checkpoint(directory, {"round": number, "update_file": str(update_file)})
        dry_run_update(repo, update)
        committed = apply_update(repo, update)
        replayed = apply_update(repo, ResearchUpdate.from_dict(json.loads(update_file.read_text())))
        assert replayed.idempotent and replayed.commit_id == committed.commit_id
        previous_finding = committed.created_ids["score"]
        assessment = [{
            "criterion": criterion, "met": number == 2 or index == 1,
            "evidence": results[0]["evidence"], "rationale": "Checked integrated score and project validation.",
        } for index, criterion in enumerate(config["goal"]["success_criteria"])]
        state = control.checkpoint(directory, {
            "round": number, "complete": True, "goal_assessment": assessment,
            "research_update": {"client_update_id": update.client_update_id,
                                "commit_id": committed.commit_id, "revision": committed.revision},
        })
        assert state["rounds_started"] == number
        assert state["status"] == ("running" if number == 1 else "goal_met")

    assert sentinel.read_bytes() == b"existing Aim metadata"
    assert {item["author_name"] for item in read_commits(repo)} == {"lead"}
    assert len(read_state(repo).findings) == 2
    assert len(read_state(repo).relations) == 1
    with pytest.raises(control.ControlError):
        control.begin_round(directory, 3, baseline)


def test_preflight_capabilities_and_observation_refusal_cases_are_recorded(
    tmp_path: Path,
) -> None:
    from aimx.aim_bridge.research_evidence import EvidenceResolver
    from aimx.research.validation import _validate_experiment_contract

    repo = tmp_path / "repo"
    repo.mkdir()
    activity = _lead_update(repo, "preflight-activity", [
        {"op": "activity.create", "local_id": "activity", "kind": "analysis",
         "title": "Capability preflight", "objective": "Check available runtime capabilities",
         "source": {"kind": "fixture", "location": "preflight"}},
    ])
    apply_update(repo, activity)
    checks = {
        "host_command": "missing", "model_support": "missing",
        "evaluator": "failed", "aim_sdk": "unavailable", "local_file_evidence": "available",
    }
    preflight = repo / "preflight.json"
    preflight.write_text(json.dumps({"status": "blocked", "checks": checks}), encoding="utf-8")
    digest = __import__("hashlib").sha256(preflight.read_bytes()).hexdigest()
    expected_missing = {name for name, status in checks.items() if status in {"missing", "failed"}}
    assert expected_missing == {"host_command", "model_support", "evaluator"}
    assert checks["aim_sdk"] == "unavailable" and checks["local_file_evidence"] == "available"
    assert all(checks[name] != "available" for name in expected_missing)

    state = read_state(repo)
    contract = {
        "contract_version": 1, "question": "Independent check", "interventions": ["candidate"],
        "fixed_controls": ["same evaluator"], "expected_evidence": ["score"],
        "disconfirmation": ["below threshold"], "confounders": [],
        "evaluation": {"protocol_id": "fixture", "protocol_sha256": "a" * 64,
            "argv": ["python", "evaluate.py"], "metric": "score", "acceptance_rule": "threshold",
            "preflight_evidence": [{"kind": "local_artifact", "location": str(preflight),
                "sha256": digest, "size_bytes": preflight.stat().st_size,
                "evidence_type": "preflight", "purpose": "capability readiness",
                "source_activity_id": next(iter(state.activities))}]},
        "data": {"scope_ids": ["scope"], "phase": "independent_confirmation",
            "candidate_fingerprint": "candidate-v1", "observation_revision": 3},
        "resources": [{"kind": "calls", "unit": "count", "planned": 1,
            "verification": "unverified", "enforcement": "declared_only"}],
        "failure_policy": "stop_pending",
    }

    class Resolver:
        def resolve_artifact(self, location: str):
            return EvidenceResolver(repo).resolve_artifact(location)

    def fixture_dispatch_decision(*, preflight_ready: bool, host_command: bool,
                                  model_supported: bool, evaluator_ready: bool,
                                  policy: str, host_policy_attested: bool = False,
                                  remaining_task_independent: bool = False) -> list[str]:
        """Mirror the skill gate in a deterministic fake host, without spawning."""
        launched = []
        if not preflight_ready or not host_command or not model_supported or not evaluator_ready:
            return launched
        launched.append("first-slot")
        if policy == "stop_pending":
            return launched
        if policy == "continue_independent" and host_policy_attested and remaining_task_independent:
            launched.append("independent-pending-slot")
        return launched

    assert fixture_dispatch_decision(preflight_ready=False, host_command=False,
        model_supported=False, evaluator_ready=False, policy="stop_pending") == []
    assert fixture_dispatch_decision(preflight_ready=True, host_command=True,
        model_supported=True, evaluator_ready=True, policy="stop_pending") == ["first-slot"]
    assert fixture_dispatch_decision(preflight_ready=True, host_command=True,
        model_supported=True, evaluator_ready=True, policy="continue_independent",
        host_policy_attested=False, remaining_task_independent=True) == ["first-slot"]
    assert fixture_dispatch_decision(preflight_ready=True, host_command=True,
        model_supported=True, evaluator_ready=True, policy="continue_independent",
        host_policy_attested=True, remaining_task_independent=True) == [
            "first-slot", "independent-pending-slot"]
    assert fixture_dispatch_decision(preflight_ready=True, host_command=True,
        model_supported=True, evaluator_ready=True, policy="continue_independent",
        host_policy_attested=True, remaining_task_independent=False) == ["first-slot"]

    event_types = ["already_scored", "promoted_to_development", "unknown_coverage", "overlap"]
    for case in event_types:
        scope = {"id": "scope", "coverage": "declared_complete",
            "initial_observation": "declared_unobserved", "overlap_keys": ["holdout"],
            "source": {"kind": "fixture", "location": "scope.json"}}
        observations = {"freeze": {"id": "freeze", "dataset_scope_id": "scope",
            "event_kind": "candidate_frozen", "candidate_fingerprint": "candidate-v1",
            "occurred_at": "2026-01-01T00:00:00Z", "created_revision": 1}}
        scopes = {"scope": scope}
        if case == "already_scored":
            observations["first_score"] = {"id": "first_score", "dataset_scope_id": "scope",
                "event_kind": "first_scored", "candidate_fingerprint": "candidate-v1",
                "occurred_at": "2026-01-02T00:00:00Z", "created_revision": 2}
        elif case == "promoted_to_development":
            observations["promotion"] = {"id": "promotion", "dataset_scope_id": "scope",
                "event_kind": "promoted_to_development", "candidate_fingerprint": "candidate-v1",
                "occurred_at": "2026-01-02T00:00:00Z", "created_revision": 2}
        elif case == "unknown_coverage":
            scopes["scope"]["coverage"] = "unknown"
        elif case == "overlap":
            scopes["other"] = {"id": "other", "coverage": "declared_complete",
                "initial_observation": "declared_unobserved", "overlap_keys": ["holdout"],
                "source": {"kind": "fixture", "location": "other-scope.json"}}
        snapshot = ResearchState(revision=3, activities=state.activities,
            dataset_scopes=scopes, observations=observations)
        with pytest.raises(ValidationError) as rejected:
            _validate_experiment_contract(contract, snapshot, {}, Resolver())
        assert rejected.value.code == "independence_claim_rejected", case
        assert rejected.value.details.get("query_commands"), case
        if case in {"already_scored", "promoted_to_development"}:
            assert rejected.value.details["first_observation"]["id"] in {"first_score", "promotion"}

    exploratory = {**contract, "data": {**contract["data"], "phase": "exploratory"}}
    assert _validate_experiment_contract(exploratory, snapshot, {}, Resolver())["data"]["phase"] == "exploratory"
    assert read_state(repo).revision == 1
    assert not (repo / ".aim").exists()


def test_execution_failure_matrix_keeps_four_axes_evidence_and_reconciliation(
    tmp_path: Path,
) -> None:
    from aimx.aim_bridge.research_evidence import make_evidence_resolver
    from aimx.research.overview import compile_overview
    repo = tmp_path / "repo"
    repo.mkdir()
    report = repo / "complete-predictions.json"
    report.write_text(json.dumps({"predictions": [0.2, 0.8]}), encoding="utf-8")
    partial = repo / "partial-predictions.json"
    partial.write_text(json.dumps({"predictions": [0.2]}), encoding="utf-8")
    preflight = repo / "failed-preflight.json"
    preflight.write_text(json.dumps({"status": "failed", "error": "evaluator unavailable"}), encoding="utf-8")
    checkpoint_dir = repo / ".aimx/autoresearch/fault-loop"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "state.json"
    checkpoint.write_text(json.dumps({"schema_version": 1, "loop_id": "fault-loop",
        "status": "interrupted", "rounds_started": 1, "config": {"max_rounds": 2},
        "rounds": [{"round": 1, "agents": [{"id": "agent-1", "dispatch_status": "finished",
            "result": {"status": "failed", "summary": "host disconnected"},
            "result_file": str(partial)}]}]}), encoding="utf-8")
    activity = {"op": "activity.create", "local_id": "activity", "kind": "loop",
        "loop_id": "fault-loop", "title": "Fault matrix", "objective": "Preserve failed attempts",
        "source": {"kind": "fixture", "location": str(preflight)},
        "checkpoint": {"path": str(checkpoint), "repo": str(repo), "loop_id": "fault-loop"}}
    operations = [activity,
        {"op": "agenda.item.create", "local_id": "agenda", "objective": "Evaluate fixture output",
         "question": "Are predictions complete?", "frontier_item_ids": ["$direction"],
         "motivating_finding_ids": [], "source_activity_id": "$activity"},
    ]
    # Agenda requires a concrete research motivation, so add its own lane/item.
    operations[1:1] = [
        {"op": "frontier.lane.create", "local_id": "lane", "name": "faults"},
        {"op": "frontier.item.create", "local_id": "direction", "lane_id": "$lane",
         "direction": "Preserve every attempt", "rationale": "Acceptance matrix"},
    ]
    base = _lead_update(repo, "fault-matrix-setup", operations)
    ids = apply_update(repo, base, make_evidence_resolver(repo)).created_ids

    def evidence(path: Path, name: str) -> dict:
        return {"kind": "local_artifact", "location": str(path),
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size, "evidence_type": "evaluation_report",
            "purpose": name, "source_activity_id": ids["activity"]}

    attempt_specs = [
        ("service_failed_complete_predictions", "failed", "complete", "passed", "failed", [evidence(report, "complete predictions")], []),
        ("partial_artifact", "succeeded", "partial", "inconclusive", "not_run", [evidence(partial, "partial predictions")], []),
        ("preflight_failed", "failed", "none", "not_run", "not_run", [evidence(preflight, "failed evaluator preflight")], []),
    ]
    outcome_ops = []
    agenda_id = ids["agenda"]
    for key, execution, artifact, evaluation, audit, refs, resources in attempt_specs:
        outcome_ops.append({"op": "execution.outcome.record", "local_id": key,
            "source_activity_id": ids["activity"], "agenda_item_id": agenda_id,
            "attempt_key": key, "execution_status": execution, "artifact_status": artifact,
            "evaluation_status": evaluation, "audit_status": audit, "summary": key,
            "evidence": refs, "resources": resources, "round": 1, "slot_id": key})
    update = _lead_update(repo, "fault-matrix-outcomes", outcome_ops)
    outcome_ids = apply_update(repo, update, make_evidence_resolver(repo)).created_ids
    repeated = _lead_update(repo, "fault-matrix-retry", [{"op": "execution.outcome.record",
        "local_id": "retry", "source_activity_id": ids["activity"], "agenda_item_id": agenda_id,
        "attempt_key": "duplicate-evaluation", "execution_status": "succeeded",
        "artifact_status": "complete", "evaluation_status": "passed", "audit_status": "not_run",
        "summary": "separate second execution", "evidence": [evidence(report, "retry result")],
        "resources": [{"kind": "calls", "unit": "count", "planned": 1, "actual": 2,
            "verification": "verified", "enforcement": "declared_only", "source": "fixture executor log"}],
        "retry_of": outcome_ids["service_failed_complete_predictions"], "round": 1,
        "slot_id": "service_failed_complete_predictions"}])
    retry_receipt = apply_update(repo, repeated, make_evidence_resolver(repo))

    loop_bytes_before = checkpoint.read_bytes()
    result = compile_overview(repo, read_state(repo), "fault-loop")
    assert {item["attempt_key"] for item in result["attempts"]["retry_chains"]} == {"duplicate-evaluation"}
    assert result["attempts"]["count"] == 4
    assert len(result["attempts"]["duplicate_slots"]) == 1
    by_key = {item["attempt_key"]: item for item in read_state(repo).outcomes.values()}
    assert by_key["service_failed_complete_predictions"]["execution_status"] == "failed"
    assert by_key["service_failed_complete_predictions"]["artifact_status"] == "complete"
    assert by_key["service_failed_complete_predictions"]["evaluation_status"] == "passed"
    assert by_key["preflight_failed"]["execution_status"] == "failed"
    assert result["attempts"]["resources"][0]["actual"] == 2
    assert result["attempts"]["resources"][0]["verification"] == "verified"
    assert result["outcome_differences"]
    assert checkpoint.read_bytes() == loop_bytes_before
    assert read_state(repo).revision == retry_receipt.revision
    assert report.is_file() and partial.is_file() and preflight.is_file()
    assert result["attempts"]["resource_summary"]["over_plan"]
    assert result["attempts"]["resource_summary"]["not_enforced"]

    # Model a successful commit whose response was lost: exact update replay
    # recovers the same receipt, and a not_found snapshot cannot authorize spawn.
    from aimx.research.store import _request_fingerprint
    receipt_query = run_research_command(["receipt", "--repo", str(repo), "--client-update-id",
        repeated.client_update_id, "--fingerprint", _request_fingerprint(repeated), "--json"])
    assert receipt_query.exit_status == 0
    assert json.loads(receipt_query.output)["status"] == "committed"
    missing_receipt = run_research_command(["receipt", "--repo", str(repo), "--client-update-id",
        "not-yet-committed", "--json"])
    assert json.loads(missing_receipt.output)["status"] == "not_found"
    recovered = apply_update(repo, repeated, make_evidence_resolver(repo))
    assert recovered.idempotent and recovered.commit_id == retry_receipt.commit_id
    assert recovered.revision == retry_receipt.revision
    conflict = run_research_command(["receipt", "--repo", str(repo), "--client-update-id",
        repeated.client_update_id, "--fingerprint", "0" * 64, "--json"])
    assert conflict.exit_status == 2
    assert "idempotency_conflict" in conflict.error_message


def test_concurrent_observation_invalidates_preflight_snapshot_before_scoring(tmp_path: Path) -> None:
    from aimx.aim_bridge.research_evidence import make_evidence_resolver
    from aimx.research.validation import compile_update

    repo = tmp_path / "repo"
    repo.mkdir()
    preflight = repo / "preflight.json"
    preflight.write_text('{"status":"ready"}\\n', encoding="utf-8")
    scope_source = repo / "scope.json"
    scope_source.write_text('{"scope":"held-out-v1"}\\n', encoding="utf-8")
    evidence_ref = lambda path, kind, purpose, activity_ref: {
        "kind": "local_artifact", "location": str(path),
        "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size, "evidence_type": kind, "purpose": purpose,
        "source_activity_id": activity_ref,
    }
    setup = _lead_update(repo, "concurrency-setup", [
        {"op": "activity.create", "local_id": "activity", "kind": "analysis",
         "title": "Concurrent observation", "objective": "Freeze then detect concurrent scoring",
         "source": {"kind": "fixture", "location": str(preflight)}},
        {"op": "dataset.scope.create", "local_id": "scope", "project_key": "fixture-project",
         "dataset_key": "holdout", "data_revision": "v1", "description": "Fixed holdout",
         "overlap_keys": ["holdout-v1"], "coverage": "declared_complete",
         "initial_observation": "declared_unobserved", "source_activity_id": "$activity",
         "source": {"kind": "fixture", "location": str(scope_source)}},
        {"op": "dataset.observation.record", "local_id": "freeze", "dataset_scope_id": "$scope",
         "event_kind": "candidate_frozen", "candidate_fingerprint": "candidate-v1",
         "occurred_at": "2026-01-01T00:00:00Z", "source_activity_id": "$activity",
         "source": {"kind": "fixture", "location": str(scope_source)},
         "evidence": [evidence_ref(scope_source, "dataset_manifest", "declared scope", "$activity")]},
        {"op": "frontier.lane.create", "local_id": "lane", "name": "validation"},
        {"op": "frontier.item.create", "local_id": "direction", "lane_id": "$lane",
         "direction": "Independent validation", "rationale": "Acceptance fixture"},
    ])
    setup_result = apply_update(repo, setup, make_evidence_resolver(repo))
    snapshot_revision = read_state(repo).revision
    assert snapshot_revision == 1

    contract = {
        "contract_version": 1, "question": "Does frozen candidate pass holdout?",
        "interventions": ["candidate-v1"], "fixed_controls": ["same evaluator"],
        "expected_evidence": ["score"], "disconfirmation": ["below threshold"],
        "confounders": [], "evaluation": {"protocol_id": "fixture-eval",
            "protocol_sha256": "a" * 64, "argv": ["python", "evaluate.py"], "metric": "score",
            "acceptance_rule": "score >= 1", "preflight_evidence": [
                evidence_ref(preflight, "preflight", "ready evaluator", setup_result.created_ids["activity"])]},
        "data": {"scope_ids": [setup_result.created_ids["scope"]],
            "phase": "independent_confirmation", "candidate_fingerprint": "candidate-v1",
            "observation_revision": snapshot_revision},
        "resources": [{"kind": "calls", "unit": "count", "planned": 1,
            "verification": "unverified", "enforcement": "declared_only"}],
        "failure_policy": "stop_pending",
    }
    lead_activity = setup_result.created_ids["activity"]
    observation = _lead_update(repo, "concurrent-promotion", [{
        "op": "dataset.observation.record", "local_id": "promotion",
        "dataset_scope_id": setup_result.created_ids["scope"],
        "event_kind": "promoted_to_development", "candidate_fingerprint": "candidate-v1",
        "occurred_at": "2026-01-02T00:00:00Z", "source_activity_id": lead_activity,
        "source": {"kind": "fixture", "location": str(scope_source)},
        "evidence": [evidence_ref(scope_source, "dataset_manifest", "concurrent promotion", lead_activity)],
    }])
    apply_update(repo, observation, make_evidence_resolver(repo))
    assert read_state(repo).revision == snapshot_revision + 1

    stale_dispatch = _lead_update(repo, "stale-independent-dispatch", [{
        "op": "agenda.item.create", "local_id": "agenda", "objective": "Independent validation",
        "question": "Does the frozen candidate generalize?", "frontier_item_ids": [setup_result.created_ids["direction"]],
        "motivating_finding_ids": [], "source_activity_id": lead_activity,
        "experiment_contract": contract,
    }])
    with pytest.raises(ValidationError) as rejected:
        compile_update(stale_dispatch, read_state(repo), revision=read_state(repo).revision + 1,
            commit_id="commit_stale", evidence_resolver=make_evidence_resolver(repo), repo_root=repo)
    assert rejected.value.code == "independence_claim_rejected"
    assert rejected.value.details["observed_revision"] == snapshot_revision + 1
    assert read_state(repo).revision == snapshot_revision + 1
    assert not read_state(repo).agenda_items


def test_failed_integration_keeps_primary_worktree_and_baseline_unchanged(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    baseline = _project(project)
    candidate = tmp_path / "candidate"
    integration = tmp_path / "integration"
    _git(project, "worktree", "add", "--detach", str(candidate), baseline)
    (candidate / "model.py").write_text("def score():\n    return -1\n", encoding="utf-8")
    _git(candidate, "add", "model.py")
    _git(candidate, "commit", "-qm", "invalid candidate")
    _git(project, "worktree", "add", "--detach", str(integration), baseline)
    _git(integration, "cherry-pick", _git(candidate, "rev-parse", "HEAD"))
    assert _validate_candidate(integration).returncode != 0
    assert _git(project, "rev-parse", "HEAD") == baseline
    assert _git(project, "status", "--porcelain") == ""
    assert (project / "model.py").read_text() == "def score():\n    return 0\n"
    assert _git(candidate, "rev-parse", "HEAD") != baseline


def test_installed_skill_can_validate_and_checkpoint_without_sibling_skills(
    autoresearch_config: Path, tmp_path: Path,
) -> None:
    import shutil

    source = Path(__file__).resolve().parents[2] / "skills/aimx-autoresearch"
    installed = tmp_path / "isolated-install" / "aimx-autoresearch"
    shutil.copytree(source, installed, ignore=shutil.ignore_patterns("__pycache__"))
    script = installed / "scripts/control.py"
    command = [sys.executable, "-I", str(script)]
    validation = subprocess.run([*command, "validate", "--config", str(autoresearch_config)],
                                cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(validation.stdout)["ok"] is True
    assert not (tmp_path / "data/.aimx").exists()
    initialization = subprocess.run([*command, "init", "--config", str(autoresearch_config)],
                                    cwd=tmp_path, capture_output=True, text=True, check=True)
    directory = Path(json.loads(initialization.stdout)["loop_dir"])
    begin = subprocess.run([*command, "begin-round", "--loop-dir", str(directory), "--round", "1",
                            "--baseline-ref", "fixture-baseline"], capture_output=True, text=True, check=True)
    assert json.loads(begin.stdout)["state"]["rounds_started"] == 1
    status = subprocess.run([*command, "status", "--loop-dir", str(directory), "--config", str(autoresearch_config)],
                            capture_output=True, text=True, check=True)
    assert json.loads(status.stdout)["state"]["rounds_started"] == 1
    assert not (directory.parent.parent / "research").exists()
