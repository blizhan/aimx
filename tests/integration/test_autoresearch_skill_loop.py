"""Exercise the skill protocol with a deterministic host fixture, not live LLMs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aimx.research.context import compile_context
from aimx.research.errors import RevisionConflict
from aimx.research.models import ResearchUpdate
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
