from __future__ import annotations

import json
from pathlib import Path

import pytest


def _config(path: Path, **changes) -> None:
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value), encoding="utf-8")


def _assessment(config: dict, *, met: bool = False) -> list[dict]:
    return [{"criterion": text, "met": met, "evidence": ["validation.json"] if met else [],
             "rationale": "Checked against project evidence."}
            for text in config["goal"]["success_criteria"]]


def _finish(control, directory: Path, *, statuses: tuple[str, ...], met: bool = False) -> dict:
    state = control.load_state(directory)
    current = state["rounds"][-1]
    agents = []
    for agent, status in zip(current["agents"], statuses):
        if status == "succeeded":
            control.checkpoint(directory, {"round": current["round"], "agents": [{
                "id": agent["id"], "dispatch_status": "reserved",
                "task_file": "task.txt", "worktree": "worktree",
            }]})
        agents.append({"id": agent["id"], "dispatch_status": "finished",
                       "result": {"status": status, "summary": "Task outcome", "evidence": []}})
    return control.checkpoint(directory, {
        "round": current["round"], "agents": agents, "complete": True,
        "update_file": f"round-{current['round']}.json",
        "research_update": {"client_update_id": f"round-{current['round']}",
                            "commit_id": f"commit-{current['round']}", "revision": current["round"]},
        "goal_assessment": _assessment(state["config"], met=met),
    })


def test_validate_normalizes_paths_and_replaces_round_agents_without_writes(
    autoresearch_control, autoresearch_config: Path,
) -> None:
    _config(autoresearch_config, round_overrides={"2": {"agents": [{"id": "explore"}]}})
    result = autoresearch_control.validate(autoresearch_config)
    config = result["config"]
    assert config["project_root"] == str(autoresearch_config.parent / "project")
    assert config["aim_repo"] == str(autoresearch_config.parent / "data")
    assert config["context_budget_bytes"] == 12000
    assert result["effective_rounds"][0]["agents"][1] == {"id": "verify"}
    assert result["effective_rounds"][1]["agents"] == [{"id": "explore"}]
    assert not (autoresearch_config.parent / "data/.aimx").exists()


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"schema_version": 2},
    {"max_rounds": 0}, {"max_rounds": -1}, {"max_rounds": True}, {"max_rounds": 2.5},
    {"context_budget_bytes": 0}, {"project_root": "missing"},
    {"default_agents": []}, {"default_agents": [{"id": "a"}, {"id": "a"}]},
    {"default_agents": [{"id": "../bad"}]}, {"default_agents": [{"id": "a", "model": ""}]},
    {"default_agents": [{"id": "a", "reasoning_effort": None}]},
    {"default_agents": [{"id": "a", "count": 2}]},
    {"round_overrides": {"0": {"agents": [{"id": "a"}]}}},
    {"round_overrides": {"3": {"agents": [{"id": "a"}]}}},
    {"round_overrides": {"01": {"agents": [{"id": "a"}]}}},
    {"round_overrides": {"1": {"agents": []}}},
    {"goal": {"description": "goal", "success_criteria": []}},
    {"max_round": 10},
])
def test_invalid_config_never_initializes_state(autoresearch_control, autoresearch_config: Path, changes) -> None:
    _config(autoresearch_config, **changes)
    with pytest.raises(autoresearch_control.ControlError):
        autoresearch_control.init(autoresearch_config)
    assert not (autoresearch_config.parent / "data/.aimx").exists()


def test_duplicate_keys_and_non_json_numbers_are_rejected(autoresearch_control, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    for content in ('{"max_rounds": 2, "max_rounds": 999}', '{"max_rounds": NaN}'):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(autoresearch_control.ControlError):
            autoresearch_control.read_json(path)


@pytest.mark.parametrize("loop_id", ["", "../loop", "with spaces", "x" * 65])
def test_invalid_loop_identity_does_not_create_sidecar(
    autoresearch_control, autoresearch_config: Path, loop_id: str,
) -> None:
    with pytest.raises(autoresearch_control.ControlError):
        autoresearch_control.init(autoresearch_config, loop_id)
    assert not (autoresearch_config.parent / "data/.aimx").exists()


@pytest.mark.parametrize("damage", ["schema", "counter", "agents", "artifacts", "assessment", "terminal"])
def test_malformed_checkpoint_is_rejected_without_repair_or_budget_reset(
    autoresearch_control, autoresearch_config: Path, capsys, damage: str,
) -> None:
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config)["loop_dir"])
    state = control.begin_round(directory, 1, "baseline")
    if damage == "schema":
        state["schema_version"] = True
    elif damage == "counter":
        state["rounds_started"] = 0
    elif damage == "agents":
        state["rounds"][0]["agents"][0]["requested"] = None
    elif damage == "artifacts":
        state["rounds"][0]["artifacts"] = "not-an-array"
    elif damage == "assessment":
        state["goal_assessment"] = {"met": True}
    else:
        state["status"] = "goal_met"
    path = directory / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    damaged = path.read_bytes()
    assert control.main(["status", "--loop-dir", str(directory)]) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False
    with pytest.raises(control.ControlError):
        control.begin_round(directory, 1, "baseline")
    assert path.read_bytes() == damaged


def test_repeated_init_begin_and_status_preserve_budget_and_file(
    autoresearch_control, autoresearch_config: Path,
) -> None:
    control = autoresearch_control
    initial = control.init(autoresearch_config, "study")
    directory = Path(initial["loop_dir"])
    state = control.begin_round(directory, 1, "baseline")
    before = (directory / "state.json").read_bytes()
    assert control.init(autoresearch_config, "study")["state"] == state
    assert control.begin_round(directory, 1, "baseline") == state
    assert control.load_state(directory, autoresearch_config) == state
    assert (directory / "state.json").read_bytes() == before
    assert state["rounds_started"] == 1
    assert not (directory.parent.parent / "research").exists()
    with pytest.raises(control.ControlError):
        control.begin_round(directory, 1, "another-baseline")
    with pytest.raises(control.ControlError, match="current round"):
        control.begin_round(directory, 2, "baseline")


@pytest.mark.parametrize("outcomes", [("succeeded", "failed"), ("failed", "failed")])
def test_failed_rounds_spend_budget_and_exhausted_loop_cannot_restart(
    autoresearch_control, autoresearch_config: Path, outcomes,
) -> None:
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config)["loop_dir"])
    for number in (1, 2):
        control.begin_round(directory, number, f"baseline-{number}")
        state = _finish(control, directory, statuses=outcomes)
    assert state["rounds_started"] == 2
    assert state["status"] == "budget_exhausted"
    with pytest.raises(control.ControlError):
        control.begin_round(directory, 3, "baseline-3")
    with pytest.raises(control.ControlError):
        control.checkpoint(directory, {"status": "running"})


@pytest.mark.parametrize("rounds", [1, 2])
def test_goal_success_takes_precedence_and_stops_new_rounds(
    autoresearch_control, autoresearch_config: Path, rounds: int,
) -> None:
    _config(autoresearch_config, max_rounds=rounds, round_overrides={})
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config)["loop_dir"])
    control.begin_round(directory, 1, "baseline")
    state = _finish(control, directory, statuses=("succeeded", "succeeded"), met=True)
    assert state["status"] == "goal_met"
    assert state["rounds_started"] == 1
    with pytest.raises(control.ControlError):
        control.begin_round(directory, 2, "next-baseline")


def test_already_achieved_goal_spends_no_rounds(autoresearch_control, autoresearch_config: Path) -> None:
    control = autoresearch_control
    initial = control.init(autoresearch_config)
    state = control.checkpoint(Path(initial["loop_dir"]), {
        "goal_assessment": _assessment(initial["state"]["config"], met=True),
    })
    assert state["status"] == "goal_met"
    assert state["rounds_started"] == 0


def test_unknown_dispatch_requires_reconciliation_and_cannot_be_replayed(
    autoresearch_control, autoresearch_config: Path,
) -> None:
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config)["loop_dir"])
    control.begin_round(directory, 1, "baseline")
    reserved = {"round": 1, "agents": [{"id": "explore", "dispatch_status": "reserved",
                                         "task_file": "task.txt", "worktree": "worktree"}]}
    control.checkpoint(directory, reserved)
    control.checkpoint(directory, {"status": "interrupted", "stop_reason": "Native task state is unknown."})
    state = control.begin_round(directory, 1, "baseline")
    assert state["rounds_started"] == 1
    assert state["rounds"][0]["agents"][0]["dispatch_status"] == "reserved"
    with pytest.raises(control.ControlError):
        control.begin_round(directory, 2, "baseline")
    with pytest.raises(control.ControlError, match="rewind"):
        control.checkpoint(directory, {"round": 1, "agents": [{"id": "explore", "dispatch_status": "pending"}]})
    control.checkpoint(directory, {"status": "running"})
    state = _finish(control, directory, statuses=("interrupted", "interrupted"))
    assert state["status"] == "running"
    control.begin_round(directory, 2, "baseline")
    assert control.load_state(directory)["rounds_started"] == 2


def test_changed_resume_config_rejected_without_resetting_state(
    autoresearch_control, autoresearch_config: Path,
) -> None:
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config, "study")["loop_dir"])
    control.begin_round(directory, 1, "baseline")
    before = (directory / "state.json").read_bytes()
    _config(autoresearch_config, max_rounds=10)
    for action in (lambda: control.load_state(directory, autoresearch_config),
                   lambda: control.init(autoresearch_config, "study")):
        with pytest.raises(control.ControlError) as error:
            action()
        assert error.value.code == "config_mismatch"
    assert (directory / "state.json").read_bytes() == before


@pytest.mark.parametrize("patch", [
    {"round": 1, "agents": [{"id": "extra-agent"}]},
    {"round": 1, "agents": [{"id": "explore", "requested": {"model": "replacement"}}]},
    {"round": 1, "complete": True},
    {"round": 1, "agents": [{"id": "explore", "dispatch_status": "started"}]},
    {"round": 1, "agents": [{"id": "explore", "dispatch_status": "finished",
                             "result": {"status": "succeeded", "summary": "Unlaunched", "evidence": []}}]},
    {"status": "goal_met"},
    {"goal_assessment": [{"criterion": "Easier criterion", "met": True, "evidence": ["fake"], "rationale": "changed"}]},
])
def test_invalid_checkpoint_preserves_original_file(
    autoresearch_control, autoresearch_config: Path, patch,
) -> None:
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config)["loop_dir"])
    control.begin_round(directory, 1, "baseline")
    before = (directory / "state.json").read_bytes()
    with pytest.raises(control.ControlError):
        control.checkpoint(directory, patch)
    assert (directory / "state.json").read_bytes() == before


def test_goal_requires_evidence_and_completed_round_requires_lead_receipt(
    autoresearch_control, autoresearch_config: Path,
) -> None:
    control = autoresearch_control
    initial = control.init(autoresearch_config)
    directory = Path(initial["loop_dir"])
    assessment = _assessment(initial["state"]["config"], met=True)
    assessment[0]["evidence"] = []
    with pytest.raises(control.ControlError):
        control.checkpoint(directory, {"goal_assessment": assessment})
    control.begin_round(directory, 1, "baseline")
    control.checkpoint(directory, {"round": 1, "agents": [
        {"id": agent, "dispatch_status": "finished",
         "result": {"status": "failed", "summary": "Spawn failed", "evidence": []}}
        for agent in ("explore", "verify")
    ]})
    with pytest.raises(control.ControlError, match="ResearchUpdate receipt"):
        control.checkpoint(directory, {"round": 1, "complete": True, "goal_assessment": assessment})


def test_completed_round_and_receipt_are_immutable_but_exact_checkpoint_is_idempotent(
    autoresearch_control, autoresearch_config: Path,
) -> None:
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config)["loop_dir"])
    control.begin_round(directory, 1, "baseline")
    state = _finish(control, directory, statuses=("failed", "failed"))
    before = (directory / "state.json").read_bytes()
    for patch in (
        {"round": 1, "update_file": "different-update.json"},
        {"round": 1, "artifacts": ["late-change"]},
        {"round": 1, "agents": [{"id": "explore", "dispatch_status": "reserved"}]},
    ):
        with pytest.raises(control.ControlError):
            control.checkpoint(directory, patch)
    repeated = control.checkpoint(directory, {
        "round": 1, "complete": True, "goal_assessment": state["goal_assessment"],
        "update_file": state["rounds"][0]["update_file"],
        "research_update": state["rounds"][0]["research_update"],
    })
    assert repeated == state
    assert (directory / "state.json").read_bytes() == before


def test_exclusive_writer_lock_and_failed_atomic_replace_preserve_state(
    autoresearch_control, autoresearch_config: Path, monkeypatch,
) -> None:
    control = autoresearch_control
    directory = Path(control.init(autoresearch_config)["loop_dir"])
    before = (directory / "state.json").read_bytes()
    with control.writer_lock(directory):
        assert control.load_state(directory)["rounds_started"] == 0
        with pytest.raises(control.ControlError) as error:
            control.begin_round(directory, 1, "baseline")
        assert error.value.code == "loop_busy"
    assert not (directory / ".writer.lock").exists()

    def fail_replace(*args):
        raise OSError("injected atomic-write failure")

    monkeypatch.setattr(control.os, "replace", fail_replace)
    with pytest.raises(OSError):
        control.begin_round(directory, 1, "baseline")
    assert (directory / "state.json").read_bytes() == before
    assert not list(directory.glob("*.tmp"))
    assert not (directory / ".writer.lock").exists()


def test_cli_reports_json_errors_and_status_is_read_only(
    autoresearch_control, autoresearch_config: Path, capsys,
) -> None:
    control = autoresearch_control
    assert control.main(["validate", "--config", str(autoresearch_config)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    initial = control.init(autoresearch_config)
    directory = Path(initial["loop_dir"])
    before = {path: path.read_bytes() for path in directory.iterdir()}
    assert control.main(["status", "--loop-dir", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["state"]["rounds_started"] == 0
    assert {path: path.read_bytes() for path in directory.iterdir()} == before
    assert control.main(["begin-round", "--loop-dir", str(directory), "--round", "3", "--baseline-ref", "base"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "budget_exhausted"
