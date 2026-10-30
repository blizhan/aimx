#!/usr/bin/env python3
"""Validate AutoResearch plans and maintain lead-owned round checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
LOOP_STATUSES = {"running", "blocked", "interrupted", "goal_met", "budget_exhausted"}
TERMINAL_STATUSES = {"goal_met", "budget_exhausted"}
RESULT_STATUSES = {"succeeded", "failed", "interrupted"}
DISPATCH_TRANSITIONS = {
    "pending": {"pending", "reserved", "finished"},
    "reserved": {"reserved", "started", "finished"},
    "started": {"started", "finished"},
    "finished": {"finished"},
}


class ControlError(ValueError):
    def __init__(self, message: str, code: str = "invalid_input") -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, message: str, code: str = "invalid_input") -> None:
    if not condition:
        raise ControlError(message, code)


def object_fields(value: Any, allowed: set[str], required: set[str], label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require(not (set(value) - allowed), f"{label} has unsupported fields: {sorted(set(value) - allowed)}")
    require(not (required - set(value)), f"{label} needs fields: {sorted(required - set(value))}")


def nonempty_text(value: Any, label: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a nonempty string")


def positive_int(value: Any, label: str) -> None:
    require(type(value) is int and value > 0, f"{label} must be a positive integer")


def text_list(value: Any, label: str, *, nonempty: bool = False) -> None:
    require(isinstance(value, list), f"{label} must be an array")
    require(not nonempty or bool(value), f"{label} must not be empty")
    for item in value:
        nonempty_text(item, label)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ControlError(f"Invalid JSON number: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"Cannot read JSON at {path}: {exc}") from exc


def validate_agents(value: Any, label: str) -> list[dict[str, str]]:
    require(isinstance(value, list) and bool(value), f"{label} must be a nonempty array")
    ids: set[str] = set()
    for agent in value:
        object_fields(agent, {"id", "model", "reasoning_effort"}, {"id"}, label)
        nonempty_text(agent["id"], f"{label}.id")
        require(bool(ID_PATTERN.fullmatch(agent["id"])), f"Invalid agent id: {agent['id']}")
        require(agent["id"] not in ids, f"Duplicate agent id: {agent['id']}")
        ids.add(agent["id"])
        for key in ("model", "reasoning_effort"):
            if key in agent:
                nonempty_text(agent[key], f"{label}.{key}")
    return copy.deepcopy(value)


def normalize_config(value: Any, directory: Path) -> dict[str, Any]:
    required = {"schema_version", "project_root", "aim_repo", "goal", "max_rounds", "default_agents"}
    object_fields(value, required | {"round_overrides", "context_budget_bytes"}, required, "config")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1,
            "config.schema_version must be 1")
    config = copy.deepcopy(value)
    positive_int(config["max_rounds"], "max_rounds")
    goal = config["goal"]
    object_fields(goal, {"description", "success_criteria"}, {"description", "success_criteria"}, "goal")
    nonempty_text(goal["description"], "goal.description")
    text_list(goal["success_criteria"], "goal.success_criteria", nonempty=True)
    require(len(set(goal["success_criteria"])) == len(goal["success_criteria"]),
            "goal.success_criteria must be unique")
    for key in ("project_root", "aim_repo"):
        nonempty_text(config[key], key)
        path = (directory / Path(config[key]).expanduser()).resolve()
        require(path.is_dir(), f"{key} is not an existing directory: {path}")
        if key == "aim_repo" and path.name == ".aim":
            path = path.parent
        config[key] = str(path)
    config["default_agents"] = validate_agents(config["default_agents"], "default_agents")
    config.setdefault("context_budget_bytes", 12000)
    positive_int(config["context_budget_bytes"], "context_budget_bytes")
    config.setdefault("round_overrides", {})
    require(isinstance(config["round_overrides"], dict), "round_overrides must be an object")
    for number, override in config["round_overrides"].items():
        require(isinstance(number, str) and number.isascii() and number.isdecimal()
                and str(int(number)) == number and 1 <= int(number) <= config["max_rounds"],
                f"Override round must be between 1 and max_rounds: {number}")
        object_fields(override, {"agents"}, {"agents"}, f"round_overrides.{number}")
        override["agents"] = validate_agents(override["agents"], f"round_overrides.{number}.agents")
    return config


def load_config(path: Path) -> dict[str, Any]:
    return normalize_config(read_json(path), path.resolve().parent)


def fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_agents(config: dict[str, Any], number: int) -> list[dict[str, str]]:
    return copy.deepcopy(config["round_overrides"].get(str(number), {}).get("agents", config["default_agents"]))


def validate(path: Path) -> dict[str, Any]:
    config = load_config(path)
    return {"config": config, "config_fingerprint": fingerprint(config), "effective_rounds": [
        {"round": number, "agents": effective_agents(config, number)}
        for number in range(1, config["max_rounds"] + 1)
    ]}


@contextmanager
def writer_lock(directory: Path) -> Iterator[None]:
    lock = directory / ".writer.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ControlError(f"Loop is locked at {lock}; verify the writer has stopped before removing it.",
                           "loop_busy") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid()}, stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def write_state(directory: Path, state: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".state-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "state.json")
        if os.name != "nt":
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_state(directory: Path, config_path: Path | None = None) -> dict[str, Any]:
    state = read_json(directory / "state.json")
    required = {"schema_version", "loop_id", "config", "config_fingerprint", "status",
                "rounds_started", "rounds", "goal_assessment", "stop_reason"}
    object_fields(state, required, required, "saved checkpoint")
    require(type(state["schema_version"]) is int and state["schema_version"] == 1, "Invalid checkpoint schema")
    require(isinstance(state.get("config"), dict), "Checkpoint config is missing")
    config = normalize_config(state["config"], directory)
    require(state.get("config_fingerprint") == fingerprint(config), "Checkpoint config fingerprint mismatch")
    if config_path is not None:
        require(fingerprint(load_config(config_path)) == state["config_fingerprint"],
                "Resume config differs from the initialized loop; use the original config or a new loop.",
                "config_mismatch")
    require(isinstance(state.get("status"), str) and state["status"] in LOOP_STATUSES, "Invalid loop status")
    require(isinstance(state["goal_assessment"], list), "Invalid saved goal assessment")
    if state["goal_assessment"]:
        validate_assessment(state["goal_assessment"], config["goal"]["success_criteria"])
    rounds = state.get("rounds")
    require(isinstance(rounds, list) and type(state.get("rounds_started")) is int
            and state["rounds_started"] == len(rounds) <= config["max_rounds"], "Invalid round counter")
    for number, item in enumerate(rounds, 1):
        fields = {"round", "status", "baseline_ref", "agents", "artifacts", "merged_ref", "update_file", "research_update"}
        object_fields(item, fields, fields, "saved round")
        nonempty_text(item["baseline_ref"], "saved baseline_ref")
        text_list(item["artifacts"], "saved artifacts")
        require(isinstance(item, dict) and type(item.get("round")) is int and item["round"] == number
                and isinstance(item.get("status"), str)
                and item["status"] in {"active", "completed"}, "Invalid round checkpoint")
        require(number == len(rounds) or item["status"] == "completed", "Earlier round is incomplete")
        agents = item.get("agents")
        require(isinstance(agents, list) and all(isinstance(agent, dict)
                and isinstance(agent.get("requested"), dict) and "id" not in agent["requested"] for agent in agents),
                "Invalid round agents")
        require([{"id": agent.get("id"), **agent["requested"]} for agent in agents]
                == effective_agents(config, number), "Round agents differ from the configured plan")
        require(all(isinstance(agent.get("dispatch_status"), str)
                    and agent["dispatch_status"] in DISPATCH_TRANSITIONS for agent in agents),
                "Invalid dispatch status")
        for agent in agents:
            require({"task_id", "task_file", "worktree", "result"} <= set(agent), "Incomplete agent checkpoint")
            require((agent["dispatch_status"] == "finished") == (agent["result"] is not None),
                    "Invalid finished agent checkpoint")
            if agent["result"] is not None:
                validate_result(agent["result"], state["loop_id"], number, legacy=True)
        require(item["status"] != "completed" or all(agent["dispatch_status"] == "finished" for agent in agents),
                "Completed round has unfinished agents")
    settled = not rounds or rounds[-1]["status"] == "completed"
    if state["status"] == "goal_met":
        require(settled and bool(state["goal_assessment"]) and all(item["met"] for item in state["goal_assessment"]),
                "Invalid saved goal completion")
    if state["status"] == "budget_exhausted":
        require(settled and len(rounds) == config["max_rounds"], "Invalid saved budget exhaustion")
    return state


def init(config_path: Path, loop_id: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    loop_id = uuid.uuid4().hex if loop_id is None else loop_id
    require(isinstance(loop_id, str) and bool(ID_PATTERN.fullmatch(loop_id)), "Invalid loop id")
    directory = Path(config["aim_repo"]) / ".aimx" / "autoresearch" / loop_id
    directory.mkdir(parents=True, exist_ok=True)
    with writer_lock(directory):
        if (directory / "state.json").exists():
            state = load_state(directory, config_path)
        else:
            state = {"schema_version": 1, "loop_id": loop_id, "config": config,
                     "config_fingerprint": fingerprint(config), "status": "running",
                     "rounds_started": 0, "rounds": [], "goal_assessment": [], "stop_reason": None}
            write_state(directory, state)
    return {"loop_dir": str(directory), "state": state}


def begin_round(directory: Path, number: int, baseline_ref: str,
                config_path: Path | None = None) -> dict[str, Any]:
    positive_int(number, "round")
    nonempty_text(baseline_ref, "baseline_ref")
    with writer_lock(directory):
        state = load_state(directory, config_path)
        if number <= state["rounds_started"]:
            require(state["rounds"][number - 1]["baseline_ref"] == baseline_ref,
                    "Registered round has a different baseline")
            return state
        require(state["status"] == "running", f"Loop is {state['status']}", "loop_stopped")
        require(number <= state["config"]["max_rounds"], "Round budget exhausted", "budget_exhausted")
        require(number == state["rounds_started"] + 1, "Rounds must start in sequence")
        require(not state["rounds"] or state["rounds"][-1]["status"] == "completed",
                "Reconcile and complete the current round before starting another", "round_in_progress")
        agents = [{"id": agent["id"], "requested": {key: value for key, value in agent.items() if key != "id"},
                   "dispatch_status": "pending", "task_id": None, "task_file": None,
                   "worktree": None, "result": None}
                  for agent in effective_agents(state["config"], number)]
        state["rounds"].append({"round": number, "status": "active", "baseline_ref": baseline_ref,
                                "agents": agents, "artifacts": [], "merged_ref": None,
                                "update_file": None, "research_update": None})
        state["rounds_started"] = number
        write_state(directory, state)
    return state


def validate_result(result: Any, loop_id: str, round_number: int, *, legacy: bool = False) -> None:
    """Validate an agent result for the checkpoint's loop and round.

    ``legacy`` accepts results saved before ``aim_runs`` became required.
    """
    required = {"status", "summary", "evidence"} if legacy else {"status", "summary", "evidence", "aim_runs"}
    object_fields(result, {"status", "summary", "evidence", "candidate_ref", "actual_settings", "aim_runs"},
                  required, "agent result")
    require(isinstance(result["status"], str) and result["status"] in RESULT_STATUSES, "Invalid agent result status")
    nonempty_text(result["summary"], "result.summary")
    text_list(result["evidence"], "result.evidence")
    if "candidate_ref" in result:
        nonempty_text(result["candidate_ref"], "result.candidate_ref")
    if "actual_settings" in result:
        object_fields(result["actual_settings"], {"model", "reasoning_effort"}, set(), "actual_settings")
        for key, value in result["actual_settings"].items():
            nonempty_text(value, f"actual_settings.{key}")
    if "aim_runs" in result:
        aim_runs = result["aim_runs"]
        require(isinstance(aim_runs, list), "result.aim_runs must be an array")
        seen_hashes: set[str] = set()
        for index, run in enumerate(aim_runs):
            label = f"result.aim_runs[{index}]"
            object_fields(run, {"hash", "experiment", "aimx", "tags"},
                          {"hash", "experiment", "aimx", "tags"}, label)
            run_hash = run["hash"]
            nonempty_text(run_hash, f"{label}.hash")
            require(run_hash not in seen_hashes, f"Duplicate Aim run hash in {label}: {run_hash}")
            seen_hashes.add(run_hash)
            params = run["aimx"]
            object_fields(params, {"loop_id", "round", "branch", "task_id", "role", "hypothesis"},
                          {"loop_id", "round", "role", "hypothesis"}, f"{label}.aimx")
            require(params["loop_id"] == loop_id, f"{label}.aimx.loop_id must equal the checkpoint loop {loop_id}")
            require(type(params["round"]) is int and params["round"] == round_number,
                    f"{label}.aimx.round must equal the current round {round_number}")
            dispatch_fields = {"branch", "task_id"} & set(params)
            require(len(dispatch_fields) == 1,
                    f"{label}.aimx must contain exactly one of branch or task_id")
            dispatch_field = next(iter(dispatch_fields))
            nonempty_text(params[dispatch_field], f"{label}.aimx.{dispatch_field}")
            role = params["role"]
            require(isinstance(role, str) and role in {"baseline", "candidate", "control"},
                    f"{label}.aimx.role must be baseline, candidate, or control")
            hypothesis = params["hypothesis"]
            nonempty_text(hypothesis, f"{label}.aimx.hypothesis")
            require("\n" not in hypothesis and "\r" not in hypothesis,
                    f"{label}.aimx.hypothesis must be one line")
            nonempty_text(run["experiment"], f"{label}.experiment")
            require(run["experiment"] == f"ar-{loop_id}",
                    f"{label}.experiment must equal ar-{loop_id}")
            text_list(run["tags"], f"{label}.tags", nonempty=True)
            required_tags = {f"loop:{loop_id}", f"round:{params['round']}"}
            require(required_tags <= set(run["tags"]),
                    f"{label}.tags must include {sorted(required_tags)}")
        aim_evidence_hashes = {reference.removeprefix("aim_run:") for reference in result["evidence"]
                               if reference.startswith("aim_run:")}
        require(aim_evidence_hashes == seen_hashes,
                "result.evidence aim_run: references must match result.aim_runs hashes exactly; "
                f"missing from evidence: {sorted(seen_hashes - aim_evidence_hashes)}, "
                f"missing from aim_runs: {sorted(aim_evidence_hashes - seen_hashes)}")


def patch_agent(agent: dict[str, Any], patch: Any, loop_id: str, round_number: int) -> None:
    object_fields(patch, {"id", "dispatch_status", "task_id", "task_file", "worktree", "result"},
                  {"id"}, "agent checkpoint")
    updated = {**agent, **copy.deepcopy(patch)}
    require(isinstance(updated["dispatch_status"], str)
            and updated["dispatch_status"] in DISPATCH_TRANSITIONS[agent["dispatch_status"]],
            "Cannot rewind or repeat an agent dispatch")
    require(agent["dispatch_status"] != "finished" or updated == agent, "Cannot change a finished slot")
    for key in ("task_id", "task_file", "worktree"):
        if key in patch:
            nonempty_text(patch[key], key)
            require(agent[key] is None or agent[key] == patch[key], f"Cannot replace recorded {key}")
    if updated["dispatch_status"] in {"reserved", "started"}:
        require(updated["task_file"] is not None and updated["worktree"] is not None,
                "Save task_file and worktree before reserving a dispatch")
    if updated["dispatch_status"] == "started":
        require(updated["task_id"] is not None, "Started dispatch needs the native task_id")
    if "result" in patch:
        validate_result(patch["result"], loop_id, round_number)
        require(agent["result"] is None or agent["result"] == patch["result"], "Cannot replace a recorded result")
    require((updated["dispatch_status"] == "finished") == (updated["result"] is not None),
            "Finished dispatch must have a result; other dispatches must not")
    if agent["dispatch_status"] == "pending" and updated["dispatch_status"] == "finished":
        require(updated["result"]["status"] != "succeeded", "An undispatched agent cannot succeed")
    agent.update(updated)


def validate_assessment(value: Any, criteria: list[str]) -> bool:
    require(isinstance(value, list) and len(value) == len(criteria), "Assess every original success criterion")
    for item, criterion in zip(value, criteria):
        object_fields(item, {"criterion", "met", "evidence", "rationale"},
                      {"criterion", "met", "evidence", "rationale"}, "goal assessment")
        require(item["criterion"] == criterion, "Cannot change or reorder the original success criteria")
        require(type(item["met"]) is bool, "assessment.met must be a boolean")
        nonempty_text(item["rationale"], "assessment.rationale")
        text_list(item["evidence"], "assessment.evidence", nonempty=item["met"])
    return all(item["met"] for item in value)


def checkpoint(directory: Path, patch: Any, config_path: Path | None = None) -> dict[str, Any]:
    allowed = {"round", "agents", "artifacts", "merged_ref", "update_file", "research_update",
               "complete", "goal_assessment", "status", "stop_reason"}
    object_fields(patch, allowed, set(), "checkpoint")
    require(bool(patch), "Checkpoint must not be empty")
    with writer_lock(directory):
        original = load_state(directory, config_path)
        require(original["status"] not in TERMINAL_STATUSES or "status" not in patch,
                "A terminal loop cannot be restarted", "loop_stopped")
        state = copy.deepcopy(original)
        if "round" in patch:
            positive_int(patch["round"], "round")
            require(patch["round"] == state["rounds_started"], "Only the current round can be checkpointed")
            require(bool(state["rounds"]), "No round has started")
            current = state["rounds"][-1]
            agent_patches = patch.get("agents", [])
            require(isinstance(agent_patches, list), "agents checkpoint must be an array")
            by_id = {agent["id"]: agent for agent in current["agents"]}
            seen: set[str] = set()
            for agent_patch in agent_patches:
                require(isinstance(agent_patch, dict) and isinstance(agent_patch.get("id"), str), "Agent id is required")
                agent_id = agent_patch["id"]
                require(agent_id in by_id and agent_id not in seen, "Unknown or duplicate checkpoint agent")
                seen.add(agent_id)
                patch_agent(by_id[agent_id], agent_patch, state["loop_id"], current["round"])
            for key in ("merged_ref", "update_file"):
                if key in patch:
                    nonempty_text(patch[key], key)
                    if key == "merged_ref":
                        require(current[key] is None or current[key] == patch[key], "Cannot replace merged_ref")
                    else:
                        require(current["research_update"] is None or current[key] == patch[key],
                                "Cannot replace the committed update_file")
                    current[key] = patch[key]
            if "artifacts" in patch:
                text_list(patch["artifacts"], "artifacts")
                current["artifacts"] = list(dict.fromkeys([*current["artifacts"], *patch["artifacts"]]))
            if "research_update" in patch:
                update = patch["research_update"]
                object_fields(update, {"client_update_id", "commit_id", "revision"},
                              {"client_update_id", "commit_id", "revision"}, "research_update receipt")
                nonempty_text(update["client_update_id"], "client_update_id")
                nonempty_text(update["commit_id"], "commit_id")
                positive_int(update["revision"], "revision")
                require(current["update_file"] is not None, "Save the exact update_file before its receipt")
                require(current["research_update"] is None or current["research_update"] == update,
                        "Cannot replace a committed ResearchUpdate receipt")
                current["research_update"] = copy.deepcopy(update)
            if "complete" in patch:
                require(type(patch["complete"]) is bool, "complete must be a boolean")
                if patch["complete"]:
                    require(all(agent["dispatch_status"] == "finished" for agent in current["agents"]),
                            "Reconcile all agent outcomes before completing a round")
                    require(current["research_update"] is not None, "Lead ResearchUpdate receipt is required")
                    require("goal_assessment" in patch, "A completed round needs a fresh goal assessment")
                    current["status"] = "completed"
            if original["rounds"][-1]["status"] == "completed":
                require(current == original["rounds"][-1], "Cannot change a completed round")
        else:
            require(not (set(patch) & {"agents", "artifacts", "merged_ref", "update_file", "research_update", "complete"}),
                    "Round fields require round")
        if "goal_assessment" in patch:
            validate_assessment(patch["goal_assessment"], state["config"]["goal"]["success_criteria"])
            state["goal_assessment"] = copy.deepcopy(patch["goal_assessment"])
        if "status" in patch:
            require(isinstance(patch["status"], str) and patch["status"] in {"running", "blocked", "interrupted"},
                    "Terminal status is derived from evidence and budget")
            if patch["status"] != "running":
                nonempty_text(patch.get("stop_reason"), "stop_reason")
            state["status"] = patch["status"]
            state["stop_reason"] = None if patch["status"] == "running" else patch["stop_reason"]
        else:
            require("stop_reason" not in patch, "stop_reason requires status")
        settled = not state["rounds"] or state["rounds"][-1]["status"] == "completed"
        if settled and state["goal_assessment"] and all(item["met"] for item in state["goal_assessment"]):
            state["status"] = "goal_met"
            state["stop_reason"] = "All original success criteria are supported by evidence."
        elif settled and state["rounds_started"] == state["config"]["max_rounds"]:
            state["status"] = "budget_exhausted"
            state["stop_reason"] = "Round budget exhausted before all success criteria were met."
        require(original["status"] not in TERMINAL_STATUSES or state == original,
                "A terminal loop cannot be changed or restarted", "loop_stopped")
        if state != original:
            write_state(directory, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "init"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "init":
            command.add_argument("--loop-id")
    for name in ("begin-round", "checkpoint", "status"):
        command = commands.add_parser(name)
        command.add_argument("--loop-dir", type=Path, required=True)
        command.add_argument("--config", type=Path, help="Verify the original config when resuming")
        if name == "begin-round":
            command.add_argument("--round", type=int, required=True)
            command.add_argument("--baseline-ref", required=True)
        elif name == "checkpoint":
            command.add_argument("--file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate(args.config)
        elif args.command == "init":
            result = init(args.config, args.loop_id)
        else:
            directory = args.loop_dir.resolve()
            if args.command == "begin-round":
                state = begin_round(directory, args.round, args.baseline_ref, args.config)
            elif args.command == "checkpoint":
                state = checkpoint(directory, read_json(args.file), args.config)
            else:
                state = load_state(directory, args.config)
            result = {"loop_dir": str(directory), "state": state}
    except (ControlError, OSError) as exc:
        print(json.dumps({"ok": False, "error": {"code": getattr(exc, "code", "io_error"), "message": str(exc)}}))
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
