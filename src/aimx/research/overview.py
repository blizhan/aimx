from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from aimx.research.errors import ValidationError
from aimx.research.models import ResearchState
from aimx.research.store import database_path, normalize_repo_root


def compile_overview(
    repo_value: str | Path,
    state: ResearchState | None,
    loop_id: str | None = None,
    *,
    state_error: str | None = None,
) -> dict[str, Any]:
    root = normalize_repo_root(Path(repo_value))
    activities = state.activities if state is not None else {}
    loops = [activity for activity in activities.values() if activity.get("kind") == "loop"]
    loops.sort(key=lambda activity: (int(activity.get("created_revision", 0)), str(activity.get("loop_id", ""))))
    if loop_id is None:
        discovered = []
        known_ids: set[str] = set()
        for activity in loops:
            current_id = str(activity.get("loop_id"))
            known_ids.add(current_id)
            checkpoint = _checkpoint_location(root, current_id)
            checkpoint_status, checkpoint_value, error = _read_checkpoint(Path(checkpoint), current_id)
            activity_id = activity.get("id")
            activity_annotations = [item for item in state.annotations.values() if item.get("source_activity_id") == activity_id] if state is not None else []
            activity_findings = [item for item in state.findings.values() if item.get("source_activity_id") == activity_id] if state is not None else []
            discovered.append({"loop_id": current_id, "registered": True, "source_activity_id": activity_id, "objective": activity.get("objective"), "title": activity.get("title"), "checkpoint_location": checkpoint, "checkpoint_status": checkpoint_status, "report_locations": list(activity.get("report_locations", [])), "artifact_locations": list(activity.get("artifact_locations", [])), "annotation_count": len(activity_annotations), "finding_count": len(activity_findings), "finding_summaries": [item.get("claim") for item in activity_findings], **({"error": error} if error else {}), "loop_status": checkpoint_value.get("status") if checkpoint_value else None, "goal_status": _goal_status(checkpoint_value) if checkpoint_value else "unknown"})
        for checkpoint_path in sorted((root / ".aimx" / "autoresearch").glob("*/state.json")):
            directory_id = checkpoint_path.parent.name
            if directory_id in known_ids:
                continue
            checkpoint_status, checkpoint_value, error = _read_checkpoint(checkpoint_path, directory_id)
            discovered.append({"loop_id": checkpoint_value.get("loop_id") if checkpoint_value else directory_id, "registered": False, "source_activity_id": None, "objective": None, "title": None, "checkpoint_location": str(checkpoint_path), "checkpoint_status": checkpoint_status, **({"error": error} if error else {}), "loop_status": checkpoint_value.get("status") if checkpoint_value else None, "goal_status": _goal_status(checkpoint_value) if checkpoint_value else "unknown", "ownership": "checkpoint discovered; research ownership is unregistered"})
        incomplete = state_error is not None or any(item["checkpoint_status"] in {"missing", "unreadable"} for item in discovered)
        result = {
            "schema_version": 1,
            "status": "partial" if incomplete else "complete",
            "repo": str(root),
            "revision": state.revision if state is not None else None,
            "state_location": str(database_path(root)),
            "loops": discovered,
            "counts": _counts(state) if state is not None else None,
            "unattributed": _unattributed(state) if state is not None else None,
            "query_commands": _query_commands(root),
        }
        if state_error:
            result["error"] = state_error
        return result
    matches = [activity for activity in loops if activity.get("loop_id") == loop_id]
    if state is not None and not matches:
        raise ValidationError(f"Unknown loop: {loop_id}", code="unknown_loop")
    activity = matches[0] if matches else None
    checkpoint = _checkpoint_location(root, loop_id)
    checkpoint_status, checkpoint_state, checkpoint_error = _read_checkpoint(Path(checkpoint), loop_id)
    pending = [value for value in (state_error, checkpoint_error) if value]
    findings = [item for item in state.findings.values() if item.get("source_activity_id") == activity["id"]] if state is not None and activity else []
    annotations = [item for item in state.annotations.values() if item.get("source_activity_id") == activity["id"]] if state is not None and activity else []
    agenda = [item for item in state.agenda_items.values() if item.get("source_activity_id") == activity["id"]] if state is not None and activity else []
    config = checkpoint_state.get("config", {}) if checkpoint_state else {}
    result = {
        "schema_version": 1,
        "status": "complete" if checkpoint_status == "readable" and state_error is None and activity is not None else "partial",
        "repo": str(root),
        "revision": state.revision if state is not None else None,
        "loop_id": loop_id,
        "source_activity_id": activity.get("id") if activity else None,
        "state_location": str(database_path(root)),
        "checkpoint_location": checkpoint,
        "checkpoint_status": checkpoint_status,
        "loop_status": checkpoint_state.get("status") if checkpoint_state else None,
        "goal_status": _goal_status(checkpoint_state) if checkpoint_state else "unknown",
        "direction_status": _direction_status(state, activity) if state is not None and activity else "unknown",
        "stop_reason": checkpoint_state.get("stop_reason") if checkpoint_state else None,
        "rounds_started": checkpoint_state.get("rounds_started") if checkpoint_state else None,
        "max_rounds": config.get("max_rounds"),
        "goal_assessment": checkpoint_state.get("goal_assessment", []) if checkpoint_state else [],
        "original_criteria": config.get("goal", {}).get("success_criteria", []),
        "goal_assessment_details": checkpoint_state.get("goal_assessment", []) if checkpoint_state else [],
        "goal": config.get("goal", {}),
        "counts": {"findings": len(findings), "annotations": len(annotations), "agenda_items": len(agenda)} if state is not None else None,
        "unattributed": _unattributed(state) if state is not None else None,
        "summaries": [item.get("claim") for item in findings],
        "persisted_receipts": _persisted_receipts(checkpoint_state),
        "latest_persisted_receipt": (_persisted_receipts(checkpoint_state)[-1] if _persisted_receipts(checkpoint_state) else None),
        "external_artifacts": _external_artifacts(root, checkpoint_state, activity),
        "report_locations": list(activity.get("report_locations", [])) if activity else [],
        "attempts": _attempt_summary(state, activity["id"]) if state is not None and activity else None,
        "outcome_differences": _outcome_differences(checkpoint_state, state, activity["id"]) if state is not None and activity else [],
        "pending_reconciliation": pending,
        "follow_up_activities": [
            {
                "id": item["id"],
                "title": item.get("title"),
                "objective": item.get("objective"),
                "findings": [finding for finding in state.findings.values() if finding.get("source_activity_id") == item.get("id")],
                "annotation_count": sum(annotation.get("source_activity_id") == item.get("id") for annotation in state.annotations.values()),
                "artifact_locations": list(item.get("artifact_locations", [])),
                "report_locations": list(item.get("report_locations", [])),
            }
            for item in activities.values()
            if item.get("kind") == "follow_up" and item.get("related_loop_id") == loop_id
        ],
        "query_commands": _query_commands(root, loop_id),
    }
    if not activity and state_error:
        result["ownership"] = "research state unreadable; loop registration could not be verified"
    return result


def _checkpoint_location(root: Path, loop_id: Any) -> str:
    return str(root / ".aimx" / "autoresearch" / str(loop_id) / "state.json")


def _read_checkpoint(path: Path, loop_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "missing", None, "registered loop has no standard checkpoint"
    except (OSError, UnicodeError) as exc:
        return "unreadable", None, f"checkpoint unavailable: {exc}"
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("loop_id") != loop_id:
            raise ValueError("checkpoint root or loop_id mismatch")
        return "readable", value, None
    except (json.JSONDecodeError, ValueError) as exc:
        return "unreadable", None, f"checkpoint unavailable: {exc}"


def _query_commands(root: Path, loop_id: str | None = None) -> list[str]:
    repo = shlex.quote(str(root))
    commands = [f"aimx finding ls --repo {repo} --json", f"aimx research annotations --repo {repo} --json"]
    if loop_id:
        loop = shlex.quote(loop_id)
        commands = [f"aimx finding ls --repo {repo} --loop {loop} --json", f"aimx research annotations --repo {repo} --loop {loop} --json"]
    return commands


def _direction_status(state: ResearchState, activity: dict[str, Any]) -> str:
    related = [item for item in state.frontier_items.values() if item.get("source_activity_id") == activity.get("id")]
    if not related:
        return "active"
    statuses = {str(item.get("status", "active")) for item in related}
    return "retired" if statuses == {"retired"} else "paused" if statuses == {"paused"} else "active"


def _external_artifacts(root: Path, checkpoint: dict[str, Any] | None, activity: dict[str, Any] | None) -> list[dict[str, Any]]:
    paths = [value for value in activity.get("artifact_locations", []) if isinstance(value, str)] if activity else []
    paths += [value for value in activity.get("report_locations", []) if isinstance(value, str)] if activity else []
    if checkpoint:
        rounds = checkpoint.get("rounds", [])
        if isinstance(rounds, list):
            for round_state in rounds:
                if not isinstance(round_state, dict):
                    continue
                artifacts = round_state.get("artifacts", [])
                if isinstance(artifacts, list):
                    paths.extend(value for value in artifacts if isinstance(value, str))
                agents = round_state.get("agents", [])
                if isinstance(agents, list):
                    paths.extend(slot["result_file"] for slot in agents if isinstance(slot, dict) and isinstance(slot.get("result_file"), str))
    result = []
    for value in dict.fromkeys(path for path in paths if isinstance(path, str) and path):
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            if activity is not None and value in activity.get("report_locations", []) + activity.get("artifact_locations", []):
                result.append({"location": value, "persistence": "external_artifact", "availability": "unknown", "reason": "relative activity artifact has no recorded base"})
            else:
                candidate = root / value
        if not candidate.is_absolute():
            result.append({"location": value, "persistence": "external_artifact", "availability": "unknown", "reason": "relative artifact location has no recorded base"})
            continue
        try:
            candidate.stat()
            availability = "available"
            reason = None
        except FileNotFoundError:
            availability = "missing"
            reason = "artifact path does not exist"
        except OSError as exc:
            availability = "unreadable"
            reason = str(exc)
        result.append({"location": value, "persistence": "external_artifact", "availability": availability, **({"reason": reason} if reason else {})})
    return result


def _persisted_receipts(checkpoint: dict[str, Any] | None) -> list[dict[str, Any]]:
    if checkpoint is None:
        return []
    receipts = []
    rounds = checkpoint.get("rounds", [])
    if not isinstance(rounds, list):
        return receipts
    for round_state in rounds:
        if not isinstance(round_state, dict):
            continue
        receipt = round_state.get("research_update")
        if isinstance(receipt, dict):
            receipts.append({key: receipt[key] for key in ("client_update_id", "commit_id", "revision") if key in receipt})
    return receipts


def _attempt_summary(state: ResearchState, activity_id: str) -> dict[str, Any]:
    attempts = [item for item in state.outcomes.values() if item.get("source_activity_id") == activity_id]
    attempts.sort(key=lambda value: (int(value.get("created_revision", 0)), str(value.get("id", ""))))
    axes = ("execution_status", "artifact_status", "evaluation_status", "audit_status")
    resources = [dict(resource, attempt_id=item.get("id"), attempt_key=item.get("attempt_key")) for item in attempts for resource in item.get("resources", []) if isinstance(resource, dict)]
    duplicates: dict[tuple[Any, Any], list[str]] = {}
    repeated_attempt_keys: dict[str, list[str]] = {}
    for item in attempts:
        slot = item.get("slot_id")
        if slot is not None:
            duplicates.setdefault((item.get("round"), slot), []).append(str(item.get("id")))
        key = item.get("attempt_key")
        if isinstance(key, str):
            repeated_attempt_keys.setdefault(key, []).append(str(item.get("id")))
    duplicate_slots = [{"round": key[0], "slot_id": key[1], "attempt_ids": ids} for key, ids in sorted(duplicates.items(), key=lambda pair: (str(pair[0][0]), str(pair[0][1]))) if len(ids) > 1]
    over_plan = [resource for resource in resources if isinstance(resource.get("actual"), (int, float)) and isinstance(resource.get("planned"), (int, float)) and resource["actual"] > resource["planned"]]
    return {
        "count": len(attempts),
        "attempt_keys": [item.get("attempt_key") for item in attempts],
        "axes": {axis: {value: sum(item.get(axis, "unknown") == value for item in attempts) for value in sorted({str(item.get(axis, "unknown")) for item in attempts})} for axis in axes},
        "checkpoint_statuses": {str(item.get("status", "unknown")): sum(item.get("status", "unknown") == value for item in attempts) for value in sorted({str(item.get("status", "unknown")) for item in attempts})},
        "failure_policies": [
            {"agenda_item_id": item.get("id"), "failure_policy": item.get("experiment_contract", {}).get("failure_policy", "unknown")}
            for item in state.agenda_items.values() if item.get("source_activity_id") == activity_id
        ],
        "requested_settings": [{"attempt_id": item.get("id"), "settings": item["requested_settings"]} for item in attempts if "requested_settings" in item],
        "actual_settings": [{"attempt_id": item.get("id"), "settings": item["actual_settings"]} for item in attempts if "actual_settings" in item],
        "resources": resources,
        "retry_chains": [
            {"attempt_id": item.get("id"), "attempt_key": item.get("attempt_key"), "retry_of": item.get("retry_of"), "round": item.get("round"), "slot_id": item.get("slot_id")}
            for item in attempts if item.get("retry_of") is not None
        ],
        "resource_summary": {
            "over_plan": over_plan,
            "unverified": [resource for resource in resources if resource.get("verification") != "verified"],
            "not_enforced": [resource for resource in resources if resource.get("enforcement") != "executor_enforced"],
            "missing_logs": [item.get("id") for item in attempts if not item.get("resources")],
        },
        "duplicate_slots": duplicate_slots,
        "duplicate_attempt_keys": [{"attempt_key": key, "attempt_ids": ids} for key, ids in sorted(repeated_attempt_keys.items()) if len(ids) > 1],
    }


def _outcome_differences(checkpoint: dict[str, Any] | None, state: ResearchState, activity_id: str) -> list[dict[str, Any]]:
    if checkpoint is None:
        return []
    attempts = [item for item in state.outcomes.values() if item.get("source_activity_id") == activity_id]
    differences = []
    rounds = checkpoint.get("rounds", [])
    if not isinstance(rounds, list):
        return [{"research_attempt_recorded": False, "reason": "checkpoint rounds are malformed"}]
    finished = []
    for round_state in rounds:
        if not isinstance(round_state, dict) or not isinstance(round_state.get("agents", []), list):
            continue
        finished.extend((round_state.get("round"), slot) for slot in round_state.get("agents", []) if isinstance(slot, dict) and slot.get("dispatch_status") == "finished")
    matched_ids: set[str] = set()
    for round_number, slot in finished:
        result = slot.get("result", {})
        if not isinstance(result, dict):
            result = {}
        match = next((item for item in attempts if item.get("id") not in matched_ids and item.get("round") == round_number and item.get("slot_id") == slot.get("id")), None)
        if match is None and result.get("summary"):
            match = next((item for item in attempts if item.get("id") not in matched_ids and item.get("summary") == result.get("summary")), None)
        if match is None:
            differences.append({"round": round_number, "slot_id": slot.get("id"), "checkpoint_summary": result.get("summary"), "research_attempt_recorded": False, "checkpoint_status": result.get("status"), "research_axes": None})
            continue
        matched_ids.add(str(match.get("id")))
        checkpoint_status = result.get("status")
        attempt_status = match.get("execution_status", "unknown")
        axes = {axis: match.get(axis, "unknown") for axis in ("execution_status", "artifact_status", "evaluation_status", "audit_status")}
        if checkpoint_status != attempt_status:
            differences.append({"round": round_number, "slot_id": slot.get("id"), "checkpoint_summary": result.get("summary"), "research_attempt_id": match.get("id"), "research_attempt_recorded": True, "checkpoint_status": checkpoint_status, "research_axes": axes, "status_difference": True})
        elif match.get("artifact_status", "unknown") != "unknown" or match.get("evaluation_status", "unknown") != "unknown" or match.get("audit_status", "unknown") != "unknown":
            differences.append({"round": round_number, "slot_id": slot.get("id"), "checkpoint_summary": result.get("summary"), "research_attempt_id": match.get("id"), "research_attempt_recorded": True, "checkpoint_status": checkpoint_status, "research_axes": axes, "status_difference": False})
    return differences


def _counts(state: ResearchState) -> dict[str, int]:
    return {"findings": len(state.findings), "annotations": len(state.annotations), "agenda_items": len(state.agenda_items)}


def _unattributed(state: ResearchState) -> dict[str, int]:
    return {
        "findings": sum("source_activity_id" not in item for item in state.findings.values()),
        "annotations": sum("source_activity_id" not in item for item in state.annotations.values()),
    }


def _goal_status(checkpoint: dict[str, Any] | None) -> str:
    if checkpoint is None:
        return "unknown"
    assessment = checkpoint.get("goal_assessment", [])
    criteria = checkpoint.get("config", {}).get("goal", {}).get("success_criteria")
    if isinstance(assessment, list) and assessment:
        assessed = {
            item.get("criterion"): item.get("met")
            for item in assessment
            if isinstance(item, dict) and isinstance(item.get("criterion"), str)
        }
        if isinstance(criteria, list) and criteria:
            if len(assessed) == len(criteria) and all(
                assessed.get(criterion) is True for criterion in criteria
            ):
                return "met"
            if any(assessed.get(criterion) is False for criterion in criteria):
                return "not_met"
        else:
            if all(isinstance(item, dict) and item.get("met") is True for item in assessment):
                return "met"
            if any(isinstance(item, dict) and item.get("met") is False for item in assessment):
                return "not_met"
    if checkpoint.get("status") in {"blocked", "interrupted", "running"}:
        return "unknown"
    return "unknown"
