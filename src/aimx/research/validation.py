from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from aimx.research.errors import ValidationError
from aimx.research.events import SUPPORTED_OPERATIONS, event_type_for_operation
from aimx.research.ids import is_local_name, is_local_ref, local_name, new_id
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.replay import apply_event

_EPISTEMIC = {"candidate", "validated", "contradicted"}
_CONFIDENCE = {"low", "medium", "high"}
_GOVERNANCE = {"proposed", "accepted", "rejected"}
_RELATIONS = {
    "derived_from",
    "supports",
    "challenges",
    "refines",
    "supersedes",
    "tests",
    "related_to",
}
_FRONTIER_STATUS = {"active", "paused", "retired"}
_AGENDA_STATUS = {"proposed", "active", "completed", "failed", "abandoned"}
_AGENDA_TRANSITIONS = {
    "proposed": {"active", "abandoned"},
    "active": {"completed", "failed", "abandoned"},
    "failed": {"proposed"},
    "abandoned": {"proposed"},
    "completed": set(),
}
_AGENDA_TERMINAL_STATUS = {"completed"}


@dataclass(frozen=True)
class CompiledUpdate:
    events: tuple[dict[str, Any], ...]
    created_ids: dict[str, str]


def compile_update(
    update: ResearchUpdate,
    state: ResearchState,
    *,
    revision: int,
    commit_id: str,
    evidence_resolver: Callable[[str], str] | None = None,
    repo_root: Path | None = None,
) -> CompiledUpdate:
    _validate_update_header(update)
    working = state.clone()
    local_refs: dict[str, tuple[str, str]] = {}
    created_ids: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    attempt_keys: set[tuple[str, str]] = set()
    finding_refs: dict[str, set[str]] = {}

    for index, operation in enumerate(update.operations):
        if not isinstance(operation, dict):
            raise ValidationError(f"Operation {index} must be an object", code="invalid_schema")
        op_name = operation.get("op")
        if op_name not in SUPPORTED_OPERATIONS:
            raise ValidationError(
                f"Unsupported research operation at index {index}: {op_name!r}",
                code="invalid_schema",
            )

        event_type = event_type_for_operation(op_name)
        payload, entity_id = _compile_operation(
            operation,
            op_name,
            working,
            local_refs,
            created_ids,
            evidence_resolver,
            repo_root,
        )
        _copy_optional_source(operation, payload, working, local_refs)
        _reject_unknown_operation_fields(operation)
        if event_type == "execution.outcome.record":
            attempt_identity = (str(payload["source_activity_id"]), str(payload["attempt_key"]))
            if attempt_identity in attempt_keys:
                raise ValidationError("Attempt key is already recorded for this activity", code="attempt_conflict")
            attempt_keys.add(attempt_identity)
        if event_type == "finding.create" and entity_id is not None:
            finding_refs[str(entity_id)] = {hashlib.sha256(json.dumps(ref, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() for ref in payload.get("evidence", [])}
        if event_type == "evidence.review":
            finding_id = str(payload["finding_id"])
            reference_fingerprints = finding_refs.get(finding_id)
            if reference_fingerprints is None:
                reference_fingerprints = {hashlib.sha256(json.dumps(ref, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() for ref in working.findings[finding_id].get("evidence", [])}
            if payload["evidence_fingerprint"] not in reference_fingerprints:
                raise ValidationError("Evidence review fingerprint does not match a finding evidence reference", code="evidence_identity_mismatch")
        payload.setdefault("author", update.author.as_dict())
        payload.setdefault("created_commit_id", commit_id)
        payload.setdefault("created_revision", revision)
        apply_event(
            working,
            event_type,
            payload,
            commit_id=commit_id,
            revision=revision,
        )
        events.append(
            {
                "event_type": event_type,
                "entity_id": entity_id,
                "payload": payload,
            }
        )

    return CompiledUpdate(events=tuple(events), created_ids=created_ids)


def _validate_update_header(update: ResearchUpdate) -> None:
    if update.schema_version != 1 or isinstance(update.schema_version, bool):
        raise ValidationError(
            f"Unsupported ResearchUpdate schema version: {update.schema_version!r}",
            code="invalid_schema",
        )
    if not isinstance(update.base_revision, int) or isinstance(update.base_revision, bool) or update.base_revision < 0:
        raise ValidationError("base_revision must be a non-negative integer", code="invalid_schema")
    if not update.operations:
        raise ValidationError("ResearchUpdate operations must not be empty", code="invalid_schema")
    if update.author.kind not in {"agent", "human", "system"}:
        raise ValidationError(
            f"Unsupported author kind: {update.author.kind!r}",
            code="invalid_schema",
        )
    if not update.author.name.strip():
        raise ValidationError(
            "ResearchUpdate author.name must not be empty",
            code="invalid_schema",
        )
    if update.client_update_id is not None and not update.client_update_id.strip():
        raise ValidationError(
            "client_update_id must not be empty",
            code="invalid_schema",
        )
    if update.message is not None and not isinstance(update.message, str):
        raise ValidationError("message must be a string", code="invalid_schema")


def _compile_operation(
    operation: dict[str, Any],
    op_name: str,
    state: ResearchState,
    local_refs: dict[str, tuple[str, str]],
    created_ids: dict[str, str],
    evidence_resolver: Callable[[str], str] | None,
    repo_root: Path | None,
) -> tuple[dict[str, Any], str | None]:
    if op_name == "finding.create":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "finding", local_refs, created_ids)
        claim = _required_text(operation, "claim")
        payload = {
            "id": entity_id,
            "claim": claim,
            "evidence": _canonical_evidence(operation.get("evidence", []), evidence_resolver, state=state, local_refs=local_refs),
            "epistemic_status": _enum(operation, "epistemic_status", _EPISTEMIC, "candidate"),
            "confidence": _enum(operation, "confidence", _CONFIDENCE, "low"),
            "governance_status": _enum(operation, "governance_status", _GOVERNANCE, "proposed"),
        }
        _copy_optional_source(operation, payload, state, local_refs)
        semantic_fields = {"result_kind", "scope", "limitations"}
        if semantic_fields.intersection(operation):
            missing = semantic_fields | {"topics", "source_activity_id"}
            if missing - set(operation) or not payload["evidence"]:
                raise ValidationError(
                    "Structured findings require result_kind, scope, limitations, topics, source_activity_id and evidence",
                    code="incomplete_finding",
                )
            result_kind = operation["result_kind"]
            if result_kind not in {"improvement", "degradation", "insufficient_evidence", "scope_boundary"}:
                raise ValidationError("Unsupported result_kind", code="invalid_schema")
            payload["result_kind"] = result_kind
            payload["scope"] = _validate_scope(operation["scope"], state, local_refs)
            payload["limitations"] = _text_list(operation["limitations"], "limitations")
            payload["topics"] = _text_list(operation["topics"], "topics", nonempty=True)
            annotation_ids = operation.get("source_annotation_ids", [])
            payload["source_annotation_ids"] = _entity_list(annotation_ids, "annotation", state, local_refs)
        elif "topics" in operation or "source_annotation_ids" in operation:
            if "topics" in operation:
                payload["topics"] = _text_list(operation["topics"], "topics", nonempty=True)
            if "source_annotation_ids" in operation:
                payload["source_annotation_ids"] = _entity_list(operation["source_annotation_ids"], "annotation", state, local_refs)
        return payload, entity_id

    if op_name == "finding.assess":
        _reject_unknown_operation_fields(operation)
        finding_id = _resolve(operation.get("finding_id"), "finding", state, local_refs)
        if "epistemic_status" not in operation and "confidence" not in operation:
            raise ValidationError("finding.assess needs epistemic_status or confidence")
        payload = {"finding_id": finding_id}
        if "epistemic_status" in operation:
            payload["epistemic_status"] = _enum(operation, "epistemic_status", _EPISTEMIC)
        if "confidence" in operation:
            payload["confidence"] = _enum(operation, "confidence", _CONFIDENCE)
        _optional_reason(operation, payload)
        return payload, finding_id

    if op_name == "finding.governance":
        _reject_unknown_operation_fields(operation)
        finding_id = _resolve(operation.get("finding_id"), "finding", state, local_refs)
        payload = {
            "finding_id": finding_id,
            "governance_status": _enum(operation, "governance_status", _GOVERNANCE),
        }
        _optional_reason(operation, payload)
        return payload, finding_id

    if op_name == "relation.create":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "relation", local_refs, created_ids)
        source = _resolve(operation.get("source"), "finding", state, local_refs)
        target = _resolve(operation.get("target"), "finding", state, local_refs)
        if source == target:
            raise ValidationError("A relation cannot connect a finding to itself")
        relation_type = operation.get("relation_type")
        if relation_type not in _RELATIONS:
            raise ValidationError(f"Unsupported relation type: {relation_type!r}")
        for relation in state.relations.values():
            if (
                relation.get("active", True)
                and relation.get("source_finding_id") == source
                and relation.get("target_finding_id") == target
                and relation.get("type") == relation_type
            ):
                raise ValidationError("An identical active relation already exists")
        payload = {
            "id": entity_id,
            "source_finding_id": source,
            "target_finding_id": target,
            "type": relation_type,
            "active": True,
        }
        _optional_reason(operation, payload)
        return payload, entity_id

    if op_name == "relation.retract":
        _reject_unknown_operation_fields(operation)
        relation_id = _resolve(operation.get("relation_id"), "relation", state, local_refs)
        relation = state.relations[relation_id]
        if not relation.get("active", True):
            raise ValidationError("Relation is already retracted", code="invalid_transition")
        payload = {"relation_id": relation_id}
        _optional_reason(operation, payload)
        return payload, relation_id

    if op_name == "annotation.create":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "annotation", local_refs, created_ids)
        target_kind = operation.get("annotation_target_kind", "research_state")
        if target_kind not in {"research_state", "finding", "relation", "frontier_item", "agenda_item"}:
            raise ValidationError(f"Unsupported annotation target kind: {target_kind!r}")
        target_id = operation.get("annotation_target_id")
        if target_kind == "research_state":
            target_id = None
        elif target_id is None:
            raise ValidationError("Annotation target_id is required for this target kind")
        else:
            target_id = _resolve(target_id, _kind_for_target(target_kind), state, local_refs)
        return {
            "id": entity_id,
            "target_kind": target_kind,
            "target_id": target_id,
            "text": _required_text(operation, "text"),
        }, entity_id

    if op_name == "frontier.lane.create":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "lane", local_refs, created_ids)
        name = _required_text(operation, "name")
        if any(lane.get("active", True) and lane.get("name") == name for lane in state.lanes.values()):
            raise ValidationError(f"Active frontier lane already exists: {name}")
        return {
            "id": entity_id,
            "name": name,
            "description": operation.get("description"),
            "active": True,
        }, entity_id

    if op_name in {"frontier.lane.rename", "frontier.lane.retire"}:
        _reject_unknown_operation_fields(operation)
        lane_id = _resolve(operation.get("lane_id"), "lane", state, local_refs)
        lane = state.lanes[lane_id]
        if not lane.get("active", True):
            raise ValidationError("Frontier lane is retired", code="invalid_transition")
        payload = {"lane_id": lane_id}
        if op_name.endswith("rename"):
            name = _required_text(operation, "name")
            if any(
                other_id != lane_id
                and other.get("active", True)
                and other.get("name") == name
                for other_id, other in state.lanes.items()
            ):
                raise ValidationError(f"Active frontier lane already exists: {name}")
            payload["name"] = name
        return payload, lane_id

    if op_name == "frontier.item.create":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "frontier", local_refs, created_ids)
        lane_id = _resolve(operation.get("lane_id"), "lane", state, local_refs)
        if not state.lanes[lane_id].get("active", True):
            raise ValidationError("Frontier item must use an active lane")
        finding_id = operation.get("finding_id")
        direction = operation.get("direction")
        if bool(finding_id) == bool(direction):
            raise ValidationError("Frontier item needs exactly one of finding_id or direction")
        if finding_id is not None:
            finding_id = _resolve(finding_id, "finding", state, local_refs)
        payload = {
            "id": entity_id,
            "lane_id": lane_id,
            "subject": {
                "kind": "finding" if finding_id else "direction",
                "finding_id": finding_id,
                "title": direction,
            },
            "rationale": _required_text(operation, "rationale"),
            "priority": _priority(operation),
            "status": _enum(operation, "frontier_status", _FRONTIER_STATUS, "active"),
            "supporting_finding_ids": _finding_list(operation.get("supporting_finding_ids", []), state, local_refs),
            "challenging_finding_ids": _finding_list(operation.get("challenging_finding_ids", []), state, local_refs),
        }
        return payload, entity_id

    if op_name in {
        "frontier.item.move",
        "frontier.item.set_priority",
        "frontier.item.set_status",
        "frontier.item.update",
    }:
        _reject_unknown_operation_fields(operation)
        item_id = _resolve(operation.get("frontier_item_id"), "frontier", state, local_refs)
        item = state.frontier_items[item_id]
        payload: dict[str, Any] = {"frontier_item_id": item_id}
        if op_name == "frontier.item.move":
            lane_id = _resolve(operation.get("lane_id"), "lane", state, local_refs)
            if not state.lanes[lane_id].get("active", True):
                raise ValidationError("Frontier item must use an active lane")
            payload["lane_id"] = lane_id
        elif op_name == "frontier.item.set_priority":
            payload["priority"] = _priority(operation)
        elif op_name == "frontier.item.set_status":
            payload["status"] = _enum(operation, "frontier_status", _FRONTIER_STATUS)
        else:
            if not any(key in operation for key in ("rationale", "supporting_finding_ids", "challenging_finding_ids")):
                raise ValidationError("frontier.item.update needs a changed field")
            if "rationale" in operation:
                payload["rationale"] = _required_text(operation, "rationale")
            if "supporting_finding_ids" in operation:
                payload["supporting_finding_ids"] = _finding_list(operation["supporting_finding_ids"], state, local_refs)
            if "challenging_finding_ids" in operation:
                payload["challenging_finding_ids"] = _finding_list(operation["challenging_finding_ids"], state, local_refs)
        if not item:
            raise ValidationError("Unknown frontier item", code="unknown_entity")
        return payload, item_id

    if op_name == "agenda.item.create":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "agenda", local_refs, created_ids)
        finding_ids = _finding_list(operation.get("motivating_finding_ids", []), state, local_refs)
        frontier_ids = _entity_list(operation.get("frontier_item_ids", []), "frontier", state, local_refs)
        if not finding_ids and not frontier_ids:
            raise ValidationError("Agenda item needs a finding or frontier motivation")
        evidence = _canonical_evidence(operation.get("evidence", []), evidence_resolver, state=state, local_refs=local_refs)
        payload = {
            "id": entity_id,
            "objective": _required_text(operation, "objective"),
            "hypothesis": operation.get("hypothesis"),
            "question": operation.get("question"),
            "motivating_finding_ids": finding_ids,
            "frontier_item_ids": frontier_ids,
            "controls": copy.deepcopy(operation.get("controls")),
            "factors": copy.deepcopy(operation.get("factors")),
            "constraints": copy.deepcopy(operation.get("constraints")),
            "success_criteria": copy.deepcopy(operation.get("success_criteria")),
            "priority": _priority(operation),
            "status": _enum(operation, "agenda_status", _AGENDA_STATUS, "proposed"),
            "evidence": evidence,
            "result_commit_ids": [],
        }
        if "experiment_contract" in operation:
            payload["experiment_contract"] = _validate_experiment_contract(operation["experiment_contract"], state, local_refs, evidence_resolver)
        if "source_activity_id" in operation:
            payload["source_activity_id"] = _resolve_activity(operation["source_activity_id"], state, local_refs)
        return payload, entity_id

    if op_name == "agenda.item.transition":
        _reject_unknown_operation_fields(operation)
        agenda_id = _resolve(operation.get("agenda_item_id"), "agenda", state, local_refs)
        current = state.agenda_items[agenda_id].get("status", "proposed")
        target = operation.get("agenda_status")
        if target not in _AGENDA_STATUS:
            raise ValidationError(f"Unsupported agenda status: {target!r}")
        if (target == current and current in _AGENDA_TERMINAL_STATUS) or (
            target != current and target not in _AGENDA_TRANSITIONS[current]
        ):
            raise ValidationError(
                f"Invalid agenda transition: {current} -> {target}", code="invalid_transition"
            )
        payload: dict[str, Any] = {"agenda_item_id": agenda_id, "status": target}
        if "evidence" in operation:
            payload["evidence"] = _canonical_evidence(operation["evidence"], evidence_resolver, state=state, local_refs=local_refs)
        _optional_reason(operation, payload)
        return payload, agenda_id

    if op_name == "agenda.item.attach_evidence":
        _reject_unknown_operation_fields(operation)
        agenda_id = _resolve(operation.get("agenda_item_id"), "agenda", state, local_refs)
        return {
            "agenda_item_id": agenda_id,
            "evidence": _canonical_evidence(operation.get("evidence", []), evidence_resolver, state=state, local_refs=local_refs),
        }, agenda_id

    if op_name == "agenda.item.set_priority":
        _reject_unknown_operation_fields(operation)
        agenda_id = _resolve(operation.get("agenda_item_id"), "agenda", state, local_refs)
        return {"agenda_item_id": agenda_id, "priority": _priority(operation)}, agenda_id

    if op_name == "activity.create":
        _reject_unknown_operation_fields(operation)
        if operation.get("kind") == "follow_up" and is_local_ref(operation.get("related_loop_id")):
            raise ValidationError("follow_up must reference a previously registered loop")
        entity_id = _declare(operation, "activity", local_refs, created_ids)
        kind = operation.get("kind")
        if kind not in {"loop", "analysis", "follow_up"}:
            raise ValidationError("Unsupported activity kind", code="invalid_schema")
        source = _validate_source(operation.get("source"))
        payload = {
            "id": entity_id,
            "kind": kind,
            "title": _required_text(operation, "title"),
            "objective": _required_text(operation, "objective"),
            "source": source,
            "report_locations": _text_list(operation.get("report_locations", []), "report_locations"),
            "artifact_locations": _text_list(operation.get("artifact_locations", []), "artifact_locations"),
        }
        if kind == "loop":
            loop_id = _required_text(operation, "loop_id")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", loop_id):
                raise ValidationError("loop_id must match the AutoResearch checkpoint ID format")
            if any(item.get("loop_id") == loop_id for item in state.activities.values()):
                raise ValidationError("A loop activity with this loop_id already exists")
            _validate_loop_checkpoint(state, operation.get("checkpoint"), loop_id, repo_root=repo_root)
            payload["loop_id"] = loop_id
        elif "loop_id" in operation:
            raise ValidationError("loop_id is only valid for loop activities")
        if kind == "follow_up":
            payload["related_loop_id"] = _resolve_loop_id(operation.get("related_loop_id"), state, local_refs)
        elif "related_loop_id" in operation:
            raise ValidationError("related_loop_id is only valid for follow_up activities")
        return payload, entity_id

    if op_name == "activity.artifact.add":
        _reject_unknown_operation_fields(operation)
        activity_id = _resolve(operation.get("activity_id"), "activity", state, local_refs)
        if activity_id not in state.activities:
            raise ValidationError("New activities cannot receive artifacts in the same update")
        payload = {
            "activity_id": activity_id,
            "locations": _text_list(operation.get("locations"), "locations", nonempty=True),
            "purpose": _required_text(operation, "purpose"),
            "source": _validate_source(operation.get("source")),
        }
        return payload, activity_id

    if op_name == "evidence.review":
        _reject_unknown_operation_fields(operation)
        finding_id = _resolve(operation.get("finding_id"), "finding", state, local_refs)
        scientific_review = operation.get("scientific_review")
        if scientific_review not in {"accepted", "rejected", "inconclusive"}:
            raise ValidationError("Unsupported scientific_review", code="invalid_schema")
        reason = _required_text(operation, "reason")
        basis = _canonical_evidence(operation.get("basis"), evidence_resolver, state=state, local_refs=local_refs)
        fingerprint = _required_text(operation, "evidence_fingerprint")
        source_activity_id = _resolve_activity(operation["source_activity_id"], state, local_refs) if "source_activity_id" in operation else None
        if not basis:
            raise ValidationError("Evidence review needs structured basis")
        payload = {"id": new_id("event"), "finding_id": finding_id, "evidence_fingerprint": fingerprint, "scientific_review": scientific_review, "reason": reason, "basis": basis}
        if source_activity_id is not None:
            payload["source_activity_id"] = source_activity_id
        return payload, finding_id

    if op_name == "dataset.scope.create":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "dataset_scope", local_refs, created_ids)
        coverage = operation.get("coverage")
        initial = operation.get("initial_observation")
        if coverage not in {"declared_complete", "partial", "unknown"}:
            raise ValidationError("Unsupported dataset scope coverage")
        if initial not in {"declared_unobserved", "unknown"}:
            raise ValidationError("Unsupported initial_observation")
        payload = {
            "id": entity_id,
            "project_key": _required_text(operation, "project_key"),
            "dataset_key": _required_text(operation, "dataset_key"),
            "data_revision": _required_text(operation, "data_revision"),
            "description": _required_text(operation, "description"),
            "overlap_keys": _text_list(operation.get("overlap_keys"), "overlap_keys", nonempty=True),
            "coverage": coverage,
            "initial_observation": initial,
            "source_activity_id": _resolve_activity(operation.get("source_activity_id"), state, local_refs),
            "source": _validate_source(operation.get("source")),
        }
        identity = (payload["project_key"], payload["dataset_key"], payload["data_revision"])
        if any((item.get("project_key"), item.get("dataset_key"), item.get("data_revision")) == identity for item in state.dataset_scopes.values()):
            raise ValidationError("Dataset scope identity is immutable and already registered")
        return payload, entity_id

    if op_name == "dataset.observation.record":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "observation", local_refs, created_ids)
        scope_id = _resolve(operation.get("dataset_scope_id"), "dataset_scope", state, local_refs)
        event_kind = operation.get("event_kind")
        if event_kind not in {"scope_declared", "candidate_frozen", "first_scored", "promoted_to_development"}:
            raise ValidationError("Unsupported dataset observation event_kind")
        fingerprint = operation.get("candidate_fingerprint")
        attempt_id = operation.get("attempt_id")
        if event_kind in {"candidate_frozen", "first_scored", "promoted_to_development"} and (not isinstance(fingerprint, str) or not fingerprint.strip()):
            raise ValidationError("candidate_fingerprint is required for this observation")
        if event_kind == "first_scored" and (not isinstance(attempt_id, str) or not attempt_id.strip()):
            raise ValidationError("attempt_id is required for first_scored")
        if event_kind != "first_scored" and attempt_id is not None:
            raise ValidationError("attempt_id is only valid for first_scored")
        payload = {
            "id": entity_id,
            "dataset_scope_id": scope_id,
            "event_kind": event_kind,
            "occurred_at": _required_text(operation, "occurred_at"),
            "source_activity_id": _resolve_activity(operation.get("source_activity_id"), state, local_refs),
            "source": _validate_source(operation.get("source")),
            "evidence": _canonical_evidence(operation.get("evidence"), evidence_resolver, state=state, local_refs=local_refs),
        }
        if fingerprint is not None:
            payload["candidate_fingerprint"] = fingerprint
        if attempt_id is not None:
            payload["attempt_id"] = _resolve(attempt_id, "attempt", state, local_refs) if event_kind == "first_scored" else attempt_id
        scope = state.dataset_scopes.get(scope_id)
        if scope is None:
            raise ValidationError(f"Unknown dataset scope: {scope_id}", code="unknown_entity")
        prior = [item for item in state.observations.values() if item.get("dataset_scope_id") == scope_id]
        if not payload["evidence"]:
            raise ValidationError("Observation events require source evidence", code="incomplete_experiment_contract")
        _parse_observation_time(payload["occurred_at"])
        first_observation = _first_relevant_observation(prior)
        if event_kind == "scope_declared" and any(item.get("event_kind") == "scope_declared" for item in prior):
            raise ValidationError("Dataset scope declaration is immutable", code="observation_conflict", first_observation=prior[0], source=scope.get("source"))
        if event_kind == "scope_declared":
            if prior or scope.get("initial_observation") not in {"declared_unobserved", "unknown"}:
                raise ValidationError("Dataset scope can only be declared before observations", code="observation_conflict", first_observation=first_observation, source=scope.get("source"))
        if event_kind == "candidate_frozen":
            if scope.get("coverage") != "declared_complete" or scope.get("initial_observation") != "declared_unobserved":
                raise ValidationError("Dataset scope is not complete and declared unobserved", code="independence_claim_rejected", first_observation=first_observation, source=scope.get("source"), query_commands=["aimx research state --json"])
            if any(item.get("event_kind") in {"first_scored", "promoted_to_development"} for item in prior):
                raise ValidationError("Candidate cannot be frozen after this scope was observed", code="independence_claim_rejected", first_observation=first_observation, source=scope.get("source"), query_commands=["aimx research state --json"])
            if any(item.get("event_kind") == "candidate_frozen" and item.get("candidate_fingerprint") == fingerprint for item in prior):
                raise ValidationError("Candidate is already frozen for this scope", code="observation_conflict", first_observation=first_observation, source=scope.get("source"))
        if event_kind in {"first_scored", "promoted_to_development"}:
            frozen_events = [item for item in prior if item.get("event_kind") == "candidate_frozen" and item.get("candidate_fingerprint") == fingerprint]
            if not frozen_events:
                raise ValidationError("Candidate must be frozen before it is scored or promoted", code="independence_claim_rejected", first_observation=first_observation, source=scope.get("source"), query_commands=["aimx research state --json"])
        if event_kind == "first_scored":
            if any(item.get("event_kind") in {"first_scored", "promoted_to_development"} for item in prior):
                raise ValidationError("Dataset scope has already been scored or promoted; use exploratory phase", code="independence_claim_rejected", first_observation=first_observation, query_commands=["aimx research state --json"])
            matching_attempt = state.outcomes.get(str(payload.get("attempt_id")))
            if matching_attempt is None:
                matching_attempt = next((item for item in state.outcomes.values() if item.get("id") == payload.get("attempt_id")), None)
            if matching_attempt is None:
                raise ValidationError("first_scored must reference a recorded execution attempt", code="incomplete_experiment_contract")
            if matching_attempt.get("source_activity_id") != payload["source_activity_id"]:
                raise ValidationError("first_scored attempt must belong to the observation activity", code="attempt_conflict", attempt_id=matching_attempt.get("id"), source=matching_attempt.get("source_activity_id"))
        return payload, entity_id

    if op_name == "execution.outcome.record":
        _reject_unknown_operation_fields(operation)
        entity_id = _declare(operation, "attempt", local_refs, created_ids)
        activity_id = _resolve_activity(operation.get("source_activity_id"), state, local_refs)
        agenda_id = _resolve(operation.get("agenda_item_id"), "agenda", state, local_refs)
        attempt_key = _required_text(operation, "attempt_key")
        if any(item.get("source_activity_id") == activity_id and item.get("attempt_key") == attempt_key for item in state.outcomes.values()):
            raise ValidationError("Attempt key is already recorded for this activity", code="attempt_conflict")
        status_fields = {
            "execution_status": {"succeeded", "failed", "interrupted", "unknown"},
            "artifact_status": {"none", "partial", "complete", "unknown"},
            "evaluation_status": {"not_run", "passed", "failed", "inconclusive", "unknown"},
            "audit_status": {"not_run", "passed", "failed", "unknown"},
        }
        payload = {"id": entity_id, "source_activity_id": activity_id, "agenda_item_id": agenda_id, "attempt_key": attempt_key, "summary": _required_text(operation, "summary"), "evidence": _canonical_evidence(operation.get("evidence"), evidence_resolver, state=state, local_refs=local_refs), "resources": _validate_resources(operation.get("resources"))}
        for field, allowed in status_fields.items():
            payload[field] = _enum(operation, field, allowed)
        if payload["evaluation_status"] == "passed" and not payload["evidence"]:
            raise ValidationError("A passed scientific evaluation requires evidence", code="incomplete_experiment_contract")
        retry_of = operation.get("retry_of")
        if retry_of is not None:
            retry_id = _resolve(retry_of, "attempt", state, local_refs)
            prior_attempt = state.outcomes[retry_id]
            if prior_attempt.get("source_activity_id") != activity_id:
                raise ValidationError("retry_of must reference an attempt from the same activity", code="attempt_conflict")
            payload["retry_of"] = retry_id
        if "round" in operation and (type(operation["round"]) is not int or operation["round"] < 1):
            raise ValidationError("round must be a positive integer", code="invalid_schema")
        for key, expected_type in (("slot_id", str), ("native_task_id", str), ("requested_settings", dict), ("actual_settings", dict)):
            if key in operation and not isinstance(operation[key], expected_type):
                raise ValidationError(f"{key} must be a {expected_type.__name__}", code="invalid_schema")
        for key in ("round", "slot_id", "native_task_id", "requested_settings", "actual_settings"):
            if key in operation:
                payload[key] = copy.deepcopy(operation[key])
        return payload, entity_id

    raise ValidationError(f"Unsupported research operation: {op_name}", code="invalid_schema")


def _declare(
    operation: dict[str, Any],
    kind: str,
    local_refs: dict[str, tuple[str, str]],
    created_ids: dict[str, str],
) -> str:
    local_id = operation.get("local_id")
    if local_id is not None:
        if not is_local_name(local_id):
            raise ValidationError(f"Invalid local_id: {local_id!r}")
        if local_id in local_refs:
            raise ValidationError(f"Duplicate local_id: {local_id}")
    entity_id = new_id(kind)
    if local_id is not None:
        local_refs[local_id] = (entity_id, kind)
        created_ids[local_id] = entity_id
    return entity_id


def _resolve(
    value: Any,
    kind: str,
    state: ResearchState,
    local_refs: dict[str, tuple[str, str]],
) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{kind} reference is required")
    if is_local_ref(value):
        local_id = local_name(value)
        try:
            entity_id, local_kind = local_refs[local_id]
        except KeyError as exc:
            raise ValidationError(f"Unknown same-update reference: {value}", code="unknown_entity") from exc
        if local_kind != kind:
            raise ValidationError(f"Reference {value} is a {local_kind}, expected {kind}")
        return entity_id
    collection = _collection(state, kind)
    if value not in collection:
        raise ValidationError(f"Unknown {kind}: {value}", code="unknown_entity")
    return value


def _collection(state: ResearchState, kind: str) -> dict[str, dict[str, Any]]:
    return {
        "finding": state.findings,
        "relation": state.relations,
        "lane": state.lanes,
        "frontier": state.frontier_items,
        "annotation": state.annotations,
        "agenda": state.agenda_items,
        "activity": state.activities,
        "dataset_scope": state.dataset_scopes,
        "observation": state.observations,
        "attempt": state.outcomes,
    }[kind]


def _kind_for_target(target_kind: str) -> str:
    return {
        "finding": "finding",
        "relation": "relation",
        "frontier_item": "frontier",
        "agenda_item": "agenda",
    }[target_kind]


def _entity_list(value: Any, kind: str, state: ResearchState, local_refs: dict[str, tuple[str, str]]) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{kind} references must be an array")
    result = [_resolve(item, kind, state, local_refs) for item in value]
    if len(result) != len(set(result)):
        raise ValidationError(f"Duplicate {kind} reference")
    return result


def _finding_list(value: Any, state: ResearchState, local_refs: dict[str, tuple[str, str]]) -> list[str]:
    return _entity_list(value, "finding", state, local_refs)


def _canonical_evidence(
    value: Any,
    resolver: Callable[[str], str] | None,
    *,
    state: ResearchState | None = None,
    local_refs: dict[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError("evidence must be an array")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValidationError("Evidence reference must be an object")
        kind = item.get("kind")
        if kind == "aim_run":
            allowed = {"kind", "run_hash", "role", "note", "evidence_type", "purpose", "source_activity_id"}
            _reject_unknown_fields(item, allowed, "Aim evidence")
            raw_hash = item.get("run_hash")
            if not isinstance(raw_hash, str) or not raw_hash.strip():
                raise ValidationError("Aim evidence requires run_hash")
            if item.get("role") is not None and not isinstance(item.get("role"), str):
                raise ValidationError("Aim evidence role must be a string")
            if item.get("note") is not None and not isinstance(item.get("note"), str):
                raise ValidationError("Aim evidence note must be a string")
            canonical = resolver(raw_hash) if resolver is not None else raw_hash.lower()
            identity = ("aim_run", canonical, item.get("role"), item.get("note"))
            if identity in seen:
                continue
            seen.add(identity)
            evidence = {"kind": "aim_run", "run_hash": canonical}
            if item.get("role") is not None:
                evidence["role"] = item["role"]
            if item.get("note") is not None:
                evidence["note"] = item["note"]
            for key in ("evidence_type", "purpose"):
                if key in item:
                    evidence[key] = _required_text(item, key)
            if "source_activity_id" in item:
                evidence["source_activity_id"] = _resolve_activity(item["source_activity_id"], state, local_refs)
            result.append(evidence)
            continue
        if kind != "local_artifact":
            raise ValidationError(f"Unsupported evidence kind: {kind!r}", code="invalid_schema")
        allowed = {"kind", "location", "sha256", "size_bytes", "evidence_type", "purpose", "source_activity_id"}
        _reject_unknown_fields(item, allowed, "Local artifact evidence")
        location = _required_text(item, "location")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError("Local artifact sha256 must be 64 lowercase hexadecimal characters")
        size = item.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ValidationError("Local artifact size_bytes must be a non-negative integer")
        evidence_type = _required_text(item, "evidence_type")
        purpose = _required_text(item, "purpose")
        activity_id = _resolve_activity(item.get("source_activity_id"), state, local_refs)
        if resolver is None or not hasattr(resolver, "resolve_artifact"):
            raise ValidationError("Local artifact validation is unavailable", code="artifact_unreadable")
        resolved, actual_digest, actual_size = resolver.resolve_artifact(location)
        if (actual_digest, actual_size) != (digest, size):
            raise ValidationError("Local artifact identity does not match its declaration", code="artifact_identity_mismatch")
        identity = ("local_artifact", str(resolved), digest, evidence_type, purpose)
        if identity in seen:
            continue
        seen.add(identity)
        result.append({"kind": kind, "location": location, "resolved_location": str(resolved), "sha256": digest, "size_bytes": size, "evidence_type": evidence_type, "purpose": purpose, "source_activity_id": activity_id})
    return result


def _reject_unknown_fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValidationError(f"{label} has unsupported fields: {', '.join(unknown)}", code="invalid_schema")


def _reject_unknown_operation_fields(operation: dict[str, Any]) -> None:
    common = {"op", "local_id", "source_activity_id"}
    fields = {
        "finding.create": {"claim", "evidence", "epistemic_status", "confidence", "governance_status", "topics", "result_kind", "scope", "limitations", "source_annotation_ids"},
        "finding.assess": {"finding_id", "epistemic_status", "confidence", "reason"},
        "finding.governance": {"finding_id", "governance_status", "reason"},
        "relation.create": {"source", "target", "relation_type", "reason"},
        "relation.retract": {"relation_id", "reason"},
        "annotation.create": {"annotation_target_kind", "annotation_target_id", "text"},
        "activity.create": {"kind", "title", "objective", "source", "loop_id", "related_loop_id", "report_locations", "artifact_locations", "checkpoint"},
        "activity.artifact.add": {"activity_id", "locations", "purpose", "source"},
        "evidence.review": {"finding_id", "evidence_fingerprint", "scientific_review", "reason", "basis"},
        "dataset.scope.create": {"project_key", "dataset_key", "data_revision", "description", "overlap_keys", "coverage", "initial_observation", "source"},
        "dataset.observation.record": {"dataset_scope_id", "event_kind", "occurred_at", "source", "evidence", "candidate_fingerprint", "attempt_id"},
        "execution.outcome.record": {"agenda_item_id", "attempt_key", "execution_status", "artifact_status", "evaluation_status", "audit_status", "summary", "evidence", "resources", "retry_of", "round", "slot_id", "native_task_id", "requested_settings", "actual_settings"},
        "agenda.item.create": {"objective", "hypothesis", "question", "motivating_finding_ids", "frontier_item_ids", "controls", "factors", "constraints", "success_criteria", "priority", "agenda_status", "evidence", "experiment_contract"},
    }
    op_name = operation.get("op")
    common.update(fields.get(op_name, set()))
    if op_name == "frontier.lane.create": common.update({"name", "description"})
    elif op_name in {"frontier.lane.rename", "frontier.lane.retire"}: common.update({"lane_id", "name", "reason"})
    elif op_name == "frontier.item.create": common.update({"lane_id", "finding_id", "direction", "rationale", "priority", "frontier_status", "supporting_finding_ids", "challenging_finding_ids"})
    elif op_name == "frontier.item.move": common.update({"frontier_item_id", "lane_id"})
    elif op_name == "frontier.item.set_priority": common.update({"frontier_item_id", "priority"})
    elif op_name == "frontier.item.set_status": common.update({"frontier_item_id", "frontier_status", "reason"})
    elif op_name == "frontier.item.update": common.update({"frontier_item_id", "rationale", "supporting_finding_ids", "challenging_finding_ids"})
    elif op_name in {"agenda.item.transition", "agenda.item.attach_evidence", "agenda.item.set_priority"}: common.update({"agenda_item_id", "agenda_status", "evidence", "result_commit_id", "priority", "reason"})
    _reject_unknown_fields(operation, common, f"{op_name} operation")


def _text_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValidationError(f"{label} must be {'a non-empty' if nonempty else 'an'} array of strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValidationError(f"{label} values must be non-empty strings")
        if item not in result:
            result.append(item)
    return result


def _validate_source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("source must be an object")
    _reject_unknown_fields(value, {"kind", "location", "note"}, "source")
    result = {"kind": _required_text(value, "kind"), "location": _required_text(value, "location")}
    if "note" in value:
        if not isinstance(value["note"], str):
            raise ValidationError("source.note must be a string")
        result["note"] = value["note"]
    return result


def _resolve_activity(value: Any, state: ResearchState, local_refs: dict[str, tuple[str, str]], *, kinds: set[str] | None = None) -> str:
    activity_id = _resolve(value, "activity", state, local_refs)
    if kinds and state.activities.get(activity_id, {}).get("kind") not in kinds:
        raise ValidationError(f"Activity {activity_id} has the wrong kind", code="unknown_entity")
    return activity_id


def _resolve_loop_id(value: Any, state: ResearchState, local_refs: dict[str, tuple[str, str]]) -> str:
    if is_local_ref(value):
        activity_id = _resolve(value, "activity", state, local_refs)
        activity = next((item for item in state.activities.values() if item.get("id") == activity_id), None)
        if activity is None:
            raise ValidationError("follow_up cannot reference a loop created earlier in the same update")
        if activity.get("kind") != "loop":
            raise ValidationError("related_loop_id must reference a loop activity")
        return str(activity["loop_id"])
    if isinstance(value, str) and value in state.activities and state.activities[value].get("kind") == "loop":
        return str(state.activities[value]["loop_id"])
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("related_loop_id is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValidationError("loop_id must match the AutoResearch checkpoint ID format")
    activity = next((item for item in state.activities.values() if item.get("loop_id") == value), None)
    if activity is None:
        raise ValidationError(f"Unknown loop activity: {value}", code="unknown_entity")
    return value


def _copy_optional_source(operation: dict[str, Any], payload: dict[str, Any], state: ResearchState, local_refs: dict[str, tuple[str, str]]) -> None:
    if "source_activity_id" in operation:
        payload["source_activity_id"] = _resolve_activity(operation["source_activity_id"], state, local_refs)


def _validate_scope(value: Any, state: ResearchState, local_refs: dict[str, tuple[str, str]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("scope must be an object")
    _reject_unknown_fields(value, {"description", "dataset_scope_ids", "conditions"}, "scope")
    if set(value) != {"description", "dataset_scope_ids", "conditions"}:
        raise ValidationError("scope needs description, dataset_scope_ids, and conditions")
    return {"description": _required_text(value, "description"), "dataset_scope_ids": _entity_list(value["dataset_scope_ids"], "dataset_scope", state, local_refs), "conditions": _text_list(value["conditions"], "conditions")}


def _validate_loop_checkpoint(
    state: ResearchState, value: Any, loop_id: str, *, repo_root: Path | None = None
) -> None:
    if value is not None:
        if not isinstance(value, dict):
            raise ValidationError("checkpoint must be an object")
        _reject_unknown_fields(value, {"path", "repo", "loop_id"}, "checkpoint")
        if value.get("loop_id") != loop_id:
            raise ValidationError("Checkpoint loop_id does not match activity loop_id")
        location = _required_text(value, "path")
        declared_root = Path(value.get("repo", ".")).expanduser().resolve()
        expected_root = repo_root.expanduser().resolve() if repo_root is not None else declared_root
        if declared_root != expected_root:
            raise ValidationError("Loop checkpoint repo does not match the Research State repository", code="checkpoint_unreadable")
        standard_path = expected_root / ".aimx" / "autoresearch" / loop_id / "state.json"
        checkpoint = Path(location).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = expected_root / checkpoint
        if checkpoint.resolve() != standard_path.resolve():
            raise ValidationError("Loop checkpoint must use the standard repository checkpoint location", code="checkpoint_unreadable")
        if not checkpoint.is_file():
            raise ValidationError("Loop checkpoint is missing", code="checkpoint_unreadable")
        try:
            content = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"Loop checkpoint is unreadable: {exc}", code="checkpoint_unreadable") from exc
        if content.get("loop_id") != loop_id:
            raise ValidationError("Checkpoint loop_id does not match activity loop_id")
    elif not any(item.get("loop_id") == loop_id for item in state.activities.values()):
        raise ValidationError("A standard checkpoint location and matching repo/ID are required for loop registration", code="checkpoint_unreadable")


def _parse_observation_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("occurred_at must be an ISO-8601 timestamp", code="invalid_schema")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("occurred_at must be an ISO-8601 timestamp", code="invalid_schema") from exc
    if parsed.tzinfo is None:
        raise ValidationError("occurred_at must include a timezone", code="invalid_schema")
    return parsed.astimezone(timezone.utc)


def _first_relevant_observation(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    relevant = [item for item in observations if item.get("event_kind") in {"first_scored", "promoted_to_development"}]
    if not relevant:
        return None
    return min(relevant, key=lambda item: (int(item.get("created_revision", 0)), _parse_observation_time(item.get("occurred_at")), str(item.get("id", ""))))


def _validate_experiment_contract(value: Any, state: ResearchState, local_refs: dict[str, tuple[str, str]], evidence_resolver: Callable[[str], str] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("experiment_contract must be an object", code="incomplete_experiment_contract")
    required = {"contract_version", "question", "interventions", "fixed_controls", "expected_evidence", "disconfirmation", "confounders", "evaluation", "data", "resources", "failure_policy"}
    allowed = required | {"extensions"}
    _reject_unknown_fields(value, allowed, "experiment_contract")
    missing = required - set(value)
    if missing:
        raise ValidationError(f"experiment_contract needs fields: {sorted(missing)}", code="incomplete_experiment_contract")
    if "extensions" in value and not isinstance(value["extensions"], dict):
        raise ValidationError("experiment_contract.extensions must be an object", code="incomplete_experiment_contract")
    extensions = copy.deepcopy(value.get("extensions", {}))
    if type(value.get("contract_version")) is not int or value["contract_version"] != 1:
        raise ValidationError("experiment_contract.contract_version must be 1", code="incomplete_experiment_contract")
    _required_text(value, "question")
    normalized_lists = {
        field_name: _text_list(value.get(field_name), f"experiment_contract.{field_name}", nonempty=True)
        for field_name in ("interventions", "fixed_controls", "expected_evidence", "disconfirmation")
    }
    confounders = _text_list(value.get("confounders"), "experiment_contract.confounders")
    if len(normalized_lists["interventions"]) > 1 and not confounders:
        raise ValidationError("Joint interventions require an explicit confounder explanation", code="incomplete_experiment_contract")
    if value.get("failure_policy") not in {"stop_pending", "continue_independent"}:
        raise ValidationError("Unsupported experiment_contract.failure_policy", code="incomplete_experiment_contract")
    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {"protocol_id", "protocol_sha256", "argv", "metric", "acceptance_rule", "preflight_evidence"}:
        raise ValidationError("experiment_contract.evaluation is incomplete", code="incomplete_experiment_contract")
    _required_text(evaluation, "protocol_id")
    if not isinstance(evaluation.get("protocol_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", evaluation["protocol_sha256"]):
        raise ValidationError("evaluation.protocol_sha256 must be a SHA-256 digest", code="incomplete_experiment_contract")
    _argv(evaluation.get("argv"), "evaluation.argv")
    _required_text(evaluation, "metric")
    _required_text(evaluation, "acceptance_rule")
    raw_preflight = evaluation.get("preflight_evidence")
    if not isinstance(raw_preflight, list) or not any(isinstance(item, dict) and item.get("kind") == "local_artifact" for item in raw_preflight):
        raise ValidationError("evaluation.preflight_evidence must include a file-backed preflight report identity", code="incomplete_experiment_contract")
    preflight = _canonical_evidence(raw_preflight, evidence_resolver, state=state, local_refs=local_refs)
    if not preflight:
        raise ValidationError("evaluation.preflight_evidence must not be empty", code="incomplete_experiment_contract")
    data = value.get("data")
    if not isinstance(data, dict) or set(data) != {"scope_ids", "phase", "candidate_fingerprint", "observation_revision"}:
        raise ValidationError("experiment_contract.data is incomplete", code="incomplete_experiment_contract")
    scopes = _entity_list(data.get("scope_ids"), "dataset_scope", state, local_refs)
    phase = data.get("phase")
    if phase not in {"exploratory", "independent_confirmation"}:
        raise ValidationError("Unsupported experiment data phase", code="incomplete_experiment_contract")
    fingerprint = _required_text(data, "candidate_fingerprint")
    observation_revision = data.get("observation_revision")
    if type(observation_revision) is not int or observation_revision < 0:
        raise ValidationError("observation_revision must be a non-negative integer", code="incomplete_experiment_contract")
    resources = _validate_resources(value.get("resources"))
    if value.get("failure_policy") == "continue_independent":
        preflight_digests = {item.get("sha256") for item in preflight if item.get("kind") == "local_artifact"}
        attested_digest = extensions.get("host_failure_policy_preflight_sha256")
        if extensions.get("host_failure_policy_verified") is not True or not isinstance(attested_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", attested_digest) or attested_digest not in preflight_digests:
            raise ValidationError("continue_independent requires host support attested against a cited preflight artifact", code="incomplete_experiment_contract", failure_policy="continue_independent", preflight_digests=sorted(value for value in preflight_digests if isinstance(value, str)))
    if not resources:
        raise ValidationError("experiment_contract.resources must not be empty", code="incomplete_experiment_contract")
    if phase == "independent_confirmation":
        if not scopes:
            raise ValidationError("Independent confirmation needs at least one registered data scope", code="independence_claim_rejected", query_commands=["aimx research state --json"])
        if data.get("observation_revision") != state.revision:
            raise ValidationError("Independent confirmation rejected: observation snapshot revision is stale", code="independence_claim_rejected", observed_revision=state.revision, query_commands=["aimx research state --json"])
        observed_scopes = list(state.dataset_scopes.values())
        selected_scopes = [state.dataset_scopes[scope_id] for scope_id in scopes]
        candidate_keys = {key for scope in selected_scopes for key in scope.get("overlap_keys", [])}
        for index, scope in enumerate(selected_scopes):
            keys = set(scope.get("overlap_keys", []))
            for other in selected_scopes[index + 1:]:
                overlap = keys.intersection(other.get("overlap_keys", []))
                if overlap:
                    raise ValidationError("Independent confirmation rejected: selected dataset scopes overlap", code="independence_claim_rejected", conflicting_scope_id=other.get("id"), overlap_keys=sorted(overlap), source=other.get("source"), query_commands=["aimx research state --json"])
        for scope_id in scopes:
            scope = state.dataset_scopes[scope_id]
            prior = [event for event in state.observations.values() if event.get("dataset_scope_id") == scope_id]
            first = _first_relevant_observation(prior)
            if scope.get("coverage") != "declared_complete" or scope.get("initial_observation") != "declared_unobserved":
                raise ValidationError("Independent confirmation rejected: scope is observed or coverage is unknown", code="independence_claim_rejected", first_observation=first, source=scope.get("source"), query_commands=["aimx research state --json"])
            if any(event.get("event_kind") in {"first_scored", "promoted_to_development"} for event in prior):
                raise ValidationError("Independent confirmation rejected: scope was already scored or promoted", code="independence_claim_rejected", first_observation=first, source=scope.get("source"), query_commands=["aimx research state --json"])
            freezes = [event for event in prior if event.get("event_kind") == "candidate_frozen" and event.get("candidate_fingerprint") == fingerprint]
            if not freezes:
                raise ValidationError("Independent confirmation rejected: candidate is not frozen for this scope", code="independence_claim_rejected", first_observation=first, source=scope.get("source"), query_commands=["aimx research state --json"])
            for other in observed_scopes:
                if other.get("id") in scopes:
                    continue
                other_keys = set(other.get("overlap_keys", []))
                overlapping = candidate_keys.intersection(other_keys)
                if not other_keys or overlapping:
                    other_observations = [event for event in state.observations.values() if event.get("dataset_scope_id") == other.get("id") and event.get("event_kind") in {"first_scored", "promoted_to_development"}]
                    first_other = min(other_observations, key=lambda item: (_parse_observation_time(item.get("occurred_at")), int(item.get("created_revision", 0)), str(item.get("id", ""))), default=None)
                    raise ValidationError("Independent confirmation rejected: overlap with another registered scope cannot be ruled out", code="independence_claim_rejected", first_observation=first_other, source=other.get("source"), conflicting_scope_id=other.get("id"), overlap_keys=sorted(overlapping), query_commands=["aimx research state --json"])
    return {"contract_version": 1, "question": value["question"].strip(), **normalized_lists, "confounders": confounders, "evaluation": {**evaluation, "preflight_evidence": preflight}, "data": {**data, "scope_ids": scopes, "candidate_fingerprint": fingerprint}, "resources": resources, "failure_policy": value["failure_policy"], **({"extensions": extensions} if "extensions" in value else {})}


def _argv(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValidationError(f"{label} must be a non-empty argv array", code="incomplete_experiment_contract")
    return [item.strip() for item in value]


def _validate_resources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError("resources must be an array")
    output = []
    for resource in value:
        if not isinstance(resource, dict):
            raise ValidationError("resource must be an object")
        _reject_unknown_fields(resource, {"kind", "unit", "planned", "actual", "verification", "enforcement", "source"}, "resource")
        result = {"kind": _required_text(resource, "kind"), "unit": _required_text(resource, "unit"), "planned": resource.get("planned"), "verification": resource.get("verification"), "enforcement": resource.get("enforcement")}
        if type(result["planned"]) not in {int, float} or not math.isfinite(result["planned"]) or result["planned"] < 0:
            raise ValidationError("resource planned must be a non-negative number")
        if result["verification"] not in {"verified", "unverified"} or result["enforcement"] not in {"executor_enforced", "declared_only", "unknown"}:
            raise ValidationError("Invalid resource verification/enforcement")
        if "actual" in resource:
            if type(resource["actual"]) not in {int, float} or not math.isfinite(resource["actual"]) or resource["actual"] < 0:
                raise ValidationError("resource actual must be a non-negative number")
            result["actual"] = resource["actual"]
        if result["verification"] == "verified" and "actual" not in result:
            raise ValidationError("Verified resources require an actual count")
        if result["verification"] == "verified" and not isinstance(resource.get("source"), str):
            raise ValidationError("Verified resources require executor source evidence")
        if "source" in resource:
            result["source"] = _required_text(resource, "source")
        if result["enforcement"] == "executor_enforced" and result["verification"] != "verified":
            raise ValidationError("Executor-enforced resources must be verified")
        output.append(result)
    return output


def _required_text(operation: dict[str, Any], key: str) -> str:
    value = operation.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _enum(
    operation: dict[str, Any], key: str, allowed: set[str], default: str | None = None
) -> str:
    value = operation.get(key, default)
    if value not in allowed:
        raise ValidationError(f"Unsupported {key}: {value!r}")
    return value


def _priority(operation: dict[str, Any]) -> int:
    value = operation.get("priority", 0)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
        raise ValidationError("priority must be an integer from 0 to 100")
    return value


def _optional_reason(operation: dict[str, Any], payload: dict[str, Any]) -> None:
    if "reason" in operation:
        reason = operation["reason"]
        if not isinstance(reason, str):
            raise ValidationError("reason must be a string")
        payload["reason"] = reason
