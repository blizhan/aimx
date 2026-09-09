from __future__ import annotations

import copy
from dataclasses import dataclass
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
) -> CompiledUpdate:
    _validate_update_header(update)
    working = state.clone()
    local_refs: dict[str, tuple[str, str]] = {}
    created_ids: dict[str, str] = {}
    events: list[dict[str, Any]] = []

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
        )
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
) -> tuple[dict[str, Any], str | None]:
    if op_name == "finding.create":
        entity_id = _declare(operation, "finding", local_refs, created_ids)
        claim = _required_text(operation, "claim")
        payload = {
            "id": entity_id,
            "claim": claim,
            "evidence": _canonical_evidence(operation.get("evidence", []), evidence_resolver),
            "epistemic_status": _enum(operation, "epistemic_status", _EPISTEMIC, "candidate"),
            "confidence": _enum(operation, "confidence", _CONFIDENCE, "low"),
            "governance_status": _enum(operation, "governance_status", _GOVERNANCE, "proposed"),
        }
        return payload, entity_id

    if op_name == "finding.assess":
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
        finding_id = _resolve(operation.get("finding_id"), "finding", state, local_refs)
        payload = {
            "finding_id": finding_id,
            "governance_status": _enum(operation, "governance_status", _GOVERNANCE),
        }
        _optional_reason(operation, payload)
        return payload, finding_id

    if op_name == "relation.create":
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
        relation_id = _resolve(operation.get("relation_id"), "relation", state, local_refs)
        relation = state.relations[relation_id]
        if not relation.get("active", True):
            raise ValidationError("Relation is already retracted", code="invalid_transition")
        payload = {"relation_id": relation_id}
        _optional_reason(operation, payload)
        return payload, relation_id

    if op_name == "annotation.create":
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
        entity_id = _declare(operation, "agenda", local_refs, created_ids)
        finding_ids = _finding_list(operation.get("motivating_finding_ids", []), state, local_refs)
        frontier_ids = _entity_list(operation.get("frontier_item_ids", []), "frontier", state, local_refs)
        if not finding_ids and not frontier_ids:
            raise ValidationError("Agenda item needs a finding or frontier motivation")
        evidence = _canonical_evidence(operation.get("evidence", []), evidence_resolver)
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
        return payload, entity_id

    if op_name == "agenda.item.transition":
        agenda_id = _resolve(operation.get("agenda_item_id"), "agenda", state, local_refs)
        current = state.agenda_items[agenda_id].get("status", "proposed")
        target = operation.get("agenda_status")
        if target not in _AGENDA_STATUS:
            raise ValidationError(f"Unsupported agenda status: {target!r}")
        if target != current and target not in _AGENDA_TRANSITIONS[current]:
            raise ValidationError(
                f"Invalid agenda transition: {current} -> {target}", code="invalid_transition"
            )
        payload: dict[str, Any] = {"agenda_item_id": agenda_id, "status": target}
        if "evidence" in operation:
            payload["evidence"] = _canonical_evidence(operation["evidence"], evidence_resolver)
        _optional_reason(operation, payload)
        return payload, agenda_id

    if op_name == "agenda.item.attach_evidence":
        agenda_id = _resolve(operation.get("agenda_item_id"), "agenda", state, local_refs)
        return {
            "agenda_item_id": agenda_id,
            "evidence": _canonical_evidence(operation.get("evidence", []), evidence_resolver),
        }, agenda_id

    if op_name == "agenda.item.set_priority":
        agenda_id = _resolve(operation.get("agenda_item_id"), "agenda", state, local_refs)
        return {"agenda_item_id": agenda_id, "priority": _priority(operation)}, agenda_id

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
    value: Any, resolver: Callable[[str], str] | None
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValidationError("evidence must be an array")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for item in value:
        if not isinstance(item, dict) or item.get("kind") != "aim_run":
            raise ValidationError("Only aim_run evidence references are supported")
        raw_hash = item.get("run_hash")
        if not isinstance(raw_hash, str) or not raw_hash.strip():
            raise ValidationError("Aim evidence requires run_hash")
        if item.get("role") is not None and not isinstance(item.get("role"), str):
            raise ValidationError("Aim evidence role must be a string")
        if item.get("note") is not None and not isinstance(item.get("note"), str):
            raise ValidationError("Aim evidence note must be a string")
        canonical = resolver(raw_hash) if resolver is not None else raw_hash.lower()
        identity = (canonical, item.get("role"), item.get("note"))
        if identity in seen:
            continue
        seen.add(identity)
        evidence = {"kind": "aim_run", "run_hash": canonical}
        if item.get("role") is not None:
            evidence["role"] = item["role"]
        if item.get("note") is not None:
            evidence["note"] = item["note"]
        result.append(evidence)
    return result


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
