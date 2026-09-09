from __future__ import annotations

import copy
from typing import Any, Iterable

from aimx.research.errors import ValidationError
from aimx.research.models import ResearchState


def apply_event(
    state: ResearchState,
    event_type: str,
    payload: dict[str, Any],
    *,
    commit_id: str | None = None,
    revision: int | None = None,
) -> None:
    """Apply one already validated event to the projected state."""

    p = copy.deepcopy(payload)
    if commit_id is not None:
        p.setdefault("created_commit_id", commit_id)
    if revision is not None:
        p.setdefault("created_revision", revision)

    if event_type == "finding.create":
        entity_id = _required_id(p)
        if entity_id in state.findings:
            raise ValidationError(f"Finding already exists: {entity_id}", code="duplicate_entity")
        p.setdefault("history", [_history_entry("created", p, revision)])
        state.findings[entity_id] = p
        return

    if event_type == "finding.assess":
        finding = _entity(state.findings, p.get("finding_id"), "finding")
        _set_if_present(finding, p, "epistemic_status")
        _set_if_present(finding, p, "confidence")
        _append_history(finding, "assessed", p, revision)
        return

    if event_type == "finding.governance":
        finding = _entity(state.findings, p.get("finding_id"), "finding")
        _set_if_present(finding, p, "governance_status")
        _append_history(finding, "governance", p, revision)
        return

    if event_type == "relation.create":
        entity_id = _required_id(p)
        if entity_id in state.relations:
            raise ValidationError(f"Relation already exists: {entity_id}", code="duplicate_entity")
        p["active"] = True
        p.setdefault("history", [_history_entry("created", p, revision)])
        state.relations[entity_id] = p
        return

    if event_type == "relation.retract":
        relation = _entity(state.relations, p.get("relation_id"), "relation")
        if not relation.get("active", True):
            raise ValidationError(
                f"Relation is already retracted: {p.get('relation_id')}",
                code="invalid_transition",
            )
        relation["active"] = False
        relation["retracted_commit_id"] = p.get("created_commit_id")
        relation["retracted_revision"] = p.get("created_revision")
        relation.setdefault("history", []).append(_history_entry("retracted", p, revision))
        return

    if event_type == "annotation.create":
        entity_id = _required_id(p)
        if entity_id in state.annotations:
            raise ValidationError(f"Annotation already exists: {entity_id}", code="duplicate_entity")
        p.setdefault("history", [_history_entry("created", p, revision)])
        state.annotations[entity_id] = p
        return

    if event_type == "frontier.lane.create":
        entity_id = _required_id(p)
        if entity_id in state.lanes:
            raise ValidationError(f"Frontier lane already exists: {entity_id}", code="duplicate_entity")
        p.setdefault("history", [_history_entry("created", p, revision)])
        state.lanes[entity_id] = p
        return

    if event_type == "frontier.lane.rename":
        lane = _entity(state.lanes, p.get("lane_id"), "frontier lane")
        lane["name"] = p["name"]
        _append_history(lane, "renamed", p, revision)
        return

    if event_type == "frontier.lane.retire":
        lane = _entity(state.lanes, p.get("lane_id"), "frontier lane")
        lane["active"] = False
        _append_history(lane, "retired", p, revision)
        return

    if event_type == "frontier.item.create":
        entity_id = _required_id(p)
        if entity_id in state.frontier_items:
            raise ValidationError(f"Frontier item already exists: {entity_id}", code="duplicate_entity")
        p.setdefault("history", [_history_entry("created", p, revision)])
        state.frontier_items[entity_id] = p
        return

    if event_type == "frontier.item.move":
        item = _entity(state.frontier_items, p.get("frontier_item_id"), "frontier item")
        item["lane_id"] = p["lane_id"]
        _append_history(item, "moved", p, revision)
        return

    if event_type == "frontier.item.set_priority":
        item = _entity(state.frontier_items, p.get("frontier_item_id"), "frontier item")
        item["priority"] = p["priority"]
        _append_history(item, "priority_changed", p, revision)
        return

    if event_type == "frontier.item.set_status":
        item = _entity(state.frontier_items, p.get("frontier_item_id"), "frontier item")
        item["status"] = p["status"]
        _append_history(item, "status_changed", p, revision)
        return

    if event_type == "frontier.item.update":
        item = _entity(state.frontier_items, p.get("frontier_item_id"), "frontier item")
        for key in ("rationale", "supporting_finding_ids", "challenging_finding_ids"):
            if key in p:
                item[key] = p[key]
        _append_history(item, "updated", p, revision)
        return

    if event_type == "agenda.item.create":
        entity_id = _required_id(p)
        if entity_id in state.agenda_items:
            raise ValidationError(f"Agenda item already exists: {entity_id}", code="duplicate_entity")
        p.setdefault("history", [_history_entry("created", p, revision)])
        state.agenda_items[entity_id] = p
        return

    if event_type == "agenda.item.transition":
        item = _entity(state.agenda_items, p.get("agenda_item_id"), "agenda item")
        item["status"] = p["status"]
        if p.get("evidence"):
            item.setdefault("evidence", []).extend(copy.deepcopy(p["evidence"]))
        if p.get("result_commit_id"):
            item.setdefault("result_commit_ids", []).append(p["result_commit_id"])
        elif p.get("status") == "completed" and p.get("created_commit_id"):
            item.setdefault("result_commit_ids", []).append(p["created_commit_id"])
        _append_history(item, "transitioned", p, revision)
        return

    if event_type == "agenda.item.attach_evidence":
        item = _entity(state.agenda_items, p.get("agenda_item_id"), "agenda item")
        item.setdefault("evidence", []).extend(copy.deepcopy(p.get("evidence", [])))
        _append_history(item, "evidence_attached", p, revision)
        return

    if event_type == "agenda.item.set_priority":
        item = _entity(state.agenda_items, p.get("agenda_item_id"), "agenda item")
        item["priority"] = p["priority"]
        _append_history(item, "priority_changed", p, revision)
        return

    raise ValidationError(f"Unknown research event type: {event_type}", code="invalid_schema")


def replay(
    commits: Iterable[dict[str, Any]], events: Iterable[dict[str, Any]]
) -> ResearchState:
    state = ResearchState()
    commit_list = sorted(commits, key=lambda value: (value["revision"], value["id"]))
    event_list = sorted(events, key=lambda value: (value["revision"], value["sequence"]))
    events_by_revision: dict[int, list[dict[str, Any]]] = {}
    for event in event_list:
        events_by_revision.setdefault(int(event["revision"]), []).append(event)

    known_commit_ids = {commit["id"] for commit in commit_list}
    if any(event.get("commit_id") not in known_commit_ids for event in event_list):
        raise ValidationError("Research event references an unknown commit", code="invalid_schema")

    for expected_revision, commit in enumerate(commit_list, start=1):
        revision = int(commit["revision"])
        if revision != expected_revision:
            raise ValidationError(
                "Research commit revisions must be contiguous",
                code="invalid_schema",
            )
        if commit.get("schema_version") != 1:
            raise ValidationError(
                f"Unsupported Research commit schema version: {commit.get('schema_version')!r}",
                code="invalid_schema",
            )
        revision_events = events_by_revision.get(revision, [])
        if not revision_events:
            raise ValidationError(
                f"Research commit {commit.get('id')} has no events",
                code="invalid_schema",
            )
        if any(event.get("commit_id") != commit.get("id") for event in revision_events):
            raise ValidationError(
                f"Research events for revision {revision} reference the wrong commit",
                code="invalid_schema",
            )
        if [int(event.get("sequence", -1)) for event in revision_events] != list(range(len(revision_events))):
            raise ValidationError(
                f"Research event sequences for revision {revision} are not contiguous",
                code="invalid_schema",
            )
        state.revision = revision
        state.commits.append(copy.deepcopy(commit))
        for event in revision_events:
            apply_event(
                state,
                event["event_type"],
                event["payload"],
                commit_id=commit["id"],
                revision=revision,
            )
    return state


def _required_id(payload: dict[str, Any]) -> str:
    value = payload.get("id")
    if not isinstance(value, str) or not value:
        raise ValidationError("Research event is missing an entity id", code="invalid_schema")
    return value


def _entity(collection: dict[str, dict[str, Any]], entity_id: Any, label: str) -> dict[str, Any]:
    if not isinstance(entity_id, str) or entity_id not in collection:
        raise ValidationError(f"Unknown {label}: {entity_id!r}", code="unknown_entity")
    return collection[entity_id]


def _set_if_present(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    if key in source:
        target[key] = source[key]


def _append_history(target: dict[str, Any], event: str, payload: dict[str, Any], revision: int | None) -> None:
    target.setdefault("history", []).append(_history_entry(event, payload, revision))


def _history_entry(event: str, payload: dict[str, Any], revision: int | None) -> dict[str, Any]:
    return {
        "event": event,
        "revision": revision,
        "reason": payload.get("reason"),
        "commit_id": payload.get("created_commit_id"),
        "author": copy.deepcopy(payload.get("author")),
    }
