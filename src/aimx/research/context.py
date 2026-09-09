from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from aimx.research.errors import BudgetError, ValidationError
from aimx.research.models import ResearchState, json_bytes

COMPILER_VERSION = "context-v1"
_TOKEN_RE = re.compile(r"[\w][\w/-]*", re.UNICODE)
_CLOSURE_RELATIONS = {"challenges", "refines", "supersedes", "supports", "tests"}
_MATERIAL_RELATIONS = {"challenges", "refines", "supersedes"}


@dataclass(frozen=True)
class _ContextIndexes:
    relation_ids_by_finding: dict[str, tuple[str, ...]]
    annotation_ids_by_target: dict[tuple[str, str], tuple[str, ...]]
    frontier_ids_by_finding: dict[str, tuple[str, ...]]
    agenda_ids_by_finding: dict[str, tuple[str, ...]]
    agenda_ids_by_frontier: dict[str, tuple[str, ...]]
    adjacency: dict[str, tuple[str, ...]]


def compile_context(state: ResearchState, objective: str, budget: int) -> dict[str, Any]:
    if not isinstance(objective, str) or not objective.strip():
        raise ValidationError("Context objective must not be empty")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValidationError("Context budget must be a positive integer")

    indexes = _build_indexes(state)
    tokens = _tokens(objective)
    findings = list(state.findings.values())
    scores = {
        item["id"]: _finding_score(item, tokens, state, indexes)
        for item in findings
    }
    seeds = sorted(
        (item for item in findings if scores.get(item["id"], 0) > 0),
        key=lambda item: (
            -scores.get(item["id"], 0),
            -_priority_for_finding(item, state),
            int(item.get("created_revision", 0)),
            item["id"],
        ),
    )

    groups: list[tuple[list[dict[str, Any]], int, bool]] = []
    covered: set[str] = set()
    for finding in seeds:
        finding_id = finding["id"]
        if finding_id in covered:
            continue
        group_ids = _closure_ids(state, finding_id, indexes)
        group_items = _items_for_group(state, group_ids, indexes)
        group_items = _add_related_frontier_and_agenda(
            state, group_items, group_ids, indexes
        )
        group_items = _sorted_items(_dedupe_items(group_items))
        groups.append(
            (
                group_items,
                json_bytes(group_items),
                _group_is_materially_required(group_items),
            )
        )
        covered.update(group_ids)

    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    selected_size = _items_byte_size([])
    required_groups = 0
    for group_items, group_size, material in groups:
        if group_size > budget and material:
            raise BudgetError(group_size, budget)
        new_items = [
            item
            for item in group_items
            if (item["kind"], item["id"]) not in selected_by_key
        ]
        candidate_size = _appended_items_size(selected_size, new_items)
        if candidate_size <= budget:
            for item in new_items:
                selected_by_key[(item["kind"], item["id"])] = item
            selected_size = candidate_size
            if material:
                required_groups += 1

    # Active policy artifacts remain visible even if there are no matching
    # findings. They are packed in stable priority order after knowledge.
    for item in _policy_items(state):
        key = (item["kind"], item["id"])
        if key in selected_by_key:
            continue
        candidate_size = _appended_items_size(selected_size, [item])
        if candidate_size <= budget:
            selected_by_key[key] = item
            selected_size = candidate_size

    selected = _sorted_items(list(selected_by_key.values()))
    return {
        "schema_version": 1,
        "compiler_version": COMPILER_VERSION,
        "revision": state.revision,
        "objective": objective,
        "budget_unit": "utf8_bytes",
        "budget_limit": budget,
        "budget_used": json_bytes(selected),
        "items": selected,
        "evidence_refs": _evidence_refs(selected),
        "selection_notes": {
            "selection": "deterministic lexical and structural relevance",
            "contradiction_closure": "active challenges/refinements/supersessions are packed together",
                "required_group_count": required_groups,
        },
    }


def _build_indexes(state: ResearchState) -> _ContextIndexes:
    relation_ids: dict[str, set[str]] = defaultdict(set)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for relation in state.relations.values():
        if not relation.get("active", True):
            continue
        relation_id = str(relation["id"])
        source = relation.get("source_finding_id")
        target = relation.get("target_finding_id")
        if source in state.findings:
            relation_ids[source].add(relation_id)
        if target in state.findings:
            relation_ids[target].add(relation_id)
        if relation.get("type") in _CLOSURE_RELATIONS:
            if source in state.findings and target in state.findings:
                adjacency[source].add(target)
                adjacency[target].add(source)

    annotation_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for annotation in state.annotations.values():
        target_kind = annotation.get("target_kind")
        target_id = annotation.get("target_id")
        if isinstance(target_kind, str) and isinstance(target_id, str):
            annotation_ids[(target_kind, target_id)].add(str(annotation["id"]))

    frontier_ids: dict[str, set[str]] = defaultdict(set)
    for frontier in state.frontier_items.values():
        subject_id = frontier.get("subject", {}).get("finding_id")
        if subject_id in state.findings:
            frontier_ids[subject_id].add(frontier["id"])
        for finding_id in [
            *frontier.get("supporting_finding_ids", []),
            *frontier.get("challenging_finding_ids", []),
        ]:
            if finding_id in state.findings:
                frontier_ids[finding_id].add(frontier["id"])

    agenda_by_finding: dict[str, set[str]] = defaultdict(set)
    agenda_by_frontier: dict[str, set[str]] = defaultdict(set)
    for agenda in state.agenda_items.values():
        for finding_id in agenda.get("motivating_finding_ids", []):
            if finding_id in state.findings:
                agenda_by_finding[finding_id].add(agenda["id"])
        for frontier_id in agenda.get("frontier_item_ids", []):
            if frontier_id in state.frontier_items:
                agenda_by_frontier[frontier_id].add(agenda["id"])

    return _ContextIndexes(
        relation_ids_by_finding=_freeze_index(relation_ids),
        annotation_ids_by_target=_freeze_index(annotation_ids),
        frontier_ids_by_finding=_freeze_index(frontier_ids),
        agenda_ids_by_finding=_freeze_index(agenda_by_finding),
        agenda_ids_by_frontier=_freeze_index(agenda_by_frontier),
        adjacency={key: tuple(sorted(value)) for key, value in adjacency.items()},
    )


def _freeze_index(value: dict[Any, set[str]]) -> dict[Any, tuple[str, ...]]:
    return {key: tuple(sorted(items)) for key, items in value.items()}


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value)}


def _finding_score(
    item: dict[str, Any],
    tokens: set[str],
    state: ResearchState,
    indexes: _ContextIndexes,
) -> int:
    finding_id = item["id"]
    text_parts = [
        str(item.get("claim", "")),
        str(item.get("epistemic_status", "")),
        str(item.get("governance_status", "")),
    ]
    for annotation_id in indexes.annotation_ids_by_target.get(("finding", finding_id), ()):
        text_parts.append(str(state.annotations[annotation_id].get("text", "")))
    for frontier_id in indexes.frontier_ids_by_finding.get(finding_id, ()):
        frontier = state.frontier_items[frontier_id]
        text_parts.extend(
            [
                str(frontier.get("rationale", "")),
                str(frontier.get("subject", {}).get("title", "")),
            ]
        )
    for agenda_id in indexes.agenda_ids_by_finding.get(finding_id, ()):
        agenda = state.agenda_items[agenda_id]
        text_parts.extend(
            [
                str(agenda.get("objective", "")),
                str(agenda.get("hypothesis", "")),
                str(agenda.get("question", "")),
            ]
        )
    score = len(tokens.intersection(_tokens(" ".join(text_parts)))) * 100
    if item.get("governance_status") == "accepted":
        score += 20
    if item.get("epistemic_status") == "validated":
        score += 15
    if item.get("epistemic_status") == "contradicted":
        score += 10
    if _finding_has_human_annotation(finding_id, state, indexes):
        score += 25
    return score


def _priority_for_finding(item: dict[str, Any], state: ResearchState) -> int:
    return max(
        [
            int(frontier.get("priority", 0))
            for frontier in state.frontier_items.values()
            if frontier.get("subject", {}).get("finding_id") == item.get("id")
        ]
        or [0]
    )


def _finding_has_human_annotation(
    finding_id: str, state: ResearchState, indexes: _ContextIndexes
) -> bool:
    return any(
        state.annotations[annotation_id].get("author", {}).get("kind") == "human"
        for annotation_id in indexes.annotation_ids_by_target.get(("finding", finding_id), ())
    )


def _closure_ids(state: ResearchState, seed: str, indexes: _ContextIndexes) -> list[str]:
    # V1 keeps the structural expansion bounded to the seed's direct
    # neighborhood. A later compiler version can widen this closure without
    # changing the persisted state contract.
    return sorted({seed, *indexes.adjacency.get(seed, ())})


def _items_for_group(
    state: ResearchState,
    finding_ids: list[str],
    indexes: _ContextIndexes,
) -> list[dict[str, Any]]:
    ids = set(finding_ids)
    items: list[dict[str, Any]] = [
        _item("finding", state.findings[finding_id])
        for finding_id in finding_ids
        if finding_id in state.findings
    ]
    relation_ids = {
        relation_id
        for finding_id in ids
        for relation_id in indexes.relation_ids_by_finding.get(finding_id, ())
        if state.relations[relation_id].get("source_finding_id") in ids
        and state.relations[relation_id].get("target_finding_id") in ids
    }
    for relation_id in relation_ids:
        relation = state.relations[relation_id]
        items.append(_item("relation", relation))
        for annotation_id in indexes.annotation_ids_by_target.get(("relation", relation_id), ()):
            items.append(_item("annotation", state.annotations[annotation_id]))
    for finding_id in ids:
        for annotation_id in indexes.annotation_ids_by_target.get(("finding", finding_id), ()):
            items.append(_item("annotation", state.annotations[annotation_id]))
    return items


def _add_related_frontier_and_agenda(
    state: ResearchState,
    items: list[dict[str, Any]],
    finding_ids: list[str],
    indexes: _ContextIndexes,
) -> list[dict[str, Any]]:
    ids = set(finding_ids)
    frontier_ids = {
        frontier_id
        for finding_id in ids
        for frontier_id in indexes.frontier_ids_by_finding.get(finding_id, ())
    }
    for frontier_id in sorted(frontier_ids):
        frontier = state.frontier_items[frontier_id]
        items.append(_item("frontier_item", frontier))
        lane_id = frontier.get("lane_id")
        if lane_id in state.lanes:
            items.append(_item("frontier_lane", state.lanes[lane_id]))
        for annotation_id in indexes.annotation_ids_by_target.get(("frontier_item", frontier_id), ()):
            items.append(_item("annotation", state.annotations[annotation_id]))

    agenda_ids = {
        agenda_id
        for finding_id in ids
        for agenda_id in indexes.agenda_ids_by_finding.get(finding_id, ())
    }
    agenda_ids.update(
        agenda_id
        for frontier_id in frontier_ids
        for agenda_id in indexes.agenda_ids_by_frontier.get(frontier_id, ())
    )
    for agenda_id in sorted(agenda_ids):
        agenda = state.agenda_items[agenda_id]
        items.append(_item("agenda_item", agenda))
        for annotation_id in indexes.annotation_ids_by_target.get(("agenda_item", agenda_id), ()):
            items.append(_item("annotation", state.annotations[annotation_id]))
    return items


def _policy_items(state: ResearchState) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    frontiers = [
        frontier
        for frontier in state.frontier_items.values()
        if frontier.get("status") == "active"
    ]
    frontiers.sort(
        key=lambda item: (
            -int(item.get("priority", 0)),
            int(item.get("created_revision", 0)),
            str(item.get("id", "")),
        )
    )
    for frontier in frontiers:
        items.append(_item("frontier_item", frontier))
        lane_id = frontier.get("lane_id")
        if lane_id in state.lanes:
            items.append(_item("frontier_lane", state.lanes[lane_id]))

    agendas = [
        agenda
        for agenda in state.agenda_items.values()
        if agenda.get("status") in {"active", "proposed"}
    ]
    agendas.sort(
        key=lambda item: (
            0 if item.get("status") == "active" else 1,
            -int(item.get("priority", 0)),
            int(item.get("created_revision", 0)),
            str(item.get("id", "")),
        )
    )
    items.extend(_item("agenda_item", agenda) for agenda in agendas)
    for annotation in state.annotations.values():
        if annotation.get("target_kind") == "research_state":
            items.append(_item("annotation", annotation))
    return items


def _item(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "id": data["id"], "data": data}


def _sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (item["kind"], item["id"]))


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        result[(item["kind"], item["id"])] = item
    return list(result.values())


def _group_is_materially_required(items: list[dict[str, Any]]) -> bool:
    return any(
        item["kind"] == "relation"
        and item["data"].get("type") in _MATERIAL_RELATIONS
        for item in items
    )


def _evidence_refs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for item in items:
        for ref in item.get("data", {}).get("evidence", []):
            run_hash = ref.get("run_hash")
            if run_hash:
                refs.setdefault(run_hash, ref)
    return [refs[key] for key in sorted(refs)]


def _items_byte_size(items: list[dict[str, Any]]) -> int:
    if not items:
        return 2
    return 2 + sum(json_bytes(item) for item in items) + len(items) - 1


def _appended_items_size(current_size: int, new_items: list[dict[str, Any]]) -> int:
    if not new_items:
        return current_size
    added = sum(json_bytes(item) for item in new_items)
    separators = len(new_items) if current_size > 2 else len(new_items) - 1
    return current_size + added + separators
