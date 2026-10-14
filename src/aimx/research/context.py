from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimx.research.errors import BudgetError, ValidationError
from aimx.research.models import ResearchState, json_bytes

COMPILER_VERSION = "context-v2"
_TOKEN_RE = re.compile(r"[a-z0-9_/-]+|[\u3400-\u9fff]+", re.IGNORECASE)
_CLOSURE_RELATIONS = {"challenges", "refines", "supersedes"}
_MATERIAL_RELATIONS = {"challenges", "refines", "supersedes"}


@dataclass(frozen=True)
class _ContextIndexes:
    relation_ids_by_finding: dict[str, tuple[str, ...]]
    annotation_ids_by_target: dict[tuple[str, str], tuple[str, ...]]
    frontier_ids_by_finding: dict[str, tuple[str, ...]]
    agenda_ids_by_finding: dict[str, tuple[str, ...]]
    agenda_ids_by_frontier: dict[str, tuple[str, ...]]
    adjacency: dict[str, tuple[str, ...]]


def compile_context(
    state: ResearchState,
    objective: str,
    budget: int,
    *,
    repo: str | None = None,
    loop_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(objective, str) or not objective.strip():
        raise ValidationError("Context objective must not be empty")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValidationError("Context budget must be a positive integer")

    indexes = _build_indexes(state)
    tokens = _tokens(objective)
    activities_by_id = state.activities
    allowed_loop_activity_ids = {
        activity_id
        for activity_id, activity in activities_by_id.items()
        if activity.get("kind") == "loop" and activity.get("loop_id") == loop_id
    } if loop_id else None
    allowed_follow_up_activity_ids = {
        activity_id
        for activity_id, activity in activities_by_id.items()
        if loop_id and activity.get("kind") == "follow_up" and activity.get("related_loop_id") == loop_id
    } if loop_id else set()
    if loop_id and not allowed_loop_activity_ids:
        raise ValidationError(f"Unknown loop: {loop_id}", code="unknown_loop")

    findings = [
        item for item in state.findings.values()
        if _belongs_to_loop(item, allowed_loop_activity_ids)
        or (allowed_follow_up_activity_ids and item.get("source_activity_id") in allowed_follow_up_activity_ids)
    ]
    scores = {item["id"]: _finding_score(item, tokens, state, indexes) for item in findings}
    seeds = sorted(
        (item for item in findings if scores.get(item["id"], 0) > 0),
        key=lambda item: (
            -scores[item["id"]],
            -_priority_for_finding(item, state, indexes),
            -_explanation_revision(item, state, indexes),
            str(item["id"]),
        ),
    )

    groups: list[dict[str, Any]] = []
    covered: set[str] = set()
    for finding in seeds:
        finding_id = str(finding["id"])
        if finding_id in covered:
            continue
        finding_ids = _closure_ids(state, finding_id, indexes)
        items = _items_for_group(state, finding_ids, indexes)
        items = _add_related_frontier_and_agenda(state, items, finding_ids, indexes, allowed_loop_activity_ids, allowed_follow_up_activity_ids)
        if allowed_loop_activity_ids is not None:
            allowed_sources = allowed_loop_activity_ids | (allowed_follow_up_activity_ids or set())
            items = [
                item for item in items
                if item.get("kind") != "annotation"
                or item.get("data", {}).get("source_activity_id") in allowed_sources
            ]
        items = _sorted_items(_dedupe_items(items))
        groups.append({
            "items": items,
            "seed_ids": [finding_id],
            "finding_ids": finding_ids,
            "reason": "lexical_match_with_complete_active_relation_closure",
            "material": _group_is_materially_required(items),
            "revision": _explanation_revision(finding, state, indexes),
            "score": scores[finding_id],
            "closure_dependency_ids": sorted(
                dependency_id for dependency_id in finding_ids
                if allowed_loop_activity_ids is not None
                and state.findings[dependency_id].get("source_activity_id") not in allowed_loop_activity_ids
            ),
        })
        covered.update(finding_ids)

    annotation_groups = _matching_annotation_groups(
        state, indexes, tokens, allowed_loop_activity_ids, covered,
        allowed_follow_up_activity_ids=allowed_follow_up_activity_ids,
    )
    groups.extend(annotation_groups)
    # Structured matches rank by lexical relevance first; revision breaks ties.
    groups.sort(key=lambda group: (
        0 if group.get("finding_ids") else 1,
        -int(group.get("score", 0)),
        -int(group.get("revision", 0)),
        tuple(group["seed_ids"]),
    ))

    base = _base_payload(state, objective, budget, repo, loop_id)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    selected_groups: list[dict[str, Any]] = []
    omitted_groups: list[dict[str, Any]] = []
    query_commands = _query_commands(repo, loop_id)

    all_group_items = _sorted_items(_dedupe_items([item for group in groups for item in group["items"]]))
    all_group_notes = [_group_note(group) for group in groups]
    all_policy_items = _sorted_items(_dedupe_items(_policy_items(state, indexes))) if allowed_loop_activity_ids is None else []
    full_items = _sorted_items(_dedupe_items([*all_group_items, *all_policy_items]))
    full_payload = _assemble_payload(base, full_items, all_group_notes, [], state, query_commands)
    _set_budget_used_fixed_point(full_payload)
    if _response_size(full_payload) <= budget:
        selected = {(item["kind"], item["id"]): item for item in full_items}
        selected_groups = all_group_notes
    elif groups:
        def prefix_payload(count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            included = groups[:count]
            omitted = groups[count:]
            included_notes = [_group_note(group) for group in included]
            items_by_key = {
                (item["kind"], item["id"]): item
                for group in included
                for item in group["items"]
            }
            omitted_counts: dict[str, int] = {}
            for group in omitted:
                omitted_counts["response_budget"] = omitted_counts.get("response_budget", 0) + len(group["items"])
            legacy = ["legacy_relationships_unverified: free-text annotations do not prove support or contradiction"] if any(group.get("legacy_text") for group in included) else []
            payload = _assemble_payload(
                base,
                _sorted_items(_dedupe_items(list(items_by_key.values()))),
                included_notes,
                legacy,
                state,
                query_commands,
                omitted_count=sum(len(group["items"]) for group in omitted),
                omitted_by_reason=omitted_counts,
            )
            _set_budget_used_fixed_point(payload)
            return payload, included_notes

        low, high = 0, len(groups)
        best = -1
        while low <= high:
            middle = (low + high) // 2
            candidate_payload, _ = prefix_payload(middle)
            if _response_size(candidate_payload) <= budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best < 1:
            required_only, _ = prefix_payload(1)
            required_only["selection_notes"]["omitted_count"] = 0
            required_only["selection_notes"]["omitted_by_reason"] = {}
            _set_budget_used_fixed_point(required_only)
            raise BudgetError(
                _response_size(required_only), budget,
                query_commands=query_commands, group_ids=groups[0]["seed_ids"],
            )
        best_payload, selected_groups = prefix_payload(best)
        selected = {
            (item["kind"], item["id"]): item
            for item in best_payload["items"]
        }
        omitted_groups = groups[best:]
        omitted_count = sum(len(group["items"]) for group in omitted_groups)
        omitted_by_reason = {"response_budget": omitted_count} if omitted_count else {}
        for item in all_policy_items:
            key = (item["kind"], item["id"])
            if key in selected:
                continue
            candidate = dict(selected)
            candidate[key] = item
            candidate_payload = _assemble_payload(
                base, _sorted_items(list(candidate.values())), selected_groups,
                ["legacy_relationships_unverified: free-text annotations do not prove support or contradiction"] if any(group.get("legacy_text") for group in groups[:best]) else [],
                state, query_commands, omitted_count=omitted_count, omitted_by_reason=omitted_by_reason,
            )
            _set_budget_used_fixed_point(candidate_payload)
            if _response_size(candidate_payload) <= budget:
                selected = candidate
    else:
        selected = {}
        # Unrelated policy items are optional and ordered by stable priority.
        for item in all_policy_items:
            candidate_items = _sorted_items([*selected.values(), item])
            candidate_payload = _assemble_payload(base, candidate_items, [], [], state, query_commands)
            _set_budget_used_fixed_point(candidate_payload)
            if _response_size(candidate_payload) <= budget:
                selected[(item["kind"], item["id"])] = item

    omitted_by_reason: dict[str, int] = {}
    for group in omitted_groups:
        reason = "response_budget"
        omitted_by_reason[reason] = omitted_by_reason.get(reason, 0) + len(group["items"])
    limitations = []
    if any(group.get("legacy_text") for group in selected_groups):
        limitations.append("legacy_relationships_unverified: free-text annotations do not prove support or contradiction")
    payload = _assemble_payload(
        base,
        _sorted_items(list(selected.values())),
        selected_groups,
        limitations,
        state,
        query_commands,
        omitted_count=sum(len(group["items"]) for group in omitted_groups),
        omitted_by_reason=omitted_by_reason,
    )
    _set_budget_used_fixed_point(payload)
    actual_size = _response_size(payload)
    if actual_size > budget:
        raise BudgetError(actual_size, budget, query_commands=query_commands)
    return payload


def _base_payload(state: ResearchState, objective: str, budget: int, repo: str | None, loop_id: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "compiler_version": COMPILER_VERSION,
        "repo": repo,
        "revision": state.revision,
        "objective": objective,
        "loop_id": loop_id,
        "budget_unit": "utf8_bytes",
        "budget_scope": "response",
        "budget_limit": budget,
    }


def _assemble_payload(
    base: dict[str, Any],
    items: list[dict[str, Any]],
    selected_groups: list[dict[str, Any]],
    limitations: list[str],
    state: ResearchState,
    query_commands: list[str],
    *,
    omitted_count: int = 0,
    omitted_by_reason: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload = dict(base)
    payload.update({
        "budget_used": 0,
        "items": items,
        "evidence_refs": _evidence_refs(items, state, str(base.get("repo", "."))),
        "selection_notes": {
            "policy": "lexical_relevance_then_recursive_active_relation_closure",
            "match_status": "matched" if selected_groups else "no_matching_knowledge",
            "selected_groups": selected_groups,
            "omitted_count": omitted_count,
            "omitted_by_reason": dict(sorted((omitted_by_reason or {}).items())),
            "limitations": list(limitations),
            "query_commands": query_commands,
        },
    })
    return payload


def _set_budget_used_fixed_point(payload: dict[str, Any]) -> None:
    # JSON encodes budget_used as one integer token. Replacing its one-byte
    # placeholder with the final decimal width gives an exact fixed point
    # without serializing a potentially large response repeatedly.
    payload["budget_used"] = 0
    baseline = _response_size(payload)
    value = baseline
    for _ in range(4):
        measured = baseline - 1 + len(str(value))
        if measured == value:
            payload["budget_used"] = value
            return
        value = measured
    payload["budget_used"] = value
    if _response_size(payload) != value:
        raise ValidationError("Unable to stabilize context response byte count", code="invalid_context")


def _response_size(payload: dict[str, Any]) -> int:
    # The CLI writes exactly one LF after the compact JSON output.
    return json_bytes(payload) + 1


def _query_commands(repo: str | None, loop_id: str | None) -> list[str]:
    args = ["aimx", "finding", "ls"]
    if repo:
        args += ["--repo", repo]
    if loop_id:
        args += ["--loop", loop_id]
    args.append("--json")
    return [" ".join(shlex.quote(value) for value in args), "aimx research annotations" + (f" --repo {shlex.quote(repo)}" if repo else "") + " --json"]


def _build_indexes(state: ResearchState) -> _ContextIndexes:
    relation_ids: dict[str, set[str]] = defaultdict(set)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for relation in state.relations.values():
        if not relation.get("active", True):
            continue
        relation_id = str(relation["id"])
        source, target = relation.get("source_finding_id"), relation.get("target_finding_id")
        if source in state.findings:
            relation_ids[source].add(relation_id)
        if target in state.findings:
            relation_ids[target].add(relation_id)
        if relation.get("type") in _CLOSURE_RELATIONS and source in state.findings and target in state.findings:
            adjacency[source].add(target)
            adjacency[target].add(source)

    annotation_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for annotation in state.annotations.values():
        target_kind, target_id = annotation.get("target_kind"), annotation.get("target_id")
        if isinstance(target_kind, str) and isinstance(target_id, str):
            annotation_ids[(target_kind, target_id)].add(str(annotation["id"]))

    frontier_ids: dict[str, set[str]] = defaultdict(set)
    for frontier in state.frontier_items.values():
        subject_id = frontier.get("subject", {}).get("finding_id")
        if subject_id in state.findings:
            frontier_ids[subject_id].add(frontier["id"])
        for finding_id in [*frontier.get("supporting_finding_ids", []), *frontier.get("challenging_finding_ids", [])]:
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


def _freeze_index(values: dict[Any, set[str]]) -> dict[Any, tuple[str, ...]]:
    return {key: tuple(sorted(items)) for key, items in values.items()}


def _tokens(value: str) -> set[str]:
    result: set[str] = set()
    for token in _TOKEN_RE.findall(value.casefold()):
        if token and all(ord(char) < 128 for char in token):
            result.add(token)
        elif len(token) == 1:
            result.add(token)
        else:
            result.update(token[index : index + 2] for index in range(len(token) - 1))
    return result


def _priority_for_finding(item: dict[str, Any], state: ResearchState, indexes: _ContextIndexes) -> int:
    return max(
        (int(state.frontier_items[item_id].get("priority", 0)) for item_id in indexes.frontier_ids_by_finding.get(str(item.get("id")), ())),
        default=0,
    )


def _finding_score(item: dict[str, Any], tokens: set[str], state: ResearchState, indexes: _ContextIndexes) -> int:
    finding_id = str(item["id"])
    parts = [str(item.get(key, "")) for key in ("claim", "result_kind", "scope", "limitations", "topics")]
    for annotation_id in indexes.annotation_ids_by_target.get(("finding", finding_id), ()):
        parts.append(str(state.annotations[annotation_id].get("text", "")))
    for frontier_id in indexes.frontier_ids_by_finding.get(finding_id, ()):
        frontier = state.frontier_items[frontier_id]
        parts.extend([str(frontier.get("rationale", "")), str(frontier.get("subject", {}).get("title", ""))])
    for agenda_id in indexes.agenda_ids_by_finding.get(finding_id, ()):
        agenda = state.agenda_items[agenda_id]
        parts.extend(str(agenda.get(key, "")) for key in ("objective", "hypothesis", "question"))
    matched = tokens.intersection(_tokens(" ".join(parts)))
    if not matched:
        return 0
    score = len(matched) * 100
    # Status can order actual matches, but can never create one.
    score += 20 if item.get("governance_status") == "accepted" else 0
    score += 15 if item.get("epistemic_status") == "validated" else 0
    score += 10 if item.get("epistemic_status") == "contradicted" else 0
    score += 25 if _finding_has_human_annotation(finding_id, state, indexes) else 0
    return score


def _finding_has_human_annotation(finding_id: str, state: ResearchState, indexes: _ContextIndexes) -> bool:
    return any(
        state.annotations[annotation_id].get("author", {}).get("kind") == "human"
        for annotation_id in indexes.annotation_ids_by_target.get(("finding", finding_id), ())
    )


def _belongs_to_loop(item: dict[str, Any], allowed_activity_ids: set[str] | None) -> bool:
    return allowed_activity_ids is None or item.get("source_activity_id") in allowed_activity_ids


def _explanation_revision(item: dict[str, Any], state: ResearchState, indexes: _ContextIndexes) -> int:
    latest = int(item.get("created_revision", 0))
    for relation_id in indexes.relation_ids_by_finding.get(str(item.get("id")), ()):
        latest = max(latest, int(state.relations[relation_id].get("created_revision", 0)))
    for annotation_id in indexes.annotation_ids_by_target.get(("finding", str(item.get("id"))), ()):
        latest = max(latest, int(state.annotations[annotation_id].get("created_revision", 0)))
    return latest


def _closure_ids(state: ResearchState, seed: str, indexes: _ContextIndexes) -> list[str]:
    visited = {seed}
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        for neighbor in indexes.adjacency.get(current, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return sorted(visited)


def _items_for_group(state: ResearchState, finding_ids: list[str], indexes: _ContextIndexes) -> list[dict[str, Any]]:
    ids = set(finding_ids)
    items = [_item("finding", state.findings[finding_id]) for finding_id in finding_ids]
    relation_ids = {
        relation_id for finding_id in ids for relation_id in indexes.relation_ids_by_finding.get(finding_id, ())
        if state.relations[relation_id].get("source_finding_id") in ids and state.relations[relation_id].get("target_finding_id") in ids
    }
    for relation_id in sorted(relation_ids):
        items.append(_item("relation", state.relations[relation_id]))
        items.extend(_item("annotation", state.annotations[ann_id]) for ann_id in indexes.annotation_ids_by_target.get(("relation", relation_id), ()))
    for finding_id in sorted(ids):
        items.extend(_item("annotation", state.annotations[ann_id]) for ann_id in indexes.annotation_ids_by_target.get(("finding", finding_id), ()))
    return items


def _add_related_frontier_and_agenda(
    state: ResearchState,
    items: list[dict[str, Any]],
    finding_ids: list[str],
    indexes: _ContextIndexes,
    allowed_activity_ids: set[str] | None = None,
    allowed_follow_up_activity_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    ids = set(finding_ids)
    frontier_ids = {value for finding_id in ids for value in indexes.frontier_ids_by_finding.get(finding_id, ())}
    for frontier_id in sorted(frontier_ids):
        frontier = state.frontier_items[frontier_id]
        items.append(_item("frontier_item", frontier))
        lane_id = frontier.get("lane_id")
        if lane_id in state.lanes:
            items.append(_item("frontier_lane", state.lanes[lane_id]))
        items.extend(
            _item("annotation", state.annotations[ann_id]) for ann_id in indexes.annotation_ids_by_target.get(("frontier_item", frontier_id), ())
            if allowed_activity_ids is None or state.annotations[ann_id].get("source_activity_id") in (allowed_activity_ids | (allowed_follow_up_activity_ids or set()))
        )
    agenda_ids = {value for finding_id in ids for value in indexes.agenda_ids_by_finding.get(finding_id, ())}
    agenda_ids.update(value for frontier_id in frontier_ids for value in indexes.agenda_ids_by_frontier.get(frontier_id, ()))
    for agenda_id in sorted(agenda_ids):
        items.append(_item("agenda_item", state.agenda_items[agenda_id]))
        items.extend(
            _item("annotation", state.annotations[ann_id]) for ann_id in indexes.annotation_ids_by_target.get(("agenda_item", agenda_id), ())
            if allowed_activity_ids is None or state.annotations[ann_id].get("source_activity_id") in (allowed_activity_ids | (allowed_follow_up_activity_ids or set()))
        )
    return items


def _matching_annotation_groups(
    state: ResearchState,
    indexes: _ContextIndexes,
    tokens: set[str],
    allowed_activity_ids: set[str] | None,
    covered_findings: set[str],
    *,
    allowed_follow_up_activity_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for annotation in state.annotations.values():
        if allowed_activity_ids is not None and annotation.get("source_activity_id") not in (allowed_activity_ids | (allowed_follow_up_activity_ids or set())):
            continue
        text = str(annotation.get("text", ""))
        matches = len(tokens.intersection(_tokens(text)))
        if matches:
            rows.append((matches, int(annotation.get("created_revision", 0)), annotation))
    rows.sort(key=lambda row: (-row[0], -row[1], str(row[2].get("id", ""))))
    groups = []
    consumed: set[str] = set()
    for matches, revision, annotation in rows:
        ann_id = str(annotation["id"])
        if ann_id in consumed:
            continue
        target_kind, target_id = annotation.get("target_kind"), annotation.get("target_id")
        if target_kind == "finding" and target_id in covered_findings:
            continue
        if allowed_activity_ids is not None and target_kind == "finding" and target_id in state.findings:
            allowed_sources = allowed_activity_ids | (allowed_follow_up_activity_ids or set())
            target_source = state.findings[str(target_id)].get("source_activity_id")
            if target_source not in allowed_sources:
                continue
        if isinstance(target_kind, str) and isinstance(target_id, str):
            related = [state.annotations[ann] for ann in indexes.annotation_ids_by_target.get((target_kind, target_id), ())]
        else:
            related = [annotation]
        if allowed_activity_ids is not None:
            allowed_sources = allowed_activity_ids | (allowed_follow_up_activity_ids or set())
            related = [item for item in related if item.get("source_activity_id") in allowed_sources]
        related.sort(key=lambda value: (int(value.get("created_revision", 0)), str(value.get("id", ""))))
        consumed.update(str(value.get("id")) for value in related)
        items = [_item("annotation", value) for value in related]
        groups.append({"items": _sorted_items(_dedupe_items(items)), "seed_ids": [ann_id], "finding_ids": [], "reason": "lexical_match_legacy_annotation", "material": False, "revision": revision, "score": matches, "legacy_text": True})
    return groups


def _policy_items(state: ResearchState, indexes: _ContextIndexes) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    frontiers = [item for item in state.frontier_items.values() if item.get("status") == "active"]
    frontiers.sort(key=lambda item: (-int(item.get("priority", 0)), int(item.get("created_revision", 0)), str(item.get("id", ""))))
    for frontier in frontiers:
        items.append(_item("frontier_item", frontier))
        items.extend(_item("annotation", state.annotations[ann_id]) for ann_id in indexes.annotation_ids_by_target.get(("frontier_item", frontier["id"]), ()))
        lane_id = frontier.get("lane_id")
        if lane_id in state.lanes:
            items.append(_item("frontier_lane", state.lanes[lane_id]))
    agendas = [item for item in state.agenda_items.values() if item.get("status") in {"active", "proposed"}]
    agendas.sort(key=lambda item: (0 if item.get("status") == "active" else 1, -int(item.get("priority", 0)), int(item.get("created_revision", 0)), str(item.get("id", ""))))
    for agenda in agendas:
        items.append(_item("agenda_item", agenda))
        items.extend(_item("annotation", state.annotations[ann_id]) for ann_id in indexes.annotation_ids_by_target.get(("agenda_item", agenda["id"]), ()))
    return items


def _item(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "id": data["id"], "data": data}


def _sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (item["kind"], item["id"]))


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = {(item["kind"], item["id"]): item for item in items}
    return list(result.values())


def _group_is_materially_required(items: list[dict[str, Any]]) -> bool:
    return any(item["kind"] == "relation" and item["data"].get("type") in _MATERIAL_RELATIONS for item in items)


def _group_note(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed_ids": list(group["seed_ids"]),
        "finding_ids": list(group["finding_ids"]),
        "reason": group["reason"],
        "legacy_text": bool(group.get("legacy_text")),
        "closure_dependency_ids": list(group.get("closure_dependency_ids", [])),
    }


def _evidence_refs(items: list[dict[str, Any]], state: ResearchState, repo: str) -> list[dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for item in items:
        for ref in item.get("data", {}).get("evidence", []):
            identity = ref.get("sha256") or ref.get("run_hash")
            if identity:
                projected = dict(ref)
                if ref.get("kind") == "local_artifact":
                    location = Path(str(ref.get("location", ""))).expanduser()
                    root_path = Path(repo).resolve()
                    is_relative = not location.is_absolute()
                    candidate = Path(os.path.abspath(location if not is_relative else root_path / location))
                    if is_relative and not candidate.is_relative_to(root_path):
                        projected["availability"] = "unsupported"
                        projected["identity_status"] = "not_checked"
                    else:
                        current = Path(candidate.anchor)
                        components = candidate.parts[1:]
                        availability = "available"
                        for index, component in enumerate(components):
                            current = current / component
                            try:
                                info = current.lstat()
                            except FileNotFoundError:
                                availability = "missing"
                                break
                            except OSError:
                                availability = "unreadable"
                                break
                            if stat.S_ISLNK(info.st_mode):
                                availability = "unknown"
                                break
                            if index < len(components) - 1 and not stat.S_ISDIR(info.st_mode):
                                availability = "unsupported"
                                break
                            if index == len(components) - 1 and not stat.S_ISREG(info.st_mode):
                                availability = "unsupported"
                        projected.setdefault("availability", availability)
                        projected.setdefault("identity_status", "not_checked")
                else:
                    projected.setdefault("availability", "unknown")
                    projected.setdefault("identity_status", "not_checked")
                projected.setdefault("availability", "unknown")
                projected.setdefault("identity_status", "not_checked")
                reviews = state.evidence_reviews.get(str(item["id"]), [])
                fingerprint = hashlib.sha256(json.dumps({key: value for key, value in ref.items() if key != "availability"}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                matching_reviews = [review for review in reviews if review.get("evidence_fingerprint") == fingerprint]
                projected.setdefault("scientific_review", matching_reviews[-1].get("scientific_review") if matching_reviews else "not_reviewed")
                projected.setdefault("evidence_fingerprint", fingerprint)
                projected.setdefault("finding_id", item["id"])
                projected["uses"] = [{"finding_id": item["id"], "purpose": ref.get("purpose")}]
                key = str(identity)
                if key in references:
                    references[key]["uses"].extend(projected["uses"])
                else:
                    references[key] = projected
    return [references[key] for key in sorted(references)]
