from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from aimx.research.errors import ResearchError
from aimx.research.models import ResearchState


def utc_timestamp(value: datetime | None = None) -> str:
    """Format a timestamp used by research envelopes and history."""

    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def sort_entities(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic creation-revision then stable-ID ordering."""

    return sorted(
        values,
        key=lambda value: (
            int(value.get("created_revision", 0)),
            str(value.get("id", "")),
        ),
    )


def render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def render_error(error: ResearchError) -> str:
    return render_json(
        {
            "schema_version": 1,
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
            },
        }
    )


def render_state_json(state: ResearchState, repo: str) -> str:
    return render_json(state.as_dict(repo=repo))


def render_state_human(state: ResearchState, repo: str) -> str:
    lines = [
        "Research State",
        f"  Repository: {repo}",
        f"  Revision:   {state.revision}",
        f"  Findings:   {len(state.findings)}",
        f"  Relations:  {len([r for r in state.relations.values() if r.get('active', True)])}",
        f"  Annotations:{len(state.annotations)}",
        f"  Frontier:   {len([i for i in state.frontier_items.values() if i.get('status') == 'active'])} active",
        f"  Agenda:     {len([i for i in state.agenda_items.values() if i.get('status') in {'active', 'proposed'}])} actionable",
    ]
    return "\n".join(lines)


def render_findings(state: ResearchState, *, output_json: bool, repo: str) -> str:
    rows = sort_entities(list(state.findings.values()))
    payload = {"schema_version": 1, "revision": state.revision, "repo": repo, "findings": rows}
    if output_json:
        return render_json(payload)
    if not rows:
        return "No findings."
    return "\n".join(
        f"{row['id']}\t{row.get('epistemic_status')}\t{row.get('governance_status')}\t{row.get('claim')}"
        for row in rows
    )


def render_finding(state: ResearchState, finding_id: str, *, output_json: bool, repo: str) -> str:
    finding = state.findings[finding_id]
    relations = [
        relation
        for relation in state.relations.values()
        if relation.get("active", True)
        and (relation.get("source_finding_id") == finding_id or relation.get("target_finding_id") == finding_id)
    ]
    annotations = [
        annotation
        for annotation in state.annotations.values()
        if annotation.get("target_kind") == "finding" and annotation.get("target_id") == finding_id
    ]
    payload = {
        "schema_version": 1,
        "revision": state.revision,
        "repo": repo,
        "finding": finding,
        "relations": sort_entities(relations),
        "annotations": sort_entities(annotations),
    }
    if output_json:
        return render_json(payload)
    lines = [
        f"{finding['id']}: {finding.get('claim')}",
        f"  epistemic: {finding.get('epistemic_status')} ({finding.get('confidence')})",
        f"  governance: {finding.get('governance_status')}",
        f"  evidence: {', '.join(_format_evidence_reference(ref) for ref in finding.get('evidence', [])) or '-'}",
    ]
    if annotations:
        lines.append("  annotations:")
        lines.extend(f"    - {item.get('text')}" for item in annotations)
    return "\n".join(lines)


def _format_evidence_reference(reference: dict[str, Any]) -> str:
    run_hash = str(reference.get("run_hash", ""))
    if reference.get("availability") == "unavailable":
        return f"{run_hash} (unavailable)"
    return run_hash


def render_relations(
    state: ResearchState, finding_id: str, *, output_json: bool, repo: str
) -> str:
    relations = [
        relation
        for relation in state.relations.values()
        if relation.get("source_finding_id") == finding_id
        or relation.get("target_finding_id") == finding_id
    ]
    payload = {
        "schema_version": 1,
        "revision": state.revision,
        "repo": repo,
        "relations": sort_entities(relations),
    }
    if output_json:
        return render_json(payload)
    if not relations:
        return "No lineage relations."
    return "\n".join(
        f"{item['id']}\t{item.get('source_finding_id')}\t{item.get('type')}\t{item.get('target_finding_id')}\tactive={item.get('active', True)}"
        for item in sort_entities(relations)
    )


def render_frontier(state: ResearchState, *, output_json: bool, repo: str) -> str:
    payload = {
        "schema_version": 1,
        "revision": state.revision,
        "repo": repo,
        "frontier": {
            "lanes": sort_entities(list(state.lanes.values())),
            "items": sort_entities(list(state.frontier_items.values())),
        },
    }
    if output_json:
        return render_json(payload)
    lines = ["Frontier"]
    for item in payload["frontier"]["items"]:
        subject = item.get("subject", {}).get("finding_id") or item.get("subject", {}).get("title")
        lines.append(f"{item['id']}\t{item.get('status')}\tpriority={item.get('priority')}\t{subject}")
    return "\n".join(lines)


def render_agenda(payload: dict[str, Any], *, output_json: bool) -> str:
    if output_json:
        return render_json(payload)
    items = payload.get("items", [])
    if not items:
        return "No agenda items."
    return "\n".join(
        f"{item['id']}\t{item.get('status')}\tpriority={item.get('priority')}\t{item.get('objective')}"
        for item in items
    )


def render_context(payload: dict[str, Any], *, output_json: bool) -> str:
    if output_json:
        return render_json(payload)
    lines = [
        f"Research Context revision {payload['revision']}",
        f"Objective: {payload['objective']}",
        f"Budget: {payload['budget_used']}/{payload['budget_limit']} {payload['budget_unit']}",
    ]
    for item in payload.get("items", []):
        lines.append(f"- {item['kind']} {item['id']}")
    return "\n".join(lines)
