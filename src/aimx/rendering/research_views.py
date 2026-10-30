from __future__ import annotations

import json
import shlex
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


def render_findings(
    state: ResearchState,
    *,
    output_json: bool,
    repo: str,
    findings: list[dict[str, Any]] | None = None,
    match_reasons: dict[str, list[str]] | None = None,
    loop_id: str | None = None,
    follow_up_findings: list[dict[str, Any]] | None = None,
) -> str:
    rows = sort_entities(findings if findings is not None else list(state.findings.values()))
    payload = {"schema_version": 1, "revision": state.revision, "repo": repo, "findings": rows}
    if match_reasons is not None:
        payload["match_reasons"] = match_reasons
    if loop_id is not None:
        payload["loop_id"] = loop_id
    if follow_up_findings:
        payload["follow_up_findings"] = sort_entities(follow_up_findings)
        payload["related_findings"] = {
            str(item["id"]): _related_findings(state, str(item["id"]))
            for item in payload["follow_up_findings"]
        }
    other_annotations = state.annotations.values()
    if loop_id is not None:
        loop_activity_ids = {key for key, item in state.activities.items() if item.get("kind") == "loop" and item.get("loop_id") == loop_id}
        follow_up_activity_ids = {key for key, item in state.activities.items() if item.get("kind") == "follow_up" and item.get("related_loop_id") == loop_id}
        other_annotations = [item for item in state.annotations.values() if item.get("source_activity_id") in loop_activity_ids | follow_up_activity_ids]
    payload["other_record_counts"] = {"annotations": len(list(other_annotations)), "relations": len(state.relations), "agenda_items": len(state.agenda_items)}
    annotations = state.annotations.values()
    if loop_id is not None:
        loop_activity_ids = {key for key, item in state.activities.items() if item.get("kind") == "loop" and item.get("loop_id") == loop_id}
        follow_up_activity_ids = {key for key, item in state.activities.items() if item.get("kind") == "follow_up" and item.get("related_loop_id") == loop_id}
        annotations = [item for item in state.annotations.values() if item.get("source_activity_id") in loop_activity_ids | follow_up_activity_ids or "source_activity_id" not in item]
    payload["unattributed"] = {
        "findings": sum("source_activity_id" not in item for item in state.findings.values()),
        "annotations": sum("source_activity_id" not in item for item in annotations),
    }
    quoted_repo = shlex.quote(repo)
    payload["query_commands"] = [f"aimx research annotations --repo {quoted_repo} --json"]
    if output_json:
        return render_json(payload)
    if not rows and not follow_up_findings:
        other = len(state.annotations) + len(state.relations) + len(state.agenda_items)
        return f"No matching findings. {other} other research records are available; run aimx research annotations --repo {quoted_repo} --json."
    lines = [
        f"{row['id']}\t{row.get('epistemic_status')}\t{row.get('governance_status')}\t{row.get('claim')}"
        for row in rows
    ]
    lines.extend(f"follow-up {row['id']}\t{row.get('result_kind', 'unknown')}\t{row.get('claim')}" for row in sort_entities(follow_up_findings or []))
    for item in payload.get("follow_up_findings", []):
        for related in payload.get("related_findings", {}).get(str(item["id"]), []):
            lines.append(f"  related {related['relation'].get('type')} {related['finding'].get('id')}\t{related['finding'].get('claim')}\tcross_activity={related.get('cross_activity')}")
    return "\n".join(lines)


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
        "related_findings": _related_findings(state, finding_id),
    }
    if output_json:
        return render_json(payload)
    lines = [
        f"{finding['id']}: {finding.get('claim')}",
        f"  epistemic: {finding.get('epistemic_status')} ({finding.get('confidence')})",
        f"  governance: {finding.get('governance_status')}",
        f"  result_kind: {finding.get('result_kind', 'unknown')}",
        f"  topics: {', '.join(finding.get('topics', [])) or '-'}",
        f"  scope: {finding.get('scope', 'unknown')}",
        f"  limitations: {', '.join(finding.get('limitations', [])) or '-'}",
        f"  source_activity_id: {finding.get('source_activity_id', 'unattributed')}",
        f"  evidence: {', '.join(_format_evidence_reference(ref) for ref in finding.get('evidence', [])) or '-'}",
    ]
    if annotations:
        lines.append("  annotations:")
        lines.extend(f"    - {item.get('text')}" for item in annotations)
    return "\n".join(lines)


def _related_findings(state: ResearchState, finding_id: str) -> list[dict[str, Any]]:
    related = []
    for relation in state.relations.values():
        if not relation.get("active", True):
            continue
        if finding_id not in {relation.get("source_finding_id"), relation.get("target_finding_id")}:
            continue
        other_id = relation.get("target_finding_id") if relation.get("source_finding_id") == finding_id else relation.get("source_finding_id")
        other = state.findings.get(other_id)
        if other is not None:
            related.append({"relation": relation, "finding": other, "cross_activity": other.get("source_activity_id") != state.findings[finding_id].get("source_activity_id")})
    return sorted(related, key=lambda item: (int(item["finding"].get("created_revision", 0)), str(item["finding"].get("id", ""))))


def render_overview_human(payload: dict[str, Any]) -> str:
    attempts = payload.get("attempts") or {}
    if "loop_id" not in payload:
        lines = [f"Research overview for {payload['repo']} (revision {payload['revision']})"]
        lines.extend(f"- {loop.get('loop_id')}: {loop.get('objective') or loop.get('title')} [{loop['checkpoint_location']}] ({loop['checkpoint_status']}, goal={loop['goal_status']}; findings={loop.get('finding_count', 'unknown')}, annotations={loop.get('annotation_count', 'unknown')})" for loop in payload.get("loops", []))
        if not payload.get("loops"):
            lines.append("No registered research loops.")
        if payload.get("status") == "partial":
            lines.append("Some checkpoint or Research State sources are incomplete; query partial results and reconcile before claiming a complete handoff.")
            if payload.get("error"):
                lines.append(f"Research State: {payload['error']}")
        return "\n".join(lines)
    lines = [
        f"Research receipt for {payload['loop_id']} ({payload['status']})",
        f"Repository: {payload['repo']}",
        f"Loop status: {payload.get('loop_status') or 'unknown'}; goal: {payload.get('goal_status')}; direction: {payload.get('direction_status')}",
        f"Rounds: {payload.get('rounds_started')}/{payload.get('max_rounds')}",
        f"Stop reason: {payload.get('stop_reason') or 'unknown'}",
        (f"Findings: {payload['counts']['findings']}; annotations: {payload['counts']['annotations']}; agenda: {payload['counts']['agenda_items']}" if isinstance(payload.get('counts'), dict) else "Knowledge counts: unknown"),
        f"Unattributed: {payload.get('unattributed') if payload.get('unattributed') is not None else 'unknown'}",
        f"State: {payload['state_location']}",
        f"Checkpoint: {payload['checkpoint_location']} ({payload['checkpoint_status']})",
        *[f"Summary: {summary}" for summary in payload.get("summaries", [])],
        *[f"Persisted receipt: key={item.get('client_update_id', 'unknown')} commit={item.get('commit_id', 'unknown')} revision={item.get('revision', 'unknown')}" for item in payload.get("persisted_receipts", [])],
        *[f"Report: {item}" for item in payload.get("report_locations", [])],
        *[f"Follow-up activity: {activity.get('id')} {activity.get('title') or ''} findings={len(activity.get('findings', []))} annotations={activity.get('annotation_count', 0)} artifacts={activity.get('artifact_locations', []) + activity.get('report_locations', [])}" for activity in payload.get("follow_up_activities", [])],
        f"Attempts: {payload.get('attempts', {}).get('count', 0) if isinstance(payload.get('attempts'), dict) else 'unknown'}",
        *[f"Attempt axes ({axis}): {counts}" for axis, counts in attempts.get("axes", {}).items()],
        *[f"Checkpoint result statuses (separate from research axes): {counts}" for counts in [attempts.get("checkpoint_statuses", {})] if counts],
        *[f"Requested settings: attempt={item.get('attempt_id')} settings={item.get('settings')}" for item in attempts.get("requested_settings", [])],
        *[f"Actual settings: attempt={item.get('attempt_id')} settings={item.get('settings')}" for item in attempts.get("actual_settings", [])],
        *[f"Resource: attempt={item.get('attempt_key')} kind={item.get('kind')} planned={item.get('planned')} actual={item.get('actual', 'unknown')} unit={item.get('unit')} source={item.get('source', 'unknown')} verification={item.get('verification')} enforcement={item.get('enforcement')}" for item in attempts.get("resources", [])],
        *[f"Failure policy: {item.get('failure_policy')} (Agenda {item.get('agenda_item_id')})" for item in attempts.get("failure_policies", [])],
        *[f"Resource over plan: {item.get('kind')} planned={item.get('planned')} actual={item.get('actual')}" for item in attempts.get("resource_summary", {}).get("over_plan", [])],
        *[f"Resource not verified: {item.get('kind')} ({item.get('verification')}/{item.get('enforcement')})" for item in attempts.get("resource_summary", {}).get("unverified", [])],
        *[f"Duplicate slot: round={item.get('round')} slot={item.get('slot_id')} attempts={','.join(item.get('attempt_ids', []))}" for item in attempts.get("duplicate_slots", [])],
        *[f"Duplicate attempt key: {item.get('attempt_key')} attempts={','.join(item.get('attempt_ids', []))}" for item in attempts.get("duplicate_attempt_keys", [])],
        *[f"Retry: attempt={item.get('attempt_key')} retries={item.get('retry_of')} round={item.get('round')} slot={item.get('slot_id')}" for item in attempts.get("retry_chains", [])],
        *[f"Checkpoint/research difference: {item}" for item in payload.get("outcome_differences", [])],
        *[f"External artifact: {item.get('location')} ({item.get('availability')})" for item in payload.get("external_artifacts", [])],
        *[f"Pending: {item}" for item in payload.get("pending_reconciliation", [])],
    ]
    return "\n".join(lines)


def render_annotations_human(payload: dict[str, Any]) -> str:
    annotations = payload.get("annotations", [])
    if not annotations:
        return "No annotations."
    return "\n".join(f"{item.get('id')}\t{item.get('created_revision')}\t{item.get('author', {}).get('name', 'unknown')}\t{item.get('text')}" for item in annotations)


def render_receipt_human(payload: dict[str, Any]) -> str:
    return f"Receipt {payload.get('status')} for {payload.get('client_update_id')}: revision={payload.get('revision', 'unknown')} commit={payload.get('commit_id', 'unknown')}"


def render_evidence_human(payload: dict[str, Any]) -> str:
    if not payload.get("references"):
        return "Finding has no evidence references."
    return "\n".join(f"{row['reference'].get('kind')}\tavailability={row.get('availability')}\tidentity={row.get('identity_status')}\treview={row.get('scientific_review')}" for row in payload["references"])


def _format_evidence_reference(reference: dict[str, Any]) -> str:
    if reference.get("kind") == "local_artifact":
        return f"{reference.get('location')} (sha256={reference.get('sha256')})"
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
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    lines = [
        f"Research Context revision {payload['revision']}",
        f"Objective: {payload['objective']}",
        f"JSON response budget: {payload['budget_used']}/{payload['budget_limit']} UTF-8 bytes (budget scope: response_json)",
    ]
    for item in payload.get("items", []):
        lines.append(f"- {item['kind']} {item['id']}")
    return "\n".join(lines)
