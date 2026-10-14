import hashlib
import json
import shlex

import pytest

from aimx.commands.research import run_research_command
from aimx.research.context import _evidence_refs, _query_commands
from aimx.research.errors import ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.overview import compile_overview
from aimx.research.validation import compile_update
from aimx.rendering.research_views import render_finding, render_findings, render_overview_human


def fingerprint(ref):
    return hashlib.sha256(json.dumps(ref, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("field,value", [("round", []), ("round", True), ("round", 0), ("slot_id", {}), ("native_task_id", []), ("requested_settings", []), ("actual_settings", None)])
def test_invalid_attempt_metadata_is_rejected(field, value):
    state = ResearchState(activities={"act": {"id": "act", "kind": "analysis"}}, agenda_items={"agenda": {"id": "agenda", "status": "active"}})
    operation = {"op": "execution.outcome.record", "source_activity_id": "act", "agenda_item_id": "agenda", "attempt_key": "one", "execution_status": "failed", "artifact_status": "none", "evaluation_status": "unknown", "audit_status": "not_run", "summary": "failed", "evidence": [], "resources": [], field: value}
    update = ResearchUpdate.from_dict({"schema_version": 1, "base_revision": 0, "author": {"kind": "agent", "name": "test"}, "operations": [operation]})
    with pytest.raises(ValidationError, match=field):
        compile_update(update, state, revision=1, commit_id="test")
    assert not state.outcomes


@pytest.mark.parametrize("availability", ["available", "unavailable"])
def test_context_preserves_review_of_decorated_aim_evidence(availability):
    ref = {"kind": "aim_run", "run_hash": "a" * 32, "role": "support"}
    digest = fingerprint(ref)
    state = ResearchState(evidence_reviews={"f": [{"evidence_fingerprint": digest, "scientific_review": "rejected"}]})
    rows = _evidence_refs([{"id": "f", "data": {"evidence": [{**ref, "availability": availability}]}}], state, ".")
    assert rows[0]["evidence_fingerprint"] == digest
    assert rows[0]["scientific_review"] == "rejected"


@pytest.mark.parametrize("new_finding", [False, True])
def test_reviews_accept_each_reference_to_same_run(new_finding):
    refs = [{"kind": "aim_run", "run_hash": "a" * 32, "role": role} for role in ("support", "comparison")]
    operations = [{"op": "finding.create", "local_id": "f", "claim": "claim", "evidence": refs}] if new_finding else []
    operations += [{"op": "evidence.review", "finding_id": "$f" if new_finding else "f", "evidence_fingerprint": fingerprint(ref), "scientific_review": "rejected", "reason": "insufficient", "basis": [refs[0]]} for ref in refs]
    state = ResearchState() if new_finding else ResearchState(findings={"f": {"id": "f", "evidence": refs}})
    update = ResearchUpdate.from_dict({"schema_version": 1, "base_revision": 0, "author": {"kind": "human", "name": "test"}, "operations": operations})
    compiled = compile_update(update, state, revision=1, commit_id="test")
    assert sum(event["event_type"] == "evidence.review" for event in compiled.events) == 2


def test_partial_overview_renders_without_attempts(tmp_path):
    payload = compile_overview(tmp_path, None, "loop", state_error="unreadable")
    assert payload["attempts"] is None
    output = render_overview_human(payload)
    assert "Attempts: unknown" in output
    assert "partial" in output


def test_follow_up_only_findings_and_relations_are_rendered():
    primary = {"id": "primary", "claim": "original"}
    follow = {"id": "follow", "claim": "matching follow up"}
    state = ResearchState(findings={"primary": primary, "follow": follow}, relations={"r": {"id": "r", "source_finding_id": "follow", "target_finding_id": "primary", "type": "supports"}})
    output = render_findings(state, output_json=False, repo=".", findings=[], follow_up_findings=[follow])
    assert "matching follow up" in output
    assert "related supports primary" in output


def test_recovery_command_uses_root_finding_command():
    assert shlex.split(_query_commands("repo with spaces", "loop")[0]) == ["aimx", "finding", "ls", "--repo", "repo with spaces", "--loop", "loop", "--json"]


def test_finding_details_show_artifact_location_and_identity():
    state = ResearchState(findings={"f": {"id": "f", "claim": "claim", "evidence": [{"kind": "local_artifact", "location": "result.json", "sha256": "a" * 64}]}})
    output = render_finding(state, "f", output_json=False, repo=".")
    assert "result.json" in output
    assert "a" * 64 in output


@pytest.mark.parametrize("args", [["context", "--objective", "accuracy", "--budget", "12000", "--help"], ["evidence", "check", "--finding", "f123", "--help"]])
def test_help_ignores_populated_options(args):
    result = run_research_command(args)
    assert result.exit_status == 0
    assert result.output
