from __future__ import annotations

from pathlib import Path

from aimx.research.models import ResearchUpdate
from aimx.research.store import apply_update, read_state


def test_replay_preserves_assessment_governance_lineage_and_history(
    research_repo: Path, research_update_factory
) -> None:
    first = apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                operations=[
                    {
                        "op": "finding.create",
                        "local_id": "f1",
                        "claim": "The candidate improves accuracy.",
                        "epistemic_status": "candidate",
                        "confidence": "low",
                        "governance_status": "proposed",
                    },
                    {
                        "op": "finding.create",
                        "local_id": "f2",
                        "claim": "The gain disappears on a second seed.",
                    },
                    {
                        "op": "relation.create",
                        "local_id": "rel",
                        "source": "$f2",
                        "target": "$f1",
                        "relation_type": "challenges",
                        "reason": "Counterexample run.",
                    },
                    {
                        "op": "annotation.create",
                        "local_id": "ann",
                        "annotation_target_kind": "finding",
                        "annotation_target_id": "$f1",
                        "text": "Verify the seed and split before accepting.",
                    },
                ],
            )
        ),
    )
    finding_id = first.created_ids["f1"]
    relation_id = first.created_ids["rel"]

    apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                base_revision=1,
                operations=[
                    {
                        "op": "finding.assess",
                        "finding_id": finding_id,
                        "epistemic_status": "validated",
                        "confidence": "medium",
                        "reason": "The original comparison is reproducible.",
                    },
                    {
                        "op": "finding.governance",
                        "finding_id": finding_id,
                        "governance_status": "accepted",
                        "reason": "Useful working result.",
                    },
                    {
                        "op": "relation.retract",
                        "relation_id": relation_id,
                        "reason": "Use a more specific counterexample relation.",
                    },
                ],
            )
        ),
    )

    state = read_state(research_repo)
    finding = state.findings[finding_id]
    assert finding["claim"] == "The candidate improves accuracy."
    assert finding["epistemic_status"] == "validated"
    assert finding["governance_status"] == "accepted"
    assert len(finding["history"]) == 3
    assert state.relations[relation_id]["active"] is False
    assert state.relations[relation_id]["retracted_commit_id"] is not None
    assert any(annotation["text"].startswith("Verify") for annotation in state.annotations.values())


def test_new_activity_scope_and_attempt_events_replay_without_checkpoint_migration(
    research_repo: Path, research_update_factory
) -> None:
    update = ResearchUpdate.from_dict(research_update_factory(operations=[
        {"op": "activity.create", "local_id": "act", "kind": "analysis", "title": "Audit", "objective": "Check result", "source": {"kind": "test", "location": "fixture"}},
        {"op": "finding.create", "local_id": "f", "claim": "Candidate result"},
        {"op": "agenda.item.create", "local_id": "agenda", "objective": "score candidate", "motivating_finding_ids": ["$f"]},
        {"op": "dataset.scope.create", "local_id": "scope", "project_key": "project", "dataset_key": "holdout", "data_revision": "r1", "description": "fixed scope", "overlap_keys": ["candidate"], "coverage": "declared_complete", "initial_observation": "declared_unobserved", "source_activity_id": "$act", "source": {"kind": "test", "location": "fixture"}},
        {"op": "dataset.observation.record", "local_id": "freeze", "dataset_scope_id": "$scope", "event_kind": "candidate_frozen", "occurred_at": "2026-10-13T12:00:00Z", "source_activity_id": "$act", "source": {"kind": "test", "location": "fixture"}, "evidence": [{"kind": "aim_run", "run_hash": "a" * 32}], "candidate_fingerprint": "candidate-sha256"},
        {"op": "execution.outcome.record", "local_id": "attempt", "source_activity_id": "$act", "agenda_item_id": "$agenda", "attempt_key": "attempt-1", "execution_status": "failed", "artifact_status": "complete", "evaluation_status": "passed", "audit_status": "not_run", "summary": "Executor failed after saving complete predictions", "evidence": [{"kind": "aim_run", "run_hash": "a" * 32}], "resources": [{"kind": "calls", "unit": "count", "planned": 1, "actual": 2, "verification": "unverified", "enforcement": "declared_only"}], "round": 1, "slot_id": "slot-a", "requested_settings": {"model": "requested"}},
        {"op": "dataset.observation.record", "local_id": "score", "dataset_scope_id": "$scope", "event_kind": "first_scored", "occurred_at": "2026-10-13T12:01:00Z", "source_activity_id": "$act", "source": {"kind": "test", "location": "fixture"}, "evidence": [{"kind": "aim_run", "run_hash": "a" * 32}], "candidate_fingerprint": "candidate-sha256", "attempt_id": "$attempt"},
    ]))
    result = apply_update(research_repo, update)
    state = read_state(research_repo)
    assert result.revision == 1
    assert state.activities[result.created_ids["act"]]["kind"] == "analysis"
    assert state.dataset_scopes[result.created_ids["scope"]]["coverage"] == "declared_complete"
    assert [row["event_kind"] for row in state.observations.values()] == ["candidate_frozen", "first_scored"]
    outcome = state.outcomes[result.created_ids["attempt"]]
    assert outcome["execution_status"] == "failed"
    assert outcome["artifact_status"] == "complete"
    assert outcome["evaluation_status"] == "passed"


def test_evidence_review_replays_as_append_only_research_history(
    research_repo: Path,
) -> None:
    import hashlib
    import json
    import pytest
    from aimx.research.errors import ValidationError

    ref = {"kind": "aim_run", "run_hash": "a" * 32}
    fingerprint = hashlib.sha256(json.dumps(ref, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    update = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 0, "author": {"kind": "human", "name": "reviewer"},
        "operations": [
            {"op": "finding.create", "local_id": "finding", "claim": "Candidate result", "evidence": [ref]},
            {"op": "evidence.review", "finding_id": "$finding", "evidence_fingerprint": fingerprint, "scientific_review": "inconclusive", "reason": "The report does not establish the threshold", "basis": [ref]},
        ],
    })
    result = apply_update(research_repo, update)
    state = read_state(research_repo)
    finding_id = result.created_ids["finding"]
    review = state.evidence_reviews[finding_id][0]
    assert review["scientific_review"] == "inconclusive"
    assert review["reason"] == "The report does not establish the threshold"
    changed_fingerprint = "0" * 64
    invalid = ResearchUpdate.from_dict({
        "schema_version": 1, "base_revision": 1, "author": {"kind": "human", "name": "reviewer"},
        "operations": [{"op": "evidence.review", "finding_id": finding_id, "evidence_fingerprint": changed_fingerprint, "scientific_review": "accepted", "reason": "stale evidence", "basis": [ref]}],
    })
    with pytest.raises(ValidationError) as mismatch:
        apply_update(research_repo, invalid)
    assert mismatch.value.code == "evidence_identity_mismatch"
    assert read_state(research_repo).revision == 1


def test_state_projection_order_is_creation_revision_then_id(
    research_repo: Path, research_update_factory
) -> None:
    apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                operations=[
                    {"op": "finding.create", "local_id": "first", "claim": "first"},
                    {"op": "finding.create", "local_id": "second", "claim": "second"},
                ]
            )
        ),
    )
    payload = read_state(research_repo).as_dict()
    assert [item["id"] for item in payload["findings"]] == sorted(
        item["id"] for item in payload["findings"]
    )
