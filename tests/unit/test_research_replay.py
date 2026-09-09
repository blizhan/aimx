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
