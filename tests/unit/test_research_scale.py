from __future__ import annotations

from aimx.research.context import compile_context
from aimx.research.models import ResearchState, json_bytes
from aimx.research.replay import replay


def test_replay_and_context_handle_research_scale() -> None:
    commits = []
    events = []
    for revision in range(1, 10_001):
        commit_id = f"commit_{revision:032x}"
        commits.append(
            {
                "id": commit_id,
                "revision": revision,
                "base_revision": revision - 1,
                "client_update_id": None,
                "author_kind": "agent",
                "author_name": "scale-test",
                "created_at": "2026-09-09T00:00:00+00:00",
                "message": None,
                "schema_version": 1,
            }
        )
        for sequence in range(5):
            annotation_id = f"ann_{revision:016x}{sequence:016x}"
            events.append(
                {
                    "id": f"event_{revision:016x}{sequence:016x}",
                    "commit_id": commit_id,
                    "revision": revision,
                    "sequence": sequence,
                    "event_type": "annotation.create",
                    "entity_id": annotation_id,
                    "payload": {
                        "id": annotation_id,
                        "target_kind": "research_state",
                        "target_id": None,
                        "text": f"scale note {revision}-{sequence}",
                    },
                }
            )

    replayed = replay(commits, events)
    assert replayed.revision == 10_000
    assert len(replayed.commits) == 10_000
    assert len(replayed.annotations) == 50_000

    findings = {
        f"f_{index:032x}": {
            "id": f"f_{index:032x}",
            "claim": f"accuracy result {index}",
            "evidence": [],
            "epistemic_status": "candidate",
            "confidence": "medium",
            "governance_status": "proposed",
            "created_revision": index + 1,
        }
        for index in range(10_000)
    }
    state = ResearchState(revision=10_000, findings=findings)
    context = compile_context(state, "accuracy results", 5_000_000)
    assert context["budget_used"] == json_bytes(context["items"])
    assert context["budget_used"] <= context["budget_limit"]
    assert len([item for item in context["items"] if item["kind"] == "finding"]) == 10_000
