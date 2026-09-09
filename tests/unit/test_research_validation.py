from __future__ import annotations

import pytest

from aimx.research.errors import ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.validation import compile_update


def _compile(payload: dict):
    return compile_update(
        ResearchUpdate.from_dict(payload),
        ResearchState(),
        revision=1,
        commit_id="commit_test",
    )


def test_local_references_resolve_in_order_and_created_ids_are_returned() -> None:
    compiled = _compile(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent-a"},
            "operations": [
                {"op": "finding.create", "local_id": "f1", "claim": "A claim."},
                {"op": "finding.create", "local_id": "f2", "claim": "Another claim."},
                {
                    "op": "relation.create",
                    "local_id": "r1",
                    "source": "$f1",
                    "target": "$f2",
                    "relation_type": "supports",
                },
            ],
        }
    )
    assert compiled.created_ids["f1"].startswith("f_")


def test_self_relation_is_rejected_before_commit() -> None:
    with pytest.raises(ValidationError, match="cannot connect"):
        _compile(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent-a"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "A claim."},
                    {
                        "op": "relation.create",
                        "source": "$f1",
                        "target": "$f1",
                        "relation_type": "supports",
                    },
                ],
            }
        )


def test_duplicate_local_ids_and_unknown_references_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate local_id"):
        _compile(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent-a"},
                "operations": [
                    {"op": "finding.create", "local_id": "f1", "claim": "one"},
                    {"op": "finding.create", "local_id": "f1", "claim": "two"},
                ],
            }
        )

    with pytest.raises(ValidationError, match="Unknown same-update reference"):
        _compile(
            {
                "schema_version": 1,
                "base_revision": 0,
                "author": {"kind": "agent", "name": "agent-a"},
                "operations": [
                    {"op": "relation.create", "source": "$later", "target": "$other", "relation_type": "supports"},
                ],
            }
        )


def test_claim_meaning_cannot_be_overwritten() -> None:
    state = ResearchState(findings={"f_existing": {"id": "f_existing", "claim": "old"}})
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "human", "name": "reviewer"},
            "operations": [
                {
                    "op": "finding.update",
                    "finding_id": "f_existing",
                    "claim": "new",
                }
            ],
        }
    )
    with pytest.raises(ValidationError, match="Unsupported research operation"):
        compile_update(update, state, revision=1, commit_id="commit_test")


def test_agenda_transition_rules_are_enforced() -> None:
    state = ResearchState(
        findings={"f1": {"id": "f1"}},
        agenda_items={"agenda_1": {"id": "agenda_1", "status": "completed"}},
    )
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent-a"},
            "operations": [
                {
                    "op": "agenda.item.transition",
                    "agenda_item_id": "agenda_1",
                    "agenda_status": "active",
                }
            ],
        }
    )
    with pytest.raises(ValidationError) as error:
        compile_update(update, state, revision=1, commit_id="commit_test")
    assert error.value.code == "invalid_transition"
