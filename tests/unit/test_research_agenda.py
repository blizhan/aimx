from __future__ import annotations

import pytest

from aimx.research.agenda import choose_next, ordered_agenda
from aimx.research.errors import ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.store import apply_update, read_state


def test_agenda_experiment_contract_lifecycle_and_result_provenance(
    research_repo, research_update_factory
) -> None:
    first = apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                operations=[
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy claim"},
                    {
                        "op": "agenda.item.create",
                        "local_id": "agenda1",
                        "objective": "Test a second seed.",
                        "hypothesis": "Accuracy survives seed changes.",
                        "motivating_finding_ids": ["$f1"],
                        "controls": {"dataset": "fixed"},
                        "factors": {"seed": [42, 43]},
                        "constraints": ["same training budget"],
                        "success_criteria": {"metric": "val/accuracy", "comparison": ">"},
                        "priority": 80,
                        "agenda_status": "proposed",
                    },
                ]
            )
        ),
    )
    agenda_id = first.created_ids["agenda1"]

    second = apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                base_revision=1,
                operations=[
                    {
                        "op": "agenda.item.transition",
                        "agenda_item_id": agenda_id,
                        "agenda_status": "active",
                    }
                ],
            )
        ),
    )
    third = apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                base_revision=2,
                operations=[
                    {
                        "op": "agenda.item.transition",
                        "agenda_item_id": agenda_id,
                        "agenda_status": "completed",
                        "evidence": [{"kind": "aim_run", "run_hash": "a" * 32}],
                    }
                ],
            )
        ),
    )

    item = read_state(research_repo).agenda_items[agenda_id]
    assert item["status"] == "completed"
    assert item["evidence"][0]["run_hash"] == "a" * 32
    assert item["result_commit_ids"] == [third.commit_id]
    assert item["history"][-1]["event"] == "transitioned"
    assert second.revision == 2


def test_agenda_requires_motivation_and_valid_transitions(research_repo, research_update_factory) -> None:
    with pytest.raises(ValidationError, match="motivation"):
        apply_update(
            research_repo,
            ResearchUpdate.from_dict(
                research_update_factory(
                    operations=[
                        {
                            "op": "agenda.item.create",
                            "objective": "No motivation",
                        }
                    ]
                )
            ),
        )

    state = ResearchState(
        agenda_items={"agenda_1": {"id": "agenda_1", "status": "completed"}}
    )
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 0,
            "author": {"kind": "agent", "name": "agent"},
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
        from aimx.research.validation import compile_update

        compile_update(update, state, revision=1, commit_id="commit_test")
    assert error.value.code == "invalid_transition"


def test_next_prefers_active_then_priority_and_is_stable() -> None:
    state = ResearchState(
        agenda_items={
            "agenda_proposed": {
                "id": "agenda_proposed",
                "status": "proposed",
                "priority": 100,
                "created_revision": 1,
            },
            "agenda_active_late": {
                "id": "agenda_active_late",
                "status": "active",
                "priority": 20,
                "created_revision": 3,
            },
            "agenda_active_early": {
                "id": "agenda_active_early",
                "status": "active",
                "priority": 20,
                "created_revision": 2,
            },
        }
    )
    assert choose_next(state)["id"] == "agenda_active_early"
    assert ordered_agenda(state)[0]["id"] == "agenda_active_early"
    assert ordered_agenda(state, status="proposed")[0]["id"] == "agenda_proposed"


def test_next_returns_no_work_for_completed_or_abandoned_items() -> None:
    state = ResearchState(
        agenda_items={
            "agenda_done": {"id": "agenda_done", "status": "completed", "priority": 100},
            "agenda_abandoned": {"id": "agenda_abandoned", "status": "abandoned", "priority": 90},
        }
    )
    assert choose_next(state) is None

    with pytest.raises(ValidationError, match="Unsupported agenda status"):
        ordered_agenda(state, status="invented")
