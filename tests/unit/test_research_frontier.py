from __future__ import annotations

import pytest

from aimx.research.errors import ValidationError
from aimx.research.models import ResearchUpdate
from aimx.research.store import apply_update, read_state


def test_frontier_lanes_items_and_history_are_replayable(
    research_repo, research_update_factory
) -> None:
    first = apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                operations=[
                    {"op": "finding.create", "local_id": "f1", "claim": "accuracy claim"},
                    {"op": "frontier.lane.create", "local_id": "lane1", "name": "promising"},
                    {
                        "op": "frontier.item.create",
                        "local_id": "front1",
                        "lane_id": "$lane1",
                        "finding_id": "$f1",
                        "rationale": "Test the gain under another seed.",
                        "priority": 80,
                        "frontier_status": "active",
                        "supporting_finding_ids": ["$f1"],
                        "challenging_finding_ids": [],
                    },
                ]
            )
        ),
    )
    frontier_id = first.created_ids["front1"]
    lane_id = first.created_ids["lane1"]

    second = apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                base_revision=1,
                operations=[
                    {
                        "op": "frontier.lane.rename",
                        "lane_id": lane_id,
                        "name": "validate",
                    },
                    {
                        "op": "frontier.item.set_priority",
                        "frontier_item_id": frontier_id,
                        "priority": 95,
                    },
                    {
                        "op": "frontier.item.set_status",
                        "frontier_item_id": frontier_id,
                        "frontier_status": "paused",
                        "reason": "Wait for the controlled baseline.",
                    },
                ],
            )
        ),
    )

    state = read_state(research_repo)
    assert second.revision == 2
    assert state.lanes[lane_id]["name"] == "validate"
    assert state.frontier_items[frontier_id]["priority"] == 95
    assert state.frontier_items[frontier_id]["status"] == "paused"
    assert len(state.frontier_items[frontier_id]["history"]) == 3
    assert state.frontier_items[frontier_id]["created_commit_id"] == first.commit_id


def test_frontier_lane_names_are_unique_and_retired_lanes_cannot_receive_items(
    research_repo, research_update_factory
) -> None:
    apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                operations=[
                    {"op": "frontier.lane.create", "local_id": "lane", "name": "one"},
                ]
            )
        ),
    )
    state = read_state(research_repo)
    lane_id = next(iter(state.lanes))

    with pytest.raises(ValidationError, match="already exists"):
        apply_update(
            research_repo,
            ResearchUpdate.from_dict(
                research_update_factory(
                    base_revision=1,
                    operations=[
                        {"op": "frontier.lane.create", "name": "one"},
                    ],
                )
            ),
        )

    apply_update(
        research_repo,
        ResearchUpdate.from_dict(
            research_update_factory(
                base_revision=1,
                operations=[{"op": "frontier.lane.retire", "lane_id": lane_id}],
            )
        ),
    )
    with pytest.raises(ValidationError, match="active lane"):
        apply_update(
            research_repo,
            ResearchUpdate.from_dict(
                research_update_factory(
                    base_revision=2,
                    operations=[
                        {
                            "op": "frontier.item.create",
                            "lane_id": lane_id,
                            "direction": "try another seed",
                            "rationale": "The lane is no longer active.",
                        }
                    ],
                )
            ),
        )


@pytest.mark.parametrize("priority", [-1, 101])
def test_frontier_priority_range_is_enforced(priority: int, research_repo, research_update_factory) -> None:
    with pytest.raises(ValidationError, match="priority"):
        apply_update(
            research_repo,
            ResearchUpdate.from_dict(
                research_update_factory(
                    operations=[
                        {"op": "frontier.lane.create", "local_id": "lane", "name": "lane"},
                        {
                            "op": "frontier.item.create",
                            "lane_id": "$lane",
                            "direction": "test direction",
                            "rationale": "A rationale.",
                            "priority": priority,
                        },
                    ]
                )
            ),
        )
