from __future__ import annotations

from aimx.research.events import deserialize_update, serialize_update
from aimx.research.models import ResearchUpdate


def test_research_update_serialization_is_canonical_and_round_trips() -> None:
    update = ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": 4,
            "author": {"kind": "agent", "name": "agent"},
            "operations": [
                {"op": "finding.create", "claim": "中文 accuracy claim"},
            ],
        }
    )
    encoded = serialize_update(update)
    assert encoded == serialize_update(deserialize_update(encoded))
    assert "中文" in encoded
