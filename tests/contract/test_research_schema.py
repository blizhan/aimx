from __future__ import annotations

import json
from pathlib import Path

from aimx.research.events import SUPPORTED_OPERATIONS


def test_research_update_schema_matches_supported_operation_catalog() -> None:
    path = Path("specs/007-research-state/contracts/research-update.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    values = set(schema["$defs"]["operation"]["properties"]["op"]["enum"])
    assert values == set(SUPPORTED_OPERATIONS)
    assert schema["properties"]["schema_version"]["const"] == 1
    assert schema["properties"]["operations"]["minItems"] == 1
