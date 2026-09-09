from __future__ import annotations

import json
from typing import Any

from aimx.research.errors import ValidationError
from aimx.research.models import ResearchUpdate

SUPPORTED_OPERATIONS = frozenset(
    {
        "finding.create",
        "finding.assess",
        "finding.governance",
        "relation.create",
        "relation.retract",
        "annotation.create",
        "frontier.lane.create",
        "frontier.lane.rename",
        "frontier.lane.retire",
        "frontier.item.create",
        "frontier.item.move",
        "frontier.item.set_priority",
        "frontier.item.set_status",
        "frontier.item.update",
        "agenda.item.create",
        "agenda.item.transition",
        "agenda.item.attach_evidence",
        "agenda.item.set_priority",
    }
)


def event_type_for_operation(operation: str) -> str:
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported research operation: {operation}")
    return operation


def serialize_update(update: ResearchUpdate) -> str:
    """Serialize a ResearchUpdate with stable, compact JSON ordering."""

    return json.dumps(
        update.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deserialize_update(value: str | bytes | bytearray | dict[str, Any]) -> ResearchUpdate:
    """Deserialize JSON or an already decoded object into a ResearchUpdate."""

    if isinstance(value, (str, bytes, bytearray)):
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationError("ResearchUpdate JSON is invalid", code="invalid_schema") from exc
    else:
        decoded = value
    if not isinstance(decoded, dict):
        raise ValidationError("ResearchUpdate JSON must be an object", code="invalid_schema")
    return ResearchUpdate.from_dict(decoded)
