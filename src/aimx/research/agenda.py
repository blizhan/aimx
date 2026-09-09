from __future__ import annotations

from typing import Any

from aimx.research.errors import ValidationError
from aimx.research.models import ResearchState

AGENDA_STATUSES = frozenset({"proposed", "active", "completed", "failed", "abandoned"})


def ordered_agenda(
    state: ResearchState, *, status: str | None = None
) -> list[dict[str, Any]]:
    if status is not None and status not in AGENDA_STATUSES:
        raise ValidationError(f"Unsupported agenda status: {status}")
    items = list(state.agenda_items.values())
    if status is not None:
        items = [item for item in items if item.get("status") == status]
    return sorted(
        items,
        key=lambda item: (
            _status_rank(item.get("status")),
            -int(item.get("priority", 0)),
            int(item.get("created_revision", 0)),
            str(item.get("id", "")),
        ),
    )


def choose_next(state: ResearchState) -> dict[str, Any] | None:
    active = [item for item in state.agenda_items.values() if item.get("status") == "active"]
    if active:
        return _best(active)
    proposed = [item for item in state.agenda_items.values() if item.get("status") == "proposed"]
    if proposed:
        return _best(proposed)
    return None


def _best(items: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        items,
        key=lambda item: (
            -int(item.get("priority", 0)),
            int(item.get("created_revision", 0)),
            str(item.get("id", "")),
        ),
    )[0]


def _status_rank(status: str | None) -> int:
    return {"active": 0, "proposed": 1, "failed": 2, "abandoned": 3, "completed": 4}.get(
        status or "", 5
    )
