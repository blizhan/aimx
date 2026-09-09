from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchError(Exception):
    """Expected, user-facing failure from the research control plane."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    exit_status: int = 2

    def __post_init__(self) -> None:
        super().__init__(self.message)


class RevisionConflict(ResearchError):
    def __init__(self, base_revision: int, current_revision: int) -> None:
        super().__init__(
            code="revision_conflict",
            message=(
                f"Research state advanced from revision {base_revision} "
                f"to {current_revision}. Refresh context before retrying."
            ),
            details={
                "base_revision": base_revision,
                "current_revision": current_revision,
            },
            exit_status=3,
        )


class ValidationError(ResearchError):
    def __init__(self, message: str, *, code: str = "invalid_input", **details: Any) -> None:
        super().__init__(code=code, message=message, details=details, exit_status=2)


class StoreError(ResearchError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(
            code="research_store_unreadable",
            message=message,
            details=details,
            exit_status=2,
        )


class BudgetError(ResearchError):
    def __init__(self, required: int, limit: int) -> None:
        super().__init__(
            code="budget_too_small_for_required_context",
            message=(
                f"Context budget {limit} bytes is too small for required "
                f"context closure ({required} bytes minimum)."
            ),
            details={"required_bytes": required, "budget_limit": limit},
            exit_status=2,
        )
