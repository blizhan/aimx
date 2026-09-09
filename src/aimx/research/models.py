from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

from aimx.research.errors import ValidationError

SchemaVersion = Literal[1]
AuthorKind = Literal["agent", "human", "system"]


class EpistemicStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    CONTRADICTED = "contradicted"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GovernanceStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RelationType(str, Enum):
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CHALLENGES = "challenges"
    REFINES = "refines"
    SUPERSEDES = "supersedes"
    TESTS = "tests"
    RELATED_TO = "related_to"


class FrontierStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    RETIRED = "retired"


class AgendaStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class Author:
    kind: AuthorKind
    name: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Author":
        if not isinstance(value, dict):
            raise ValidationError("author must be an object", code="invalid_schema")
        kind = value.get("kind")
        name = value.get("name")
        if not isinstance(kind, str) or not isinstance(name, str):
            raise ValidationError(
                "author.kind and author.name are required strings",
                code="invalid_schema",
            )
        return cls(kind=kind, name=name)

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name}


@dataclass(frozen=True)
class EvidenceReference:
    kind: str
    run_hash: str
    role: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, str]:
        result: dict[str, str] = {"kind": self.kind, "run_hash": self.run_hash}
        if self.role is not None:
            result["role"] = self.role
        if self.note is not None:
            result["note"] = self.note
        return result


@dataclass(frozen=True)
class Finding:
    id: str
    claim: str
    evidence: tuple[EvidenceReference, ...]
    epistemic_status: str
    confidence: str
    governance_status: str
    created_commit_id: str
    created_revision: int
    author: Author
    history: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Relation:
    id: str
    source_finding_id: str
    type: str
    target_finding_id: str
    active: bool
    reason: str | None
    created_commit_id: str
    created_revision: int
    retracted_commit_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Annotation:
    id: str
    target_kind: str
    target_id: str | None
    text: str
    author: Author
    created_commit_id: str
    created_revision: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierLane:
    id: str
    name: str
    description: str | None
    active: bool
    created_commit_id: str
    created_revision: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontierItem:
    id: str
    lane_id: str
    subject: dict[str, Any]
    rationale: str
    priority: int
    status: str
    supporting_finding_ids: tuple[str, ...]
    challenging_finding_ids: tuple[str, ...]
    created_commit_id: str
    created_revision: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgendaItem:
    id: str
    objective: str
    hypothesis: str | None
    question: str | None
    motivating_finding_ids: tuple[str, ...]
    frontier_item_ids: tuple[str, ...]
    controls: Any
    factors: Any
    constraints: Any
    success_criteria: Any
    priority: int
    status: str
    evidence: tuple[EvidenceReference, ...]
    created_commit_id: str
    created_revision: int
    result_commit_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchUpdate:
    schema_version: SchemaVersion
    base_revision: int
    author: Author
    operations: tuple[dict[str, Any], ...]
    client_update_id: str | None = None
    message: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResearchUpdate":
        if not isinstance(value, dict):
            raise ValidationError("ResearchUpdate must be an object", code="invalid_schema")
        allowed = {
            "schema_version",
            "base_revision",
            "client_update_id",
            "author",
            "message",
            "operations",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValidationError(
                f"ResearchUpdate has unsupported fields: {', '.join(unknown)}",
                code="invalid_schema",
            )
        operations = value.get("operations", [])
        if not isinstance(operations, list):
            raise ValidationError(
                "ResearchUpdate operations must be an array",
                code="invalid_schema",
            )
        client_update_id = value.get("client_update_id")
        if client_update_id is not None and not isinstance(client_update_id, str):
            raise ValidationError(
                "client_update_id must be a string",
                code="invalid_schema",
            )
        message = value.get("message")
        if message is not None and not isinstance(message, str):
            raise ValidationError("message must be a string", code="invalid_schema")
        return cls(
            schema_version=value.get("schema_version", -1),
            base_revision=value.get("base_revision", -1),
            author=Author.from_dict(value.get("author", {})),
            operations=tuple(copy.deepcopy(operations)),
            client_update_id=client_update_id,
            message=message,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "base_revision": self.base_revision,
            "author": self.author.as_dict(),
            "operations": copy.deepcopy(list(self.operations)),
        }
        if self.client_update_id is not None:
            result["client_update_id"] = self.client_update_id
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class ResearchState:
    revision: int = 0
    findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    relations: dict[str, dict[str, Any]] = field(default_factory=dict)
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    lanes: dict[str, dict[str, Any]] = field(default_factory=dict)
    frontier_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    agenda_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    commits: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "ResearchState":
        return copy.deepcopy(self)

    def as_dict(self, *, repo: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "revision": self.revision,
            "findings": _sorted_values(self.findings),
            "relations": _sorted_values(self.relations),
            "annotations": _sorted_values(self.annotations),
            "frontier": {
                "lanes": _sorted_values(self.lanes),
                "items": _sorted_values(self.frontier_items),
            },
            "agenda": {"items": _sorted_values(self.agenda_items)},
        }
        if repo is not None:
            result["repo"] = repo
        return result

    def json_items(self) -> list[dict[str, Any]]:
        """Return compact, deterministic context candidates."""
        items: list[dict[str, Any]] = []
        for kind, values in (
            ("finding", self.findings),
            ("relation", self.relations),
            ("annotation", self.annotations),
            ("frontier_lane", self.lanes),
            ("frontier_item", self.frontier_items),
            ("agenda_item", self.agenda_items),
        ):
            for value in _sorted_values(values):
                items.append({"kind": kind, "id": value.get("id"), "data": value})
        return items


def _sorted_values(values: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        values.values(),
        key=lambda value: (int(value.get("created_revision", 0)), str(value.get("id", ""))),
    )
    return [copy.deepcopy(value) for value in ordered]


def json_bytes(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
