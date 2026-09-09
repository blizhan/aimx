from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aimx.research.errors import ValidationError

if TYPE_CHECKING:
    from aimx.research.models import ResearchState

try:
    from aim import Repo
except ModuleNotFoundError:  # pragma: no cover - exercised by existing Aim-missing tests
    Repo = None  # type: ignore[assignment,misc]


def canonicalize_run_hash(raw_hash: str, repo_path: Path) -> str:
    if not isinstance(raw_hash, str) or not raw_hash.strip():
        raise ValidationError("Aim run hash must not be empty", code="unresolved_evidence")
    if Repo is None:
        raise ValidationError(
            "`aimx` requires the Python `aim` package to validate Aim evidence.",
            code="unresolved_evidence",
        )
    root = _repo_root(repo_path)
    try:
        hashes = _list_run_hashes(root)
    except Exception as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(
            f"Unable to resolve Aim evidence in {root}: {exc}",
            code="unresolved_evidence",
        ) from exc
    value = raw_hash.strip().lower()
    matches = [item for item in hashes if item == value or item.startswith(value)]
    if not matches:
        raise ValidationError(
            f"Aim run '{raw_hash}' did not match a run in the repository.",
            code="unresolved_evidence",
        )
    if len(matches) > 1:
        raise ValidationError(
            f"Aim run prefix '{raw_hash}' is ambiguous ({len(matches)} matches).",
            code="ambiguous_evidence",
            matches=matches[:5],
        )
    return matches[0]


def make_evidence_resolver(repo_path: Path):
    return lambda raw_hash: canonicalize_run_hash(raw_hash, repo_path)


def mark_evidence_availability(state: "ResearchState", repo_path: Path) -> "ResearchState":
    """Decorate persisted Aim references with current read-time availability."""

    references: list[dict] = []
    for finding in state.findings.values():
        references.extend(finding.get("evidence", []))
    for item in state.agenda_items.values():
        references.extend(item.get("evidence", []))
    aim_references = [ref for ref in references if ref.get("kind") == "aim_run"]
    if not aim_references:
        return state

    try:
        available_hashes = set(_list_run_hashes(_repo_root(repo_path)))
    except Exception:
        available_hashes = set()

    for reference in aim_references:
        run_hash = str(reference.get("run_hash", "")).lower()
        reference["availability"] = (
            "available" if run_hash and run_hash in available_hashes else "unavailable"
        )
    return state


def _repo_root(repo_path: Path) -> Path:
    return repo_path.parent if repo_path.name == ".aim" else repo_path


def _list_run_hashes(root: Path) -> list[str]:
    repository = _open_repo_read_only(root)
    return [str(value).lower() for value in repository.list_all_runs()]


def _open_repo_read_only(root: Path):
    if Repo is None:
        raise ValidationError(
            "`aimx` requires the Python `aim` package to inspect Aim evidence.",
            code="unresolved_evidence",
        )
    try:
        return Repo(str(root), read_only=True)
    except (NotImplementedError, TypeError) as exc:
        check_status = getattr(Repo, "check_repo_status", None)
        if check_status is None:
            raise ValidationError(
                "Installed Aim cannot guarantee read-only repository access.",
                code="unresolved_evidence",
            ) from exc
        status = check_status(str(root))
        if getattr(status, "name", None) != "UPDATED":
            raise ValidationError(
                f"Aim repository at {root} requires migration or patching before it can be inspected safely.",
                code="unresolved_evidence",
            ) from exc
        return Repo(str(root))
