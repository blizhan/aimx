from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

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


@dataclass(frozen=True)
class EvidenceResolver:
    repo_path: Path

    def __call__(self, raw_hash: str) -> str:
        return canonicalize_run_hash(raw_hash, self.repo_path)

    def resolve_artifact(self, location: str) -> tuple[Path, str, int]:
        original = Path(location).expanduser()
        is_relative = not original.is_absolute()
        candidate = Path(os.path.abspath(original if not is_relative else self.repo_path / original))
        if is_relative and not candidate.is_relative_to(self.repo_path):
            raise ValidationError("Relative artifact paths must stay inside the repository", code="artifact_unsupported")
        try:
            parts = list(reversed(candidate.parents)) + [candidate]
            for component in parts:
                if stat.S_ISLNK(component.lstat().st_mode):
                    raise ValidationError("Symlink artifacts cannot be identity-verified safely", code="artifact_changed_during_read")
            before = candidate.stat()
        except ValidationError:
            raise
        except OSError as exc:
            raise ValidationError(f"Unable to read local artifact at {candidate}: {exc}", code="artifact_unreadable") from exc
        resolved = candidate.resolve(strict=True)
        if not stat.S_ISREG(before.st_mode):
            raise ValidationError("Local artifact must be a regular file", code="artifact_unsupported")
        digest = hashlib.sha256()
        size = 0
        try:
            stream_context = candidate.open("rb")
        except OSError as exc:
            raise ValidationError(f"Unable to read local artifact at {candidate}: {exc}", code="artifact_unreadable") from exc
        with stream_context as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                raise ValidationError("Local artifact changed before reading", code="artifact_changed_during_read")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            after_open = os.fstat(stream.fileno())
        try:
            after = candidate.stat()
        except OSError as exc:
            raise ValidationError(f"Local artifact disappeared while reading: {candidate}", code="artifact_changed_during_read") from exc
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity_before != (after_open.st_dev, after_open.st_ino, after_open.st_size, after_open.st_mtime_ns) or identity_before != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != before.st_size:
            raise ValidationError("Local artifact changed while reading", code="artifact_changed_during_read")
        return resolved, digest.hexdigest(), size


def make_evidence_resolver(repo_path: Path) -> EvidenceResolver:
    return EvidenceResolver(repo_path.resolve())


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
