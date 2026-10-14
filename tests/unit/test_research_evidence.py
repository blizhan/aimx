from __future__ import annotations

from pathlib import Path

import pytest

import aimx.aim_bridge.research_evidence as evidence
from aimx.research.errors import ValidationError


class _FakeRepo:
    calls: list[tuple[str, bool | None]] = []
    hashes = ["a" * 32, "abcdef" + "1" * 26, "abcdef" + "2" * 26]

    def __init__(self, path: str, *, read_only: bool | None = None) -> None:
        self.calls.append((path, read_only))

    def list_all_runs(self) -> list[str]:
        return self.hashes


class _PatchRequiredRepo:
    calls: list[tuple[str, bool | None]] = []

    def __init__(self, path: str, *, read_only: bool | None = None) -> None:
        self.calls.append((path, read_only))
        if read_only is True:
            raise NotImplementedError
        raise AssertionError("unsafe writable Repo fallback was attempted")

    @classmethod
    def check_repo_status(cls, path: str):
        class _Status:
            name = "PATCH_REQUIRED"

        return _Status()


def test_full_and_unambiguous_short_hashes_are_canonicalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _FakeRepo.calls = []
    monkeypatch.setattr(evidence, "Repo", _FakeRepo)

    assert evidence.canonicalize_run_hash("A" * 32, tmp_path) == "a" * 32
    assert evidence.canonicalize_run_hash("abcdef1", tmp_path) == "abcdef" + "1" * 26
    assert all(read_only is True for _, read_only in _FakeRepo.calls)


def test_unknown_and_ambiguous_hashes_have_stable_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evidence, "Repo", _FakeRepo)

    with pytest.raises(ValidationError) as unknown:
        evidence.canonicalize_run_hash("deadbeef", tmp_path)
    assert unknown.value.code == "unresolved_evidence"

    with pytest.raises(ValidationError) as ambiguous:
        evidence.canonicalize_run_hash("abcdef", tmp_path)
    assert ambiguous.value.code == "ambiguous_evidence"


def test_local_artifact_identity_is_computed_without_opening_aim(tmp_path: Path) -> None:
    import hashlib

    artifact = tmp_path / "report.json"
    artifact.write_bytes(b'{"metric": 0.1}\\n')
    resolver = evidence.make_evidence_resolver(tmp_path)

    resolved, digest, size = resolver.resolve_artifact("report.json")

    assert resolved == artifact.resolve()
    assert digest == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert size == artifact.stat().st_size


def test_local_artifact_rejects_symlinks_and_detects_changed_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "report.json"
    artifact.write_text("report", encoding="utf-8")
    link = tmp_path / "report-link"
    link.symlink_to(artifact)
    resolver = evidence.make_evidence_resolver(tmp_path)

    with pytest.raises(ValidationError) as changed:
        resolver.resolve_artifact("report-link")
    assert changed.value.code == "artifact_changed_during_read"

    with pytest.raises(ValidationError) as missing:
        resolver.resolve_artifact("missing.json")
    assert missing.value.code == "artifact_unreadable"


def test_local_artifact_rejects_special_files(tmp_path: Path) -> None:
    import os
    import pytest

    fifo = tmp_path / "pipe"
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    os.mkfifo(fifo)
    with pytest.raises(ValidationError) as error:
        evidence.EvidenceResolver(tmp_path).resolve_artifact("pipe")
    assert error.value.code == "artifact_unsupported"


def test_local_artifact_reports_read_permission_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = tmp_path / "report.json"
    artifact.write_text("report", encoding="utf-8")
    resolver = evidence.EvidenceResolver(tmp_path)
    original_open = Path.open

    def fail_read(path: Path, *args, **kwargs):
        if path == artifact:
            raise PermissionError("injected permission denial")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_read)
    with pytest.raises(ValidationError) as error:
        resolver.resolve_artifact("report.json")
    assert error.value.code == "artifact_unreadable"
    assert "permission denial" in error.value.message


def test_local_artifact_rejects_symlink_component_inside_repository(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "report.json").write_text("report", encoding="utf-8")
    (tmp_path / "linked-dir").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValidationError) as error:
        evidence.EvidenceResolver(tmp_path).resolve_artifact("linked-dir/report.json")
    assert error.value.code == "artifact_changed_during_read"


def test_patch_required_repo_never_falls_back_to_writable_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _PatchRequiredRepo.calls = []
    monkeypatch.setattr(evidence, "Repo", _PatchRequiredRepo)

    with pytest.raises(ValidationError) as error:
        evidence.canonicalize_run_hash("a" * 32, tmp_path)

    assert error.value.code == "unresolved_evidence"
    assert _PatchRequiredRepo.calls == [(str(tmp_path), True)]
