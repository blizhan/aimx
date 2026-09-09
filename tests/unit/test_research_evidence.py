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


def test_patch_required_repo_never_falls_back_to_writable_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _PatchRequiredRepo.calls = []
    monkeypatch.setattr(evidence, "Repo", _PatchRequiredRepo)

    with pytest.raises(ValidationError) as error:
        evidence.canonicalize_run_hash("a" * 32, tmp_path)

    assert error.value.code == "unresolved_evidence"
    assert _PatchRequiredRepo.calls == [(str(tmp_path), True)]
