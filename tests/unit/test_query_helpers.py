from __future__ import annotations

from pathlib import Path

import pytest

from aimx.commands.query import QueryInvocation, normalize_repo_path


def test_normalize_repo_path_keeps_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    normalized = normalize_repo_path(repo_root)

    assert normalized == repo_root


def test_normalize_repo_path_converts_dot_aim_directory_to_parent(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dot_aim = repo_root / ".aim"
    dot_aim.mkdir(parents=True)

    normalized = normalize_repo_path(dot_aim)

    assert normalized == repo_root


def test_normalize_repo_path_rejects_missing_path() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        normalize_repo_path(Path("missing-repo"))


def test_query_invocation_rejects_unsupported_target() -> None:
    with pytest.raises(ValueError, match="Unsupported query target"):
        QueryInvocation(
            target="artifacts",
            expression="metric.name == 'loss'",
            repo_path=Path("data"),
            output_json=False,
        )
