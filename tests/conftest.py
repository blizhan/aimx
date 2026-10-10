from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest


@pytest.fixture
def autoresearch_control():
    path = Path(__file__).resolve().parents[1] / "skills/aimx-autoresearch/scripts/control.py"
    spec = importlib.util.spec_from_file_location("autoresearch_control_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def autoresearch_config(tmp_path: Path) -> Path:
    (tmp_path / "project").mkdir()
    (tmp_path / "data" / ".aim").mkdir(parents=True)
    path = tmp_path / "autoresearch.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "project_root": "project",
        "aim_repo": "data/.aim",
        "goal": {
            "description": "Improve score with validated changes",
            "success_criteria": ["Score reaches 4", "Project validation passes"],
        },
        "max_rounds": 2,
        "default_agents": [
            {"id": "explore", "model": "fixture/explorer", "reasoning_effort": "low"},
            {"id": "verify"},
        ],
        "round_overrides": {"2": {"agents": [
            {"id": "explore", "model": "fixture/researcher", "reasoning_effort": "high"},
            {"id": "verify", "model": "fixture/reviewer", "reasoning_effort": "medium"},
        ]}},
    }), encoding="utf-8")
    return path


def _sample_repo_root() -> Path:
    return Path("data")


def _sample_repo_dot_aim() -> Path:
    return _sample_repo_root() / ".aim"


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch):
    from aimx.__main__ import main

    def _run(args: Sequence[str]) -> int:
        monkeypatch.setattr("sys.argv", ["aimx", *args])
        return main(list(args))

    return _run


@pytest.fixture
def fake_aim_script(tmp_path: Path) -> Path:
    script = tmp_path / "aim"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "if sys.argv[1:] == ['version']:\n"
        "    print(os.environ.get('FAKE_AIM_VERSION', '3.29.1'))\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}))\n"
        "print('fake-aim-stderr', file=sys.stderr)\n"
        "sys.exit(int(os.environ.get('FAKE_AIM_EXIT', '0')))\n"
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def sample_repo_root() -> Path:
    repo_root = _sample_repo_root()
    if not _sample_repo_dot_aim().exists():
        pytest.skip("sample Aim repository is not available in this environment")
    return repo_root


@pytest.fixture
def sample_repo_dot_aim() -> Path:
    dot_aim = _sample_repo_dot_aim()
    if not dot_aim.exists():
        pytest.skip("sample Aim repository is not available in this environment")
    return dot_aim


@pytest.fixture
def research_repo(tmp_path: Path) -> Path:
    """Return a fresh repository root without pre-creating Aimx state."""

    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def research_update_factory() -> Callable[..., dict[str, Any]]:
    def _make(
        base_revision: int = 0,
        *,
        operations: list[dict[str, Any]] | None = None,
        author_kind: str = "agent",
        author_name: str = "test-agent",
        client_update_id: str | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "base_revision": base_revision,
            "author": {"kind": author_kind, "name": author_name},
            "operations": operations
            or [
                {
                    "op": "finding.create",
                    "local_id": "finding",
                    "claim": "The candidate improves accuracy.",
                    "epistemic_status": "candidate",
                    "confidence": "medium",
                    "governance_status": "proposed",
                }
            ],
        }
        if client_update_id is not None:
            value["client_update_id"] = client_update_id
        return value

    return _make
