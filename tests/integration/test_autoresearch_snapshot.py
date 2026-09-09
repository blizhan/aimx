from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


def _load_snapshot_module():
    path = Path("skills/aimx/scripts/collect_experiment_snapshot.py").resolve()
    spec = importlib.util.spec_from_file_location("aimx_snapshot_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_snapshot_collector_remains_read_only_and_does_not_write_research_state(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_snapshot_module()
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_run_json(base_cmd, args, timeout):
        return module.CommandPayload(
            argv=[*base_cmd, *args],
            ok=True,
            data={"runs": [], "metrics": []},
        )

    monkeypatch.setattr(module, "run_json", fake_run_json)
    args = Namespace(
        repo=str(repo),
        base_expr="run.hash != ''",
        metric=["accuracy"],
        metric_expr=[],
        trace_metric=["accuracy"],
        trace_expr=[],
        trace_tail=5,
        param=[],
        include_images=False,
        image_expr="images",
        image_head=20,
        aimx="python -m aimx",
        timeout=60,
        pretty=False,
    )

    snapshot, ok = module.collect_snapshot(args)
    assert ok is True
    assert snapshot["read_only"] is True
    assert not (repo / ".aimx").exists()
