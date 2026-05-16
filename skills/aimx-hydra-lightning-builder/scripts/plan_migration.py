#!/usr/bin/env python3
"""Generate a Hydra Lightning migration plan from audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STACK = "hydra-lightning"
STAGES = {
    "uv_or_pyproject": "Normalize project metadata and dependency workflow around `pyproject.toml` and `uv`.",
    "hydra_dependency": "Add Hydra and OmegaConf dependencies, then introduce a `configs/` tree.",
    "lightning_dependency": "Add Lightning and move training lifecycle into Lightning modules and Trainer.",
    "aim_dependency": "Add Aim as the experiment evidence backend.",
    "hydra_entrypoint": "Create `src/train.py` with `@hydra.main` as the canonical entrypoint.",
    "config_groups": "Split config into datamodule/model/plmodule/trainer/logger/paths groups.",
    "lightning_module": "Wrap task logic in a LightningModule with stable train/val/test/predict hooks.",
    "lightning_datamodule": "Move dataset split and DataLoader policy into LightningDataModule classes.",
    "trainer_orchestration": "Instantiate Trainer, callbacks, loggers, datamodule, and plmodule from Hydra config.",
    "aim_logger": "Add `configs/logger/aim.yaml` with `aim.pytorch_lightning.AimLogger`.",
    "scalar_metrics": "Log objective and diagnostic metrics with `self.log` using stable names.",
    "explicit_artifacts": "Add optional `experiment.track(...)` hooks for images or distributions.",
    "hyperparameters": "Log model/data/trainer/optimizer/task config to all configured loggers.",
    "tests": "Add a dummy-data fast-dev-run test that does not require private datasets.",
}


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(audit: dict[str, Any], target_stack: str) -> str:
    lines = [
        f"# Hydra Lightning Migration Plan: {Path(audit['repo']).name}",
        "",
        f"- Target stack: `{target_stack}`",
        f"- Audit score: {audit['score']}/{audit['max_score']}",
        "",
        "## Ordered Stages",
    ]
    number = 1
    for check in audit["checks"]:
        if check["status"] == "pass":
            continue
        stage = STAGES.get(check["id"])
        if stage:
            lines.append(f"{number}. {stage}")
            number += 1
    if number == 1:
        lines.append("1. No missing core stages from the audit. Review naming and evidence quality before changing code.")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "- `uv run python src/train.py trainer.fast_dev_run=true trainer.logger=false` succeeds on dummy or tiny data.",
            "- AimLogger can be enabled by config without changing code.",
            "- `aimx query params` and `aimx query metrics` can inspect the generated Aim repo.",
            "- Existing model/data behavior is preserved or covered by focused migration tests.",
            "",
            "## Safety",
            "- This output is a plan only.",
            "- Do not edit the audited repository until the user approves execution.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--target-stack", default=STACK, choices=[STACK])
    args = parser.parse_args()

    audit = load_audit(args.audit)
    audit_stack = audit.get("stack")
    if audit_stack and audit_stack != args.target_stack:
        parser.error(f"audit stack {audit_stack!r} does not match target stack {args.target_stack!r}")
    print(build_plan(audit, args.target_stack), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
