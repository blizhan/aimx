#!/usr/bin/env python3
"""Read-only Hydra + Lightning + Aim repository audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]


TEXT_EXTENSIONS = {".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "multirun",
    "outputs",
}
CONFIG_GROUPS = {
    "accelerate",
    "callbacks",
    "data",
    "datamodule",
    "experiment",
    "logger",
    "loss",
    "metrics",
    "model",
    "opt",
    "paths",
    "plmodule",
    "trainer",
}
STACK = "hydra-lightning"


def read_text(path: Path, limit: int = 300_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def iter_text_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo.rglob("*"):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_EXTENSIONS:
            files.append(path)
    return files


def rel(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def grep(files: list[Path], pattern: str) -> list[Path]:
    regex = re.compile(pattern)
    return [path for path in files if regex.search(read_text(path))]


def status(condition: bool) -> str:
    return "pass" if condition else "missing"


def normalize_dist_name(requirement: str) -> str:
    name = requirement.strip()
    name = re.split(r"\s*(?:\[|==|~=|!=|<=|>=|<|>|=|;|\s)", name, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def normalized_dependency_set(requirements: list[str]) -> set[str]:
    return {name for item in requirements if (name := normalize_dist_name(item))}


def collect_quoted_dependency_specs(text: str) -> set[str]:
    dependencies: list[str] = []
    for value in re.findall(r"[\"']([^\"']+)[\"']", text):
        if re.search(r"[<>=~!]", value):
            dependencies.append(value)
    return normalized_dependency_set(dependencies)


def collect_pyproject_dependencies(repo: Path) -> set[str]:
    path = repo / "pyproject.toml"
    if not path.exists():
        return set()
    text = read_text(path)
    if tomllib is None:
        return collect_quoted_dependency_specs(text)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return collect_quoted_dependency_specs(text)

    dependencies: list[str] = []
    project = data.get("project", {})
    dependencies.extend(project.get("dependencies", []))
    for values in project.get("optional-dependencies", {}).values():
        dependencies.extend(item for item in values if isinstance(item, str))
    for values in data.get("dependency-groups", {}).values():
        dependencies.extend(item for item in values if isinstance(item, str))

    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    dependencies.extend(name for name in poetry_deps if name.lower() != "python")

    return normalized_dependency_set(dependencies)


def collect_requirements_dependencies(repo: Path) -> set[str]:
    dependencies: set[str] = set()
    candidates = list(repo.glob("requirements*.txt"))
    requirements_dir = repo / "requirements"
    if requirements_dir.is_dir():
        candidates.extend(requirements_dir.glob("*.txt"))

    for path in candidates:
        for line in read_text(path).splitlines():
            line = line.split("#", maxsplit=1)[0].strip()
            if not line or line.startswith(("-", ".")):
                continue
            name = normalize_dist_name(line)
            if name:
                dependencies.add(name)
    return dependencies


def collect_dependencies(repo: Path) -> set[str]:
    return collect_pyproject_dependencies(repo) | collect_requirements_dependencies(repo)


def audit(repo: Path, stack: str = STACK) -> dict[str, Any]:
    files = iter_text_files(repo)
    rel_files = {rel(repo, path) for path in files}
    configs_dir = repo / "configs"
    config_groups = sorted(path.name for path in configs_dir.iterdir() if path.is_dir()) if configs_dir.exists() else []
    yaml_files = [path for path in files if path.suffix in {".yaml", ".yml"}]
    py_files = [path for path in files if path.suffix == ".py"]

    dependencies = collect_dependencies(repo)
    hydra_entrypoints = grep(py_files, r"@hydra\.main|hydra\.main\(")
    lightning_modules = grep(py_files, r"LightningModule|lightning\.LightningModule|L\.LightningModule")
    lightning_datamodules = grep(py_files, r"LightningDataModule|lightning\.LightningDataModule")
    trainer_usage = grep(py_files, r"\bTrainer\b|trainer\.fit|trainer\.validate|trainer\.test")
    aim_logger_configs = grep(yaml_files, r"aim\.pytorch_lightning\.AimLogger")
    direct_aim_tracks = grep(py_files, r"experiment\.track|aim_run\.track|from aim import")
    scalar_logs = grep(py_files, r"\bself\.log\(")
    hparam_logs = grep(py_files, r"log_hyperparams|log_hyperparameters")

    required_groups = {"datamodule", "model", "plmodule", "trainer", "logger", "paths"}
    present_required_groups = required_groups.intersection(config_groups)

    checks = [
        (
            "uv_or_pyproject",
            "pass" if (repo / "pyproject.toml").exists() else "partial",
            "Python project metadata exists." if (repo / "pyproject.toml").exists() else "No pyproject.toml detected.",
        ),
        (
            "hydra_dependency",
            status("hydra-core" in dependencies),
            "Hydra dependency detected." if "hydra-core" in dependencies else "No exact hydra-core dependency detected.",
        ),
        (
            "lightning_dependency",
            status(bool({"lightning", "pytorch-lightning"}.intersection(dependencies))),
            "Lightning dependency detected."
            if {"lightning", "pytorch-lightning"}.intersection(dependencies)
            else "No exact lightning dependency detected.",
        ),
        (
            "aim_dependency",
            status("aim" in dependencies),
            "Aim dependency detected." if "aim" in dependencies else "No exact aim dependency detected.",
        ),
        ("hydra_entrypoint", status(bool(hydra_entrypoints)), "Hydra main entrypoint detected."),
        ("config_groups", "pass" if len(present_required_groups) == len(required_groups) else "partial", f"Found config groups: {', '.join(config_groups) or '-'}"),
        ("lightning_module", status(bool(lightning_modules)), "LightningModule implementation detected."),
        ("lightning_datamodule", status(bool(lightning_datamodules)), "LightningDataModule implementation detected."),
        ("trainer_orchestration", status(bool(trainer_usage)), "Trainer orchestration detected."),
        ("aim_logger", status(bool(aim_logger_configs)), "AimLogger config detected."),
        ("scalar_metrics", status(bool(scalar_logs)), "Lightning self.log metrics detected."),
        (
            "explicit_artifacts",
            "pass" if direct_aim_tracks else "partial",
            "Explicit Aim artifact/distribution traces detected."
            if direct_aim_tracks
            else "No explicit Aim artifact/distribution traces detected.",
        ),
        (
            "hyperparameters",
            "pass" if hparam_logs else "partial",
            "Hyperparameter logging detected." if hparam_logs else "No explicit hyperparameter logging helper detected.",
        ),
        (
            "tests",
            "pass" if (repo / "tests").exists() else "partial",
            "Tests directory detected." if (repo / "tests").exists() else "No top-level tests directory detected.",
        ),
    ]

    return {
        "repo": str(repo),
        "stack": stack,
        "score": sum(1 for _, item_status, _ in checks if item_status == "pass"),
        "max_score": len(checks),
        "checks": [{"id": item_id, "status": item_status, "detail": detail} for item_id, item_status, detail in checks],
        "evidence": {
            "entrypoints": sorted(path for path in rel_files if path in {"src/train.py", "train.py", "main.py"}),
            "hydra_entrypoints": [rel(repo, path) for path in hydra_entrypoints[:20]],
            "config_groups": config_groups,
            "lightning_modules": [rel(repo, path) for path in lightning_modules[:30]],
            "lightning_datamodules": [rel(repo, path) for path in lightning_datamodules[:30]],
            "aim_logger_configs": [rel(repo, path) for path in aim_logger_configs[:20]],
            "direct_aim_tracks": [rel(repo, path) for path in direct_aim_tracks[:30]],
            "dependencies": sorted(dependencies),
        },
    }


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Hydra Lightning Audit: {Path(payload['repo']).name}",
        "",
        f"- Score: {payload['score']}/{payload['max_score']}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(f"| `{check['id']}` | {check['status']} | {check['detail']} |")
    lines.extend(["", "## Evidence", ""])
    for key, value in payload["evidence"].items():
        if isinstance(value, list):
            shown = ", ".join(value[:8]) or "-"
            lines.append(f"- `{key}`: {shown}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default=STACK, choices=[STACK])
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.exists() or not repo.is_dir():
        parser.error(f"--repo must be an existing directory: {repo}")

    payload = audit(repo, args.stack)
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(to_markdown(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
