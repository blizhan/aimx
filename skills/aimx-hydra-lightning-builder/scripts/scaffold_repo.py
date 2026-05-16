#!/usr/bin/env python3
"""Scaffold a minimal Hydra + Lightning + Aim repository."""

from __future__ import annotations

import argparse
import keyword
import shutil
import sys
from pathlib import Path


STACK = "hydra-lightning"
TEXT_EXTENSIONS = {".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def validate_package(name: str) -> str:
    normalized = name.replace("-", "_")
    if not normalized.isidentifier() or keyword.iskeyword(normalized):
        raise argparse.ArgumentTypeError(f"Package must be a valid Python identifier: {name}")
    return normalized


def copy_template(template: Path, output: Path, replacements: dict[str, str]) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for src in template.rglob("*"):
        relative = src.relative_to(template)
        parts = [replacements.get(part, part) for part in relative.parts]
        dst = output.joinpath(*parts)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT_EXTENSIONS:
            text = src.read_text(encoding="utf-8")
            for key, value in replacements.items():
                text = text.replace(f"{{{{ {key} }}}}", value)
            dst.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default=STACK, choices=[STACK])
    parser.add_argument("--name", required=True, help="Project distribution name.")
    parser.add_argument("--package", required=True, type=validate_package, help="Python import package name.")
    parser.add_argument("--preset", default="classification", choices=["classification", "forecast"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    skill_dir = Path(__file__).resolve().parents[1]
    template = skill_dir / "assets" / "template-repo"
    if not template.exists():
        print(f"Template not found: {template}", file=sys.stderr)
        return 2

    copy_template(
        template,
        args.output.resolve(),
        {
            "__package__": args.package,
            "project_name": args.name,
            "package_name": args.package,
            "preset": args.preset,
            "stack": args.stack,
        },
    )
    print(f"Created {args.stack} repository at {args.output.resolve()}")
    print("Next: cd into the repo, run `uv sync`, then `uv run pytest`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
