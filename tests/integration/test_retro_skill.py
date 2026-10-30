from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

SKILL_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "aimx-retro"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "retro" / "reports"
MINIMAL_JSON = FIXTURES_DIR / "minimal.json"


def test_isolated_skill_install_and_validate(tmp_path):
    """Prove that copying ONLY skills/aimx-retro works under isolated python -I."""
    isolated_skill = tmp_path / "aimx-retro"
    shutil.copytree(SKILL_DIR, isolated_skill)

    report_script = isolated_skill / "scripts" / "report.py"
    assert report_script.is_file()

    # Create a draft inside tmp_path
    draft = tmp_path / "draft.json"
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    draft.write_text(json.dumps(data), encoding="utf-8")

    # Run python -I on report.py validate
    proc = subprocess.run(
        [sys.executable, "-I", str(report_script), "validate", "--file", str(draft)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    res = json.loads(proc.stdout)
    assert res["ok"] is True
    assert res["report_id"] == data["report_id"]


def test_isolated_skill_publish(tmp_path):
    """Test publish with isolated python -I and explicit output root."""
    isolated_skill = tmp_path / "aimx-retro"
    shutil.copytree(SKILL_DIR, isolated_skill)
    report_script = isolated_skill / "scripts" / "report.py"

    out_root = tmp_path / "output_reports"
    draft = tmp_path / "draft.json"
    data = json.loads(MINIMAL_JSON.read_text(encoding="utf-8"))
    data["scope"]["repo"] = str(tmp_path)
    draft.write_text(json.dumps(data), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(report_script),
            "publish",
            "--file",
            str(draft),
            "--output-root",
            str(out_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    res = json.loads(proc.stdout)
    assert res["ok"] is True
    published_dir = Path(res["report_dir"])
    assert (published_dir / "retro.json").is_file()
    assert (published_dir / "RETRO.md").is_file()
