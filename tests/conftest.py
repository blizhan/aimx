from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest


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
