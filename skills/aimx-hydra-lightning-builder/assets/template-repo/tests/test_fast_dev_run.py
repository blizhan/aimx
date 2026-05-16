from __future__ import annotations

import subprocess
import sys


def test_fast_dev_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "src/train.py",
            "trainer.fast_dev_run=true",
            "trainer.logger=false",
            "trainer.enable_progress_bar=false",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
