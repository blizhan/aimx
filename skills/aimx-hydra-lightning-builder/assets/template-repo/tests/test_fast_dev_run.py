from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("overrides", [(), ("experiment=exp",)])
def test_fast_dev_run(overrides: tuple[str, ...]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "src/train.py",
            *overrides,
            "trainer.fast_dev_run=true",
            "trainer.logger=false",
            "trainer.enable_progress_bar=false",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_plmodule_exports() -> None:
    from {{ package_name }}.plmodules import BaseLitModule, ClassificationModule

    assert BaseLitModule.__name__ == "BaseLitModule"
    assert ClassificationModule.__name__ == "ClassificationModule"
