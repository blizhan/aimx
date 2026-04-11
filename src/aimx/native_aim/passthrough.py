from __future__ import annotations

from dataclasses import dataclass
import subprocess

from aimx.native_aim.locator import NativeAimResolution


@dataclass(frozen=True)
class DelegatedExecutionResult:
    process_started: bool
    exit_status: int
    error_message: str | None = None


def run_passthrough(args: list[str], resolution: NativeAimResolution) -> DelegatedExecutionResult:
    if resolution.status == "missing":
        return DelegatedExecutionResult(
            process_started=False,
            exit_status=127,
            error_message=resolution.diagnostic_message,
        )
    if resolution.status == "unusable":
        return DelegatedExecutionResult(
            process_started=False,
            exit_status=126,
            error_message=resolution.diagnostic_message,
        )

    result = subprocess.run(
        [resolution.executable_path, *args],
        check=False,
    )
    return DelegatedExecutionResult(process_started=True, exit_status=result.returncode)
