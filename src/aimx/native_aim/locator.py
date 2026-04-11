from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess


@dataclass(frozen=True)
class NativeAimResolution:
    status: str
    executable_path: str | None
    version_text: str | None
    diagnostic_message: str


def resolve_native_aim() -> NativeAimResolution:
    executable = shutil.which("aim")
    if not executable:
        return NativeAimResolution(
            status="missing",
            executable_path=None,
            version_text=None,
            diagnostic_message="native aim is not available; install native Aim or make `aim` discoverable in PATH.",
        )

    try:
        result = subprocess.run(
            [executable, "version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return NativeAimResolution(
            status="unusable",
            executable_path=executable,
            version_text=None,
            diagnostic_message=f"native aim at {executable} could not be executed: {exc}",
        )

    if result.returncode != 0:
        return NativeAimResolution(
            status="unusable",
            executable_path=executable,
            version_text=None,
            diagnostic_message=f"native aim at {executable} could not be executed successfully.",
        )

    version_text = (result.stdout or "").strip() or None
    return NativeAimResolution(
        status="available",
        executable_path=executable,
        version_text=version_text,
        diagnostic_message=f"native aim available at {executable}",
    )
