from __future__ import annotations

from aimx.native_aim.locator import NativeAimResolution


def render_doctor(resolution: NativeAimResolution) -> str:
    ready = "ready" if resolution.status == "available" else "not ready"
    lines = [
        f"native aim status: {resolution.status}",
        f"passthrough: {ready}",
        f"message: {resolution.diagnostic_message}",
    ]
    if resolution.executable_path:
        lines.insert(1, f"native aim path: {resolution.executable_path}")
    if resolution.version_text:
        lines.insert(2, f"native aim version: {resolution.version_text}")
    return "\n".join(lines)
