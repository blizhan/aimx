from __future__ import annotations

from aimx import __version__


def render_version(native_aim_version: str | None = None) -> str:
    lines = [f"aimx {__version__}"]
    if native_aim_version:
        lines.append(f"native aim {native_aim_version}")
    else:
        lines.append("native aim unavailable")
    return "\n".join(lines)
