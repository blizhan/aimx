from __future__ import annotations


def render_help() -> str:
    return "\n".join(
        [
            "aimx - a safe companion CLI for native Aim",
            "",
            "Owned commands:",
            "  help       Show this help message",
            "  version    Show the aimx version and detected native Aim version",
            "  doctor     Show native Aim availability and passthrough readiness",
            "  query      Query metrics or images from a local Aim repository",
            "             Usage: aimx query <metrics|images> <expression> --repo <path> [--json]",
            "             Repo paths may point at either the repo root or its .aim directory",
            "",
            "All other commands are delegated to native `aim`.",
        ]
    )
