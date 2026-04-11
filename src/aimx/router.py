from __future__ import annotations

from dataclasses import dataclass


OWNED_COMMANDS = {"help", "--help", "-h", "version", "doctor"}


@dataclass(frozen=True)
class CommandRoute:
    route_kind: str
    owned_command: str | None = None
    delegated_args: list[str] | None = None
    reason: str = ""


def route_args(args: list[str]) -> CommandRoute:
    if not args:
        return CommandRoute("owned", owned_command="help", reason="empty invocation")

    command = args[0]
    if command in {"help", "--help", "-h"}:
        return CommandRoute("owned", owned_command="help", reason="reserved help command")
    if command in {"version", "doctor"}:
        return CommandRoute("owned", owned_command=command, reason="reserved aimx command")

    return CommandRoute(
        "passthrough",
        delegated_args=list(args),
        reason="unowned command delegated to native aim",
    )
