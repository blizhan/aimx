from __future__ import annotations

import sys

from aimx.commands.doctor import render_doctor
from aimx.commands.help import render_help
from aimx.commands.query import run_query_command
from aimx.commands.trace import run_trace_command
from aimx.commands.version import render_version
from aimx.native_aim.locator import resolve_native_aim
from aimx.native_aim.passthrough import run_passthrough
from aimx.router import route_args


def run_cli(args: list[str]) -> int:
    route = route_args(args)
    resolution = resolve_native_aim()

    if route.route_kind == "owned":
        command = route.owned_command
        if command == "help":
            sys.stdout.write(f"{render_help()}\n")
            return 0
        if command == "version":
            sys.stdout.write(f"{render_version(resolution.version_text)}\n")
            return 0
        if command == "doctor":
            sys.stdout.write(f"{render_doctor(resolution)}\n")
            return 0 if resolution.status == "available" else 1
        if command == "query":
            result = run_query_command(route.owned_args or [])
            if result.output:
                sys.stdout.write(f"{result.output}\n")
            if result.error_message:
                sys.stderr.write(f"{result.error_message}\n")
            return result.exit_status
        if command == "trace":
            result = run_trace_command(route.owned_args or [])
            if result.output:
                sys.stdout.write(f"{result.output}\n")
            if result.error_message:
                sys.stderr.write(f"{result.error_message}\n")
            return result.exit_status
        raise ValueError(f"Unsupported owned command: {command}")

    result = run_passthrough(route.delegated_args or [], resolution)
    if not result.process_started and result.error_message:
        sys.stderr.write(f"{result.error_message}\n")
    return result.exit_status
