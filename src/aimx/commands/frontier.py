from __future__ import annotations

import os

from aimx.commands.finding import _options
from aimx.commands.research import (
    ResearchCommandResult,
    build_update,
    execute_update,
    parse_repo_json,
    read_repo_state,
)
from aimx.research.errors import ResearchError, ValidationError
from aimx.rendering.research_views import render_error, render_frontier


def run_frontier_command(args: list[str]) -> ResearchCommandResult:
    if not args:
        return ResearchCommandResult(2, error_message="Usage: aimx frontier <show|lane-add|add|move|pause|retire> ...")
    try:
        command = args[0]
        rest, repo_value, output_json = parse_repo_json(args[1:])
        root, state = read_repo_state(str(repo_value))
        if command == "show":
            if rest:
                raise ValidationError(f"Unsupported frontier show option: {rest[0]}")
            return ResearchCommandResult(0, output=render_frontier(state, output_json=output_json, repo=str(root)))

        options = _options(rest)
        if command == "lane-add":
            name = _positional(options)
            operation = {"op": "frontier.lane.create", "local_id": "lane", "name": name}
            if options.get("description"):
                operation["description"] = options["description"]
        elif command == "add":
            lane_ref = _required_option(options, "lane")
            lane_id = _lane_id(state, lane_ref)
            finding = options.get("finding")
            direction = options.get("direction")
            if bool(finding) == bool(direction):
                raise ValidationError("frontier add needs exactly one of --finding or --direction")
            operation = {
                "op": "frontier.item.create",
                "local_id": "frontier",
                "lane_id": lane_id,
                "rationale": _required_option(options, "rationale"),
                "priority": int(options.get("priority", 0)),
            }
            if finding:
                operation["finding_id"] = finding
            else:
                operation["direction"] = direction
        elif command in {"move", "pause", "retire"}:
            item_id = _positional(options)
            if item_id not in state.frontier_items:
                raise ValidationError(f"Unknown frontier item: {item_id}", code="unknown_entity")
            if command == "move":
                operation = {
                    "op": "frontier.item.move",
                    "frontier_item_id": item_id,
                    "lane_id": _lane_id(state, _required_option(options, "lane")),
                }
            else:
                operation = {
                    "op": "frontier.item.set_status",
                    "frontier_item_id": item_id,
                    "frontier_status": "paused" if command == "pause" else "retired",
                }
                if options.get("reason"):
                    operation["reason"] = options["reason"]
        else:
            raise ValidationError(f"Unsupported frontier command: {command}")

        update = build_update(
            state,
            [operation],
            author_kind="human",
            author_name=os.environ.get("AIMX_AUTHOR", "human"),
            message=f"human frontier {command}",
        )
        return execute_update(root, update, output_json=output_json)
    except (ResearchError, ValueError) as error:
        if isinstance(error, ResearchError):
            return ResearchCommandResult(
                error.exit_status,
                error_message=render_error(error) if "--json" in args else error.message,
            )
        return ResearchCommandResult(2, error_message=str(error))


def _positional(options: dict[str, object]) -> str:
    values = options.get("_", [])
    if not values:
        raise ValidationError("A positional identifier/name is required")
    return str(values[0])


def _required_option(options: dict[str, object], key: str) -> str:
    value = options.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"--{key.replace('_', '-')} is required")
    return value


def _lane_id(state, value: str) -> str:
    if value in state.lanes:
        return value
    matches = [lane_id for lane_id, lane in state.lanes.items() if lane.get("active", True) and lane.get("name") == value]
    if len(matches) != 1:
        raise ValidationError(f"Unknown or ambiguous frontier lane: {value}", code="unknown_entity")
    return matches[0]
