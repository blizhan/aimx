from __future__ import annotations

import os
from pathlib import Path

from aimx.commands.research import (
    ResearchCommandResult,
    build_update,
    execute_update,
    parse_repo_json,
    read_repo_state,
)
from aimx.research.errors import ResearchError, ValidationError
from aimx.rendering.research_views import render_error, render_finding, render_findings

_ALLOWED_OPTIONS = {
    "comment": frozenset(),
    "accept": frozenset({"reason"}),
    "reject": frozenset({"reason"}),
    "assess": frozenset({"status", "confidence", "reason"}),
}


def run_finding_command(args: list[str]) -> ResearchCommandResult:
    if not args:
        return ResearchCommandResult(2, error_message="Usage: aimx finding <ls|show|comment|accept|reject|assess> ...")
    try:
        command = args[0]
        rest, repo_value, output_json = parse_repo_json(args[1:])
        root, state = read_repo_state(str(repo_value))
        if command == "ls":
            if rest:
                raise ValidationError(f"Unsupported finding ls option: {rest[0]}")
            return ResearchCommandResult(0, output=render_findings(state, output_json=output_json, repo=str(root)))
        if command == "show":
            if len(rest) != 1:
                raise ValidationError("finding show requires a finding id")
            if rest[0] not in state.findings:
                raise ValidationError(f"Unknown finding: {rest[0]}", code="unknown_entity")
            return ResearchCommandResult(0, output=render_finding(state, rest[0], output_json=output_json, repo=str(root)))
        update = _build_human_update(command, rest, state)
        return execute_update(root, update, output_json=output_json)
    except ResearchError as error:
        return ResearchCommandResult(
            error.exit_status,
            error_message=render_error(error) if "--json" in args else error.message,
        )
    except (ValueError, IndexError) as error:
        validation_error = ValidationError(str(error))
        return ResearchCommandResult(
            validation_error.exit_status,
            error_message=(
                render_error(validation_error)
                if "--json" in args
                else validation_error.message
            ),
        )


def _build_human_update(command: str, args: list[str], state):
    if not args:
        raise ValidationError(f"finding {command} requires a finding id")
    finding_id = args[0]
    if finding_id not in state.findings:
        raise ValidationError(f"Unknown finding: {finding_id}", code="unknown_entity")
    if command not in _ALLOWED_OPTIONS:
        raise ValidationError(f"Unsupported finding command: {command}")
    options = _options(args[1:], allowed=_ALLOWED_OPTIONS[command])
    if command == "comment":
        text = " ".join(options.get("_", []))
        if not text.strip():
            raise ValidationError("finding comment requires text")
        operation = {
            "op": "annotation.create",
            "annotation_target_kind": "finding",
            "annotation_target_id": finding_id,
            "text": text.strip(),
        }
    elif command in {"accept", "reject"}:
        operation = {
            "op": "finding.governance",
            "finding_id": finding_id,
            "governance_status": command + "ed",
        }
        if options.get("reason"):
            operation["reason"] = options["reason"]
    elif command == "assess":
        if not options.get("status") and not options.get("confidence"):
            raise ValidationError("finding assess requires --status or --confidence")
        operation = {"op": "finding.assess", "finding_id": finding_id}
        if options.get("status"):
            operation["epistemic_status"] = options["status"]
        if options.get("confidence"):
            operation["confidence"] = options["confidence"]
        if options.get("reason"):
            operation["reason"] = options["reason"]
    else:
        raise ValidationError(f"Unsupported finding command: {command}")
    return build_update(
        state,
        [operation],
        author_kind="human",
        author_name=os.environ.get("AIMX_AUTHOR", "human"),
        message=f"human finding {command}",
    )


def _options(args: list[str], *, allowed: frozenset[str]) -> dict[str, object]:
    result: dict[str, object] = {"_": []}
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if key not in allowed:
                raise ValidationError(f"Unsupported option: {token}")
            if index + 1 >= len(args):
                raise ValidationError(f"Missing value for {token}")
            result[key] = args[index + 1]
            index += 2
        else:
            result["_"].append(token)  # type: ignore[union-attr]
            index += 1
    return result
