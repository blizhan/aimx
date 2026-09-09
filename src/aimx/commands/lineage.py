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
from aimx.rendering.research_views import render_error, render_relations


def run_lineage_command(args: list[str]) -> ResearchCommandResult:
    if not args:
        return ResearchCommandResult(2, error_message="Usage: aimx lineage <show|link|retract> ...")
    try:
        command = args[0]
        rest, repo_value, output_json = parse_repo_json(args[1:])
        root, state = read_repo_state(str(repo_value))
        if command == "show":
            if len(rest) != 1:
                raise ValidationError("lineage show requires a finding id")
            if rest[0] not in state.findings:
                raise ValidationError(f"Unknown finding: {rest[0]}", code="unknown_entity")
            return ResearchCommandResult(0, output=render_relations(state, rest[0], output_json=output_json, repo=str(root)))
        if command == "link":
            if len(rest) < 3:
                raise ValidationError("lineage link requires source, relation type, and target")
            source, relation_type, target = rest[:3]
            options = _options(rest[3:], allowed=frozenset({"reason"}))
            operation = {
                "op": "relation.create",
                "local_id": "relation",
                "source": source,
                "relation_type": relation_type,
                "target": target,
            }
            if options.get("reason"):
                operation["reason"] = options["reason"]
        elif command == "retract":
            if not rest:
                raise ValidationError("lineage retract requires a relation id")
            options = _options(rest[1:], allowed=frozenset({"reason"}))
            operation = {"op": "relation.retract", "relation_id": rest[0]}
            if options.get("reason"):
                operation["reason"] = options["reason"]
            else:
                raise ValidationError("lineage retract requires --reason")
        else:
            raise ValidationError(f"Unsupported lineage command: {command}")
        update = build_update(
            state,
            [operation],
            author_kind="human",
            author_name=os.environ.get("AIMX_AUTHOR", "human"),
            message=f"human lineage {command}",
        )
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
