from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aimx.aim_bridge.research_evidence import make_evidence_resolver, mark_evidence_availability
from aimx.research.agenda import choose_next, ordered_agenda
from aimx.research.context import compile_context
from aimx.research.errors import ResearchError, ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.store import CommitResult, apply_update, dry_run_update, normalize_repo_root, read_state
from aimx.rendering.research_views import (
    render_agenda,
    render_context,
    render_error,
    render_json,
    render_state_human,
    render_state_json,
)


@dataclass(frozen=True)
class ResearchCommandResult:
    exit_status: int
    output: str | None = None
    error_message: str | None = None


def run_research_command(args: list[str]) -> ResearchCommandResult:
    if not args:
        return ResearchCommandResult(2, error_message="Usage: aimx research <state|update|context|agenda|next> ...")
    subcommand = args[0]
    try:
        if subcommand == "state":
            return _run_state(args[1:])
        if subcommand == "update":
            return _run_update(args[1:])
        if subcommand == "context":
            return _run_context(args[1:])
        if subcommand == "agenda":
            return _run_agenda(args[1:])
        if subcommand == "next":
            return _run_next(args[1:])
        raise ValidationError(
            f"Unsupported research command: {subcommand}", code="invalid_input"
        )
    except ResearchError as error:
        output_json = "--json" in args
        return ResearchCommandResult(
            error.exit_status,
            error_message=render_error(error) if output_json else error.message,
        )
    except (ValueError, json.JSONDecodeError) as error:
        validation_error = ValidationError(str(error))
        return ResearchCommandResult(
            validation_error.exit_status,
            error_message=(
                render_error(validation_error)
                if "--json" in args
                else validation_error.message
            ),
        )
    except Exception as error:  # pragma: no cover - defensive CLI boundary
        return ResearchCommandResult(1, error_message=f"Research command failed: {error}")


def execute_update(
    repo_path: Path,
    update: ResearchUpdate,
    *,
    dry_run: bool = False,
    output_json: bool = False,
) -> ResearchCommandResult:
    root = normalize_repo_root(repo_path)
    resolver = make_evidence_resolver(root)
    if dry_run:
        state, _created_ids = dry_run_update(root, update, resolver)
        entity_kinds = {
            "finding.create": "finding",
            "relation.create": "relation",
            "annotation.create": "annotation",
            "frontier.lane.create": "lane",
            "frontier.item.create": "frontier",
            "agenda.item.create": "agenda",
        }
        payload = {
            "schema_version": 1,
            "status": "valid",
            "base_revision": state.revision,
            "would_create": [
                entity_kinds[operation["op"]]
                for operation in update.operations
                if operation.get("op") in entity_kinds
            ],
        }
        return ResearchCommandResult(0, output=render_json(payload) if output_json else "Research update is valid.")
    result = apply_update(root, update, resolver)
    payload = result.as_dict()
    if output_json:
        return ResearchCommandResult(0, output=render_json(payload))
    return ResearchCommandResult(
        0,
        output=f"Committed research revision {result.revision} ({result.commit_id}).",
    )


def build_update(
    state: ResearchState,
    operations: list[dict[str, Any]],
    *,
    author_kind: str,
    author_name: str,
    message: str | None = None,
) -> ResearchUpdate:
    return ResearchUpdate.from_dict(
        {
            "schema_version": 1,
            "base_revision": state.revision,
            "author": {"kind": author_kind, "name": author_name},
            "message": message,
            "operations": operations,
        }
    )


def read_repo_state(repo_value: str) -> tuple[Path, ResearchState]:
    normalized = normalize_repo_root(Path(repo_value))
    state = read_state(normalized)
    return normalized, mark_evidence_availability(state, normalized)


def parse_repo_json(args: list[str], *, default_json: bool = False) -> tuple[list[str], Path, bool]:
    rest: list[str] = []
    repo_value = "."
    output_json = default_json
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--repo":
            if index + 1 >= len(args):
                raise ValidationError("Missing value for --repo")
            repo_value = args[index + 1]
            index += 2
        elif token == "--json":
            output_json = True
            index += 1
        else:
            rest.append(token)
            index += 1
    return rest, Path(repo_value), output_json


def _run_state(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    if rest:
        raise ValidationError(f"Unsupported research state option: {rest[0]}")
    root, state = read_repo_state(str(repo_value))
    output = render_state_json(state, str(root)) if output_json else render_state_human(state, str(root))
    return ResearchCommandResult(0, output=output)


def _run_update(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    dry_run = False
    source: str | None = None
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--dry-run":
            dry_run = True
            index += 1
        elif token == "--stdin":
            if source is not None:
                raise ValidationError("--stdin and --file are mutually exclusive")
            source = "-"
            index += 1
        elif token == "--file":
            if index + 1 >= len(rest):
                raise ValidationError("Missing value for --file")
            if source is not None:
                raise ValidationError("--stdin and --file are mutually exclusive")
            source = rest[index + 1]
            index += 2
        else:
            raise ValidationError(f"Unsupported research update option: {token}")
    if source is None:
        raise ValidationError("Research update requires --stdin or --file")
    try:
        raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Unable to read research update: {exc}") from exc
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValidationError("Research update JSON must be an object", code="invalid_schema")
    update = ResearchUpdate.from_dict(value)
    return execute_update(repo_value, update, dry_run=dry_run, output_json=output_json)


def _run_context(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    objective: str | None = None
    budget: int | None = None
    index = 0
    while index < len(rest):
        token = rest[index]
        if token in {"--objective", "--budget"}:
            if index + 1 >= len(rest):
                raise ValidationError(f"Missing value for {token}")
            if token == "--objective":
                objective = rest[index + 1]
            else:
                try:
                    budget = int(rest[index + 1])
                except ValueError as exc:
                    raise ValidationError("--budget must be an integer") from exc
            index += 2
        else:
            raise ValidationError(f"Unsupported research context option: {token}")
    if objective is None or not objective.strip():
        raise ValidationError("--objective is required")
    if budget is None:
        raise ValidationError("--budget is required")
    root, state = read_repo_state(str(repo_value))
    payload = compile_context(state, objective, budget)
    return ResearchCommandResult(0, output=render_context(payload, output_json=output_json))


def _run_agenda(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    status: str | None = None
    if rest:
        if len(rest) != 2 or rest[0] != "--status":
            raise ValidationError(f"Unsupported research agenda option: {rest[0]}")
        status = rest[1]
    root, state = read_repo_state(str(repo_value))
    payload = {
        "schema_version": 1,
        "revision": state.revision,
        "repo": str(root),
        "items": ordered_agenda(state, status=status),
    }
    return ResearchCommandResult(0, output=render_agenda(payload, output_json=output_json))


def _run_next(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    if rest:
        raise ValidationError(f"Unsupported research next option: {rest[0]}")
    root, state = read_repo_state(str(repo_value))
    item = choose_next(state)
    payload = {
        "schema_version": 1,
        "revision": state.revision,
        "repo": str(root),
        "status": "selected" if item else "no_actionable_agenda",
        "item": item,
    }
    if output_json:
        output = render_json(payload)
    elif item:
        output = f"{item['id']}\t{item.get('status')}\t{item.get('objective')}"
    else:
        output = "No actionable agenda item."
    return ResearchCommandResult(0, output=output)
