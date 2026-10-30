from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aimx.aim_bridge.research_evidence import make_evidence_resolver, mark_evidence_availability
from aimx.research.agenda import choose_next, ordered_agenda
from aimx.research.context import compile_context
from aimx.research.overview import compile_overview
from aimx.research.errors import ResearchError, ValidationError
from aimx.research.models import ResearchState, ResearchUpdate
from aimx.research.store import CommitResult, apply_update, dry_run_update, normalize_repo_root, read_receipt, read_state
from aimx.commands.help import render_command_help
from aimx.rendering.research_views import (
    render_agenda,
    render_annotations_human,
    render_context,
    render_evidence_human,
    render_error,
    render_json,
    render_state_human,
    render_state_json,
    render_overview_human,
    render_receipt_human,
)


@dataclass(frozen=True)
class ResearchCommandResult:
    exit_status: int
    output: str | None = None
    error_message: str | None = None


def run_research_command(args: list[str]) -> ResearchCommandResult:
    if not args:
        return ResearchCommandResult(2, error_message="Usage: aimx research <state|update|context|agenda|next|overview|annotations|receipt|evidence> ...")
    help_result = _research_help(args)
    if help_result is not None:
        return help_result
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
        if subcommand == "overview":
            return _run_overview(args[1:])
        if subcommand == "annotations":
            return _run_annotations(args[1:])
        if subcommand == "receipt":
            return _run_receipt(args[1:])
        if subcommand == "evidence":
            return _run_evidence(args[1:])
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


def _research_help(args: list[str]) -> ResearchCommandResult | None:
    if not any(token in {"-h", "--help"} for token in args):
        return None
    command_tokens = []
    for token in args:
        if token.startswith("-"):
            break
        command_tokens.append(token)
    command = "research" + (" " + " ".join(command_tokens) if command_tokens else "")
    rendered = render_command_help(command, output_json="--json" in args)
    if rendered is None:
        return ResearchCommandResult(2, error_message=f"Unsupported research help target: {command}")
    return ResearchCommandResult(0, output=rendered)


def _run_overview(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    loop_id = None
    if rest:
        if len(rest) != 2 or rest[0] != "--loop":
            raise ValidationError(f"Unsupported research overview option: {rest[0]}")
        loop_id = rest[1]
    root = normalize_repo_root(repo_value)
    try:
        state = read_state(root)
    except ResearchError as error:
        payload = compile_overview(root, None, loop_id, state_error=error.message)
        return ResearchCommandResult(2, error_message=render_json(payload) if output_json else render_overview_human(payload))
    payload = compile_overview(root, state, loop_id)
    rendered = render_json(payload) if output_json else render_overview_human(payload)
    status = 2 if payload.get("status") == "partial" else 0
    if status == 2:
        return ResearchCommandResult(status, error_message=rendered)
    return ResearchCommandResult(status, output=rendered)


def _run_annotations(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    topic = loop_id = None
    index = 0
    while index < len(rest):
        if rest[index] in {"--topic", "--loop"} and index + 1 < len(rest):
            if rest[index] == "--topic":
                topic = rest[index + 1]
            else:
                loop_id = rest[index + 1]
            index += 2
        else:
            raise ValidationError(f"Unsupported research annotations option: {rest[index]}")
    root = normalize_repo_root(repo_value)
    state = read_state(root)
    annotations = list(state.annotations.values())
    if loop_id is not None:
        activities = {key for key, value in state.activities.items() if value.get("kind") == "loop" and value.get("loop_id") == loop_id}
        if not activities:
            raise ValidationError(f"Unknown loop: {loop_id}", code="unknown_loop")
        follow_up_activities = {key for key, value in state.activities.items() if value.get("kind") == "follow_up" and value.get("related_loop_id") == loop_id}
        annotations = [item for item in annotations if item.get("source_activity_id") in activities | follow_up_activities]
    if topic:
        terms = _topic_terms(topic)
        annotations = [item for item in annotations if terms.intersection(_topic_terms(str(item.get("text", ""))))]
    annotations.sort(key=lambda item: (int(item.get("created_revision", 0)), str(item.get("id", ""))))
    payload = {"schema_version": 1, "repo": str(root), "revision": state.revision, "loop_id": loop_id, "annotations": annotations, "unattributed_count": sum("source_activity_id" not in item for item in annotations), "unattributed_total": sum("source_activity_id" not in item for item in state.annotations.values())}
    return ResearchCommandResult(0, output=render_json(payload) if output_json else render_annotations_human(payload))


def _run_receipt(args: list[str]) -> ResearchCommandResult:
    rest, repo_value, output_json = parse_repo_json(args)
    values: dict[str, str] = {}
    index = 0
    while index < len(rest):
        key = rest[index]
        if key not in {"--client-update-id", "--fingerprint"} or index + 1 >= len(rest):
            raise ValidationError(f"Unsupported research receipt option: {key}")
        values[key] = rest[index + 1]
        index += 2
    client_update_id = values.get("--client-update-id")
    if not client_update_id:
        raise ValidationError("--client-update-id is required")
    if "--fingerprint" in values and (len(values["--fingerprint"]) != 64 or any(char not in "0123456789abcdef" for char in values["--fingerprint"])):
        raise ValidationError("--fingerprint must be a lowercase SHA-256 digest")
    root = normalize_repo_root(repo_value)
    payload = read_receipt(root, client_update_id, values.get("--fingerprint"))
    if payload["status"] == "receipt_unverifiable":
        error = ValidationError("Stored receipt does not contain verifiable coordination data", code="receipt_unverifiable", client_update_id=client_update_id)
        return ResearchCommandResult(2, error_message=render_error(error) if output_json else error.message)
    if payload["status"] == "committed" and values.get("--fingerprint") is not None and payload.get("request_fingerprint") != values["--fingerprint"]:
        error = ValidationError("Receipt fingerprint does not match the committed request", code="idempotency_conflict", client_update_id=client_update_id)
        return ResearchCommandResult(2, error_message=render_error(error) if output_json else error.message)
    return ResearchCommandResult(0, output=render_json(payload) if output_json else render_receipt_human(payload))


def _run_evidence(args: list[str]) -> ResearchCommandResult:
    if not args or args[0] != "check":
        raise ValidationError("Usage: aimx research evidence check --finding FINDING_ID [--repo PATH] [--json]")
    rest, repo_value, output_json = parse_repo_json(args[1:])
    if len(rest) != 2 or rest[0] != "--finding":
        raise ValidationError("research evidence check requires --finding FINDING_ID")
    finding_id = rest[1]
    root = normalize_repo_root(repo_value)
    state = read_state(root)
    finding = state.findings.get(finding_id)
    if finding is None:
        raise ValidationError(f"Unknown finding: {finding_id}", code="unknown_entity")
    rows = check_finding_evidence(finding, root, state)
    payload = {"schema_version": 1, "repo": str(root), "revision": state.revision, "finding_id": finding_id, "references": rows}
    return ResearchCommandResult(0, output=render_json(payload) if output_json else render_evidence_human(payload))


def check_finding_evidence(finding: dict[str, Any], root: Path, state: ResearchState) -> list[dict[str, Any]]:
    import hashlib
    from aimx.aim_bridge.research_evidence import Repo, make_evidence_resolver
    from aimx.research.errors import ValidationError

    resolver = make_evidence_resolver(root)
    reviews = state.evidence_reviews.get(str(finding.get("id")), [])
    results: list[dict[str, Any]] = []
    for reference in finding.get("evidence", []):
        fingerprint = hashlib.sha256(json.dumps(reference, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        matching_reviews = [review for review in reviews if review.get("evidence_fingerprint") == fingerprint]
        latest_review = matching_reviews[-1].get("scientific_review") if matching_reviews else "not_reviewed"
        row = {"reference": reference, "evidence_fingerprint": fingerprint, "scientific_review": latest_review}
        if reference.get("kind") == "local_artifact":
            try:
                resolved, digest, size = resolver.resolve_artifact(reference["location"])
                row.update({"availability": "available", "identity_status": "matches" if digest == reference.get("sha256") and size == reference.get("size_bytes") else "changed", "resolved_location": str(resolved), "observed_sha256": digest, "observed_size_bytes": size})
            except ValidationError as error:
                if error.code == "artifact_unsupported":
                    availability = "unsupported"
                elif error.code == "artifact_unreadable":
                    availability = "missing" if "no such file" in error.message.casefold() or "not found" in error.message.casefold() else "unreadable"
                else:
                    availability = "unknown"
                row.update({"availability": availability, "identity_status": "unknown", "reason": error.message})
        elif reference.get("kind") == "aim_run":
            if Repo is None:
                row.update({"availability": "unsupported", "identity_status": "not_checked", "reason": "Aim SDK is unavailable"})
            else:
                try:
                    from aimx.aim_bridge.research_evidence import canonicalize_run_hash
                    actual = canonicalize_run_hash(str(reference.get("run_hash", "")), root)
                    row.update({"availability": "available", "identity_status": "matches" if actual == reference.get("run_hash") else "mismatch", "canonical_run_hash": actual})
                except ValidationError as error:
                    availability = "missing" if error.code == "unresolved_evidence" and "did not match a run" in error.message else "unknown"
                    row.update({"availability": availability, "identity_status": "unknown", "reason": error.message})
        else:
            row.update({"availability": "unsupported", "identity_status": "not_checked", "reason": "Unsupported evidence kind"})
        results.append(row)
    return results


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
            "activity.create": "activity",
            "activity.artifact.add": "activity_artifact",
            "dataset.scope.create": "dataset_scope",
            "dataset.observation.record": "observation",
            "execution.outcome.record": "attempt",
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
    loop_id = None
    objective: str | None = None
    budget: int | None = None
    index = 0
    while index < len(rest):
        token = rest[index]
        if token in {"--objective", "--budget", "--loop"}:
            if index + 1 >= len(rest):
                raise ValidationError(f"Missing value for {token}")
            if token == "--objective":
                objective = rest[index + 1]
            elif token == "--loop":
                loop_id = rest[index + 1]
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
    payload = compile_context(state, objective, budget, repo=str(root), loop_id=loop_id)
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
