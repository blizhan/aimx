#!/usr/bin/env python3
"""Aimx Research Retro Report Helper.

Provides pure standard-library validation and publication for Retro reports.
Schema version: 1
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import string
from typing import Any

SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class ReportValidationError(Exception):
    def __init__(self, message: str, path: str = "", code: str = "invalid_report") -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.code = code


def _json_duplicate_key_checker(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for key, value in pairs:
        if key in res:
            raise ReportValidationError(f"Duplicate key in JSON: {key!r}", path=key)
        res[key] = value
    return res


def load_report_file(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        raise ReportValidationError(f"Cannot read file: {e}", code="io_error") from e

    try:
        data = json.loads(content, object_pairs_hook=_json_duplicate_key_checker)
    except ReportValidationError:
        raise
    except json.JSONDecodeError as e:
        raise ReportValidationError(f"JSON decode error: {e}", code="invalid_report") from e
    return data


def _check_fields(
    d: dict[str, Any], allowed: set[str], path: str, required: set[str] | None = None
) -> None:
    required_fields = allowed if required is None else required
    for k in sorted(required_fields - d.keys()):
        field_path = f"{path}.{k}" if path else k
        raise ReportValidationError(f"Missing required field: {k!r}", path=field_path)
    for k in d.keys():
        if k not in allowed:
            raise ReportValidationError(f"Unknown field: {k!r} in {path}", path=f"{path}.{k}" if path else k)

def _check_string(v: Any, path: str, allow_empty: bool = False, allow_null: bool = False) -> None:
    if v is None and allow_null:
        return
    if not isinstance(v, str):
        raise ReportValidationError(f"Field {path} must be string, got {type(v).__name__}", path=path)
    if not allow_empty and not v.strip():
        raise ReportValidationError(f"Field {path} cannot be empty string", path=path)


def _check_int(v: Any, path: str, allow_null: bool = False, min_val: int | None = None) -> None:
    if v is None and allow_null:
        return
    if isinstance(v, bool) or not isinstance(v, int):
        raise ReportValidationError(f"Field {path} must be integer, got {type(v).__name__}", path=path)
    if min_val is not None and v < min_val:
        raise ReportValidationError(f"Field {path} must be >= {min_val}, got {v}", path=path)


def _check_iso_timestamp(v: Any, path: str, allow_null: bool = False) -> datetime | None:
    if v is None and allow_null:
        return None
    _check_string(v, path)
    try:
        # Check finite representation
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("Missing timezone")
        return dt
    except Exception as e:
        raise ReportValidationError(f"Field {path} must be ISO 8601 timestamp with timezone: {e}", path=path) from e


def _check_finite_numbers(obj: Any, path: str = "") -> None:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ReportValidationError(f"Non-finite float at {path}", path=path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite_numbers(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _check_finite_numbers(item, f"{path}[{i}]")


def _check_statement(
    stmt: Any, path: str, evidence_map: dict[str, dict[str, Any]]
) -> None:
    if not isinstance(stmt, dict):
        raise ReportValidationError(f"Statement at {path} must be object", path=path)
    _check_fields(stmt, {"text", "basis", "evidence_ids"}, path)
    _check_string(stmt.get("text"), f"{path}.text")
    basis = stmt.get("basis")
    if basis not in {"observed", "reported", "inferred", "unknown"}:
        raise ReportValidationError(f"Invalid statement basis {basis!r} at {path}.basis", path=f"{path}.basis")
    ev_ids = stmt.get("evidence_ids")
    if not isinstance(ev_ids, list):
        raise ReportValidationError(f"Statement evidence_ids at {path} must be array", path=f"{path}.evidence_ids")

    # Check that all evidence_ids resolve
    for ev_id in ev_ids:
        if ev_id not in evidence_map:
            raise ReportValidationError(f"Statement at {path} references unknown evidence ID {ev_id!r}", path=path)

    if basis == "observed":
        # Observed requires at least one primary read source
        has_primary_read = any(
            evidence_map[eid].get("availability") == "read" and evidence_map[eid].get("provenance") == "primary"
            for eid in ev_ids
        )
        if not has_primary_read:
            raise ReportValidationError(
                f"Observed statement at {path} requires at least one read primary evidence item",
                path=path,
            )
    elif basis == "reported":
        # Reported requires at least one read source
        has_read = any(evidence_map[eid].get("availability") == "read" for eid in ev_ids)
        if not has_read:
            raise ReportValidationError(
                f"Reported statement at {path} requires at least one read evidence item",
                path=path,
            )
    elif basis == "inferred":
        has_read = any(evidence_map[eid].get("availability") == "read" for eid in ev_ids)
        if not has_read:
            raise ReportValidationError(
                f"Inferred statement at {path} requires at least one read evidence item",
                path=path,
            )


def validate_report_dict(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ReportValidationError("Top level report must be a JSON object")

    _check_finite_numbers(data)

    allowed_top = {
        "schema_version",
        "report_id",
        "created_at",
        "mode",
        "scope",
        "evidence",
        "decisions",
        "knowledge_updates",
        "reflections",
        "next_step",
        "limitations",
        "aim_views",
    }
    _check_fields(data, allowed_top, "", required=allowed_top - {"aim_views"})

    _check_int(data.get("schema_version"), "schema_version")
    if data.get("schema_version") != 1:
        raise ReportValidationError(f"Unsupported schema_version: {data.get('schema_version')}", path="schema_version")

    report_id = data.get("report_id")
    _check_string(report_id, "report_id")
    if not SAFE_ID_RE.match(report_id):
        raise ReportValidationError(f"Invalid report_id {report_id!r}: must match ^[a-zA-Z0-9_-]+$", path="report_id")

    _check_iso_timestamp(data.get("created_at"), "created_at")

    mode = data.get("mode")
    if mode not in {"standalone", "between_rounds"}:
        raise ReportValidationError(f"Invalid mode {mode!r}: must be 'standalone' or 'between_rounds'", path="mode")

    # Scope
    scope = data.get("scope")
    if not isinstance(scope, dict):
        raise ReportValidationError("Scope must be an object", path="scope")
    allowed_scope = {
        "repo",
        "objective",
        "selection",
        "read_started_at",
        "read_finished_at",
        "revision_start",
        "revision_end",
        "loop_id",
        "rounds",
        "constraints",
        "coverage",
    }
    _check_fields(scope, allowed_scope, "scope")
    _check_string(scope.get("repo"), "scope.repo")
    _check_string(scope.get("objective"), "scope.objective", allow_null=True)
    _check_string(scope.get("selection"), "scope.selection")
    t_start = _check_iso_timestamp(scope.get("read_started_at"), "scope.read_started_at")
    t_fin = _check_iso_timestamp(scope.get("read_finished_at"), "scope.read_finished_at")
    if t_start and t_fin and t_fin < t_start:
        raise ReportValidationError(
            f"scope.read_finished_at ({t_fin}) is before scope.read_started_at ({t_start})",
            path="scope.read_finished_at",
        )
    _check_int(scope.get("revision_start"), "scope.revision_start", allow_null=True, min_val=0)
    _check_int(scope.get("revision_end"), "scope.revision_end", allow_null=True, min_val=0)
    _check_string(scope.get("loop_id"), "scope.loop_id", allow_null=True)
    rounds = scope.get("rounds")
    if not isinstance(rounds, list) or not all(isinstance(r, int) and not isinstance(r, bool) and r > 0 for r in rounds):
        raise ReportValidationError("scope.rounds must be array of positive integers", path="scope.rounds")
    constraints = scope.get("constraints")
    if not isinstance(constraints, list) or not all(isinstance(c, str) for c in constraints):
        raise ReportValidationError("scope.constraints must be array of strings", path="scope.constraints")
    _check_string(scope.get("coverage"), "scope.coverage")

    # Evidence
    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        raise ReportValidationError("evidence must be array", path="evidence")
    evidence_map: dict[str, dict[str, Any]] = {}
    allowed_ev = {
        "id",
        "kind",
        "locator",
        "selector",
        "captured_at",
        "source_time",
        "source_revision",
        "availability",
        "provenance",
        "summary",
    }
    for i, ev in enumerate(evidence):
        p = f"evidence[{i}]"
        if not isinstance(ev, dict):
            raise ReportValidationError(f"Evidence item at {p} must be object", path=p)
        _check_fields(ev, allowed_ev, p)
        eid = ev.get("id")
        _check_string(eid, f"{p}.id")
        if eid in evidence_map:
            raise ReportValidationError(f"Duplicate evidence ID {eid!r}", path=f"{p}.id")
        evidence_map[eid] = ev

        kind = ev.get("kind")
        if kind not in {"research_state", "aim_run", "artifact", "user_input"}:
            raise ReportValidationError(f"Invalid kind {kind!r} at {p}.kind", path=f"{p}.kind")
        _check_string(ev.get("locator"), f"{p}.locator")
        _check_string(ev.get("selector"), f"{p}.selector", allow_empty=True)
        _check_iso_timestamp(ev.get("captured_at"), f"{p}.captured_at")
        _check_iso_timestamp(ev.get("source_time"), f"{p}.source_time", allow_null=True)
        _check_int(ev.get("source_revision"), f"{p}.source_revision", allow_null=True, min_val=0)
        avail = ev.get("availability")
        if avail not in {"read", "missing", "unreadable"}:
            raise ReportValidationError(f"Invalid availability {avail!r} at {p}.availability", path=f"{p}.availability")
        prov = ev.get("provenance")
        if prov not in {"primary", "researcher_summary"}:
            raise ReportValidationError(f"Invalid provenance {prov!r} at {p}.provenance", path=f"{p}.provenance")
        _check_string(ev.get("summary"), f"{p}.summary")

    # Decisions
    decisions = data.get("decisions")
    if not isinstance(decisions, list):
        raise ReportValidationError("decisions must be array", path="decisions")
    decision_map: dict[str, dict[str, Any]] = {}
    allowed_dec = {
        "id",
        "position",
        "source_time",
        "available_information",
        "agent_awareness",
        "alternatives",
        "constraints",
        "choice",
        "recorded_rationale",
        "expectation",
        "execution",
        "comparability",
        "observation",
        "interpretation",
        "subsequent_action",
    }
    for i, dec in enumerate(decisions):
        p = f"decisions[{i}]"
        if not isinstance(dec, dict):
            raise ReportValidationError(f"Decision item at {p} must be object", path=p)
        _check_fields(dec, allowed_dec, p)
        did = dec.get("id")
        _check_string(did, f"{p}.id")
        if did in decision_map:
            raise ReportValidationError(f"Duplicate decision ID {did!r}", path=f"{p}.id")
        decision_map[did] = dec
        _check_string(dec.get("position"), f"{p}.position")
        _check_iso_timestamp(dec.get("source_time"), f"{p}.source_time", allow_null=True)

        for arr_field in ["available_information", "alternatives", "constraints"]:
            arr = dec.get(arr_field)
            if not isinstance(arr, list):
                raise ReportValidationError(f"{p}.{arr_field} must be array", path=f"{p}.{arr_field}")
            for j, stmt in enumerate(arr):
                _check_statement(stmt, f"{p}.{arr_field}[{j}]", evidence_map)

        for single_field in [
            "agent_awareness",
            "choice",
            "recorded_rationale",
            "expectation",
            "execution",
            "comparability",
            "observation",
            "interpretation",
            "subsequent_action",
        ]:
            _check_statement(dec.get(single_field), f"{p}.{single_field}", evidence_map)

    # Optional Aim UI views
    if "aim_views" in data:
        aim_views = data["aim_views"]
        if not isinstance(aim_views, list):
            raise ReportValidationError("aim_views must be array", path="aim_views")
        valid_pages = {"runs", "metrics", "params", "scatters", "images", "figures", "text", "distributions"}
        allowed_view = {
            "id", "page", "purpose", "query", "group_by", "color_by", "axes",
            "evidence_ids", "decision_ids", "url",
        }
        required_view = {"id", "page", "purpose", "query", "url"}
        view_ids: set[str] = set()
        for i, view in enumerate(aim_views):
            p = f"aim_views[{i}]"
            if not isinstance(view, dict):
                raise ReportValidationError(f"Aim view at {p} must be object", path=p)
            _check_fields(view, allowed_view, p, required=required_view)
            view_id = view.get("id")
            _check_string(view_id, f"{p}.id")
            if view_id in view_ids:
                raise ReportValidationError(f"Duplicate Aim view ID {view_id!r}", path=f"{p}.id")
            view_ids.add(view_id)

            page = view.get("page")
            if not isinstance(page, str) or page not in valid_pages:
                raise ReportValidationError(f"Invalid page {page!r} at {p}.page", path=f"{p}.page")
            _check_string(view.get("purpose"), f"{p}.purpose")
            _check_string(view.get("query"), f"{p}.query")
            _check_string(view.get("url"), f"{p}.url", allow_null=True)

            for field in ("group_by", "color_by", "axes"):
                if field not in view:
                    continue
                values = view[field]
                if not isinstance(values, list):
                    raise ReportValidationError(f"{p}.{field} must be array of strings", path=f"{p}.{field}")
                for j, value in enumerate(values):
                    _check_string(value, f"{p}.{field}[{j}]")
            if "axes" in view and page not in {"params", "scatters"}:
                raise ReportValidationError(f"{p}.axes is only valid for params or scatters pages", path=f"{p}.axes")

            for field, known_ids in (("evidence_ids", evidence_map), ("decision_ids", decision_map)):
                if field not in view:
                    continue
                references = view[field]
                if not isinstance(references, list):
                    raise ReportValidationError(f"{p}.{field} must be array of strings", path=f"{p}.{field}")
                for j, reference in enumerate(references):
                    _check_string(reference, f"{p}.{field}[{j}]")
                    if reference not in known_ids:
                        raise ReportValidationError(f"{p}.{field} references unknown ID {reference!r}", path=f"{p}.{field}[{j}]")
    # Knowledge Updates
    kupdates = data.get("knowledge_updates")
    if not isinstance(kupdates, list):
        raise ReportValidationError("knowledge_updates must be array", path="knowledge_updates")
    allowed_ku = {"change", "statement", "decision_ids"}
    for i, ku in enumerate(kupdates):
        p = f"knowledge_updates[{i}]"
        if not isinstance(ku, dict):
            raise ReportValidationError(f"Knowledge update at {p} must be object", path=p)
        _check_fields(ku, allowed_ku, p)
        change = ku.get("change")
        if change not in {"added", "weakened", "unresolved", "retained"}:
            raise ReportValidationError(f"Invalid change {change!r} at {p}.change", path=f"{p}.change")
        _check_statement(ku.get("statement"), f"{p}.statement", evidence_map)
        dids = ku.get("decision_ids")
        if not isinstance(dids, list):
            raise ReportValidationError(f"{p}.decision_ids must be array", path=f"{p}.decision_ids")
        for did in dids:
            if did not in decision_map:
                raise ReportValidationError(f"{p}.decision_ids references unknown decision {did!r}", path=p)

    # Reflections (0 to 3)
    reflections = data.get("reflections")
    if not isinstance(reflections, list):
        raise ReportValidationError("reflections must be array", path="reflections")
    if len(reflections) > 3:
        raise ReportValidationError(f"reflections array cannot exceed 3 items, got {len(reflections)}", path="reflections")
    reflection_map: dict[str, dict[str, Any]] = {}
    allowed_ref = {
        "id",
        "dimensions",
        "decision_ids",
        "observation",
        "causal_assessment",
        "explanation",
        "alternative_explanations",
        "counterevidence",
        "behavior",
        "action",
        "conditions",
        "exceptions",
        "validation",
        "validation_status",
        "validation_evidence_ids",
    }
    valid_dims = {"hypothesis", "strategy", "experiment", "execution", "reasoning", "decision"}
    for i, ref in enumerate(reflections):
        p = f"reflections[{i}]"
        if not isinstance(ref, dict):
            raise ReportValidationError(f"Reflection item at {p} must be object", path=p)
        _check_fields(ref, allowed_ref, p)
        rid = ref.get("id")
        _check_string(rid, f"{p}.id")
        if rid in reflection_map:
            raise ReportValidationError(f"Duplicate reflection ID {rid!r}", path=f"{p}.id")
        reflection_map[rid] = ref

        dims = ref.get("dimensions")
        if not isinstance(dims, list) or not dims or not all(d in valid_dims for d in dims):
            raise ReportValidationError(f"Invalid reflection dimensions at {p}.dimensions", path=f"{p}.dimensions")

        dids = ref.get("decision_ids")
        if not isinstance(dids, list):
            raise ReportValidationError(f"{p}.decision_ids must be array", path=f"{p}.decision_ids")
        for did in dids:
            if did not in decision_map:
                raise ReportValidationError(f"{p}.decision_ids references unknown decision {did!r}", path=p)

        _check_statement(ref.get("observation"), f"{p}.observation", evidence_map)

        ca = ref.get("causal_assessment")
        if ca not in {"supported", "plausible", "undetermined"}:
            raise ReportValidationError(f"Invalid causal_assessment {ca!r} at {p}.causal_assessment", path=f"{p}.causal_assessment")
        _check_statement(ref.get("explanation"), f"{p}.explanation", evidence_map)

        for stmt_arr_f in ["alternative_explanations", "counterevidence"]:
            sarr = ref.get(stmt_arr_f)
            if not isinstance(sarr, list):
                raise ReportValidationError(f"{p}.{stmt_arr_f} must be array", path=f"{p}.{stmt_arr_f}")
            for j, s in enumerate(sarr):
                _check_statement(s, f"{p}.{stmt_arr_f}[{j}]", evidence_map)

        beh = ref.get("behavior")
        if beh not in {"retain", "adjust"}:
            raise ReportValidationError(f"Invalid behavior {beh!r} at {p}.behavior", path=f"{p}.behavior")
        _check_string(ref.get("action"), f"{p}.action")

        for sarr_f in ["conditions", "exceptions"]:
            arr = ref.get(sarr_f)
            if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
                raise ReportValidationError(f"{p}.{sarr_f} must be array of strings", path=f"{p}.{sarr_f}")

        _check_string(ref.get("validation"), f"{p}.validation")
        vstatus = ref.get("validation_status")
        if vstatus not in {"untested", "tested"}:
            raise ReportValidationError(f"Invalid validation_status {vstatus!r} at {p}.validation_status", path=f"{p}.validation_status")
        vev_ids = ref.get("validation_evidence_ids")
        if not isinstance(vev_ids, list):
            raise ReportValidationError(f"{p}.validation_evidence_ids must be array", path=f"{p}.validation_evidence_ids")
        for veid in vev_ids:
            if veid not in evidence_map:
                raise ReportValidationError(f"{p}.validation_evidence_ids references unknown evidence {veid!r}", path=p)
        if vstatus == "tested":
            has_read = any(evidence_map[veid].get("availability") == "read" for veid in vev_ids)
            if not has_read:
                raise ReportValidationError(f"{p}: Tested validation requires at least one read evidence item", path=p)

    # NextStep
    next_step = data.get("next_step")
    if not isinstance(next_step, dict):
        raise ReportValidationError("next_step must be object", path="next_step")
    allowed_ns = {"priority_question", "disposition", "rationale", "recommended_option_id", "options", "recheck_conditions"}
    _check_fields(next_step, allowed_ns, "next_step")
    _check_string(next_step.get("priority_question"), "next_step.priority_question")
    disp = next_step.get("disposition")
    if disp not in {"recommend", "gather_evidence", "stop", "clarify"}:
        raise ReportValidationError(f"Invalid disposition {disp!r} at next_step.disposition", path="next_step.disposition")
    _check_statement(next_step.get("rationale"), "next_step.rationale", evidence_map)

    options = next_step.get("options")
    if not isinstance(options, list):
        raise ReportValidationError("next_step.options must be array", path="next_step.options")
    if len(options) > 3:
        raise ReportValidationError(f"next_step.options cannot exceed 3 items, got {len(options)}", path="next_step.options")

    option_map: dict[str, dict[str, Any]] = {}
    allowed_opt = {
        "id",
        "kind",
        "question",
        "action",
        "rationale",
        "controls",
        "cost",
        "prerequisites",
        "applicability",
        "switch_conditions",
        "outcome_branches",
        "independent_value",
    }
    for i, opt in enumerate(options):
        p = f"next_step.options[{i}]"
        if not isinstance(opt, dict):
            raise ReportValidationError(f"Option item at {p} must be object", path=p)
        _check_fields(opt, allowed_opt, p)
        oid = opt.get("id")
        _check_string(oid, f"{p}.id")
        if oid in option_map:
            raise ReportValidationError(f"Duplicate option ID {oid!r}", path=f"{p}.id")
        option_map[oid] = opt

        okind = opt.get("kind")
        if okind not in {"experiment", "inspection", "continue", "stop", "clarify"}:
            raise ReportValidationError(f"Invalid option kind {okind!r} at {p}.kind", path=f"{p}.kind")
        _check_string(opt.get("question"), f"{p}.question")
        _check_string(opt.get("action"), f"{p}.action")
        _check_statement(opt.get("rationale"), f"{p}.rationale", evidence_map)

        for sarr_f in ["controls", "prerequisites", "applicability", "switch_conditions"]:
            arr = opt.get(sarr_f)
            if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
                raise ReportValidationError(f"{p}.{sarr_f} must be array of strings", path=f"{p}.{sarr_f}")

        _check_string(opt.get("cost"), f"{p}.cost", allow_null=True)
        _check_string(opt.get("independent_value"), f"{p}.independent_value", allow_null=True)

        obranches = opt.get("outcome_branches")
        if not isinstance(obranches, list):
            raise ReportValidationError(f"{p}.outcome_branches must be array", path=f"{p}.outcome_branches")
        for j, br in enumerate(obranches):
            bp = f"{p}.outcome_branches[{j}]"
            if not isinstance(br, dict):
                raise ReportValidationError(f"{bp} must be object", path=bp)
            _check_fields(br, {"result", "next_decision"}, bp)
            _check_string(br.get("result"), f"{bp}.result")
            _check_string(br.get("next_decision"), f"{bp}.next_decision")

        # Experiments require at least 2 branches unless independent_value is provided
        if okind == "experiment":
            ind_val = opt.get("independent_value")
            if (not ind_val or not ind_val.strip()) and len(obranches) < 2:
                raise ReportValidationError(
                    f"Experiment option {oid!r} requires at least 2 outcome_branches or an independent_value explanation",
                    path=p,
                )

    rec_opt_id = next_step.get("recommended_option_id")
    if disp in {"recommend", "gather_evidence"}:
        if not rec_opt_id or rec_opt_id not in option_map:
            raise ReportValidationError(
                f"Disposition {disp!r} requires recommended_option_id pointing to a valid option",
                path="next_step.recommended_option_id",
            )
    else:
        if rec_opt_id is not None and rec_opt_id not in option_map:
            raise ReportValidationError(
                f"recommended_option_id {rec_opt_id!r} does not resolve to an option",
                path="next_step.recommended_option_id",
            )

    recheck = next_step.get("recheck_conditions")
    if not isinstance(recheck, list) or not recheck or not all(isinstance(x, str) and x.strip() for x in recheck):
        raise ReportValidationError(
            "next_step.recheck_conditions must be a nonempty array of nonempty strings",
            path="next_step.recheck_conditions",
        )

    # Limitations
    limitations = data.get("limitations")
    if not isinstance(limitations, list) or not all(isinstance(lim, str) and lim.strip() for lim in limitations):
        raise ReportValidationError("limitations must be array of nonempty strings", path="limitations")

    return {
        "ok": True,
        "schema_version": 1,
        "report_id": report_id,
    }


def validate_report_dict_safe(data: dict[str, Any]) -> dict[str, Any]:
    try:
        return validate_report_dict(data)
    except ReportValidationError as e:
        return {
            "ok": False,
            "error": {
                "code": e.code,
                "message": e.message,
                "path": e.path,
            },
        }


def _markdown_values(value: Any) -> Any:
    """Copy report text as literal Markdown content, without changing the JSON."""
    if isinstance(value, str):
        # Entities display literally without introducing Markdown or raw HTML syntax.
        return "".join(
            f"&#{ord(char)};" if char in string.punctuation else
            "<br>" if char in "\r\n" else char
            for char in value
        )
    if isinstance(value, list):
        return [_markdown_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _markdown_values(item) for key, item in value.items()}
    return value


def _render_statement(statement: dict[str, Any]) -> str:
    references = ", ".join(statement["evidence_ids"]) or "none"
    return f"[{statement['basis']}] {statement['text']} (ev: {references})"


def _code_fence(text: str) -> list[str]:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence, text, fence]


def render_markdown(data: dict[str, Any]) -> str:
    raw_aim_views = data.get("aim_views", [])
    data = _markdown_values(data)
    lines: list[str] = []
    lines.append(f"# Research Retrospective: {data['report_id']}")
    lines.append("")
    lines.append(f"- **Mode**: {data['mode']}")
    lines.append(f"- **Created At**: {data['created_at']}")
    lines.append(f"- **Schema Version**: {data['schema_version']}")
    lines.append("")

    # Scope
    scope = data["scope"]
    lines.append("## Scope")
    lines.append(f"- **Repository**: {scope['repo']}")
    lines.append(f"- **Objective**: {scope['objective'] or '_None specified_'}")
    lines.append(f"- **Selection**: {scope['selection']}")
    lines.append(f"- **Read Window**: {scope['read_started_at']} to {scope['read_finished_at']}")
    lines.append(f"- **Revisions**: Start {scope['revision_start']}, End {scope['revision_end']}")
    lines.append(f"- **Loop ID**: {scope['loop_id'] or 'None'}")
    lines.append(f"- **Rounds**: {', '.join(str(r) for r in scope['rounds']) if scope['rounds'] else 'None'}")
    lines.append(f"- **Constraints**: {', '.join(scope['constraints']) if scope['constraints'] else 'None'}")
    lines.append(f"- **Coverage**: {scope['coverage']}")
    lines.append("")

    # Evidence
    lines.append("## Evidence")
    for ev in data["evidence"]:
        lines.append(f"### Evidence {ev['id']} ({ev['kind']})")
        lines.append(f"- **Locator**: {ev['locator']}")
        if ev.get("selector"):
            lines.append(f"- **Selector**: {ev['selector']}")
        lines.append(f"- **Captured**: {ev['captured_at']} | **Source Time**: {ev['source_time'] or 'Unknown'} | **Revision**: {ev['source_revision']}")
        lines.append(f"- **Status**: {ev['availability']} / {ev['provenance']}")
        lines.append(f"- **Summary**: {ev['summary']}")
        lines.append("")

    # Decisions
    lines.append("## Decisions")
    for dec in data["decisions"]:
        lines.append(f"### Decision {dec['id']}: {dec['position']}")
        if dec.get("source_time"):
            lines.append(f"_Time: {dec['source_time']}_")
        lines.append("#### Available Information")
        for item in dec["available_information"]:
            lines.append(f"- {_render_statement(item)}")
        lines.append(f"#### Agent Awareness\n- {_render_statement(dec['agent_awareness'])}")
        lines.append("#### Alternatives Considered")
        for item in dec["alternatives"]:
            lines.append(f"- {_render_statement(item)}")
        lines.append("#### Constraints")
        for item in dec["constraints"]:
            lines.append(f"- {_render_statement(item)}")
        lines.append(f"#### Choice\n- {_render_statement(dec['choice'])}")
        lines.append(f"#### Recorded Rationale\n- {_render_statement(dec['recorded_rationale'])}")
        lines.append(f"#### Expectation\n- {_render_statement(dec['expectation'])}")
        lines.append(f"#### Execution\n- {_render_statement(dec['execution'])}")
        lines.append(f"#### Comparability\n- {_render_statement(dec['comparability'])}")
        lines.append(f"#### Observation\n- {_render_statement(dec['observation'])}")
        lines.append(f"#### Interpretation\n- {_render_statement(dec['interpretation'])}")
        lines.append(f"#### Subsequent Action\n- {_render_statement(dec['subsequent_action'])}")
        lines.append("")

    # Knowledge Updates
    lines.append("## Knowledge Updates")
    for ku in data["knowledge_updates"]:
        lines.append(f"- **[{ku['change'].upper()}]**: {_render_statement(ku['statement'])} (dec: {', '.join(ku['decision_ids'])})")
    lines.append("")

    # Reflections
    lines.append("## Reflections")
    if not data["reflections"]:
        lines.append("_No reflections recorded._")
        lines.append("")
    else:
        for ref in data["reflections"]:
            lines.append(f"### Reflection {ref['id']}")
            lines.append(f"- **Dimensions**: {', '.join(ref['dimensions'])}")
            lines.append(f"- **Decisions**: {', '.join(ref['decision_ids'])}")
            lines.append(f"- **Observation**: {_render_statement(ref['observation'])}")
            lines.append(f"- **Causal Assessment**: {ref['causal_assessment']}")
            lines.append(f"- **Explanation**: {_render_statement(ref['explanation'])}")
            if ref["alternative_explanations"]:
                lines.append("- **Alternative Explanations**:")
                for a in ref["alternative_explanations"]:
                    lines.append(f"  - {_render_statement(a)}")
            if ref["counterevidence"]:
                lines.append("- **Counterevidence**:")
                for c in ref["counterevidence"]:
                    lines.append(f"  - {_render_statement(c)}")
            lines.append(f"- **Adjustment**: {ref['behavior']} -> {ref['action']}")
            lines.append(f"- **Conditions**: {', '.join(ref['conditions']) if ref['conditions'] else 'None'}")
            lines.append(f"- **Exceptions**: {', '.join(ref['exceptions']) if ref['exceptions'] else 'None'}")
            lines.append(f"- **Validation ({ref['validation_status']})**: {ref['validation']}")
            lines.append(f"- **Validation Evidence**: {', '.join(ref['validation_evidence_ids']) or 'none'}")
            lines.append("")

    # Aim UI Views
    lines.append("## Aim Views")
    aim_views = data.get("aim_views", [])
    if not aim_views:
        lines.append("_No Aim views recorded._")
        lines.append("")
    else:
        for view, raw_view in zip(aim_views, raw_aim_views):
            lines.append(f"### {view['id']}: {view['purpose']}")
            lines.append(f"- **Page**: {view['page']}")
            lines.append("- **AimQL query**:")
            lines.append("")
            lines.extend(_code_fence(raw_view["query"]))
            lines.append("")
            lines.append(f"- **Group by**: {', '.join(view.get('group_by', [])) or 'None'}")
            lines.append(f"- **Color by**: {', '.join(view.get('color_by', [])) or 'None'}")
            lines.append(f"- **Axes**: {', '.join(view.get('axes', [])) or 'None'}")
            lines.append(f"- **Evidence**: {', '.join(view.get('evidence_ids', [])) or 'None'}")
            lines.append(f"- **Decisions**: {', '.join(view.get('decision_ids', [])) or 'None'}")
            if view["url"] is None:
                lines.append(
                    f"- **Aim UI URL**: None. Open the {view['page']} page in Aim and paste the query into its search box."
                )
            elif re.fullmatch(r"https?://[^\s<>]+", raw_view["url"]):
                lines.append(f"- **Aim UI URL**: <{raw_view['url']}>")
            else:
                lines.append(f"- **Aim UI URL**: {view['url']}")
            lines.append("")

    # NextStep
    ns = data["next_step"]
    lines.append("## Next Steps & Recommendation")
    lines.append(f"### Priority Question\n{ns['priority_question']}")
    lines.append(f"\n- **Disposition**: {ns['disposition']}")
    lines.append(f"- **Rationale**: {_render_statement(ns['rationale'])}")
    lines.append(f"- **Recommended Option ID**: {ns['recommended_option_id'] or 'None'}")
    lines.append("\n#### Options")
    for opt in ns["options"]:
        lines.append(f"##### Option {opt['id']} ({opt['kind']})")
        lines.append(f"- **Question**: {opt['question']}")
        lines.append(f"- **Action**: {opt['action']}")
        lines.append(f"- **Rationale**: {_render_statement(opt['rationale'])}")
        lines.append(f"- **Controls**: {', '.join(opt['controls']) if opt['controls'] else 'None'}")
        lines.append(f"- **Cost**: {opt['cost'] or 'Unknown'}")
        lines.append(f"- **Prerequisites**: {', '.join(opt['prerequisites']) if opt['prerequisites'] else 'None'}")
        lines.append(f"- **Applicability**: {', '.join(opt['applicability']) if opt['applicability'] else 'None'}")
        lines.append(f"- **Switch Conditions**: {', '.join(opt['switch_conditions']) if opt['switch_conditions'] else 'None'}")
        if opt["outcome_branches"]:
            lines.append("- **Outcome Branches**:")
            for br in opt["outcome_branches"]:
                lines.append(f"  - If {br['result']} -> {br['next_decision']}")
        if opt.get("independent_value"):
            lines.append(f"- **Independent Value**: {opt['independent_value']}")
        lines.append("")

    lines.append("#### Recheck Conditions")
    for rc in ns["recheck_conditions"]:
        lines.append(f"- {rc}")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    for lim in data["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    return "\n".join(lines)


def _protected_directory(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == ".aim" or (
            candidate.name in {"research", "autoresearch"}
            and candidate.parent.name == ".aimx"
        ):
            return candidate
    return None


def publish_report(draft_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    data = load_report_file(draft_path)
    validate_report_dict(data)

    report_id = data["report_id"]
    repo_root = Path(data["scope"]["repo"]).resolve()
    if (_protected_directory(repo_root) is not None
            or _protected_directory(Path(data["scope"]["repo"]).absolute()) is not None):
        raise ReportValidationError(
            "scope.repo must identify a repository root outside protected metadata",
            path="scope.repo", code="unsafe_output",
        )

    if output_root is None:
        root = (repo_root / ".aimx" / "retro").resolve()
    else:
        root = Path(output_root).resolve()

    target_dir = root / report_id
    requested_root = Path(output_root).absolute() if output_root is not None else root
    protected = (
        _protected_directory(root)
        or _protected_directory(requested_root)
        or _protected_directory(target_dir.resolve())
    )
    for metadata in (".aim", ".aimx/research", ".aimx/autoresearch"):
        subtree = (repo_root / metadata).resolve()
        if root == subtree or subtree in root.parents:
            protected = subtree
    if protected is not None:
        raise ReportValidationError(
            f"Output root {root} cannot target protected directory {protected}",
            code="unsafe_output",
        )

    root.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        raise ReportValidationError(
            f"Destination report directory already exists: {target_dir}",
            code="output_exists",
        )

    json_content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    md_content = render_markdown(data)

    created_dir = False
    try:
        target_dir.mkdir(parents=False, exist_ok=False)
        created_dir = True
        json_path = target_dir / "retro.json"
        md_path = target_dir / "RETRO.md"
        json_path.write_text(json_content, encoding="utf-8")
        md_path.write_text(md_content, encoding="utf-8")
    except Exception as e:
        if created_dir:
            for f in target_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                target_dir.rmdir()
            except Exception:
                pass
        raise ReportValidationError(f"Failed to publish report: {e}", code="io_error") from e

    return {
        "ok": True,
        "report_id": report_id,
        "report_dir": str(target_dir.resolve()),
        "json_path": str((target_dir / "retro.json").resolve()),
        "markdown_path": str((target_dir / "RETRO.md").resolve()),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Aimx Retro Report Helper")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a draft retro report")
    validate_parser.add_argument("--file", required=True, type=Path, help="Path to report JSON draft")

    publish_parser = subparsers.add_parser("publish", help="Publish a retro report")
    publish_parser.add_argument("--file", required=True, type=Path, help="Path to report JSON draft")
    publish_parser.add_argument("--output-root", type=Path, default=None, help="Explicit output root directory")

    args = parser.parse_args()

    if args.subcommand == "validate":
        try:
            data = load_report_file(args.file)
            res = validate_report_dict(data)
            print(json.dumps(res, indent=2))
            sys.exit(0)
        except ReportValidationError as e:
            err_res = {
                "ok": False,
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "path": e.path,
                },
            }
            print(json.dumps(err_res, indent=2))
            sys.exit(2)
        except Exception as e:
            err_res = {
                "ok": False,
                "error": {
                    "code": "error",
                    "message": str(e),
                    "path": "",
                },
            }
            print(json.dumps(err_res, indent=2))
            sys.exit(2)
    elif args.subcommand == "publish":
        try:
            res = publish_report(args.file, output_root=args.output_root)
            print(json.dumps(res, indent=2))
            sys.exit(0)
        except ReportValidationError as e:
            err_res = {
                "ok": False,
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "path": e.path,
                },
            }
            print(json.dumps(err_res, indent=2))
            sys.exit(2)
        except Exception as e:
            err_res = {
                "ok": False,
                "error": {
                    "code": "error",
                    "message": str(e),
                    "path": "",
                },
            }
            print(json.dumps(err_res, indent=2))
            sys.exit(2)


if __name__ == "__main__":
    main()
