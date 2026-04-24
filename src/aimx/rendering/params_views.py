"""Renderers for `aimx query params` output."""

from __future__ import annotations

import io
import json
import shutil
from typing import Any

from rich.console import Console
from rich.table import Table

from aimx.aim_bridge.run_params import RunParams, default_param_keys
from aimx.rendering import colors

DEFAULT_PARAM_COLUMN_LIMIT = 6


def _short_hash(value: str) -> str:
    return value[:8]


def _display(value: Any, *, max_len: int = 48) -> str:
    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    if len(text) > max_len:
        return f"{text[: max_len - 1]}…"
    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _keys_for_display(rows: list[RunParams], limit: int | None = DEFAULT_PARAM_COLUMN_LIMIT) -> tuple[tuple[str, ...], int]:
    selected = next((row.selected_keys for row in rows if row.selected_keys), ())
    keys = selected or default_param_keys(rows)
    if limit is not None and not selected and len(keys) > limit:
        return keys[:limit], len(keys) - limit
    return keys, 0


def render_params_rich_table(
    rows: list[RunParams],
    header_info: dict[str, Any],
    *,
    no_color: bool = False,
) -> str:
    width = 120 if no_color else shutil.get_terminal_size(fallback=(120, 24)).columns
    buf = io.StringIO()
    console = Console(
        file=buf,
        no_color=no_color,
        force_terminal=not no_color,
        width=width,
        highlight=False,
    )

    expr = header_info.get("expression", "")
    repo = header_info.get("repo", "")
    target = header_info.get("target", "params")
    total = len(rows)
    console.print(
        f"[{colors.HEADER}]Repo:[/] {repo}  [{colors.HEADER}]·[/]  "
        f"[{colors.NUMBER_EMPH}]{total}[/] [{colors.HEADER}]match{'es' if total != 1 else ''}[/]  "
        f"[{colors.HEADER}]·[/]  [{colors.HEADER}]{target} where[/] {expr}"
    )

    if not rows:
        return buf.getvalue()

    keys, omitted = _keys_for_display(rows)
    table = Table(
        show_header=True,
        header_style=colors.HEADER,
        box=None,
        pad_edge=True,
        show_edge=False,
        padding=(0, 1),
    )
    table.add_column("RUN", style=colors.RUN_HASH, no_wrap=True)
    table.add_column("EXPERIMENT", style=colors.EXPERIMENT, no_wrap=True)
    table.add_column("NAME", style=colors.METRIC_NAME, no_wrap=True)
    for key in keys:
        table.add_column(key, style=colors.CONTEXT_VAL)
    if not keys:
        table.add_column("PARAMS", style=colors.CONTEXT_VAL)

    for row in rows:
        cells = [
            _short_hash(row.run.hash),
            row.run.experiment or "",
            row.run.name or "",
        ]
        if keys:
            for key in keys:
                cells.append(_display(row.params[key]) if key in row.params else "-")
        else:
            cells.append("no params")
        table.add_row(*cells)

    console.print(table)
    if omitted:
        console.print(f"[{colors.HEADER}]... omitted {omitted} parameter columns; use --json for all[/]")
    return buf.getvalue()


def render_params_oneline(rows: list[RunParams], header_info: dict[str, Any]) -> str:
    repo = header_info.get("repo", "")
    keys, _ = _keys_for_display(rows, limit=None)
    lines: list[str] = []
    for row in rows:
        cells = [
            repo,
            _short_hash(row.run.hash),
            row.run.experiment or "",
            row.run.name or "",
        ]
        for key in keys:
            value = _display(row.params[key]) if key in row.params else "-"
            cells.append(f"{key}={value}")
        if not keys:
            cells.append("params=-")
        lines.append("\t".join(cells))
    return "\n".join(lines)


def render_params_json(rows: list[RunParams], header_info: dict[str, Any]) -> str:
    selected = tuple(header_info.get("param_keys") or ()) or next(
        (row.selected_keys for row in rows if row.selected_keys), ()
    )
    param_keys = selected or default_param_keys(rows)
    payload: dict[str, Any] = {
        "target": header_info.get("target", "params"),
        "repo": header_info.get("repo", ""),
        "expression": header_info.get("expression", ""),
        "runs_count": len(rows),
        "param_keys": list(param_keys),
        "runs": [
            {
                "hash": row.run.hash,
                "experiment": row.run.experiment,
                "name": row.run.name,
                "params": _jsonable(row.params),
                "missing_params": list(row.missing_keys),
            }
            for row in rows
        ],
    }
    return json.dumps(payload)
