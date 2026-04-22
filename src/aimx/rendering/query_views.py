from __future__ import annotations

import io
import json
import math
import shutil
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from aimx.aim_bridge.metric_stats import MetricSeries, RunMeta
from aimx.rendering import colors


def _fmt_float(v: float) -> str:
    if math.isnan(v):
        return "—"
    if abs(v) >= 1e5 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.3e}"
    return f"{v:.4f}"


def _fmt_context(ctx: dict[str, Any]) -> str:
    if not ctx:
        return ""
    return " ".join(f"{k}={v}" for k, v in sorted(ctx.items()))


def _short_hash(h: str) -> str:
    return h[:8]


def _run_label(run: RunMeta) -> str:
    label = _short_hash(run.hash)
    if run.experiment:
        label = f"{label}  {run.experiment}"
    elif run.name:
        label = f"{label}  {run.name}"
    return label


def render_rich_table(
    groups: list[tuple[RunMeta, list[MetricSeries]]],
    header_info: dict[str, Any],
    *,
    no_color: bool = False,
) -> str:
    """Render query results as a rich-formatted table string.

    When ``no_color=True``, output is plain text (suitable for non-TTY).
    When color is on, ANSI escape codes are embedded in the returned string.
    """
    width = 120 if no_color else shutil.get_terminal_size(fallback=(120, 24)).columns
    buf = io.StringIO()
    console = Console(
        file=buf,
        no_color=no_color,
        force_terminal=not no_color,
        width=width,
        highlight=False,
    )

    total = sum(len(ms) for _, ms in groups)
    expr = header_info.get("expression", "")
    repo = header_info.get("repo", "")
    target = header_info.get("target", "")

    # Compact header
    if header_info.get("verbose"):
        console.print(f"[{colors.HEADER}]target:[/] {target}  [{colors.HEADER}]repo:[/] {repo}")
        console.print(f"[{colors.HEADER}]expression:[/] {expr}")
    console.print(
        f"[{colors.HEADER}]Repo:[/] {repo}  [{colors.HEADER}]·[/]  "
        f"[{colors.NUMBER_EMPH}]{total}[/] [{colors.HEADER}]match{'es' if total != 1 else ''}[/]  "
        f"[{colors.HEADER}]·[/]  [{colors.HEADER}]{target} where[/] {expr}"
    )

    for run, series_list in groups:
        console.print()
        label = Text()
        label.append("▌ ", style=colors.RULE_LINE)
        label.append(_short_hash(run.hash), style=colors.RUN_HASH)
        if run.experiment:
            label.append(f"  {run.experiment}", style=colors.EXPERIMENT)
        elif run.name:
            label.append(f"  {run.name}", style=colors.EXPERIMENT)
        console.print(label)

        table = Table(
            show_header=True,
            header_style=colors.HEADER,
            box=None,
            pad_edge=True,
            show_edge=False,
            padding=(0, 1),
        )
        table.add_column("NAME", style=colors.METRIC_NAME, no_wrap=True)
        table.add_column("CONTEXT", style=colors.CONTEXT_VAL, no_wrap=True)
        table.add_column("STEPS", justify="right")
        table.add_column("LAST", justify="right", style=colors.NUMBER_EMPH)
        table.add_column("MIN", justify="right", style=colors.NUMBER_DIM)
        table.add_column("@STEP", justify="right", style=colors.HEADER)
        table.add_column("MAX", justify="right", style=colors.NUMBER_DIM)
        table.add_column("@STEP", justify="right", style=colors.HEADER)

        for series in series_list:
            ctx_str = _fmt_context(series.context)
            last_val, last_step = series.last
            min_val, min_step = series.min
            max_val, max_step = series.max
            table.add_row(
                series.name,
                ctx_str,
                str(series.count),
                _fmt_float(last_val),
                _fmt_float(min_val),
                str(min_step) if min_step >= 0 else "—",
                _fmt_float(max_val),
                str(max_step) if max_step >= 0 else "—",
            )

        console.print(table)

    return buf.getvalue()


def render_image_rich_table(
    image_rows: list[dict[str, Any]],
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

    total = len(image_rows)
    expr = header_info.get("expression", "")
    repo = header_info.get("repo", "")
    target = header_info.get("target", "images")

    console.print(
        f"[{colors.HEADER}]Repo:[/] {repo}  [{colors.HEADER}]·[/]  "
        f"[{colors.NUMBER_EMPH}]{total}[/] [{colors.HEADER}]match{'es' if total != 1 else ''}[/]  "
        f"[{colors.HEADER}]·[/]  [{colors.HEADER}]{target} where[/] {expr}"
    )

    table = Table(
        show_header=True,
        header_style=colors.HEADER,
        box=None,
        pad_edge=True,
        show_edge=False,
        padding=(0, 1),
    )
    table.add_column("RUN", style=colors.RUN_HASH, no_wrap=True)
    table.add_column("EXPERIMENT", style=colors.EXPERIMENT)
    table.add_column("NAME", style=colors.METRIC_NAME)
    table.add_column("CONTEXT", style=colors.CONTEXT_VAL)

    for row in image_rows:
        run: RunMeta = row["run"]
        ctx_str = _fmt_context(row["context"])
        table.add_row(
            _short_hash(run.hash),
            run.experiment or run.name or "",
            row["name"],
            ctx_str,
        )

    console.print(table)
    return buf.getvalue()


def render_oneline(
    groups: list[tuple[RunMeta, list[MetricSeries]]],
    header_info: dict[str, Any],
) -> str:
    """Plain single-line-per-metric output, suitable for awk/grep pipelines."""
    repo = header_info.get("repo", "")
    lines: list[str] = []
    for run, series_list in groups:
        h = _short_hash(run.hash)
        exp = run.experiment or run.name or ""
        for series in series_list:
            ctx_str = _fmt_context(series.context) or "-"
            last_val, _ = series.last
            min_val, _ = series.min
            max_val, _ = series.max
            lines.append(
                f"{repo}\t{h}\t{exp}\t{series.name}\t{ctx_str}"
                f"\tsteps={series.count}"
                f"\tlast={_fmt_float(last_val)}"
                f"\tmin={_fmt_float(min_val)}"
                f"\tmax={_fmt_float(max_val)}"
            )
    return "\n".join(lines)


def render_image_oneline(
    image_rows: list[dict[str, Any]],
    header_info: dict[str, Any],
) -> str:
    repo = header_info.get("repo", "")
    lines: list[str] = []
    for row in image_rows:
        run: RunMeta = row["run"]
        h = _short_hash(run.hash)
        exp = run.experiment or run.name or ""
        ctx_str = _fmt_context(row["context"]) or "-"
        lines.append(f"{repo}\t{h}\t{exp}\t{row['name']}\t{ctx_str}")
    return "\n".join(lines)


def render_json(
    groups: list[tuple[RunMeta, list[MetricSeries]]],
    header_info: dict[str, Any],
) -> str:
    """Nested JSON output: runs → metrics."""
    metrics_count = sum(len(ms) for _, ms in groups)
    runs_json: list[dict[str, Any]] = []
    for run, series_list in groups:
        metrics_json: list[dict[str, Any]] = []
        for series in series_list:
            last_val, last_step = series.last
            min_val, min_step = series.min
            max_val, max_step = series.max
            metrics_json.append(
                {
                    "name": series.name,
                    "context": series.context,
                    "steps": series.count,
                    "last": {"value": _safe_float(last_val), "step": last_step},
                    "min": {"value": _safe_float(min_val), "step": min_step},
                    "max": {"value": _safe_float(max_val), "step": max_step},
                }
            )
        runs_json.append(
            {
                "hash": run.hash,
                "experiment": run.experiment,
                "name": run.name,
                "metrics": metrics_json,
            }
        )
    payload: dict[str, Any] = {
        "target": header_info.get("target", "metrics"),
        "repo": header_info.get("repo", ""),
        "expression": header_info.get("expression", ""),
        "runs_count": len(runs_json),
        "metrics_count": metrics_count,
        "runs": runs_json,
    }
    return json.dumps(payload)


def render_image_json(
    image_rows: list[dict[str, Any]],
    header_info: dict[str, Any],
) -> str:
    rows_json: list[dict[str, Any]] = []
    for row in image_rows:
        run: RunMeta = row["run"]
        rows_json.append(
            {
                "run_hash": run.hash,
                "experiment": run.experiment,
                "name": row["name"],
                "context": row["context"],
            }
        )
    payload: dict[str, Any] = {
        "target": header_info.get("target", "images"),
        "repo": header_info.get("repo", ""),
        "expression": header_info.get("expression", ""),
        "count": len(rows_json),
        "rows": rows_json,
    }
    return json.dumps(payload)


def _safe_float(v: float) -> float | None:
    if math.isnan(v):
        return None
    return v
