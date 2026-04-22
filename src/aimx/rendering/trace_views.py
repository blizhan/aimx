from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import shutil
from typing import Any

from rich.console import Console
from rich.table import Table

from aimx.aim_bridge.metric_stats import MetricSeries, RunMeta
from aimx.rendering import colors


def _short_hash(h: str) -> str:
    return h[:8]


def _fmt_context(ctx: dict[str, Any]) -> str:
    if not ctx:
        return ""
    return " ".join(f"{k}={v}" for k, v in sorted(ctx.items()))


def _series_label(series: MetricSeries) -> str:
    parts = [_short_hash(series.run.hash)]
    if series.run.experiment:
        parts.append(series.run.experiment)
    elif series.run.name:
        parts.append(series.run.name)
    parts.append(series.name)
    ctx = _fmt_context(series.context)
    if ctx:
        parts.append(f"[{ctx}]")
    return " · ".join(parts)


def render_plot(
    series_list: list[MetricSeries],
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Render time-series curves using plotext and return as a string."""
    import plotext as plt  # noqa: PLC0415

    plt.clt()
    plt.cld()

    term_width = shutil.get_terminal_size(fallback=(120, 30)).columns
    plot_width = width or term_width
    plot_height = height or 25
    plt.plot_size(plot_width, plot_height)

    for series in series_list:
        if series.count == 0:
            continue
        label = _series_label(series)
        x = series.steps.tolist()
        y = series.values.tolist()
        plt.plot(x, y, label=label)

    if series_list:
        first = series_list[0]
        title = first.name if all(s.name == first.name for s in series_list) else "Metrics"
        plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.theme("dark")

    output_buf = io.StringIO()
    with contextlib.redirect_stdout(output_buf):
        plt.show()
    return output_buf.getvalue()


def render_trace_table(
    series_list: list[MetricSeries],
    *,
    no_color: bool = False,
) -> str:
    """Render each series as a rich table with step/epoch/value columns."""
    width = 120 if no_color else shutil.get_terminal_size(fallback=(120, 24)).columns
    buf = io.StringIO()
    console = Console(
        file=buf,
        no_color=no_color,
        force_terminal=not no_color,
        width=width,
        highlight=False,
    )

    for series in series_list:
        label = _series_label(series)
        console.print(f"\n[{colors.HEADER}]{label}[/]  [{colors.HEADER}]{series.count} points[/]")

        table = Table(
            show_header=True,
            header_style=colors.HEADER,
            box=None,
            pad_edge=True,
            show_edge=False,
            padding=(0, 1),
        )
        table.add_column("STEP", justify="right")
        table.add_column("EPOCH", justify="right")
        table.add_column("VALUE", justify="right", style=colors.NUMBER_EMPH)

        for i in range(series.count):
            step = int(series.steps[i])
            epoch = f"{series.epochs[i]:.0f}" if series.epochs is not None else "—"
            val = series.values[i]
            val_str = "—" if math.isnan(float(val)) else f"{float(val):.6g}"
            table.add_row(str(step), epoch, val_str)

        console.print(table)

    return buf.getvalue()


def render_csv(series_list: list[MetricSeries]) -> str:
    """Render series data as CSV: run_hash,experiment,metric,context,step,epoch,value."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_hash", "experiment", "metric", "context", "step", "epoch", "value"])
    for series in series_list:
        ctx_str = json.dumps(series.context, sort_keys=True)
        for i in range(series.count):
            step = int(series.steps[i])
            epoch = float(series.epochs[i]) if series.epochs is not None else ""
            val = float(series.values[i])
            writer.writerow(
                [
                    series.run.hash,
                    series.run.experiment or series.run.name or "",
                    series.name,
                    ctx_str,
                    step,
                    epoch,
                    val,
                ]
            )
    return buf.getvalue()


def render_trace_json(series_list: list[MetricSeries]) -> str:
    """Render series data as JSON with full value arrays."""
    result: list[dict[str, Any]] = []
    for series in series_list:
        result.append(
            {
                "run": {
                    "hash": series.run.hash,
                    "experiment": series.run.experiment,
                    "name": series.run.name,
                },
                "metric": series.name,
                "context": series.context,
                "count": series.count,
                "steps": series.steps.tolist(),
                "epochs": series.epochs.tolist() if series.epochs is not None else None,
                "values": series.values.tolist(),
            }
        )
    return json.dumps(result)
