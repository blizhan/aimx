from __future__ import annotations

import contextlib
import datetime as dt
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunMeta:
    hash: str
    experiment: str | None
    name: str | None
    creation_time: float | None


@dataclass
class MetricSeries:
    run: RunMeta
    name: str
    context: dict[str, Any]
    values: np.ndarray
    steps: np.ndarray
    epochs: np.ndarray | None

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def last(self) -> tuple[float, int]:
        if len(self.values) == 0:
            return (float("nan"), -1)
        idx = len(self.values) - 1
        return (float(self.values[idx]), int(self.steps[idx]))

    @property
    def min(self) -> tuple[float, int]:
        if len(self.values) == 0:
            return (float("nan"), -1)
        idx = int(np.argmin(self.values))
        return (float(self.values[idx]), int(self.steps[idx]))

    @property
    def max(self) -> tuple[float, int]:
        if len(self.values) == 0:
            return (float("nan"), -1)
        idx = int(np.argmax(self.values))
        return (float(self.values[idx]), int(self.steps[idx]))


def _extract_run_meta(run: Any) -> RunMeta:
    creation_time = getattr(run, "creation_time", None)
    if creation_time is None:
        created_at = getattr(run, "created_at", None)
        if isinstance(created_at, dt.datetime):
            if created_at.tzinfo is None:
                creation_time = created_at.replace(tzinfo=dt.timezone.utc).timestamp()
            else:
                creation_time = created_at.timestamp()

    return RunMeta(
        hash=run.hash,
        experiment=getattr(run, "experiment", None),
        name=getattr(run, "name", None),
        creation_time=float(creation_time) if creation_time is not None else None,
    )


def _extract_values(metric: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract aligned values, steps, and epochs from an Aim metric.

    Aim exposes the canonical series order and metadata via
    ``metric.data.items_list()``. This preserves the distinction between
    user-provided steps and tracked epochs, which may differ.
    """
    try:
        steps, (values, epochs, _timestamps) = metric.data.items_list()
    except ValueError:
        return np.array([], dtype=float), np.array([], dtype=int), None

    return (
        np.array(values, dtype=float),
        np.array(steps, dtype=int),
        np.array(epochs, dtype=float),
    )


def collect_metric_series(expression: str, repo_path: Path) -> list[MetricSeries]:
    """Run the Aim query expression and return a flat list of MetricSeries.

    Short run.hash literals in *expression* are transparently expanded to full
    hashes before being forwarded to Aim. Aim's own progress output is silenced
    via stderr redirection.
    """
    from aimx.aim_bridge.hash_resolver import resolve_hash_prefixes

    expression = resolve_hash_prefixes(expression, repo_path)

    try:
        from aim import Repo
        from aim.sdk.types import QueryReportMode
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "`aimx` requires the Python `aim` package in the current environment."
        ) from error

    repo = Repo(str(repo_path))
    results: list[MetricSeries] = []

    stderr_buf = io.StringIO()
    with contextlib.redirect_stderr(stderr_buf):
        query_result = repo.query_metrics(
            expression, report_mode=QueryReportMode.DISABLED
        )
        for run_collection in query_result.iter_runs():
            for metric in run_collection:
                run_meta = _extract_run_meta(metric.run)
                values, steps, epochs = _extract_values(metric)
                results.append(
                    MetricSeries(
                        run=run_meta,
                        name=metric.name,
                        context=metric.context.to_dict(),
                        values=values,
                        steps=steps,
                        epochs=epochs,
                    )
                )

    return results


def collect_image_series(expression: str, repo_path: Path) -> list[dict[str, Any]]:
    """Run an image query and return a flat list of image record dicts.

    Short run.hash literals in *expression* are transparently expanded before
    being forwarded to Aim.
    """
    from aimx.aim_bridge.hash_resolver import resolve_hash_prefixes

    expression = resolve_hash_prefixes(expression, repo_path)

    try:
        from aim import Repo
        from aim.sdk.types import QueryReportMode
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "`aimx` requires the Python `aim` package in the current environment."
        ) from error

    repo = Repo(str(repo_path))
    rows: list[dict[str, Any]] = []

    stderr_buf = io.StringIO()
    with contextlib.redirect_stderr(stderr_buf):
        query_result = repo.query_images(
            expression, report_mode=QueryReportMode.DISABLED
        )
        for image in query_result.iter():
            run_meta = _extract_run_meta(image.run)
            rows.append(
                {
                    "run": run_meta,
                    "name": image.name,
                    "context": image.context.to_dict(),
                }
            )

    return rows


def subsample(series: MetricSeries, *, head: int | None, tail: int | None, every: int | None) -> MetricSeries:
    """Return a new MetricSeries with points filtered by head/tail/every."""
    n = len(series.values)
    if n == 0:
        return series

    indices = np.arange(n)
    if head is not None:
        indices = indices[:head]
    if tail is not None:
        indices = indices[-tail:]
    if every is not None and every > 1:
        indices = indices[::every]

    epochs_slice = series.epochs[indices] if series.epochs is not None else None
    return MetricSeries(
        run=series.run,
        name=series.name,
        context=series.context,
        values=series.values[indices],
        steps=series.steps[indices],
        epochs=epochs_slice,
    )


def parse_step_slice(s: str) -> tuple[int | None, int | None]:
    """Parse a ``start:end`` slice string into inclusive integer bounds.

    - ``"100:500"`` → ``(100, 500)``
    - ``"100:"``    → ``(100, None)``
    - ``":500"``    → ``(None, 500)``
    - ``":"``       → ``ValueError``
    - No colon     → ``ValueError``
    """
    if ":" not in s:
        raise ValueError(
            f"--steps requires 'start:end' slice syntax (e.g. '100:500', ':500', '100:'), got: {s!r}"
        )
    left, right = s.split(":", 1)
    start: int | None = None
    end: int | None = None
    if left.strip():
        try:
            start = int(left.strip())
        except ValueError:
            raise ValueError(f"--steps: left bound is not an integer: {left!r}")
    if right.strip():
        try:
            end = int(right.strip())
        except ValueError:
            raise ValueError(f"--steps: right bound is not an integer: {right!r}")
    if start is None and end is None:
        raise ValueError("--steps cannot be an open slice ':'; provide at least one bound.")
    return start, end


def filter_by_step_range(
    series: MetricSeries,
    start: int | None,
    end: int | None,
) -> MetricSeries:
    """Return a new ``MetricSeries`` keeping only points where ``start <= step <= end``.

    Open-ended bounds (``None``) mean no constraint on that side.
    """
    mask = np.ones(len(series.steps), dtype=bool)
    if start is not None:
        mask &= series.steps >= start
    if end is not None:
        mask &= series.steps <= end
    epochs_slice = series.epochs[mask] if series.epochs is not None else None
    return MetricSeries(
        run=series.run,
        name=series.name,
        context=series.context,
        values=series.values[mask],
        steps=series.steps[mask],
        epochs=epochs_slice,
    )


def group_by_run(
    series_list: list[MetricSeries],
) -> list[tuple[RunMeta, list[MetricSeries]]]:
    """Group a flat list of MetricSeries by run hash, preserving insertion order."""
    order: list[str] = []
    groups: dict[str, tuple[RunMeta, list[MetricSeries]]] = {}
    for series in series_list:
        h = series.run.hash
        if h not in groups:
            order.append(h)
            groups[h] = (series.run, [])
        groups[h][1].append(series)
    return [groups[h] for h in order]
