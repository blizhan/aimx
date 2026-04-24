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
    params: dict[str, Any] | None = None


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

    params = _as_plain_dict(getattr(run, "hparams", None))

    return RunMeta(
        hash=run.hash,
        experiment=getattr(run, "experiment", None),
        name=getattr(run, "name", None),
        creation_time=float(creation_time) if creation_time is not None else None,
        params=params,
    )


def _as_plain_dict(value: Any) -> dict[str, Any] | None:
    """Best-effort conversion of mapping-like objects to plain ``dict``."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(converted, dict):
            return dict(converted)
        try:
            return dict(converted)
        except Exception:  # noqa: BLE001
            return None

    try:
        return dict(value)
    except Exception:  # noqa: BLE001
        return None


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


def _first_iter_value(view: Any) -> Any | None:
    """Return the first item from an iterable-like view, or None."""
    if view is None:
        return None
    try:
        return next(iter(view))
    except (StopIteration, TypeError):
        return None


def _call_or_value(value: Any) -> Any:
    """Call *value* if it's callable, otherwise return it as-is."""
    if callable(value):
        try:
            return value()
        except Exception:  # noqa: BLE001
            return None
    return value


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
            context = image.context.to_dict()
            epoch_value = context.get("epoch")
            if epoch_value is None:
                epoch_value = _first_iter_value(getattr(image, "epochs", None))
            step_value = _call_or_value(getattr(image, "first_step", None))

            # Capture the image object in a closure so the accessor is lazy.
            # The private key ``_image_accessor`` is ONLY consumed by the
            # rich-TTY rendering path in ``image_render.render_inline``.
            # JSON / plain renderers MUST NOT touch this key.
            _aim_img = image

            def _make_accessor(_img: Any) -> Any:
                def _accessor() -> Any:
                    if hasattr(_img, "to_pil_image"):
                        return _img.to_pil_image()

                    values = getattr(_img, "values", None)
                    if values is None:
                        raise AttributeError(
                            f"{type(_img).__name__} has neither to_pil_image nor values"
                        )

                    try:
                        first_value = next(iter(values))
                    except StopIteration as error:
                        raise RuntimeError("image sequence is empty") from error

                    if not hasattr(first_value, "to_pil_image"):
                        raise AttributeError(
                            f"{type(first_value).__name__} has no to_pil_image"
                        )

                    return first_value.to_pil_image()
                return _accessor

            rows.append(
                {
                    "run": run_meta,
                    "name": image.name,
                    "context": context,
                    "_sort_epoch": epoch_value,
                    "_sort_step": step_value,
                    "_image_accessor": _make_accessor(_aim_img),
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


def parse_epoch_slice(s: str) -> tuple[float | None, float | None]:
    """Parse a ``start:end`` slice string into inclusive float bounds for epoch filtering.

    Accepts integer or float values, e.g. ``"5:50"``, ``"5.0:"``, ``":30"``.
    """
    if ":" not in s:
        raise ValueError(
            f"--epochs requires 'start:end' slice syntax (e.g. '5:50', ':30', '5:'), got: {s!r}"
        )
    left, right = s.split(":", 1)
    start: float | None = None
    end: float | None = None
    if left.strip():
        try:
            start = float(left.strip())
        except ValueError:
            raise ValueError(f"--epochs: left bound is not a number: {left!r}")
    if right.strip():
        try:
            end = float(right.strip())
        except ValueError:
            raise ValueError(f"--epochs: right bound is not a number: {right!r}")
    if start is None and end is None:
        raise ValueError("--epochs cannot be an open slice ':'; provide at least one bound.")
    return start, end


def filter_by_epoch_range(
    series: "MetricSeries",
    start: float | None,
    end: float | None,
) -> "MetricSeries":
    """Return a new ``MetricSeries`` keeping only points where ``start <= epoch <= end``.

    Points with ``epoch == None`` (series that has no epoch data) are all kept when
    bounds are present — the filter is a no-op on epoch-less series.
    Open-ended bounds (``None``) mean no constraint on that side.
    """
    if series.epochs is None:
        return series
    mask = np.ones(len(series.epochs), dtype=bool)
    if start is not None:
        mask &= series.epochs >= start
    if end is not None:
        mask &= series.epochs <= end
    return MetricSeries(
        run=series.run,
        name=series.name,
        context=series.context,
        values=series.values[mask],
        steps=series.steps[mask],
        epochs=series.epochs[mask],
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


# ---------------------------------------------------------------------------
# Image-row filter / subsample helpers
# ---------------------------------------------------------------------------

def filter_image_rows_by_step_range(
    rows: list[dict[str, Any]],
    start: int | None,
    end: int | None,
) -> list[dict[str, Any]]:
    """Keep only image rows whose ``_sort_step`` falls within [start, end].

    Rows with ``_sort_step == None`` are always kept (no step information to
    exclude them by).
    """
    result: list[dict[str, Any]] = []
    for row in rows:
        step = row.get("_sort_step")
        if step is None:
            result.append(row)
            continue
        try:
            step_val = int(step)
        except (TypeError, ValueError):
            result.append(row)
            continue
        if start is not None and step_val < start:
            continue
        if end is not None and step_val > end:
            continue
        result.append(row)
    return result


def filter_image_rows_by_epoch_range(
    rows: list[dict[str, Any]],
    start: float | None,
    end: float | None,
) -> list[dict[str, Any]]:
    """Keep only image rows whose ``_sort_epoch`` falls within [start, end].

    Rows with ``_sort_epoch == None`` are always kept (no epoch information to
    exclude them by).
    """
    result: list[dict[str, Any]] = []
    for row in rows:
        epoch = row.get("_sort_epoch")
        if epoch is None:
            result.append(row)
            continue
        try:
            epoch_val = float(epoch)
        except (TypeError, ValueError):
            result.append(row)
            continue
        if start is not None and epoch_val < start:
            continue
        if end is not None and epoch_val > end:
            continue
        result.append(row)
    return result


def subsample_image_rows(
    rows: list[dict[str, Any]],
    *,
    head: int | None,
    tail: int | None,
    every: int | None,
) -> list[dict[str, Any]]:
    """Return a globally subsampled view of *rows* (already sorted).

    Applies head → tail → every in order, mirroring ``subsample`` for metric
    series. All three parameters are optional; passing all ``None`` is a no-op.
    """
    if not rows:
        return rows
    result = rows
    if head is not None:
        result = result[:head]
    if tail is not None:
        result = result[-tail:]
    if every is not None and every > 1:
        result = result[::every]
    return result
