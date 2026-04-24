"""Read-only helpers for collecting Aim run parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimx.aim_bridge.metric_stats import RunMeta, _extract_run_meta


@dataclass(frozen=True)
class RunParams:
    run: RunMeta
    params: dict[str, Any]
    selected_keys: tuple[str, ...] = ()
    missing_keys: tuple[str, ...] = ()


def flatten_params(params: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries into stable dotted parameter keys."""
    flattened: dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_params(value, prefix=dotted_key))
        else:
            flattened[dotted_key] = value
    return flattened


def default_param_keys(rows: list[RunParams]) -> tuple[str, ...]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.params)
    return tuple(sorted(keys))


def select_params(params: dict[str, Any], selected_keys: tuple[str, ...]) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not selected_keys:
        return params, ()
    selected: dict[str, Any] = {}
    missing: list[str] = []
    for key in selected_keys:
        if key in params:
            selected[key] = params[key]
        else:
            missing.append(key)
    return selected, tuple(missing)


def sort_run_params(rows: list[RunParams]) -> list[RunParams]:
    return sorted(
        rows,
        key=lambda row: (
            (row.run.experiment or "").casefold(),
            row.run.name or "",
            row.run.hash,
        ),
    )


def _metadata_attrs(run: Any) -> dict[str, Any]:
    try:
        collected = run.meta_run_tree.collect()
    except Exception:  # noqa: BLE001
        return {}
    attrs = collected.get("attrs", {})
    return attrs if isinstance(attrs, dict) else {}


def collect_run_params(
    expression: str,
    repo_path: Path,
    selected_keys: tuple[str, ...] = (),
) -> list[RunParams]:
    """Run an Aim run query and return flattened params for each matched run."""
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
    rows: list[RunParams] = []
    query_result = repo.query_runs(expression, report_mode=QueryReportMode.DISABLED)
    for run_collection in query_result.iter_runs():
        run = run_collection.run
        flattened = flatten_params(_metadata_attrs(run))
        params, missing = select_params(flattened, selected_keys)
        rows.append(
            RunParams(
                run=_extract_run_meta(run),
                params=params,
                selected_keys=selected_keys,
                missing_keys=missing,
            )
        )
    return sort_run_params(rows)
