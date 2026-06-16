from __future__ import annotations

import json

import pytest

from aimx.aim_bridge.metric_stats import RunMeta
from aimx.aim_bridge.run_params import (
    RunDuration,
    RunParams,
    default_param_keys,
    extract_run_duration,
    flatten_params,
    select_params,
    sort_run_params,
)
from aimx.rendering.params_views import (
    render_params_json,
    render_params_oneline,
    render_params_rich_table,
)


class _FakeRun:
    def __init__(
        self,
        *,
        duration: object = None,
        end_time: object = None,
        creation_time: object = None,
    ) -> None:
        self.duration = duration
        self.end_time = end_time
        self.creation_time = creation_time


def test_flatten_params_preserves_scalar_values_with_dotted_keys() -> None:
    params = {
        "hparam": {"lr": 0.0001, "optimizer": "AdamW"},
        "model": "UCloudNet",
        "enabled": True,
        "nothing": None,
    }

    assert flatten_params(params) == {
        "enabled": True,
        "hparam.lr": 0.0001,
        "hparam.optimizer": "AdamW",
        "model": "UCloudNet",
        "nothing": None,
    }


def test_flatten_params_preserves_non_scalar_values() -> None:
    params = {"layers": [32, 64], "nested": {"schedule": {"milestones": [1, 2]}}}

    assert flatten_params(params) == {
        "layers": [32, 64],
        "nested.schedule.milestones": [1, 2],
    }


def test_default_param_keys_are_deterministic() -> None:
    rows = [
        RunParams(
            run=RunMeta("b", "exp", None, None),
            params={"z": 1, "a": 2},
            selected_keys=(),
            missing_keys=(),
        ),
        RunParams(
            run=RunMeta("a", "exp", None, None),
            params={"m": 3, "a": 4},
            selected_keys=(),
            missing_keys=(),
        ),
    ]

    assert default_param_keys(rows) == ("a", "m", "z")


def test_select_params_tracks_missing_requested_keys() -> None:
    selected, missing = select_params(
        {"hparam.lr": 0.0001, "hparam.optimizer": "AdamW"},
        ("hparam.lr", "hparam.weight_decay"),
    )

    assert selected == {"hparam.lr": 0.0001}
    assert missing == ("hparam.weight_decay",)


def test_sort_run_params_orders_by_experiment_name_and_hash() -> None:
    rows = [
        RunParams(RunMeta("ccc", "Zeta", "run", None), {"p": 1}),
        RunParams(RunMeta("bbb", "", "run", None), {"p": 1}),
        RunParams(RunMeta("eee", None, "run", None), {"p": 1}),
        RunParams(RunMeta("aaa", "alpha", "run-b", None), {"p": 1}),
        RunParams(RunMeta("ddd", "Alpha", "run-a", None), {"p": 1}),
    ]

    result = sort_run_params(rows)

    assert [row.run.hash for row in result] == ["bbb", "eee", "ddd", "aaa", "ccc"]


def test_extract_run_duration_prefers_duration_attribute() -> None:
    duration = extract_run_duration(
        _FakeRun(duration=12.5, end_time=200.0, creation_time=100.0)
    )

    assert duration.seconds == 12.5
    assert duration.status == "available"
    assert duration.source == "duration"


def test_extract_run_duration_falls_back_to_end_minus_creation_time() -> None:
    duration = extract_run_duration(_FakeRun(end_time=200.0, creation_time=100.0))

    assert duration.seconds == 100.0
    assert duration.status == "available"
    assert duration.source == "end_time_minus_creation_time"


@pytest.mark.parametrize(
    "run",
    [
        _FakeRun(duration=-1.0),
        _FakeRun(end_time=100.0, creation_time=200.0),
    ],
)
def test_extract_run_duration_treats_negative_values_as_unavailable(run: _FakeRun) -> None:
    duration = extract_run_duration(run)

    assert duration.seconds is None
    assert duration.status == "unavailable"
    assert duration.source == "missing_metadata"


def test_extract_run_duration_treats_non_numeric_values_as_unavailable() -> None:
    duration = extract_run_duration(
        _FakeRun(duration="slow", end_time="later", creation_time="earlier")
    )

    assert duration.seconds is None
    assert duration.status == "unavailable"
    assert duration.source == "missing_metadata"


def test_extract_run_duration_reports_missing_metadata_as_unavailable() -> None:
    duration = extract_run_duration(_FakeRun())

    assert duration.seconds is None
    assert duration.status == "unavailable"
    assert duration.source == "missing_metadata"


def test_extract_run_duration_reports_created_run_without_end_time_as_running() -> None:
    duration = extract_run_duration(_FakeRun(creation_time=100.0))

    assert duration.seconds is None
    assert duration.status == "running"
    assert duration.source == "missing_metadata"


def test_render_params_marks_runs_with_no_params() -> None:
    rows = [RunParams(RunMeta("abc123", "exp", "run", None), {})]
    header = {"target": "params", "repo": "repo", "expression": "run.hash != ''"}

    rich = render_params_rich_table(rows, header, no_color=True)
    plain = render_params_oneline(rows, header)

    assert "no params" in rich
    assert "params=-" in plain


def test_render_params_includes_duration_in_rich_and_plain_output() -> None:
    rows = [
        RunParams(
            RunMeta("abc123", "exp", "run", None),
            {"hparam.lr": 0.1},
            duration=RunDuration(seconds=65.2, status="available", source="duration"),
        )
    ]
    header = {"target": "params", "repo": "repo", "expression": "run.hash != ''"}

    rich = render_params_rich_table(rows, header, no_color=True)
    plain = render_params_oneline(rows, header)

    assert "DURATION" in rich
    assert "1m05s" in rich
    assert "\t1m05s\t" in plain


def test_render_params_json_includes_stable_duration_object() -> None:
    rows = [
        RunParams(
            RunMeta("abc123", "exp", "run", None),
            {"hparam.lr": 0.1},
            duration=RunDuration(
                seconds=65.2,
                status="available",
                source="end_time_minus_creation_time",
            ),
        )
    ]
    header = {"target": "params", "repo": "repo", "expression": "run.hash != ''"}

    payload = json.loads(render_params_json(rows, header))

    assert payload["runs"][0]["duration"] == {
        "seconds": 65.2,
        "status": "available",
        "source": "end_time_minus_creation_time",
    }


def test_render_params_marks_unavailable_and_running_duration() -> None:
    rows = [
        RunParams(
            RunMeta("abc123", "exp", "missing", None),
            {"hparam.lr": 0.1},
            duration=RunDuration(status="unavailable"),
        ),
        RunParams(
            RunMeta("def456", "exp", "active", None),
            {"hparam.lr": 0.2},
            duration=RunDuration(status="running"),
        ),
    ]
    header = {"target": "params", "repo": "repo", "expression": "run.hash != ''"}

    rich = render_params_rich_table(rows, header, no_color=True)
    plain = render_params_oneline(rows, header)

    assert "unavailable" in rich
    assert "running" in rich
    assert "\tunavailable\t" in plain
    assert "\trunning\t" in plain


def test_render_params_keeps_duration_independent_from_selected_params() -> None:
    rows = [
        RunParams(
            RunMeta("abc123", "exp", "run", None),
            {"hparam.lr": 0.1},
            duration=RunDuration(seconds=1.0, status="available", source="duration"),
            selected_keys=("hparam.lr", "missing.key"),
            missing_keys=("missing.key",),
        )
    ]
    header = {
        "target": "params",
        "repo": "repo",
        "expression": "run.hash != ''",
        "param_keys": ("hparam.lr", "missing.key"),
    }

    payload = json.loads(render_params_json(rows, header))
    run = payload["runs"][0]

    assert payload["param_keys"] == ["hparam.lr", "missing.key"]
    assert run["params"] == {"hparam.lr": 0.1}
    assert run["missing_params"] == ["missing.key"]
    assert run["duration"]["seconds"] == 1.0
