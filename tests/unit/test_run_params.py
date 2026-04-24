from __future__ import annotations

from aimx.aim_bridge.metric_stats import RunMeta
from aimx.aim_bridge.run_params import (
    RunParams,
    default_param_keys,
    flatten_params,
    select_params,
    sort_run_params,
)
from aimx.rendering.params_views import render_params_oneline, render_params_rich_table


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


def test_render_params_marks_runs_with_no_params() -> None:
    rows = [RunParams(RunMeta("abc123", "exp", "run", None), {})]
    header = {"target": "params", "repo": "repo", "expression": "run.hash != ''"}

    rich = render_params_rich_table(rows, header, no_color=True)
    plain = render_params_oneline(rows, header)

    assert "no params" in rich
    assert "params=-" in plain
