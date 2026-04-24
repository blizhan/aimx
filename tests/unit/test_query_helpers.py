from __future__ import annotations

from pathlib import Path

import pytest

from aimx.commands.query import (
    QueryInvocation,
    _sort_image_rows,
    normalize_repo_path,
    parse_query_invocation,
)
from aimx.aim_bridge.metric_stats import (
    RunMeta,
    filter_image_rows_by_epoch_range,
    filter_image_rows_by_step_range,
    parse_epoch_slice,
    subsample_image_rows,
)


def test_normalize_repo_path_keeps_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    normalized = normalize_repo_path(repo_root)

    assert normalized == repo_root


def test_normalize_repo_path_converts_dot_aim_directory_to_parent(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dot_aim = repo_root / ".aim"
    dot_aim.mkdir(parents=True)

    normalized = normalize_repo_path(dot_aim)

    assert normalized == repo_root


def test_normalize_repo_path_rejects_missing_path() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        normalize_repo_path(Path("missing-repo"))


def test_query_invocation_rejects_unsupported_target() -> None:
    with pytest.raises(ValueError, match="Unsupported query target"):
        QueryInvocation(
            target="artifacts",
            expression="metric.name == 'loss'",
            repo_path=Path("data"),
        )


def test_parse_query_invocation_params_defaults() -> None:
    inv = parse_query_invocation(["params", "run.hash != ''"])

    assert inv.target == "params"
    assert inv.expression == "run.hash != ''"
    assert inv.repo_path == Path(".")
    assert inv.param_keys == ()


def test_parse_query_invocation_params_repeated_param_keys() -> None:
    inv = parse_query_invocation(
        [
            "params",
            "run.hash != ''",
            "--repo",
            "data",
            "--param",
            "hparam.lr",
            "--param",
            " hparam.optimizer ",
        ]
    )

    assert inv.param_keys == ("hparam.lr", "hparam.optimizer")


def test_parse_query_invocation_param_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --param"):
        parse_query_invocation(["params", "run.hash != ''", "--param"])


def test_parse_query_invocation_param_empty_value_raises() -> None:
    with pytest.raises(ValueError, match="--param must not be empty"):
        parse_query_invocation(["params", "run.hash != ''", "--param", "  "])


def test_parse_query_invocation_param_duplicate_value_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate --param value"):
        parse_query_invocation(
            ["params", "run.hash != ''", "--param", "hparam.lr", "--param", " hparam.lr "]
        )


def test_parse_query_invocation_param_rejected_for_metrics_and_images() -> None:
    with pytest.raises(ValueError, match="--param is only supported for query params"):
        parse_query_invocation(["metrics", "metric.name == 'loss'", "--param", "hparam.lr"])

    with pytest.raises(ValueError, match="--param is only supported for query params"):
        parse_query_invocation(["images", "images", "--param", "hparam.lr"])


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--steps", "1:10"],
        ["--epochs", "1:10"],
        ["--head", "1"],
        ["--tail", "1"],
        ["--every", "2"],
        ["--max-images", "1"],
    ],
)
def test_parse_query_invocation_params_rejects_unimplemented_query_flags(
    extra_args: list[str],
) -> None:
    with pytest.raises(ValueError, match="not supported for query params"):
        parse_query_invocation(["params", "run.hash != ''", *extra_args])


def test_parse_query_invocation_defaults() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'"])
    assert inv.target == "metrics"
    assert inv.expression == "metric.name == 'loss'"
    assert inv.repo_path == Path(".")
    assert not inv.output_json
    assert not inv.plain
    assert not inv.no_color
    assert not inv.verbose


def test_parse_query_invocation_json_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--json"])
    assert inv.output_json is True


def test_parse_query_invocation_oneline_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--oneline"])
    assert inv.plain is True


def test_parse_query_invocation_plain_flag_alias() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--plain"])
    assert inv.plain is True


def test_parse_query_invocation_no_color_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--no-color"])
    assert inv.no_color is True


def test_parse_query_invocation_verbose_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--verbose"])
    assert inv.verbose is True


def test_parse_query_invocation_explicit_repo_overrides_default() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data"])
    assert inv.repo_path == Path("data")


def test_parse_query_invocation_rejects_unknown_flag() -> None:
    with pytest.raises(ValueError, match="Unsupported query option"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--bogus"])


def test_parse_query_invocation_steps_closed_range() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data", "--steps", "100:500"])
    assert inv.step_slice == "100:500"


def test_parse_query_invocation_steps_open_end() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data", "--steps", "100:"])
    assert inv.step_slice == "100:"


def test_parse_query_invocation_steps_open_start() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data", "--steps", ":500"])
    assert inv.step_slice == ":500"


def test_parse_query_invocation_steps_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --steps"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--steps"])


def test_parse_query_invocation_steps_defaults_to_none() -> None:
    inv = parse_query_invocation(["metrics", "metric.name=='loss'", "--repo", "data"])
    assert inv.step_slice is None


# ---------------------------------------------------------------------------
# T018: --max-images parsing
# ---------------------------------------------------------------------------

def test_parse_query_invocation_max_images_default_is_six() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data"])
    assert inv.max_images == 6


def test_parse_query_invocation_max_images_zero_means_unlimited() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data", "--max-images", "0"])
    assert inv.max_images == 0


def test_parse_query_invocation_max_images_positive_value() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data", "--max-images", "12"])
    assert inv.max_images == 12


def test_parse_query_invocation_max_images_negative_raises() -> None:
    with pytest.raises(ValueError, match="Invalid --max-images"):
        parse_query_invocation(["images", "images", "--repo", "data", "--max-images", "-1"])


def test_parse_query_invocation_max_images_non_integer_raises() -> None:
    with pytest.raises(ValueError, match="Invalid --max-images"):
        parse_query_invocation(["images", "images", "--repo", "data", "--max-images", "abc"])


def test_parse_query_invocation_max_images_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --max-images"):
        parse_query_invocation(["images", "images", "--repo", "data", "--max-images"])


def test_sort_image_rows_orders_epochs_numerically_within_each_run() -> None:
    run_b = RunMeta(hash="bbb", experiment="exp-b", name=None, creation_time=None)
    run_a = RunMeta(hash="aaa", experiment="exp-a", name=None, creation_time=None)
    rows = [
        {"run": run_b, "name": "example", "context": {"epoch": 30, "subset": "val"}, "_sort_epoch": 30, "_sort_step": None},
        {"run": run_b, "name": "example", "context": {"epoch": 10, "subset": "val"}, "_sort_epoch": 10, "_sort_step": None},
        {"run": run_a, "name": "example", "context": {"epoch": 20, "subset": "val"}, "_sort_epoch": 20, "_sort_step": None},
        {"run": run_a, "name": "example", "context": {"epoch": 5, "subset": "val"}, "_sort_epoch": 5, "_sort_step": None},
    ]

    result = _sort_image_rows(rows)

    assert [row["run"].hash for row in result] == ["bbb", "bbb", "aaa", "aaa"]
    assert [row["context"]["epoch"] for row in result] == [10, 30, 5, 20]


def test_sort_image_rows_falls_back_to_step_when_epoch_missing() -> None:
    run = RunMeta(hash="aaa", experiment="exp-a", name=None, creation_time=None)
    rows = [
        {"run": run, "name": "example", "context": {"subset": "val"}, "_sort_epoch": None, "_sort_step": 20},
        {"run": run, "name": "example", "context": {"subset": "val"}, "_sort_epoch": None, "_sort_step": 5},
        {"run": run, "name": "example", "context": {"subset": "val"}, "_sort_epoch": None, "_sort_step": 10},
    ]

    result = _sort_image_rows(rows)

    assert [row["_sort_step"] for row in result] == [5, 10, 20]


# ---------------------------------------------------------------------------
# --head / --tail / --every parsing
# ---------------------------------------------------------------------------

def test_parse_head_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--head", "5"])
    assert inv.head == 5


def test_parse_tail_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--tail", "3"])
    assert inv.tail == 3


def test_parse_every_flag() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'", "--repo", "data", "--every", "2"])
    assert inv.every == 2


def test_parse_head_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --head"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--head"])


def test_parse_tail_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --tail"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--tail"])


def test_parse_every_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --every"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--every"])


def test_parse_every_zero_raises() -> None:
    with pytest.raises(ValueError, match="--every"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--every", "0"])


def test_parse_every_negative_raises() -> None:
    with pytest.raises(ValueError, match="--every"):
        parse_query_invocation(["metrics", "loss", "--repo", "data", "--every", "-2"])


def test_head_tail_every_default_to_none() -> None:
    inv = parse_query_invocation(["metrics", "metric.name == 'loss'"])
    assert inv.head is None
    assert inv.tail is None
    assert inv.every is None


# ---------------------------------------------------------------------------
# --epochs parsing
# ---------------------------------------------------------------------------

def test_parse_epochs_flag() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data", "--epochs", "5:50"])
    assert inv.epoch_slice == "5:50"


def test_parse_epochs_open_start() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data", "--epochs", ":30"])
    assert inv.epoch_slice == ":30"


def test_parse_epochs_open_end() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data", "--epochs", "10:"])
    assert inv.epoch_slice == "10:"


def test_parse_epochs_missing_value_raises() -> None:
    with pytest.raises(ValueError, match="Missing value for --epochs"):
        parse_query_invocation(["images", "images", "--repo", "data", "--epochs"])


# ---------------------------------------------------------------------------
# --steps and --epochs mutual exclusion
# ---------------------------------------------------------------------------

def test_steps_and_epochs_mutually_exclusive_raises() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_query_invocation([
            "images", "images", "--repo", "data",
            "--steps", "1:50", "--epochs", "1:10",
        ])


def test_steps_alone_is_accepted() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data", "--steps", "10:100"])
    assert inv.step_slice == "10:100"
    assert inv.epoch_slice is None


def test_epochs_alone_is_accepted() -> None:
    inv = parse_query_invocation(["images", "images", "--repo", "data", "--epochs", "5:50"])
    assert inv.epoch_slice == "5:50"
    assert inv.step_slice is None


# ---------------------------------------------------------------------------
# parse_epoch_slice helper
# ---------------------------------------------------------------------------

def test_parse_epoch_slice_closed_range() -> None:
    start, end = parse_epoch_slice("5:50")
    assert start == 5.0
    assert end == 50.0


def test_parse_epoch_slice_open_start() -> None:
    start, end = parse_epoch_slice(":30")
    assert start is None
    assert end == 30.0


def test_parse_epoch_slice_open_end() -> None:
    start, end = parse_epoch_slice("10:")
    assert start == 10.0
    assert end is None


def test_parse_epoch_slice_open_both_raises() -> None:
    with pytest.raises(ValueError, match="open slice"):
        parse_epoch_slice(":")


def test_parse_epoch_slice_no_colon_raises() -> None:
    with pytest.raises(ValueError, match="start:end"):
        parse_epoch_slice("50")


# ---------------------------------------------------------------------------
# filter_image_rows_by_step_range
# ---------------------------------------------------------------------------

def _make_image_rows(step_or_epochs: list[tuple[int | None, float | None]]) -> list[dict]:
    run = RunMeta(hash="aaa", experiment="exp", name=None, creation_time=None)
    return [
        {"run": run, "name": "img", "context": {}, "_sort_step": step, "_sort_epoch": epoch}
        for step, epoch in step_or_epochs
    ]


def test_filter_image_rows_by_step_range_keeps_within_bounds() -> None:
    rows = _make_image_rows([(5, None), (10, None), (20, None), (30, None)])
    result = filter_image_rows_by_step_range(rows, 10, 20)
    assert [r["_sort_step"] for r in result] == [10, 20]


def test_filter_image_rows_by_step_range_open_start() -> None:
    rows = _make_image_rows([(5, None), (10, None), (30, None)])
    result = filter_image_rows_by_step_range(rows, None, 10)
    assert [r["_sort_step"] for r in result] == [5, 10]


def test_filter_image_rows_by_step_range_keeps_rows_without_step() -> None:
    rows = _make_image_rows([(None, 5.0), (10, None)])
    result = filter_image_rows_by_step_range(rows, 5, 5)
    assert len(result) == 1
    assert result[0]["_sort_step"] is None


# ---------------------------------------------------------------------------
# filter_image_rows_by_epoch_range
# ---------------------------------------------------------------------------

def test_filter_image_rows_by_epoch_range_keeps_within_bounds() -> None:
    rows = _make_image_rows([(None, 5.0), (None, 10.0), (None, 30.0)])
    result = filter_image_rows_by_epoch_range(rows, 5.0, 10.0)
    assert [r["_sort_epoch"] for r in result] == [5.0, 10.0]


def test_filter_image_rows_by_epoch_range_keeps_rows_without_epoch() -> None:
    rows = _make_image_rows([(10, None), (None, 30.0)])
    result = filter_image_rows_by_epoch_range(rows, 5.0, 20.0)
    assert len(result) == 1
    assert result[0]["_sort_step"] == 10


# ---------------------------------------------------------------------------
# subsample_image_rows
# ---------------------------------------------------------------------------

def test_subsample_image_rows_head() -> None:
    rows = _make_image_rows([(i, None) for i in range(10)])
    result = subsample_image_rows(rows, head=3, tail=None, every=None)
    assert [r["_sort_step"] for r in result] == [0, 1, 2]


def test_subsample_image_rows_tail() -> None:
    rows = _make_image_rows([(i, None) for i in range(10)])
    result = subsample_image_rows(rows, head=None, tail=3, every=None)
    assert [r["_sort_step"] for r in result] == [7, 8, 9]


def test_subsample_image_rows_every() -> None:
    rows = _make_image_rows([(i, None) for i in range(6)])
    result = subsample_image_rows(rows, head=None, tail=None, every=2)
    assert [r["_sort_step"] for r in result] == [0, 2, 4]


def test_subsample_image_rows_combined_head_then_every() -> None:
    rows = _make_image_rows([(i, None) for i in range(10)])
    result = subsample_image_rows(rows, head=6, tail=None, every=2)
    assert [r["_sort_step"] for r in result] == [0, 2, 4]


def test_subsample_image_rows_noop_when_all_none() -> None:
    rows = _make_image_rows([(i, None) for i in range(4)])
    result = subsample_image_rows(rows, head=None, tail=None, every=None)
    assert result is rows
