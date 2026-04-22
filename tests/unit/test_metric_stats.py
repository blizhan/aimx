from __future__ import annotations

import math

import numpy as np
import pytest

from aimx.aim_bridge.metric_stats import (
    MetricSeries,
    RunMeta,
    group_by_run,
    subsample,
)


def _make_run(hash_: str = "abc123", experiment: str | None = "exp") -> RunMeta:
    return RunMeta(hash=hash_, experiment=experiment, name=None, created_at=None)


def _make_series(
    run: RunMeta | None = None,
    name: str = "loss",
    values: list[float] | None = None,
    steps: list[int] | None = None,
) -> MetricSeries:
    if run is None:
        run = _make_run()
    vals = np.array(values if values is not None else [2.0, 1.5, 1.0, 0.5], dtype=float)
    stps = np.array(steps if steps is not None else list(range(len(vals))), dtype=int)
    return MetricSeries(run=run, name=name, context={}, values=vals, steps=stps, epochs=None)


class TestMetricSeriesStats:
    def test_count_matches_values_length(self) -> None:
        s = _make_series(values=[1.0, 2.0, 3.0])
        assert s.count == 3

    def test_last_returns_final_value_and_step(self) -> None:
        s = _make_series(values=[1.0, 0.5], steps=[10, 20])
        val, step = s.last
        assert val == pytest.approx(0.5)
        assert step == 20

    def test_min_returns_minimum_value_and_its_step(self) -> None:
        s = _make_series(values=[3.0, 1.0, 2.0], steps=[0, 1, 2])
        val, step = s.min
        assert val == pytest.approx(1.0)
        assert step == 1

    def test_max_returns_maximum_value_and_its_step(self) -> None:
        s = _make_series(values=[3.0, 1.0, 2.0], steps=[0, 1, 2])
        val, step = s.max
        assert val == pytest.approx(3.0)
        assert step == 0

    def test_empty_series_returns_nan_and_minus_one(self) -> None:
        s = _make_series(values=[], steps=[])
        last_val, last_step = s.last
        min_val, min_step = s.min
        max_val, max_step = s.max
        assert math.isnan(last_val)
        assert last_step == -1
        assert math.isnan(min_val)
        assert min_step == -1
        assert math.isnan(max_val)
        assert max_step == -1


class TestGroupByRun:
    def test_single_run_produces_one_group(self) -> None:
        run = _make_run("aaa")
        series_list = [_make_series(run=run, name="loss"), _make_series(run=run, name="lr")]
        groups = group_by_run(series_list)
        assert len(groups) == 1
        assert groups[0][0].hash == "aaa"
        assert len(groups[0][1]) == 2

    def test_multiple_runs_produce_separate_groups(self) -> None:
        run_a = _make_run("aaa")
        run_b = _make_run("bbb")
        series_list = [
            _make_series(run=run_a, name="loss"),
            _make_series(run=run_b, name="loss"),
            _make_series(run=run_a, name="lr"),
        ]
        groups = group_by_run(series_list)
        assert len(groups) == 2
        hashes = [g[0].hash for g in groups]
        assert hashes == ["aaa", "bbb"]
        assert len(groups[0][1]) == 2  # loss + lr for run_a
        assert len(groups[1][1]) == 1  # only loss for run_b

    def test_empty_list_returns_empty_groups(self) -> None:
        assert group_by_run([]) == []

    def test_insertion_order_is_preserved(self) -> None:
        runs = [_make_run(f"run{i}") for i in range(5)]
        series_list = [_make_series(run=r, name="loss") for r in runs]
        groups = group_by_run(series_list)
        assert [g[0].hash for g in groups] == [f"run{i}" for i in range(5)]


class TestSubsample:
    def test_head_keeps_first_n_points(self) -> None:
        s = _make_series(values=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = subsample(s, head=3, tail=None, every=None)
        assert result.count == 3
        assert result.values.tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_tail_keeps_last_n_points(self) -> None:
        s = _make_series(values=[1.0, 2.0, 3.0, 4.0, 5.0])
        result = subsample(s, head=None, tail=2, every=None)
        assert result.count == 2
        assert result.values.tolist() == pytest.approx([4.0, 5.0])

    def test_every_k_keeps_every_kth_point(self) -> None:
        s = _make_series(values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = subsample(s, head=None, tail=None, every=2)
        assert result.count == 3
        assert result.values.tolist() == pytest.approx([0.0, 2.0, 4.0])

    def test_empty_series_is_returned_unchanged(self) -> None:
        s = _make_series(values=[], steps=[])
        result = subsample(s, head=5, tail=None, every=None)
        assert result.count == 0

    def test_steps_are_sliced_consistently_with_values(self) -> None:
        s = _make_series(values=[10.0, 20.0, 30.0, 40.0], steps=[100, 200, 300, 400])
        result = subsample(s, head=2, tail=None, every=None)
        assert result.steps.tolist() == [100, 200]
