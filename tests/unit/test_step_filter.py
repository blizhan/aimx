from __future__ import annotations

import numpy as np
import pytest

from aimx.aim_bridge.metric_stats import (
    MetricSeries,
    RunMeta,
    filter_by_step_range,
    parse_step_slice,
)


def _make_series(
    steps: list[int] | None = None,
    values: list[float] | None = None,
    epochs: list[float] | None = None,
) -> MetricSeries:
    run = RunMeta(hash="a" * 32, experiment="exp", name=None, creation_time=None)
    steps = steps if steps is not None else list(range(1, 6))
    values = values if values is not None else [float(i) for i in steps]
    epochs_arr = np.array(epochs, dtype=float) if epochs is not None else None
    return MetricSeries(
        run=run,
        name="loss",
        context={},
        values=np.array(values, dtype=float),
        steps=np.array(steps, dtype=int),
        epochs=epochs_arr,
    )


# ---------------------------------------------------------------------------
# parse_step_slice
# ---------------------------------------------------------------------------


class TestParseStepSlice:
    def test_full_range(self) -> None:
        assert parse_step_slice("100:500") == (100, 500)

    def test_open_end(self) -> None:
        assert parse_step_slice("100:") == (100, None)

    def test_open_start(self) -> None:
        assert parse_step_slice(":500") == (None, 500)

    def test_no_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="slice syntax"):
            parse_step_slice("100")

    def test_fully_open_raises(self) -> None:
        with pytest.raises(ValueError, match="open slice"):
            parse_step_slice(":")

    def test_non_integer_left_raises(self) -> None:
        with pytest.raises(ValueError, match="left bound"):
            parse_step_slice("abc:500")

    def test_non_integer_right_raises(self) -> None:
        with pytest.raises(ValueError, match="right bound"):
            parse_step_slice("100:xyz")

    def test_zero_start_is_valid(self) -> None:
        assert parse_step_slice("0:10") == (0, 10)

    def test_whitespace_is_tolerated(self) -> None:
        assert parse_step_slice(" 10 : 50 ") == (10, 50)


# ---------------------------------------------------------------------------
# filter_by_step_range
# ---------------------------------------------------------------------------


class TestFilterByStepRange:
    def test_closed_range_keeps_correct_points(self) -> None:
        s = _make_series(steps=[1, 2, 3, 4, 5])
        result = filter_by_step_range(s, 2, 4)
        assert result.steps.tolist() == [2, 3, 4]
        assert result.values.tolist() == pytest.approx([2.0, 3.0, 4.0])

    def test_inclusive_lower_bound(self) -> None:
        s = _make_series(steps=[1, 2, 3])
        result = filter_by_step_range(s, 1, None)
        assert 1 in result.steps.tolist()

    def test_inclusive_upper_bound(self) -> None:
        s = _make_series(steps=[1, 2, 3])
        result = filter_by_step_range(s, None, 3)
        assert 3 in result.steps.tolist()

    def test_open_start_keeps_from_beginning(self) -> None:
        s = _make_series(steps=[1, 2, 3, 4, 5])
        result = filter_by_step_range(s, None, 3)
        assert result.steps.tolist() == [1, 2, 3]

    def test_open_end_keeps_to_end(self) -> None:
        s = _make_series(steps=[1, 2, 3, 4, 5])
        result = filter_by_step_range(s, 3, None)
        assert result.steps.tolist() == [3, 4, 5]

    def test_range_outside_data_returns_empty(self) -> None:
        s = _make_series(steps=[1, 2, 3])
        result = filter_by_step_range(s, 100, 200)
        assert result.count == 0

    def test_epochs_sliced_consistently(self) -> None:
        s = _make_series(steps=[1, 2, 3, 4, 5], epochs=[10.0, 20.0, 30.0, 40.0, 50.0])
        result = filter_by_step_range(s, 2, 4)
        assert result.epochs is not None
        assert result.epochs.tolist() == pytest.approx([20.0, 30.0, 40.0])

    def test_series_without_epochs_stays_none(self) -> None:
        s = _make_series(steps=[1, 2, 3])
        result = filter_by_step_range(s, 1, 2)
        assert result.epochs is None

    def test_no_bounds_returns_all_points(self) -> None:
        s = _make_series(steps=[1, 2, 3, 4])
        result = filter_by_step_range(s, None, None)
        assert result.count == 4

    def test_empty_series_returns_empty(self) -> None:
        s = _make_series(steps=[], values=[])
        result = filter_by_step_range(s, 1, 10)
        assert result.count == 0
