"""The small functions the pages are built from.

A column holds whatever the evaluator put in it, and a timestamp arrives as a string
from JSON and as a datetime from DuckDB. These decide how such values are shown, so
they are pinned here rather than read out of rendered HTML.
"""

from __future__ import annotations

import datetime as dt

import pytest

from cairn.web.app import _cairn_svg, _delta, _fmt_when, _metadata_json, _num, _widths, _with_dataset

# --- formatting a metric ----------------------------------------------------

@pytest.mark.parametrize("value, shown", [
    (0.8236, "0.824"),
    (1, "1"),                             # an int column holds counts, not measurements
    (1.0, "1.000"),
    (None, "—"),
    (True, "True"),                       # a bool is a label, not a number to round
    ("n/a", "n/a"),
    ({"tp": 3}, "{'tp': 3}"),             # a json column still has to render
    (float("nan"), "nan"),
    (float("inf"), "inf"),
])
def test_a_metric_is_rounded_and_anything_else_is_shown_as_is(value, shown):
    assert _num(value) == shown


def test_the_number_of_places_can_be_asked_for():
    assert _num(0.8236, 1) == "0.8"


# --- timestamps -------------------------------------------------------------

@pytest.mark.parametrize("value, shown", [
    ("2026-01-02T09:04:00+00:00", "01-02 09:04"),
    ("2026-01-02T09:04:00Z", "01-02 09:04"),
    (dt.datetime(2026, 1, 2, 9, 4, tzinfo=dt.timezone.utc), "01-02 09:04"),
    ("", ""),
    (None, ""),
    ("not a time", "not a time"),
])
def test_a_timestamp_is_shown_short_and_a_non_timestamp_untouched(value, shown):
    assert _fmt_when(value) == shown


# --- stone widths -----------------------------------------------------------

def test_widths_span_the_range_of_the_values_on_screen():
    assert _widths([0.0, 0.5, 1.0], lo=10, hi=20) == [10, 15, 20]


def test_an_absolute_scale_keeps_a_small_difference_small():
    """With scale, 0.80 vs 0.82 is a hair apart; without it, they are the whole range."""
    assert _widths([0.80, 0.82], lo=0, hi=100, scale=[0, 1]) == [80, 82]
    assert _widths([0.80, 0.82], lo=0, hi=100) == [0, 100]


def test_when_lower_is_better_the_widest_stone_is_the_smallest_value():
    assert _widths([1.0, 0.0], lo=10, hi=20, direction="lower") == [10, 20]


def test_values_outside_the_scale_are_clamped():
    assert _widths([-5.0, 5.0], lo=10, hi=20, scale=[0, 1]) == [10, 20]


def test_widths_survive_columns_that_hold_no_number():
    assert _widths([None, "x", float("nan"), 1.0], lo=0, hi=10) == [0, 0, 0, 10]


def test_no_runs_and_all_equal_runs_both_have_an_answer():
    assert _widths([]) == []
    assert _widths([0.5, 0.5], lo=10, hi=20) == [15, 15]
    assert _widths([0.5, 0.5], lo=10, hi=20, scale=[1, 1]) == [15, 15]   # a scale of zero width


# --- the stack of stones ----------------------------------------------------

def test_stones_are_drawn_newest_first_with_a_line_where_the_eval_version_changes():
    runs = [{"score": 0.9, "evaluator_version": "v2"},
            {"score": 0.8, "evaluator_version": "v2"},
            {"score": 0.7, "evaluator_version": "v1"}]

    svg = _cairn_svg(runs, "score", "v2")

    assert svg.count("<rect") == 3
    assert "stroke-dasharray" in svg          # the v2/v1 boundary, which is not comparable across


def test_a_table_with_no_runs_draws_nothing():
    assert _cairn_svg([], "score", "v2") == ""


# --- deltas -----------------------------------------------------------------

def test_a_delta_is_good_when_it_moves_the_way_the_column_prefers():
    assert _delta(0.5, 0.7, "higher") == {"value": pytest.approx(0.2), "good": True}
    assert _delta(0.5, 0.7, "lower") == {"value": pytest.approx(0.2), "good": False}
    assert _delta(0.7, 0.5, "lower") == {"value": pytest.approx(-0.2), "good": True}


def test_an_unchanged_value_is_neither_better_nor_worse():
    assert _delta(0.5, 0.5, "higher") == {"value": 0.0, "good": None}


@pytest.mark.parametrize("base, val", [("a", "b"), (0.5, None), (None, 0.5), (True, False), (0.5, True)])
def test_there_is_no_delta_between_things_that_are_not_numbers(base, val):
    assert _delta(base, val, "higher") is None


# --- run metadata -----------------------------------------------------------

def test_the_config_and_what_the_evaluator_recorded_are_shown_together():
    out = _metadata_json({"config": {"threshold": 0.5}, "metadata": {"by_site": {"A": 1}}})
    assert '"threshold": 0.5' in out and '"by_site"' in out


def test_a_run_with_neither_shows_nothing():
    assert _metadata_json({}) == ""
    assert _metadata_json({"config": None, "metadata": {}}) == ""


# --- the dataset chosen in the create form ----------------------------------

def test_the_selected_dataset_is_written_into_the_table_definition():
    out = _with_dataset("name: t\ncolumns: []\n", "clips")
    assert "dataset: clips" in out and "name: t" in out


def test_selecting_nothing_leaves_the_yaml_exactly_as_typed():
    text = "name: t   # mine\ncolumns: []\n"
    assert _with_dataset(text, "") == text


def test_yaml_that_is_not_a_mapping_is_rejected_as_input_not_as_a_crash():
    with pytest.raises(ValueError, match="must be a mapping"):
        _with_dataset("- just\n- a list\n", "clips")
