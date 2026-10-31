"""Date-range parsing, interval merging, and the derived years-of-experience sum."""

from __future__ import annotations

from datetime import date

from resume_dates import merge_intervals, parse_date_range, years_between


def test_parses_month_year_present_with_en_dash():
    start, end = parse_date_range("Nov 2023 – Present")
    assert start == date(2023, 11, 1)
    assert end == date.today()


def test_parses_bare_year_range_as_midpoints():
    start, end = parse_date_range("2017-2020")
    assert start == date(2017, 7, 2)
    assert end == date(2020, 7, 2)


def test_parses_mm_yyyy_range():
    start, end = parse_date_range("03/2018 - 06/2021")
    assert start == date(2018, 3, 1)
    assert end == date(2021, 6, 1)


def test_no_date_range_returns_none():
    assert parse_date_range("Did some unrelated work.") is None


def test_same_start_and_end_is_padded_forward_instead_of_zero_length():
    start, end = parse_date_range("2020-2020")
    assert end > start


def test_merge_intervals_combines_overlapping_and_keeps_disjoint_ranges_separate():
    merged = merge_intervals(
        [
            (date(2019, 1, 1), date(2020, 6, 1)),
            (date(2020, 1, 1), date(2021, 1, 1)),
            (date(2022, 1, 1), date(2022, 6, 1)),
        ]
    )
    assert merged == [
        (date(2019, 1, 1), date(2021, 1, 1)),
        (date(2022, 1, 1), date(2022, 6, 1)),
    ]


def test_years_between_sums_merged_durations():
    intervals = [
        (date(2019, 1, 1), date(2020, 1, 1)),
        (date(2020, 1, 1), date(2021, 1, 1)),
    ]
    assert years_between(intervals) == 2.0


# --- R16: abbreviated months, "to"/"until" separators, Current/Now/Today, injectable as_of ---


def test_parses_sept_abbreviation():
    start, _ = parse_date_range("Sept 2019 - Dec 2020")
    assert start == date(2019, 9, 1)


def test_parses_to_separator():
    start, end = parse_date_range("2019 to 2021")
    assert start == date(2019, 7, 2)
    assert end == date(2021, 7, 2)


def test_parses_until_separator():
    start, end = parse_date_range("Jan 2019 until Dec 2020")
    assert start == date(2019, 1, 1)
    assert end == date(2020, 12, 1)


def test_current_and_now_and_today_mean_present():
    for word in ("Current", "Now", "Today"):
        _, end = parse_date_range(f"Jan 2020 - {word}", as_of=date(2026, 1, 1))
        assert end == date(2026, 1, 1)


def test_as_of_is_injectable_and_pins_present():
    start, end = parse_date_range("Nov 2023 - Present", as_of=date(2026, 1, 1))
    assert start == date(2023, 11, 1)
    assert end == date(2026, 1, 1)
