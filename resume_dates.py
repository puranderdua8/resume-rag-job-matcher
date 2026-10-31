"""Date-range parsing and interval merging, shared by chunking and metadata extraction.

Resumes write date ranges as "Nov 2023 - Present", "2022-2023" or "04/2023 - 09/2024".
A bare year loses sub-year precision, so it's treated as its midpoint rather than
January 1st, which would bias every merged duration low.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

DATE_TOKEN = r"[A-Za-z]{3,9}\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}"
PRESENT_WORD = r"Present|Current|Now|Today"
SEPARATOR = r"[-–—]|\s+to\s+|\s+until\s+"
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*(?:{SEPARATOR})\s*(?P<end>{PRESENT_WORD}|{DATE_TOKEN})",
    re.IGNORECASE,
)
# Month abbreviations/spellings strptime's "%b"/"%B" don't accept on their own (R16).
_MONTH_ALIASES = {"sept": "sep"}


def _parse_date_token(token: str, as_of: date) -> date:
    token = token.strip()
    if token.lower() in {"present", "current", "now", "today"}:
        return as_of
    if re.fullmatch(r"\d{4}", token):
        # A bare year loses sub-year precision; the middle of the year is the
        # unbiased point estimate given no other info.
        return date(int(token), 7, 2)
    if re.fullmatch(r"\d{1,2}/\d{4}", token):
        month, year = token.split("/")
        return date(int(year), int(month), 1)
    token = token.replace(".", "")
    month_part, _, year_part = token.rpartition(" ")
    normalized = f"{_MONTH_ALIASES.get(month_part.lower(), month_part)} {year_part}"
    for candidate in (normalized, token):
        for fmt in ("%b %Y", "%B %Y"):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"Unrecognised date token: {token!r}")


def parse_date_range(line: str, as_of: date | None = None) -> tuple[date, date] | None:
    as_of = as_of or date.today()
    match = DATE_RANGE_RE.search(line)
    if not match:
        return None
    try:
        start = _parse_date_token(match.group("start"), as_of)
        end = _parse_date_token(match.group("end"), as_of)
    except ValueError:
        return None
    if end <= start:
        end = start + timedelta(days=182)
    return start, end


def merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def years_between(intervals: list[tuple[date, date]]) -> float:
    merged = merge_intervals(intervals)
    days = sum((end - start).days for start, end in merged)
    return round(days / 365.25, 1)
