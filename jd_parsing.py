"""Parse a job description into required/preferred skills and hard must-have requirements.

Must-have/Nice-to-have blocks are found by jd_sections.py, the same way
resume_sections.py finds resume section headers. Taxonomy matching over each
block gives required_skills/preferred_skills (used for scoring). Regexes over
the must-have block separately extract the hard requirements Part B filters
on: a minimum years figure per skill, a bare minimum on total years, and a
minimum degree level.
"""

from __future__ import annotations

import logging
import re

import skills
from jd_sections import block_lines as _block_lines
from match_models import JDRequirements
from resume_metadata import DEGREE_LEVELS

logger = logging.getLogger(__name__)

YEARS_WORD = r"(?:years?|yrs?)"
YEAR_FIGURE_RE = re.compile(rf"(\d+(?:\.\d+)?)\+?\s*{YEARS_WORD}", re.IGNORECASE)
BARE_YEARS_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\+?\s*{YEARS_WORD}\s+of\s+(?:professional\s+)?experience", re.IGNORECASE
)
DEGREE_WORD_RE = re.compile(r"\b(Ph\.?D\.?|Master'?s?|Bachelor'?s?|Associate'?s?)\b", re.IGNORECASE)
DEGREE_OR_EQUIVALENT_RE = re.compile(r"\bor\s+equivalent\b", re.IGNORECASE)


class UnparsableMustHaveError(ValueError):
    """Raised by `parse_must_have_clause` when a clause matched no supported form (R13)."""


def _skill_in_text(alias: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE) is not None


def _matched_skills(lines: list[str]) -> list[str]:
    text = "\n".join(lines)
    return [
        canonical
        for canonical, aliases in skills.SKILL_ALIASES.items()
        if any(_skill_in_text(alias, text) for alias in aliases)
    ]


def _degree_minimum(lines: list[str]) -> int:
    """Highest required degree level, or 0 if none is a hard requirement.

    Degrees joined by "or" on the same line ("Bachelor's or Master's") are
    alternatives, so that line's requirement is their *lowest* level, not
    their highest (R4). "... or equivalent experience" means there is no hard
    degree minimum at all; that line is skipped and the decision is logged.
    """
    line_minimums: list[int] = []
    for line in lines:
        if DEGREE_OR_EQUIVALENT_RE.search(line):
            logger.info(
                "No hard degree minimum from %r ('or equivalent experience').", line.strip()
            )
            continue
        levels = []
        for match in DEGREE_WORD_RE.finditer(line):
            key = re.sub(r"[.'\s]", "", match.group(0)).lower()
            if key in DEGREE_LEVELS:
                levels.append(DEGREE_LEVELS[key])
        if not levels:
            continue
        if len(levels) > 1 and re.search(r"\bor\b", line, re.IGNORECASE):
            line_minimums.append(min(levels))
        else:
            line_minimums.append(max(levels))
    return max(line_minimums, default=0)


def _line_clauses(line: str) -> list[tuple[float, str]]:
    """Split a line at each year figure; each clause runs up to the next one.

    Pairing a skill with its own clause, rather than the whole line, means
    the *nearest preceding* figure wins: "5+ years of Python and 2+ years of
    AWS" gives Python its 5 and AWS its own 2, instead of the first figure
    leaking onto every skill on the line (R3).
    """
    matches = list(YEAR_FIGURE_RE.finditer(line))
    clauses = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        clauses.append((float(match.group(1)), line[match.start() : end]))
    return clauses


def _parse_year_requirements(
    lines: list[str],
) -> tuple[dict[str, float], list[tuple[list[str], float]]]:
    """Per-skill year minimums from the must-have lines.

    A clause naming several skills for the same year figure joined by "or"
    (e.g. "3+ years of experience with TensorFlow or PyTorch") is an
    alternative: any one of them meeting the minimum is enough, so it's kept
    separate from `minimums`, which is an AND across its skills. Skills are
    matched on a word boundary (`_skill_in_text`), so "ts" can't match inside
    "producTS" and "SQL" can't match inside "PostgreSQL" (R2).
    """
    minimums: dict[str, float] = {}
    alternatives: list[tuple[list[str], float]] = []
    for line in lines:
        for years, clause in _line_clauses(line):
            matched = [
                canonical
                for canonical, aliases in skills.SKILL_ALIASES.items()
                if any(_skill_in_text(alias, clause) for alias in aliases)
            ]
            if not matched:
                continue
            if len(matched) > 1 and re.search(r"\bor\b", clause, re.IGNORECASE):
                alternatives.append((matched, years))
            else:
                for skill in matched:
                    minimums[skill] = max(minimums.get(skill, 0.0), years)
    return minimums, alternatives


def _total_years_minimum(lines: list[str]) -> float | None:
    years = [float(m.group(1)) for line in lines for m in BARE_YEARS_RE.finditer(line)]
    return max(years) if years else None


def parse_must_have_clause(
    text: str,
) -> tuple[dict[str, float], list[tuple[list[str], float]], float | None, int]:
    """Parse a single `--must-have` clause, e.g. `"5+ years Python"`, the same way as a JD line.

    Raises `UnparsableMustHaveError` if the clause matched none of the
    supported forms, instead of silently doing nothing (R13).
    """
    lines = [text]
    minimums, alternatives = _parse_year_requirements(lines)
    total = _total_years_minimum(lines)
    degree = _degree_minimum(lines)
    if not minimums and not alternatives and total is None and not degree:
        raise UnparsableMustHaveError(
            f"Could not parse --must-have clause {text!r}. Supported forms: "
            '"5+ years Python", "5+ years of experience", "Bachelor\'s degree" '
            '(also "yr"/"yrs").'
        )
    return minimums, alternatives, total, degree


def parse_jd(text: str) -> JDRequirements:
    lines = [line for line in text.splitlines() if line.strip()]
    title = lines[0].strip() if lines else "Untitled"
    required_lines, preferred_lines = _block_lines(text)
    minimums, alternatives = _parse_year_requirements(required_lines)

    return JDRequirements(
        title=title,
        raw_text=text,
        required_skills=_matched_skills(required_lines),
        preferred_skills=_matched_skills(preferred_lines),
        skill_year_minimums=minimums,
        skill_year_alternatives=alternatives,
        total_years_minimum=_total_years_minimum(required_lines),
        degree_minimum=_degree_minimum(required_lines),
    )
