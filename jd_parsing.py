"""Parse a job description into required/preferred skills and hard must-have requirements.

Must-have/Nice-to-have blocks are found with header synonyms, the same way
resume_sections.py finds resume section headers. Taxonomy matching over each
block gives required_skills/preferred_skills (used for scoring). Regexes over
the must-have block separately extract the hard requirements Part B filters
on: a minimum years figure per skill, a bare minimum on total years, and a
minimum degree level.
"""

from __future__ import annotations

import re

import skills
from match_models import JDRequirements
from resume_metadata import DEGREE_LEVELS

REQUIRED_SYNONYMS = {"must-have", "must have", "must haves", "requirements", "required skills"}
PREFERRED_SYNONYMS = {"nice-to-have", "nice to have", "preferred", "preferred skills", "bonus"}

BULLET_RE = re.compile(r"^[\s\-*•]+")
PER_SKILL_YEARS_RE_TEMPLATE = r"(\d+(?:\.\d+)?)\+?\s*years?[^.\n]{{0,40}}?{alias}"
BARE_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+(?:professional\s+)?experience", re.IGNORECASE
)
DEGREE_WORD_RE = re.compile(r"\b(Ph\.?D\.?|Master'?s?|Bachelor'?s?|Associate'?s?)\b", re.IGNORECASE)


def _block_lines(text: str) -> tuple[list[str], list[str]]:
    """Split JD text into (required_block_lines, preferred_block_lines)."""
    required: list[str] = []
    preferred: list[str] = []
    current: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = line.rstrip(":").lower()
        if header in REQUIRED_SYNONYMS:
            current = required
            continue
        if header in PREFERRED_SYNONYMS:
            current = preferred
            continue
        if current is not None:
            current.append(BULLET_RE.sub("", line).strip())
    return required, preferred


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
    levels = []
    for line in lines:
        for match in DEGREE_WORD_RE.finditer(line):
            key = re.sub(r"[.'\s]", "", match.group(0)).lower()
            if key in DEGREE_LEVELS:
                levels.append(DEGREE_LEVELS[key])
    return max(levels, default=0)


def _parse_year_requirements(
    lines: list[str],
) -> tuple[dict[str, float], list[tuple[list[str], float]]]:
    """Per-skill year minimums from the must-have lines.

    A line naming several skills for the same year figure joined by "or"
    (e.g. "3+ years of experience with TensorFlow or PyTorch") is an
    alternative: any one of them meeting the minimum is enough, so it's kept
    separate from `minimums`, which is an AND across its skills.
    """
    minimums: dict[str, float] = {}
    alternatives: list[tuple[list[str], float]] = []
    for line in lines:
        matches: list[tuple[str, float]] = []
        for canonical, aliases in skills.SKILL_ALIASES.items():
            for alias in aliases:
                pattern = re.compile(
                    PER_SKILL_YEARS_RE_TEMPLATE.format(alias=re.escape(alias)), re.IGNORECASE
                )
                match = pattern.search(line)
                if match:
                    matches.append((canonical, float(match.group(1))))
                    break
        is_alternative = bool(re.search(r"\bor\b", line, re.IGNORECASE))
        for years in {years for _, years in matches}:
            group = [canonical for canonical, y in matches if y == years]
            if is_alternative and len(group) > 1:
                alternatives.append((group, years))
            else:
                for skill in group:
                    minimums[skill] = max(minimums.get(skill, 0.0), years)
    return minimums, alternatives


def _total_years_minimum(lines: list[str]) -> float | None:
    years = [float(m.group(1)) for line in lines for m in BARE_YEARS_RE.finditer(line)]
    return max(years) if years else None


def parse_must_have_clause(
    text: str,
) -> tuple[dict[str, float], list[tuple[list[str], float]], float | None, int]:
    """Parse a single `--must-have` clause, e.g. `"5+ years Python"`, the same way as a JD line."""
    lines = [text]
    minimums, alternatives = _parse_year_requirements(lines)
    return minimums, alternatives, _total_years_minimum(lines), _degree_minimum(lines)


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
