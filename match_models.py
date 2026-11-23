"""Structured data for Part B: parsed JD requirements and one candidate's match result."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class JDRequirements:
    title: str
    raw_text: str
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    skill_year_minimums: dict[str, float] = field(default_factory=dict)
    skill_year_alternatives: list[tuple[list[str], float]] = field(default_factory=list)
    total_years_minimum: float | None = None
    degree_minimum: int = 0


def condensed_requirements_text(jd: JDRequirements) -> str:
    """Title + requirement lines only, not the whole JD text: used both to query the
    cross-encoder (R19, cheaper per-candidate inference) and as the BM25 reference
    document (R20, a stable per-JD normaliser built from the same requirement
    signal, not the full posting's narrative text)."""
    lines = [jd.title]
    for skill, min_years in jd.skill_year_minimums.items():
        lines.append(f"{skill} {min_years:g}+ years")
    for group_skills, min_years in jd.skill_year_alternatives:
        lines.append(f"{'/'.join(group_skills)} {min_years:g}+ years")
    if jd.total_years_minimum is not None:
        lines.append(f"{jd.total_years_minimum:g}+ years of experience")
    lines.extend(jd.required_skills)
    lines.extend(jd.preferred_skills)
    return "\n".join(lines)


@dataclass
class MatchResult:
    candidate_name: str
    resume_path: str
    match_score: int
    matched_skills: list[str]
    relevant_excerpts: list[str]
    reasoning: str
    score_breakdown: dict[str, float] = field(default_factory=dict)
    section_scores: dict[str, float] = field(default_factory=dict)
    must_have_check: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
