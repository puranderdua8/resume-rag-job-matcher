"""Post-filter: check the must-haves Chroma's prefilter can't express (per-skill years).

Runs on the fused candidate pool, before the top-K cut, so every eligible
candidate is still in the running when the cut happens. `--soft` keeps a
failing candidate in the ranking with a score penalty instead of dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from match_models import JDRequirements
from resume_models import CandidateProfile


@dataclass
class FilterResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    penalty: float = 0.0


def check_must_haves(jd: JDRequirements, profile: CandidateProfile) -> list[str]:
    """Human-readable reasons `profile` fails `jd`'s must-haves; empty means it passes."""
    reasons = []
    for skill, min_years in jd.skill_year_minimums.items():
        have = profile.skill_years.get(skill, 0.0)
        if have < min_years:
            reasons.append(f"{skill}: {have:.1f} yrs < {min_years:g} required")
    for group_skills, min_years in jd.skill_year_alternatives:
        best = max((profile.skill_years.get(skill, 0.0) for skill in group_skills), default=0.0)
        if best < min_years:
            reasons.append(
                f"{'/'.join(group_skills)}: {best:.1f} yrs < {min_years:g} required "
                f"(any one of {', '.join(group_skills)})"
            )
    if jd.total_years_minimum is not None and profile.experience_years < jd.total_years_minimum:
        reasons.append(
            f"Total experience: {profile.experience_years:.1f} yrs "
            f"< {jd.total_years_minimum:g} required"
        )
    if jd.degree_minimum and profile.education_level < jd.degree_minimum:
        reasons.append(f"Education level {profile.education_level} < {jd.degree_minimum} required")
    return reasons


def filter_candidates(
    jd: JDRequirements, candidates: list[CandidateProfile], soft: bool
) -> dict[str, FilterResult]:
    results: dict[str, FilterResult] = {}
    for profile in candidates:
        reasons = check_must_haves(jd, profile)
        if not reasons:
            results[profile.candidate_id] = FilterResult(passed=True)
        elif soft:
            results[profile.candidate_id] = FilterResult(
                passed=True, reasons=reasons, penalty=config.SOFT_FILTER_PENALTY
            )
        else:
            results[profile.candidate_id] = FilterResult(passed=False, reasons=reasons)
    return results
