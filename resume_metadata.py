"""Extract a CandidateProfile (name, contact, skills, years, education) from a resume."""

from __future__ import annotations

import re
from datetime import date

import skills
from resume_chunking import split_experience_entries
from resume_dates import parse_date_range, years_between
from resume_models import CandidateProfile
from resume_sections import CONTACT_SECTION

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
NAME_RE = re.compile(r"^(?:[A-Z][a-zA-Z'.-]*\s+){1,3}[A-Z][a-zA-Z'.-]*$")
EXPLICIT_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*years?", re.IGNORECASE)
EDU_RE = re.compile(
    r"^(?P<degree>Ph\.?D\.?|M\.?S\.?|M\.?A\.?|B\.?S\.?|B\.?A\.?|Associate'?s?|Bachelor'?s?|Master'?s?)"
    r"\s+(?:degree\s+)?in\s+(?P<field>[^,]+),\s*(?P<institution>.+)$",
    re.IGNORECASE,
)
DEGREE_LEVELS: dict[str, int] = {
    "phd": 4,
    "doctorate": 4,
    "ms": 3,
    "ma": 3,
    "master": 3,
    "masters": 3,
    "bs": 2,
    "ba": 2,
    "bachelor": 2,
    "bachelors": 2,
    "as": 1,
    "aa": 1,
    "associate": 1,
    "associates": 1,
}


def _skill_in_text(alias: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE) is not None


def _extract_name(sections: dict[str, list[str]], candidate_id: str) -> str:
    for line in sections.get(CONTACT_SECTION, []):
        candidate = line.strip()
        if "@" in candidate or "|" in candidate or "http" in candidate.lower():
            continue
        if any(ch.isdigit() for ch in candidate):
            continue
        words = candidate.split()
        if 2 <= len(words) <= 4 and NAME_RE.match(candidate):
            return candidate
    return candidate_id.replace("_", " ").replace("-", " ").title()


def _extract_explicit_years(sections: dict[str, list[str]]) -> float | None:
    for section in ("summary", CONTACT_SECTION):
        for line in sections.get(section, []):
            match = EXPLICIT_YEARS_RE.search(line)
            if match:
                return float(match.group(1))
    return None


def _extract_skills(sections: dict[str, list[str]]) -> tuple[list[str], dict[str, list[str]]]:
    matched: list[str] = []
    skill_sections: dict[str, list[str]] = {}
    for canonical, aliases in skills.SKILL_ALIASES.items():
        found_in: list[str] = []
        for section, lines in sections.items():
            section_text = "\n".join(lines)
            if any(_skill_in_text(alias, section_text) for alias in aliases):
                found_in.append(section)
        if found_in:
            matched.append(canonical)
            skill_sections[canonical] = found_in
    return matched, skill_sections


def _extract_education(
    sections: dict[str, list[str]],
) -> tuple[str | None, str | None, str | None, int]:
    for line in sections.get("education", []):
        match = EDU_RE.match(line.strip())
        if not match:
            continue
        degree = match.group("degree")
        degree_key = re.sub(r"[.'\s]", "", degree).lower()
        level = DEGREE_LEVELS.get(degree_key, 0)
        return degree, match.group("field").strip(), match.group("institution").strip(), level
    return None, None, None, 0


def _experience_entries_with_dates(
    sections: dict[str, list[str]],
) -> list[tuple[tuple[date, date], str]]:
    entries = split_experience_entries(sections.get("experience", []))
    result = []
    for entry in entries:
        entry_text = "\n".join(entry)
        for line in entry:
            date_range = parse_date_range(line)
            if date_range:
                result.append((date_range, entry_text))
                break
    return result


def extract_metadata(
    candidate_id: str,
    resume_path: str,
    full_text: str,
    sections: dict[str, list[str]],
) -> CandidateProfile:
    name = _extract_name(sections, candidate_id)
    email_match = EMAIL_RE.search(full_text)
    phone_match = PHONE_RE.search(full_text)
    matched_skills, skill_sections = _extract_skills(sections)
    dated_entries = _experience_entries_with_dates(sections)

    explicit_years = _extract_explicit_years(sections)
    if explicit_years is not None:
        experience_years = explicit_years
    else:
        experience_years = years_between([dr for dr, _ in dated_entries])

    skill_years: dict[str, float] = {}
    for canonical_skill, aliases in skills.SKILL_ALIASES.items():
        if canonical_skill not in matched_skills:
            continue
        intervals = [
            date_range
            for date_range, entry_text in dated_entries
            if any(_skill_in_text(alias, entry_text) for alias in aliases)
        ]
        if intervals:
            skill_years[canonical_skill] = years_between(intervals)
        elif explicit_years is not None and not dated_entries:
            skill_years[canonical_skill] = explicit_years

    degree, edu_field, institution, level = _extract_education(sections)

    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        resume_path=resume_path,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        skills=matched_skills,
        skill_sections=skill_sections,
        skill_years=skill_years,
        experience_years=experience_years,
        education_level=level,
        education_degree=degree,
        education_field=edu_field,
        education_institution=institution,
        sections={section: "\n".join(lines) for section, lines in sections.items()},
        full_text=full_text,
    )
