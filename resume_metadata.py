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
# Prefers an explicit "N years (of) (professional) experience" phrase over any other
# figure in the summary (R14): up to two filler words between "years" and "experience"
# covers "years of professional experience" and "years total experience".
EXPERIENCE_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*years?(?:\s+\w+){0,2}?\s+experience", re.IGNORECASE
)
# A bare "N years" figure whose nearby context names a team/project size rather than
# tenure ("a team of 4 for 2 years") is excluded from the fallback figure search (R14).
OTHER_NOUN_CONTEXT_RE = re.compile(r"\b(?:team|project)s?\b", re.IGNORECASE)

# Degree spellings, longest/most-specific alternatives first so e.g. "M.Sc" and
# "Master of Science" match in full rather than stopping at "M.S" or "Master's" (R6).
_DEGREE_ALTERNATION = (
    r"Ph\.?D\.?|Doctor(?:ate)?(?:\s+of\s+Philosophy)?|"
    r"M\.?B\.?A\.?|"
    r"Master'?s?\s+of\s+(?:Science|Arts)|Master'?s?|M\.?Sc\.?|M\.?S\.?|M\.?A\.?|"
    r"Bachelor'?s?\s+of\s+(?:Science|Arts)|Bachelor'?s?|B\.?Tech\.?|B\.?E\.?|B\.?Sc\.?|B\.?S\.?|B\.?A\.?|"
    r"Associate'?s?|A\.?S\.?|A\.?A\.?"
)
DEGREE_RE = re.compile(rf"(?<![A-Za-z])(?P<degree>{_DEGREE_ALTERNATION})(?![A-Za-z])")
EDU_DETAIL_RE = re.compile(
    r"\s*(?:degree\s+)?in\s+(?P<field>[^,]+),\s*(?P<institution>.+)", re.IGNORECASE
)
DEGREE_LEVELS: dict[str, int] = {
    "phd": 4,
    "doctorate": 4,
    "doctor": 4,
    "doctorofphilosophy": 4,
    "mba": 3,
    "ms": 3,
    "msc": 3,
    "ma": 3,
    "master": 3,
    "masters": 3,
    "masterofscience": 3,
    "masterofarts": 3,
    "bs": 2,
    "bsc": 2,
    "ba": 2,
    "bachelor": 2,
    "bachelors": 2,
    "bachelorofscience": 2,
    "bachelorofarts": 2,
    "btech": 2,
    "be": 2,
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


def _extract_explicit_years(
    sections: dict[str, list[str]], dated_years: float | None
) -> float | None:
    text = "\n".join(sections.get("summary", []) + sections.get(CONTACT_SECTION, []))
    experience_matches = [float(m.group(1)) for m in EXPERIENCE_YEARS_RE.finditer(text)]
    if experience_matches:
        return max(experience_matches)

    candidates: list[float] = []
    for match in EXPLICIT_YEARS_RE.finditer(text):
        context = text[max(0, match.start() - 25) : match.start()]
        if OTHER_NOUN_CONTEXT_RE.search(context):
            continue
        candidates.append(float(match.group(1)))
    if not candidates:
        return None
    if dated_years is not None:
        consistent = [c for c in candidates if c >= dated_years]
        if consistent:
            return max(consistent)
    return max(candidates)


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


def _claimed_skill_years(sections: dict[str, list[str]]) -> dict[str, float]:
    """Skill-specific year claims in the Summary, e.g. "6 years of Python" (R7)."""
    text = "\n".join(sections.get("summary", []))
    matches = list(EXPLICIT_YEARS_RE.finditer(text))
    claims: dict[str, float] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause = text[match.start() : end]
        years = float(match.group(1))
        for canonical, aliases in skills.SKILL_ALIASES.items():
            if any(_skill_in_text(alias, clause) for alias in aliases):
                claims[canonical] = max(claims.get(canonical, 0.0), years)
    return claims


def _parse_education_line(
    line: str,
) -> tuple[str, str | None, str | None, int] | None:
    match = DEGREE_RE.search(line)
    if not match:
        return None
    degree = match.group("degree")
    degree_key = re.sub(r"[.'\s]", "", degree).lower()
    level = DEGREE_LEVELS.get(degree_key, 0)
    detail = EDU_DETAIL_RE.match(line[match.end() :])
    field = detail.group("field").strip() if detail else None
    institution = detail.group("institution").strip() if detail else None
    return degree, field, institution, level


def _extract_education(
    sections: dict[str, list[str]],
) -> tuple[str | None, str | None, str | None, int]:
    best: tuple[str, str | None, str | None, int] | None = None
    for line in sections.get("education", []):
        parsed = _parse_education_line(line.strip())
        if parsed is None:
            continue
        if best is None or parsed[3] > best[3]:
            best = parsed
    if best is None:
        return None, None, None, 0
    return best


def _experience_entries_with_dates(
    sections: dict[str, list[str]], as_of: date
) -> list[tuple[tuple[date, date], str]]:
    entries = split_experience_entries(sections.get("experience", []))
    result = []
    for entry in entries:
        entry_text = "\n".join(entry)
        for line in entry:
            date_range = parse_date_range(line, as_of=as_of)
            if date_range:
                result.append((date_range, entry_text))
                break
    return result


def extract_metadata(
    candidate_id: str,
    resume_path: str,
    full_text: str,
    sections: dict[str, list[str]],
    as_of: date | None = None,
) -> CandidateProfile:
    as_of = as_of or date.today()
    name = _extract_name(sections, candidate_id)
    email_match = EMAIL_RE.search(full_text)
    phone_match = PHONE_RE.search(full_text)
    matched_skills, skill_sections = _extract_skills(sections)
    dated_entries = _experience_entries_with_dates(sections, as_of)

    dated_years = years_between([dr for dr, _ in dated_entries])
    explicit_years = _extract_explicit_years(sections, dated_years)
    experience_years = explicit_years if explicit_years is not None else dated_years

    claimed_years = _claimed_skill_years(sections)

    skill_years: dict[str, float] = {}
    for canonical_skill, aliases in skills.SKILL_ALIASES.items():
        if canonical_skill not in matched_skills:
            continue
        intervals = [
            date_range
            for date_range, entry_text in dated_entries
            if any(_skill_in_text(alias, entry_text) for alias in aliases)
        ]
        evidence_years = years_between(intervals) if intervals else 0.0
        claimed = claimed_years.get(canonical_skill, 0.0)
        # A skill can be confirmed present (matched_skills) without being named inside
        # any dated entry, e.g. a resume names a framework in a bullet but keeps the
        # base language to its one Skills line. Crediting it with the candidate's total
        # experience only when it's actually *listed* in Skills/Summary (rather than
        # unconditionally) keeps the overall skill_years policy monotonic: naming a
        # skill in one more place never lowers its years (R7).
        listed_years = (
            experience_years
            if any(sec in ("skills", "summary") for sec in skill_sections.get(canonical_skill, []))
            else 0.0
        )
        skill_years[canonical_skill] = max(evidence_years, claimed, listed_years)

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
