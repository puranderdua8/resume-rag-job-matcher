"""Split resume text into canonical sections: summary, experience, education, etc."""

from __future__ import annotations

CONTACT_SECTION = "contact"
MAX_HEADER_WORDS = 5


def _is_unrecognized_header(line: str) -> bool:
    """A short, all-caps or colon-terminated line not in SECTION_SYNONYMS (R5).

    Narrow on purpose: job titles ("Senior Software Engineer") and other
    short prose lines inside a section must not be mistaken for a header,
    so this only fires for shapes that are unambiguously headers.
    """
    stripped = line.strip()
    if not stripped or len(stripped.split()) > MAX_HEADER_WORDS:
        return False
    has_colon = stripped.endswith(":")
    body = stripped.rstrip(":").strip()
    if not body:
        return False
    if has_colon:
        return True
    return body == body.upper() and any(ch.isalpha() for ch in body)


SECTION_SYNONYMS: dict[str, str] = {
    "summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "professional summary": "summary",
    "experience": "experience",
    "work history": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "work experience": "experience",
    "employment history": "experience",
    "career history": "experience",
    "relevant experience": "experience",
    "education": "education",
    "education & training": "education",
    "skills": "skills",
    "technical skills": "skills",
    "core competencies": "skills",
    "skills & tools": "skills",
    "projects": "projects",
    "certifications": "certifications",
    "publications": "publications",
    "awards": "awards",
    "volunteer": "volunteer",
    "languages": "languages",
    "interests": "interests",
}


def _header_canonical(line: str) -> str | None:
    stripped = line.strip().rstrip(":")
    if not stripped or len(stripped.split()) > MAX_HEADER_WORDS:
        return None
    return SECTION_SYNONYMS.get(stripped.lower())


def _inline_header(line: str) -> tuple[str, str] | None:
    """Match `"Skills: Python, SQL"` style headers: label, colon, content on one line."""
    if ":" not in line:
        return None
    label, _, rest = line.partition(":")
    if not rest.strip():
        return None
    canonical = SECTION_SYNONYMS.get(label.strip().lower())
    if canonical and len(label.strip().split()) <= MAX_HEADER_WORDS:
        return canonical, rest.strip()
    return None


def detect_sections(text: str) -> dict[str, list[str]]:
    """Split resume text into canonical sections, preserving line order within each."""
    sections: dict[str, list[str]] = {CONTACT_SECTION: []}
    current = CONTACT_SECTION
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        canonical = _header_canonical(line)
        if canonical:
            current = canonical
            sections.setdefault(current, [])
            continue
        inline = _inline_header(line)
        if inline:
            canonical, content = inline
            current = canonical
            sections.setdefault(current, []).append(content)
            continue
        if current != CONTACT_SECTION and _is_unrecognized_header(line):
            current = line.strip().rstrip(":").strip().lower()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {section: lines for section, lines in sections.items() if lines}
