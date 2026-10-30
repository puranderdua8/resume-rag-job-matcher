"""Split resume text into canonical sections: summary, experience, education, etc."""

from __future__ import annotations

CONTACT_SECTION = "contact"
MAX_HEADER_WORDS = 5

SECTION_SYNONYMS: dict[str, str] = {
    "summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "experience": "experience",
    "work history": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "education": "education",
    "skills": "skills",
    "technical skills": "skills",
    "core competencies": "skills",
    "projects": "projects",
    "certifications": "certifications",
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
        sections.setdefault(current, []).append(line)
    return {section: lines for section, lines in sections.items() if lines}
