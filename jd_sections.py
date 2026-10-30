"""Find the Must-have/Nice-to-have blocks in a job description.

Mirrors resume_sections.py's header-synonym approach: a short line matching
one of the synonym sets below starts (or switches) a block. Any other short,
non-bulleted, digit-free line closes whichever block is open, so an unrelated
section ("About us", "Benefits") doesn't leak into the must-haves (R11).
"""

from __future__ import annotations

import re

MAX_HEADER_WORDS = 5

REQUIRED_SYNONYMS = {
    "must-have",
    "must have",
    "must haves",
    "requirements",
    "required skills",
    "qualifications",
    "minimum qualifications",
    "basic qualifications",
    "required qualifications",
    "requirements and qualifications",
    "what you'll need",
}
PREFERRED_SYNONYMS = {
    "nice-to-have",
    "nice to have",
    "nice to haves",
    "preferred",
    "preferred skills",
    "preferred qualifications",
    "bonus",
    "bonus points",
}

BULLET_RE = re.compile(r"^[\-*•]+")
_HAS_DIGIT_RE = re.compile(r"\d")


def _is_other_header(line: str) -> bool:
    """A short, non-bulleted, digit-free line: a header, not a requirement."""
    if BULLET_RE.match(line):
        return False
    stripped = line.rstrip(":")
    if not stripped or _HAS_DIGIT_RE.search(stripped):
        return False
    return len(stripped.split()) <= MAX_HEADER_WORDS


def block_lines(text: str) -> tuple[list[str], list[str]]:
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
        if current is not None and _is_other_header(line):
            current = None
            continue
        if current is not None:
            current.append(BULLET_RE.sub("", line).strip())
    return required, preferred
