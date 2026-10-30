"""Section detection: header synonyms, inline headers, and text before the first header."""

from __future__ import annotations

from resume_sections import CONTACT_SECTION, detect_sections


def test_canonical_headers_and_contact_text_before_them():
    text = "\n".join(
        [
            "Jane Doe",
            "jane@example.com",
            "Professional Experience:",
            "Did things.",
            "Education",
            "BS in CS, State University",
            "Technical Skills",
            "Python, SQL",
        ]
    )
    sections = detect_sections(text)
    assert sections[CONTACT_SECTION] == ["Jane Doe", "jane@example.com"]
    assert sections["experience"] == ["Did things."]
    assert sections["education"] == ["BS in CS, State University"]
    assert sections["skills"] == ["Python, SQL"]


def test_header_variants_all_map_to_experience():
    for header in ["EXPERIENCE", "Work History", "Professional Experience:", "Employment"]:
        sections = detect_sections(f"{header}\nDid the work.")
        assert sections["experience"] == ["Did the work."]


def test_inline_header_keeps_content_on_the_same_line():
    sections = detect_sections("Name\nSkills: Python, SQL")
    assert sections["skills"] == ["Python, SQL"]


def test_blank_lines_are_dropped():
    sections = detect_sections("Jane Doe\n\n\nEducation\n\nBS in CS, State University")
    assert sections[CONTACT_SECTION] == ["Jane Doe"]
    assert sections["education"] == ["BS in CS, State University"]
