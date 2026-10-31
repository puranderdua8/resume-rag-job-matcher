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


def test_work_experience_header_maps_to_experience():
    """R5: "Work Experience" was the most common missing header; its content used to
    land in whatever section preceded it, zeroing out date-derived years."""
    sections = detect_sections("Name\nSummary\nBuilt things.\nWork Experience\nDid the work.")
    assert sections["experience"] == ["Did the work."]
    assert sections["summary"] == ["Built things."]


def test_other_new_synonyms_map_to_their_canonical_section():
    for header, canonical in [
        ("Employment History", "experience"),
        ("Career History", "experience"),
        ("Relevant Experience", "experience"),
        ("Professional Summary", "summary"),
        ("Education & Training", "education"),
        ("Skills & Tools", "skills"),
    ]:
        sections = detect_sections(f"{header}\nSome content.")
        assert sections[canonical] == ["Some content."]


def test_unrecognized_header_gets_its_own_section_instead_of_merging():
    """R5: a header with no canonical meaning (not in SECTION_SYNONYMS) must not
    silently merge into whichever section preceded it."""
    text = "Name\nSummary\nBuilt things.\nVOLUNTEER WORK\nTaught kids to code."
    sections = detect_sections(text)
    assert sections["summary"] == ["Built things."]
    assert sections["volunteer work"] == ["Taught kids to code."]


def test_colon_terminated_unrecognized_header_also_gets_its_own_section():
    text = "Name\nSummary\nBuilt things.\nAwards & Honors:\nEmployee of the month."
    sections = detect_sections(text)
    assert sections["summary"] == ["Built things."]
    assert sections["awards & honors"] == ["Employee of the month."]


def test_job_titles_are_not_mistaken_for_unrecognized_headers():
    """Short Title-Case lines inside a section (job titles) must stay content."""
    text = "Name\nExperience\nSenior Software Engineer\nJan 2020 - Present\n- Did work"
    sections = detect_sections(text)
    assert "Senior Software Engineer" in sections["experience"]
