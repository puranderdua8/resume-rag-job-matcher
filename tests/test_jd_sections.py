"""jd_sections.block_lines: widened header synonyms and block-closing on unknown headers (R11)."""

from __future__ import annotations

from jd_sections import block_lines


def test_widened_required_synonyms_are_recognised():
    for header in ["Qualifications", "Minimum Qualifications", "What you'll need"]:
        required, _ = block_lines(f"Title\n\n{header}:\n- 5+ years Python\n")
        assert required == ["5+ years Python"]


def test_widened_preferred_synonyms_are_recognised():
    for header in ["Preferred Qualifications", "Bonus points"]:
        _, preferred = block_lines(f"Title\n\n{header}:\n- Experience with Redis\n")
        assert preferred == ["Experience with Redis"]


def test_unrelated_header_closes_the_required_block():
    required, _ = block_lines(
        "Title\n\nRequirements:\n- 5+ years Python\n\nAbout us:\n10+ years in the market\n"
    )
    assert required == ["5+ years Python"]


def test_bulleted_lines_do_not_close_the_block():
    required, _ = block_lines("Title\n\nRequirements:\n- Python\n- AWS\n- Docker\n")
    assert required == ["Python", "AWS", "Docker"]
