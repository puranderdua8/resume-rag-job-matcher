"""Name extraction, skill/alias matching, and education parsing from resume sections."""

from __future__ import annotations

from resume_metadata import extract_metadata
from resume_sections import detect_sections


def _profile(text: str, candidate_id: str = "some_candidate"):
    sections = detect_sections(text)
    return extract_metadata(candidate_id, "resumes/x.txt", text, sections)


def test_extracts_name_from_the_first_plain_contact_line():
    profile = _profile("Jane Doe\njane@example.com\nSkills\nPython")
    assert profile.name == "Jane Doe"


def test_falls_back_to_the_candidate_id_when_no_line_looks_like_a_name():
    profile = _profile("jane@example.com\n555-123-4567\nSkills\nPython", candidate_id="jane_doe")
    assert profile.name == "Jane Doe"


def test_matches_skills_via_their_aliases():
    profile = _profile("Name\nSkills\nk8s, ML, Postgres")
    assert "Kubernetes" in profile.skills
    assert "Machine Learning" in profile.skills
    assert "PostgreSQL" in profile.skills


def test_education_extracts_degree_field_and_institution():
    profile = _profile("Name\nEducation\nBachelor's in Computer Science, State University")
    assert profile.education_degree == "Bachelor's"
    assert profile.education_field == "Computer Science"
    assert profile.education_institution == "State University"
    assert profile.education_level == 2


def test_explicit_years_statement_overrides_date_derived_years():
    text = "Name\nSummary\n6+ years of experience.\nExperience\nJan 2019 - Present\n- Did work"
    profile = _profile(text)
    assert profile.experience_years == 6.0


# --- R6: broader degree forms, highest level across all education lines ---


def test_bachelor_of_science_phrasing_gets_the_right_level():
    profile = _profile("Name\nEducation\nBachelor of Science in Computer Science, State U")
    assert profile.education_level == 2
    assert profile.education_field == "Computer Science"
    assert profile.education_institution == "State U"


def test_mba_btech_and_msc_are_recognised():
    for line, level in [
        ("MBA, Business School", 3),
        ("B.Tech in Computer Science, IIT", 2),
        ("M.Sc in Data Science, Tech University", 3),
    ]:
        profile = _profile(f"Name\nEducation\n{line}")
        assert profile.education_level == level


def test_degree_level_does_not_require_in_field_institution_shape():
    """R6: don't require the strict "<degree> in <field>, <institution>" shape
    just to get the level."""
    profile = _profile("Name\nEducation\nDoctorate, Self-taught")
    assert profile.education_level == 4


def test_highest_degree_level_wins_across_education_lines():
    text = (
        "Name\nEducation\nB.S. in Computer Science, State University\n"
        "M.S. in Computer Science, Tech University"
    )
    profile = _profile(text)
    assert profile.education_level == 3


# --- R7: skill_years is monotonic (max of evidence, claims, listed-total) ---


def test_skill_named_only_on_skills_line_gets_total_years():
    text = (
        "Name\nSummary\n11 years of experience.\nSkills\nPython\n"
        "Experience\nJan 2015 - Present\n- Did work"
    )
    profile = _profile(text)
    assert profile.skill_years["Python"] == 11.0


def test_naming_a_skill_in_a_short_bullet_does_not_lower_its_listed_years():
    """R7: appending a skill's name to one dated bullet must never lower its years
    below what it would get from being listed on the Skills line alone."""
    text = (
        "Name\nSummary\n11 years of experience.\nSkills\nPython\n"
        "Experience\nJan 2015 - Feb 2016\n- Used Python for a short internship\n"
        "Mar 2016 - Present\n- Did other work"
    )
    profile = _profile(text)
    assert profile.skill_years["Python"] >= 11.0


def test_skill_specific_claim_in_summary_is_captured():
    text = "Name\nSummary\n6 years of Python experience.\nSkills\nPython"
    profile = _profile(text)
    assert profile.skill_years["Python"] == 6.0


def test_skill_years_is_monotonic_across_resume_variants():
    """Property-style check: for a handful of resumes, appending a matched skill's
    alias to one more dated bullet never lowers that skill's years (R7)."""
    base_variants = [
        "Name\nSkills\nPython\nExperience\nJan 2015 - Present\n- Led a team",
        "Name\nSummary\n8 years of experience.\nSkills\nPython\nExperience\n"
        "Jan 2015 - Jun 2015\n- Did setup work\nJul 2015 - Present\n- Shipped features",
        "Name\nSummary\n3 years of Python.\nSkills\nPython\nExperience\n"
        "Jan 2022 - Jun 2022\n- Shipped one feature",
    ]
    for text in base_variants:
        before = _profile(text).skill_years.get("Python", 0.0)
        augmented = text + "\n- Used Python here too"
        after = _profile(augmented).skill_years.get("Python", 0.0)
        assert after >= before


# --- R14: prefer "N years (of) experience" phrasing; ignore team/project figures ---


def test_prefers_explicit_experience_phrase_over_other_year_figures():
    text = "Name\nSummary\nLed a team of 4 for 2 years.\n12 years of professional experience."
    profile = _profile(text)
    assert profile.experience_years == 12.0


def test_ignores_a_years_figure_attached_to_a_team_or_project_noun():
    text = "Name\nSummary\nLed a team of 4 for 2 years."
    profile = _profile(text)
    assert profile.experience_years == 0.0
