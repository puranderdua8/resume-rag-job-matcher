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
