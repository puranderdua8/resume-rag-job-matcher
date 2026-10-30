"""JD must-have parsing: per-skill year minimums, "or" alternatives, total years, degree."""

from __future__ import annotations

from jd_parsing import parse_jd, parse_must_have_clause

SENIOR_ML_JD = """Senior ML Engineer

Requirements:
- 5+ years of professional experience
- 3+ years of experience with TensorFlow or PyTorch
- Master's degree in Computer Science or related field

Nice to have:
- Experience with MLOps
"""


def test_total_years_minimum_and_degree_minimum():
    jd = parse_jd(SENIOR_ML_JD)
    assert jd.total_years_minimum == 5.0
    assert jd.degree_minimum == 3


def test_or_joined_skills_become_an_alternative_group_not_independent_minimums():
    jd = parse_jd(SENIOR_ML_JD)
    assert "TensorFlow" not in jd.skill_year_minimums
    assert "PyTorch" not in jd.skill_year_minimums
    groups = [(sorted(group), years) for group, years in jd.skill_year_alternatives]
    assert (sorted(["TensorFlow", "PyTorch"]), 3.0) in groups


def test_and_joined_skills_each_get_their_own_minimum():
    jd = parse_jd("Title\n\nRequirements:\n- 4+ years Python and Django\n")
    assert jd.skill_year_minimums == {"Python": 4.0, "Django": 4.0}
    assert jd.skill_year_alternatives == []


def test_required_and_preferred_skills_come_from_their_own_blocks():
    jd = parse_jd(SENIOR_ML_JD)
    assert "MLOps" in jd.preferred_skills
    assert "MLOps" not in jd.required_skills


def test_parse_must_have_clause_sets_a_single_skill_minimum():
    minimums, alternatives, total, degree = parse_must_have_clause("5+ years Python")
    assert minimums == {"Python": 5.0}
    assert alternatives == []
    assert total is None
    assert degree == 0
