"""JD must-have parsing: per-skill year minimums, "or" alternatives, total years, degree."""

from __future__ import annotations

import pytest

from jd_parsing import UnparsableMustHaveError, parse_jd, parse_must_have_clause

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


# R2: the per-skill-years match must be word-bounded, not a bare substring search.


def test_skill_alias_does_not_match_inside_a_longer_word():
    jd = parse_jd("Title\n\nRequirements:\n- 3+ years of experience with data products\n")
    assert jd.skill_year_minimums == {}


def test_skill_alias_does_not_match_inside_an_unrelated_word():
    jd = parse_jd("Title\n\nRequirements:\n- 3+ years of excellent client communication\n")
    assert jd.skill_year_minimums == {}


def test_shorter_alias_does_not_match_inside_a_longer_skill_name():
    jd = parse_jd("Title\n\nRequirements:\n- 5+ years of experience with PostgreSQL\n")
    assert jd.skill_year_minimums == {"PostgreSQL": 5.0}
    assert "SQL" not in jd.skill_year_minimums


# R3: each skill pairs with its own, nearest preceding year figure.


def test_each_skill_gets_its_own_nearest_year_figure():
    jd = parse_jd("Title\n\nRequirements:\n- 5+ years of Python and 2+ years of AWS\n")
    assert jd.skill_year_minimums == {"Python": 5.0, "AWS": 2.0}


# R4: "or"-joined degrees take the lowest level; "or equivalent experience" is no hard minimum.


def test_or_joined_degrees_take_the_lowest_level():
    jd = parse_jd("Title\n\nRequirements:\n- Bachelor's or Master's degree required\n")
    assert jd.degree_minimum == 2


def test_or_equivalent_experience_sets_no_hard_degree_minimum():
    jd = parse_jd("Title\n\nRequirements:\n- Bachelor's degree or equivalent experience\n")
    assert jd.degree_minimum == 0


# R11: widened header synonyms, and an unrecognised header-like line closes the block.


def test_qualifications_header_is_recognised_as_the_required_block():
    jd = parse_jd("Title\n\nQualifications:\n- 5+ years Python\n")
    assert jd.skill_year_minimums == {"Python": 5.0}


def test_unrelated_header_closes_the_open_block():
    jd = parse_jd(
        "Title\n\nRequirements:\n- 5+ years Python\n\nAbout us:\n10+ years in the market\n"
    )
    assert jd.total_years_minimum is None


# R13: an unparsable --must-have clause raises instead of being silently ignored;
# "yr"/"yrs" are also accepted.


def test_unparsable_must_have_clause_raises():
    with pytest.raises(UnparsableMustHaveError):
        parse_must_have_clause("5+ years Rust")


def test_must_have_clause_accepts_yr_and_yrs():
    minimums, _, _, _ = parse_must_have_clause("5 yrs Python")
    assert minimums == {"Python": 5.0}
    minimums, _, _, _ = parse_must_have_clause("3 yr AWS")
    assert minimums == {"AWS": 3.0}


def test_cli_exits_with_a_clear_error_on_an_unparsable_must_have(capsys):
    import job_matcher

    with pytest.raises(SystemExit):
        job_matcher.main(["--text", "Title", "--must-have", "5+ years Rust"])
    assert "5+ years Rust" in capsys.readouterr().err
