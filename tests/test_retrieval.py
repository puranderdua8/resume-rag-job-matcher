"""Chroma prefilter construction and Reciprocal Rank Fusion."""

from __future__ import annotations

from match_models import JDRequirements
from retrieval import build_prefilter, rrf_fuse


def _jd(**kwargs) -> JDRequirements:
    return JDRequirements(title="t", raw_text="", **kwargs)


def test_prefilter_is_none_with_no_must_haves():
    assert build_prefilter(_jd()) is None


def test_prefilter_single_condition_is_not_wrapped_in_and():
    jd = _jd(total_years_minimum=5.0)
    assert build_prefilter(jd) == {"experience_years": {"$gte": 5.0}}


def test_prefilter_combines_skills_years_and_degree_with_and():
    jd = _jd(skill_year_minimums={"Python": 5.0}, total_years_minimum=5.0, degree_minimum=2)
    where = build_prefilter(jd)
    assert where == {
        "$and": [
            {"skills": {"$contains": "Python"}},
            {"experience_years": {"$gte": 5.0}},
            {"education_level": {"$gte": 2}},
        ]
    }


def test_prefilter_alternative_group_becomes_or():
    jd = _jd(skill_year_alternatives=[(["TensorFlow", "PyTorch"], 3.0)])
    assert build_prefilter(jd) == {
        "$or": [{"skills": {"$contains": "TensorFlow"}}, {"skills": {"$contains": "PyTorch"}}]
    }


def test_rrf_fuse_rewards_candidates_ranked_high_in_both_lists_equally():
    rankings = [["a", "b", "c"], ["b", "a", "c"]]
    scores = rrf_fuse(rankings, k=60)
    assert scores["a"] == 1 / 61 + 1 / 62
    assert scores["b"] == 1 / 62 + 1 / 61
    assert scores["a"] == scores["b"]
    assert scores["c"] < scores["a"]


def test_rrf_fuse_with_no_rankings_is_empty():
    assert rrf_fuse([], k=60) == {}
