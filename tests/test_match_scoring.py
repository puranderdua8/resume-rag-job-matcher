"""Score formula: calibration, and that the final score is always clipped to 0-100."""

from __future__ import annotations

import pytest

import config
from match_models import JDRequirements
from match_scoring import calibrate, score_candidate
from resume_models import CandidateProfile


def _profile(**overrides) -> CandidateProfile:
    defaults = {"candidate_id": "c1", "name": "Candidate", "resume_path": "r.txt"}
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def test_calibrate_clips_to_the_unit_range():
    assert calibrate(0.0, low=0.3, high=0.9) == 0.0
    assert calibrate(1.0, low=0.3, high=0.9) == 1.0
    assert calibrate(0.6, low=0.3, high=0.9) == pytest.approx(0.5)


def test_score_never_exceeds_100_even_with_perfect_components():
    jd = JDRequirements(
        title="t", raw_text="", required_skills=["Python"], preferred_skills=["SQL"]
    )
    profile = _profile(skills=["Python", "SQL"])
    calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]
    scored = score_candidate(
        profile, jd, semantic_raw=1.0, bm25_raw=10.0, bm25_max=10.0, calibration=calibration
    )
    assert scored.score == 100


def test_score_never_drops_below_0_even_with_a_large_soft_penalty():
    jd = JDRequirements(title="t", raw_text="")
    profile = _profile()
    calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]
    scored = score_candidate(
        profile,
        jd,
        semantic_raw=0.0,
        bm25_raw=0.0,
        bm25_max=0.0,
        calibration=calibration,
        filter_penalty=1000.0,
    )
    assert scored.score == 0


def test_no_required_skills_drops_the_component_instead_of_awarding_it_in_full():
    """R12: a JD with no recognised required skills must not hand out its 25 points
    for free (skill_coverage returns 1.0 for an empty target list); the weight is
    dropped and the rest renormalised instead."""
    jd = JDRequirements(title="t", raw_text="")
    profile = _profile()
    calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]
    scored = score_candidate(
        profile, jd, semantic_raw=0.0, bm25_raw=0.0, bm25_max=0.0, calibration=calibration
    )
    assert scored.score == 0
    assert scored.breakdown["required_coverage_scored"] is False
    assert scored.breakdown["preferred_coverage_scored"] is False


def test_required_skills_present_scores_the_component_normally():
    jd = JDRequirements(title="t", raw_text="", required_skills=["Python"])
    profile = _profile(skills=["Python"])
    calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]
    scored = score_candidate(
        profile, jd, semantic_raw=0.0, bm25_raw=0.0, bm25_max=0.0, calibration=calibration
    )
    assert "required_coverage_scored" not in scored.breakdown
    # Only required_coverage (0.25) and preferred_coverage's weight (dropped, no
    # preferred skills) are active alongside semantic/bm25; full required coverage
    # with zero semantic/bm25 renormalises to 0.25 / (0.5 + 0.2 + 0.25) = 26%.
    assert scored.score == round(100 * 0.25 / 0.95)


def test_semantic_override_replaces_the_calibrated_similarity():
    jd = JDRequirements(title="t", raw_text="")
    profile = _profile()
    calibration = config.MODEL_REGISTRY[config.DEFAULT_MODEL]
    scored = score_candidate(
        profile,
        jd,
        semantic_raw=0.0,
        bm25_raw=0.0,
        bm25_max=0.0,
        calibration=calibration,
        semantic_override=1.0,
    )
    assert scored.breakdown["semantic_cal"] == 1.0
