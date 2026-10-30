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
