"""Loading ground truth and resume specs, and building a true-values profile from a spec."""

from __future__ import annotations

import json
from pathlib import Path

import config
from resume_models import CandidateProfile


def load_ground_truth() -> dict[str, dict[str, int]]:
    return json.loads(config.GROUND_TRUTH_PATH.read_text())


def load_resume_specs() -> list[dict]:
    return json.loads(config.RESUME_SPECS_PATH.read_text())


def jd_id_for_path(jd_path: Path) -> str:
    return jd_path.stem.removeprefix("jd_")


def profile_from_spec(spec: dict) -> CandidateProfile:
    """A `CandidateProfile` built from the true resume_specs.json values, so the same
    `check_must_haves` logic can be run against ground truth and against extraction."""
    return CandidateProfile(
        candidate_id=spec["candidate_id"],
        name=spec["name"],
        resume_path=spec["resume_path"],
        skills=spec["skills"],
        skill_years=spec["skill_years"],
        experience_years=spec["total_years"],
        education_level=spec["education"]["level"],
    )
