"""Regression for a side effect of R9 (candidate_id now includes the file extension):
resume_specs.json's own candidate_id field still predates that and is the bare file
stem, so evaluate.py must key specs_by_id by the file name derived from resume_path,
not by the spec's own candidate_id field, or every extraction/filter metric silently
reads as 0 for everyone."""

from __future__ import annotations

from eval_extraction import extraction_accuracy
from resume_models import CandidateProfile


def test_extraction_accuracy_matches_specs_keyed_by_filename_not_bare_stem():
    profile = CandidateProfile(
        candidate_id="jane_doe.pdf",
        name="Jane Doe",
        resume_path="data/resumes/jane_doe.pdf",
        skills=["Python"],
        experience_years=5.0,
        education_level=2,
    )
    spec = {
        "candidate_id": "jane_doe",  # the bare stem, as resume_specs.json stores it
        "resume_path": "data/resumes/jane_doe.pdf",
        "name": "Jane Doe",
        "total_years": 5.0,
        "education": {"level": 2},
        "skills": ["Python"],
        "skill_years": {},
    }

    # Keying by the spec's own candidate_id (the old, stem-only convention) would
    # miss every profile and report 0 accuracy across the board.
    stale_mapping = {spec["candidate_id"]: spec}
    assert extraction_accuracy([profile], stale_mapping)["n"] == 0

    # Keying by the filename derived from resume_path matches the new candidate_id.
    correct_mapping = {"jane_doe.pdf": spec}
    result = extraction_accuracy([profile], correct_mapping)
    assert result["n"] == 1
    assert result["name_accuracy"] == 1.0
