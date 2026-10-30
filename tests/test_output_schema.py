"""The output contract AGENTS.md pins: every top_matches entry must keep these keys."""

from __future__ import annotations

from match_models import MatchResult

REQUIRED_KEYS = {
    "candidate_name",
    "resume_path",
    "match_score",
    "matched_skills",
    "relevant_excerpts",
    "reasoning",
}


def test_match_result_to_dict_keeps_the_required_output_keys():
    result = MatchResult(
        candidate_name="Jane Doe",
        resume_path="resumes/jane.txt",
        match_score=90,
        matched_skills=["Python"],
        relevant_excerpts=["..."],
        reasoning="Strong match.",
    )
    data = result.to_dict()
    assert REQUIRED_KEYS <= data.keys()
    assert isinstance(data["match_score"], int)
    assert 0 <= data["match_score"] <= 100
