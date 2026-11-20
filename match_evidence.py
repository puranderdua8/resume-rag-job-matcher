"""Evidence for a match: matched skills, excerpts, and template or LLM reasoning.

`--llm` sends only the structured evidence this module already built (no raw
resume text) to Gemini for a two-sentence paraphrase, and falls back to the
template on any error, so the feature works identically with no API key.
"""

from __future__ import annotations

import json
import os

import file_tools
from match_filtering import FilterResult
from match_models import JDRequirements
from resume_models import CandidateProfile
from retrieval import SemanticHit

EXCERPT_MAX_CHARS = 300


def matched_skills(profile: CandidateProfile, jd: JDRequirements) -> list[str]:
    """Profile skills that appear in the JD, required skills listed first."""
    required = [skill for skill in jd.required_skills if skill in profile.skills]
    preferred = [
        skill for skill in jd.preferred_skills if skill in profile.skills and skill not in required
    ]
    return required + preferred


def relevant_excerpts(
    profile: CandidateProfile, hit: SemanticHit | None, required_matches: list[str]
) -> list[str]:
    """Top semantic chunks, plus `search_in_file` snippets for up to 2 required skills."""
    excerpts: list[str] = []
    if hit is not None:
        for _, _, text in hit.top_chunks:
            excerpts.append(text[:EXCERPT_MAX_CHARS])
    for skill in required_matches[:2]:
        result = file_tools.search_in_file(profile.source_path or profile.resume_path, skill)
        if result.get("success") and result.get("matches"):
            excerpts.append(result["matches"][0]["context"])
    return excerpts


def must_have_check(filter_result: FilterResult) -> dict:
    check: dict = {"passed": filter_result.passed, "reasons": filter_result.reasons}
    if filter_result.penalty:
        check["soft_penalty"] = filter_result.penalty
    return check


def template_reasoning(
    jd: JDRequirements,
    profile: CandidateProfile,
    score: int,
    section_scores: dict[str, float],
    required_coverage: float,
    filter_result: FilterResult,
) -> str:
    if score >= 80:
        band = "Strong"
    elif score >= 60:
        band = "Good"
    else:
        band = "Partial"

    top_sections = sorted(section_scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    sections_text = (
        ", ".join(f"{section} ({sim:.2f})" for section, sim in top_sections)
        if top_sections
        else "no section-level matches"
    )

    missing_required = [s for s in jd.required_skills if s not in profile.skills]
    coverage_text = f"{required_coverage:.0%} of required skills"
    if missing_required:
        coverage_text += f"; missing {', '.join(missing_required)}"

    if not filter_result.reasons:
        must_have_text = "passed all must-haves"
    elif filter_result.penalty:
        must_have_text = f"soft-filtered ({'; '.join(filter_result.reasons)})"
    else:
        must_have_text = f"failed must-haves ({'; '.join(filter_result.reasons)})"

    return (
        f"{band} match ({score}/100). Closest sections: {sections_text}. "
        f"Required-skill coverage: {coverage_text}. Must-have check: {must_have_text}."
    )


def llm_reasoning(template_text: str, evidence: dict, model_name: str | None = None) -> str:
    """Paraphrase `evidence` into two sentences with Gemini; falls back on any error."""
    try:
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return template_text

        client = genai.Client(api_key=api_key)
        model = model_name or os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"
        prompt = (
            "Summarize this resume-to-job match evidence in exactly two sentences, "
            "for a recruiter. Be concrete about the strongest and weakest points.\n\n"
            f"{json.dumps(evidence, indent=2)}"
        )
        response = client.models.generate_content(model=model, contents=prompt)
        return (response.text or "").strip() or template_text
    except Exception:
        return template_text
