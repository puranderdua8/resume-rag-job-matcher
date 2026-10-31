"""Extraction accuracy: how the extracted CandidateProfile fields compare with the true
values in resume_specs.json (name exact match, skills precision/recall, years within
±1, degree accuracy)."""

from __future__ import annotations

from resume_models import CandidateProfile


def extraction_accuracy(candidates: list[CandidateProfile], specs_by_id: dict[str, dict]) -> dict:
    name_correct = years_correct = degree_correct = n = 0
    skill_precisions: list[float] = []
    skill_recalls: list[float] = []
    skill_year_hits = skill_year_total = 0
    for profile in candidates:
        spec = specs_by_id.get(profile.candidate_id)
        if spec is None:
            continue
        n += 1
        name_correct += profile.name == spec["name"]
        years_correct += abs(profile.experience_years - spec["total_years"]) <= 1.0
        degree_correct += profile.education_level == spec["education"]["level"]
        extracted, true = set(profile.skills), set(spec["skills"])
        skill_precisions.append(len(extracted & true) / len(extracted) if extracted else 1.0)
        skill_recalls.append(len(extracted & true) / len(true) if true else 1.0)

        # Per-skill years accuracy, within +/-1yr, over skills in both the extraction
        # and resume_specs.json's true skill_years. Not a tuning target (the
        # generator's true per-skill years come from each job's skills_used list, the
        # role's core skills, not the bullet text), but a threat-to-validity signal.
        for skill in set(profile.skill_years) & set(spec.get("skill_years", {})):
            skill_year_total += 1
            if abs(profile.skill_years[skill] - spec["skill_years"][skill]) <= 1.0:
                skill_year_hits += 1

    return {
        "n": n,
        "name_accuracy": name_correct / n if n else 0.0,
        "years_within_1yr_accuracy": years_correct / n if n else 0.0,
        "degree_accuracy": degree_correct / n if n else 0.0,
        "skills_precision": sum(skill_precisions) / n if n else 0.0,
        "skills_recall": sum(skill_recalls) / n if n else 0.0,
        "skill_years_within_1yr_accuracy": (
            skill_year_hits / skill_year_total if skill_year_total else 0.0
        ),
    }
