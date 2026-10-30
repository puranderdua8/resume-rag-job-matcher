"""Filter accuracy: a confusion matrix of the extracted-metadata must-have pass/fail
against the same check run on the true resume_specs.json values. "Positive" means
passes the must-haves; this measures how extraction noise changes the verdict,
separately from retrieval ranking quality."""

from __future__ import annotations

from eval_data import profile_from_spec
from match_filtering import check_must_haves
from match_models import JDRequirements
from resume_models import CandidateProfile


def filter_confusion(
    jd: JDRequirements, candidates: list[CandidateProfile], specs_by_id: dict[str, dict]
) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for profile in candidates:
        spec = specs_by_id.get(profile.candidate_id)
        if spec is None:
            continue
        extracted_pass = not check_must_haves(jd, profile)
        true_pass = not check_must_haves(jd, profile_from_spec(spec))
        if extracted_pass and true_pass:
            tp += 1
        elif extracted_pass and not true_pass:
            fp += 1
        elif true_pass:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def confusion_accuracy(confusion: dict[str, int]) -> float:
    total = sum(confusion.values())
    return (confusion["tp"] + confusion["tn"]) / total if total else 0.0


def sum_confusion(per_jd_filter: dict[str, dict[str, int]]) -> dict[str, int]:
    return {key: sum(d[key] for d in per_jd_filter.values()) for key in ("tp", "fp", "fn", "tn")}
