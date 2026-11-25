"""Score formula: score = 100 * (0.50*semantic_cal + 0.20*bm25_norm + 0.25*req_cov + 0.05*pref_cov).

`semantic_cal` maps cosine similarity through the model's fixed calibration
bounds; `bm25_norm` divides raw BM25 by `ranking.py`'s fixed per-JD reference
score (R20). The optional cross-encoder rerank (`--rerank`, applied by
`ranking.py` to every candidate in the filtered pool, not just a top-N) replaces
a candidate's raw cosine similarity with a cross-encoder score, calibrated by
the same `calibrate()` using `config.CROSS_ENCODER_CALIBRATION_LOW/HIGH` (R19).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from match_models import JDRequirements
from resume_models import CandidateProfile
from retrieval import SemanticHit, skill_coverage


def calibrate(raw: float, low: float, high: float) -> float:
    """Map `raw` cosine similarity from [low, high] to [0, 1], clipped."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (raw - low) / (high - low)))


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: int
    breakdown: dict[str, float] = field(default_factory=dict)


def score_candidate(
    profile: CandidateProfile,
    jd: JDRequirements,
    semantic_raw: float,
    bm25_raw: float,
    bm25_max: float,
    calibration: config.ModelSpec,
    filter_penalty: float = 0.0,
    semantic_override: float | None = None,
    weights: dict[str, float] | None = None,
) -> ScoredCandidate:
    """`semantic_override`, when given (from `--rerank`), is already in [0, 1] and replaces
    the calibrated cosine similarity as the semantic component.

    `weights` overrides `config.SCORE_WEIGHTS`, for evaluate.py's weight-sweep experiment;
    callers that don't pass it get the configured defaults.
    """
    if semantic_override is not None:
        semantic_cal = semantic_override
    else:
        semantic_cal = calibrate(
            semantic_raw, calibration.calibration_low, calibration.calibration_high
        )
    bm25_norm = (bm25_raw / bm25_max) if bm25_max > 0 else 0.0
    required_cov = skill_coverage(jd.required_skills, profile)
    preferred_cov = skill_coverage(jd.preferred_skills, profile)

    weights = weights or config.SCORE_WEIGHTS
    # R12: a JD with no recognised required (or preferred) skills would otherwise award
    # that component in full to everyone (skill_coverage returns 1.0 for an empty target
    # list), handing out free points with no skill signal behind them. Drop the component
    # and renormalise the remaining weights instead, so they still sum to 1.
    components = {"semantic": semantic_cal, "bm25": bm25_norm}
    active_weights = {"semantic": weights["semantic"], "bm25": weights["bm25"]}
    required_scored = bool(jd.required_skills)
    preferred_scored = bool(jd.preferred_skills)
    if required_scored:
        components["required_coverage"] = required_cov
        active_weights["required_coverage"] = weights["required_coverage"]
    if preferred_scored:
        components["preferred_coverage"] = preferred_cov
        active_weights["preferred_coverage"] = weights["preferred_coverage"]

    total_weight = sum(active_weights.values())
    raw_score = (
        sum(active_weights[k] * components[k] for k in components) / total_weight
        if total_weight > 0
        else 0.0
    )
    score = round(100 * raw_score) - filter_penalty
    breakdown = {
        "semantic_cal": round(semantic_cal, 3),
        "bm25_norm": round(bm25_norm, 3),
        "required_coverage": round(required_cov, 3),
        "preferred_coverage": round(preferred_cov, 3),
    }
    if not required_scored:
        breakdown["required_coverage_scored"] = False
    if not preferred_scored:
        breakdown["preferred_coverage_scored"] = False
    if filter_penalty:
        breakdown["soft_penalty"] = filter_penalty
    return ScoredCandidate(
        candidate_id=profile.candidate_id, score=int(max(0, min(100, score))), breakdown=breakdown
    )


_cross_encoder = None


def rerank_semantic(
    jd_text: str, hits: dict[str, SemanticHit], candidate_ids: list[str]
) -> dict[str, float]:
    """Cross-encoder (JD, best chunk) raw logit for every one of `candidate_ids`
    (R19: the caller passes the whole filtered pool, not a top-N subset).

    Returns the raw logit, uncalibrated: the caller maps it through `calibrate()`
    with `config.CROSS_ENCODER_CALIBRATION_LOW/HIGH`, the same function and the
    same [0, 1] target range cosine similarity uses, so semantic scores sit on one
    scale whether or not `--rerank` is on. The model is cached at module level:
    evaluate.py's latency sweep calls this many times per run, and reloading the
    model each time would dwarf the inference time it's trying to measure.
    """
    global _cross_encoder

    candidates = [cid for cid in candidate_ids if cid in hits and hits[cid].top_chunks]
    if not candidates:
        return {}

    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(
            config.CROSS_ENCODER_MODEL, revision=config.CROSS_ENCODER_REVISION
        )
    model = _cross_encoder
    pairs = [(jd_text, hits[cid].top_chunks[0][2]) for cid in candidates]
    raw_scores = model.predict(pairs)
    return {
        candidate_id: float(raw_score)
        for candidate_id, raw_score in zip(candidates, raw_scores, strict=True)
    }
