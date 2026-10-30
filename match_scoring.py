"""Score formula: score = 100 * (0.50*semantic_cal + 0.20*bm25_norm + 0.25*req_cov + 0.05*pref_cov).

`semantic_cal` maps cosine similarity through the model's fixed calibration
bounds; `bm25_norm` divides by the pool's maximum BM25 score. The optional
cross-encoder rerank (`--rerank`) replaces a candidate's raw similarity with
a cross-encoder score before calibration, for the top-N candidates only.
"""

from __future__ import annotations

import math
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
    raw_score = (
        weights["semantic"] * semantic_cal
        + weights["bm25"] * bm25_norm
        + weights["required_coverage"] * required_cov
        + weights["preferred_coverage"] * preferred_cov
    )
    score = round(100 * raw_score) - filter_penalty
    breakdown = {
        "semantic_cal": round(semantic_cal, 3),
        "bm25_norm": round(bm25_norm, 3),
        "required_coverage": round(required_cov, 3),
        "preferred_coverage": round(preferred_cov, 3),
    }
    if filter_penalty:
        breakdown["soft_penalty"] = filter_penalty
    return ScoredCandidate(
        candidate_id=profile.candidate_id, score=int(max(0, min(100, score))), breakdown=breakdown
    )


_cross_encoder = None


def rerank_semantic(
    jd_text: str, hits: dict[str, SemanticHit], candidate_ids: list[str]
) -> dict[str, float]:
    """Cross-encoder (JD, best chunk) score for `candidate_ids` (the caller picks the top-N
    of the fused ranking, to keep the expensive cross-encoder call small).

    The cross-encoder's raw logit is squashed through a sigmoid to land in
    [0, 1], the same range `calibrate()` expects for the semantic component.
    The model is cached at module level: evaluate.py's latency sweep calls
    this many times per run, and reloading the model each time would dwarf
    the inference time it's trying to measure.
    """
    global _cross_encoder

    candidates = [cid for cid in candidate_ids if cid in hits and hits[cid].top_chunks]
    if not candidates:
        return {}

    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    model = _cross_encoder
    pairs = [(jd_text, hits[cid].top_chunks[0][2]) for cid in candidates]
    raw_scores = model.predict(pairs)
    return {
        candidate_id: 1.0 / (1.0 + math.exp(-float(raw_score)))
        for candidate_id, raw_score in zip(candidates, raw_scores, strict=True)
    }
