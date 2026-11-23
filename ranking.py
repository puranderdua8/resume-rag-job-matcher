"""The one ranking pipeline (Step 11), shared by job_matcher.py, evaluate.py and the
notebooks:

    parse JD -> candidate generation -> must-have filter -> score -> (optional)
    rerank -> sort -> top-K
                union(top N_SEMANTIC by semantic, top N_BM25 by BM25), ordered by RRF

`rank()` is the only caller of semantic_search, semantic_search_for_ids, bm25_search,
rrf_fuse, filter_candidates, score_candidate and rerank_semantic. Callers that need
evidence or output formatting (job_matcher.match_jd) or a plain ordering for metrics
(evaluate.py, the notebooks) build on top of the `Ranking` this returns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
from match_filtering import FilterResult, filter_candidates
from match_models import JDRequirements, condensed_requirements_text
from match_scoring import ScoredCandidate, calibrate, rerank_semantic, score_candidate
from resume_embedding import Embedder
from resume_models import CandidateProfile
from retrieval import SemanticHit, bm25_search, rrf_fuse, semantic_search, semantic_search_for_ids


@dataclass
class RankingOptions:
    mode: str = "hybrid"  # hybrid | semantic | keyword
    soft: bool = False
    rerank: bool = False
    weights: dict[str, float] | None = None


@dataclass
class Ranking:
    order: list[str]  # candidate_ids, best first; only pool members that passed the filter
    scored: dict[str, ScoredCandidate]
    filter_results: dict[str, FilterResult]
    hits: dict[str, SemanticHit]
    rrf_scores: dict[str, float]
    pool: set[str]
    timings_s: dict[str, float] = field(default_factory=dict)


def rank(
    jd: JDRequirements,
    embedder: Embedder,
    collection,
    candidates: list[CandidateProfile],
    calibration: config.ModelSpec,
    options: RankingOptions | None = None,
) -> Ranking:
    options = options or RankingOptions()
    candidates_by_id = {c.candidate_id: c for c in candidates}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    jd_embedding = embedder.embed_query(jd.raw_text) if options.mode != "keyword" else None
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits: dict[str, SemanticHit] = {}
    if options.mode != "keyword":
        hits = semantic_search(
            collection, jd_embedding, jd, config.N_SEMANTIC, apply_prefilter=not options.soft
        )
    t_semantic = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_scores: dict[str, float] = {}
    bm25_reference = 0.0
    if options.mode != "semantic":
        bm25_scores, bm25_reference = bm25_search(
            candidates, jd.raw_text, condensed_requirements_text(jd)
        )
    t_bm25 = time.perf_counter() - t0

    # A2: the pool is the union of the top-N_SEMANTIC and top-N_BM25 candidates.
    semantic_ranking = sorted(hits, key=lambda cid: hits[cid].score, reverse=True)[
        : config.N_SEMANTIC
    ]
    bm25_ranking = sorted(bm25_scores, key=lambda cid: bm25_scores[cid], reverse=True)[
        : config.N_BM25
    ]
    pool_ids = set(semantic_ranking) | set(bm25_ranking)

    t0 = time.perf_counter()
    rankings = [ranking for ranking in (semantic_ranking, bm25_ranking) if ranking]
    rrf_scores = rrf_fuse(rankings, config.RRF_K)
    t_fusion = time.perf_counter() - t0

    # R21: every pool candidate needs a real semantic score, including ones that only
    # made the pool through BM25 (semantic_search's nearest-chunk query may not have
    # surfaced them at all).
    t0 = time.perf_counter()
    missing_semantic_ids = [cid for cid in pool_ids if cid not in hits]
    if missing_semantic_ids and options.mode != "keyword":
        hits.update(semantic_search_for_ids(collection, jd_embedding, missing_semantic_ids))
    t_semantic += time.perf_counter() - t0

    pool = [candidates_by_id[cid] for cid in pool_ids]
    t0 = time.perf_counter()
    filter_results = filter_candidates(jd, pool, options.soft)
    t_filter = time.perf_counter() - t0

    t_rerank = 0.0
    semantic_overrides: dict[str, float] = {}
    if options.rerank and hits:
        t0 = time.perf_counter()
        # R19: rerank every candidate that passed the filter, not a top-N subset.
        eligible = [cid for cid, fr in filter_results.items() if fr.passed]
        raw_rerank_scores = rerank_semantic(condensed_requirements_text(jd), hits, eligible)
        semantic_overrides = {
            cid: calibrate(
                raw, config.CROSS_ENCODER_CALIBRATION_LOW, config.CROSS_ENCODER_CALIBRATION_HIGH
            )
            for cid, raw in raw_rerank_scores.items()
        }
        t_rerank = time.perf_counter() - t0

    t0 = time.perf_counter()
    scored: dict[str, ScoredCandidate] = {}
    for candidate_id, filter_result in filter_results.items():
        if not filter_result.passed:
            continue
        profile = candidates_by_id[candidate_id]
        scored[candidate_id] = score_candidate(
            profile,
            jd,
            semantic_raw=hits[candidate_id].score if candidate_id in hits else 0.0,
            bm25_raw=bm25_scores.get(candidate_id, 0.0),
            bm25_max=bm25_reference,
            calibration=calibration,
            filter_penalty=filter_result.penalty,
            semantic_override=semantic_overrides.get(candidate_id),
            weights=options.weights,
        )
    # RRF is the tie-breaker after score, then name, so RRF_K has an effect with or
    # without rerank (A2).
    order = sorted(
        scored,
        key=lambda cid: (-scored[cid].score, -rrf_scores.get(cid, 0.0), candidates_by_id[cid].name),
    )
    t_score = time.perf_counter() - t0

    total = time.perf_counter() - t_start
    return Ranking(
        order=order,
        scored=scored,
        filter_results=filter_results,
        hits=hits,
        rrf_scores=rrf_scores,
        pool=pool_ids,
        timings_s={
            "embed": t_embed,
            "semantic_search": t_semantic,
            "bm25_search": t_bm25,
            "fusion": t_fusion,
            "filter": t_filter,
            "rerank": t_rerank,
            "score": t_score,
            "total": total,
        },
    )
