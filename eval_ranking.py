"""The timed ranking pass evaluate.py scores and measures latency with: one full pass
over every candidate, mirroring job_matcher.match_jd's pipeline but always
`soft`-filtering (so the ranking covers everyone, for retrieval-accuracy purposes)
and skipping evidence/output (not needed for metrics)."""

from __future__ import annotations

import time

import config
from match_filtering import filter_candidates
from match_models import JDRequirements
from match_scoring import score_candidate
from resume_embedding import Embedder
from resume_models import CandidateProfile
from retrieval import bm25_search, rrf_fuse, semantic_search


def rank_candidates(
    jd: JDRequirements,
    embedder: Embedder,
    collection,
    candidates: list[CandidateProfile],
    candidates_by_id: dict[str, CandidateProfile],
    calibration: config.ModelSpec,
    mode: str,
    rerank: bool,
    weights: dict[str, float] | None,
) -> dict:
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    jd_embedding = embedder.embed_query(jd.raw_text)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = (
        semantic_search(
            collection, jd_embedding, jd, config.SEMANTIC_POOL_SIZE, apply_prefilter=False
        )
        if mode != "keyword"
        else {}
    )
    t_semantic = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_scores = bm25_search(candidates, jd.raw_text) if mode != "semantic" else {}
    t_bm25 = time.perf_counter() - t0

    t0 = time.perf_counter()
    rankings = []
    if hits:
        rankings.append(sorted(hits, key=lambda cid: hits[cid].score, reverse=True))
    if bm25_scores:
        rankings.append(sorted(bm25_scores, key=lambda cid: bm25_scores[cid], reverse=True))
    fused_scores = rrf_fuse(rankings, config.RRF_K)
    t_fusion = time.perf_counter() - t0

    t0 = time.perf_counter()
    filter_results = filter_candidates(jd, candidates, soft=True)
    t_filter = time.perf_counter() - t0

    t_rerank = 0.0
    semantic_overrides: dict[str, float] = {}
    if rerank and hits:
        from match_scoring import rerank_semantic

        t0 = time.perf_counter()
        rerank_pool = sorted(
            candidates_by_id, key=lambda cid: fused_scores.get(cid, 0.0), reverse=True
        )[:20]
        semantic_overrides = rerank_semantic(jd.raw_text, hits, rerank_pool)
        t_rerank = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_max = max(bm25_scores.values(), default=0.0)
    scored = []
    for profile in candidates:
        cid = profile.candidate_id
        scored.append(
            score_candidate(
                profile,
                jd,
                semantic_raw=hits[cid].score if cid in hits else 0.0,
                bm25_raw=bm25_scores.get(cid, 0.0),
                bm25_max=bm25_max,
                calibration=calibration,
                filter_penalty=filter_results[cid].penalty,
                semantic_override=semantic_overrides.get(cid),
                weights=weights,
            )
        )
    scored.sort(key=lambda sc: (-sc.score, candidates_by_id[sc.candidate_id].name))
    t_score = time.perf_counter() - t0

    total = time.perf_counter() - t_start
    return {
        "ranking": [sc.candidate_id for sc in scored],
        "timings_s": {
            "embed": t_embed,
            "semantic_search": t_semantic,
            "bm25_search": t_bm25,
            "fusion": t_fusion,
            "filter": t_filter,
            "rerank": t_rerank,
            "score": t_score,
            "total": total,
        },
    }
