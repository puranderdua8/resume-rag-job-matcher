"""Retrieval accuracy: Precision@k, Recall@10, MRR, graded nDCG@10, Hit@1, and their
macro average across JDs. `grades` holds a JD's 0/1/2 relevance per candidate from
ground_truth.json; `relevant` (grade >= 1) drives the binary metrics."""

from __future__ import annotations

import math


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    top = ranking[:k]
    return (sum(1 for cid in top if cid in relevant) / len(top)) if top else 0.0


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def mean_reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(ranking, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_1(ranking: list[str], relevant: set[str]) -> float:
    return 1.0 if ranking and ranking[0] in relevant else 0.0


def _dcg(graded_relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(graded_relevances, 1))


def ndcg_at_k(ranking: list[str], grades: dict[str, int], k: int) -> float:
    """Graded nDCG@k: `grades` holds every candidate's 0/1/2 relevance for this JD, used both
    for the actual ranking's top-k gain and for the ideal (ground-truth-sorted) top-k gain."""
    actual = _dcg([grades.get(cid, 0) for cid in ranking[:k]])
    ideal = _dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def retrieval_metrics(ranking: list[str], grades: dict[str, int]) -> dict[str, float]:
    relevant = {cid for cid, grade in grades.items() if grade >= 1}
    return {
        "precision_at_5": precision_at_k(ranking, relevant, 5),
        "precision_at_10": precision_at_k(ranking, relevant, 10),
        "recall_at_10": recall_at_k(ranking, relevant, 10),
        "mrr": mean_reciprocal_rank(ranking, relevant),
        "ndcg_at_10": ndcg_at_k(ranking, grades, 10),
        "hit_at_1": hit_at_1(ranking, relevant),
    }


def macro_average(per_jd: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = next(iter(per_jd.values()))
    return {key: sum(d[key] for d in per_jd.values()) / len(per_jd) for key in keys}
