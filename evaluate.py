"""Step 5 metrics: retrieval accuracy, extraction accuracy, filter accuracy and latency.

`run_eval(model, chunking, mode, weights, rerank)` re-ingests the given
(model, chunking) pair, then for every JD in data/job_descriptions/ builds a
full ranking over all candidates (retrieval + fusion + scoring, with the
must-have filter run *soft* so nobody is dropped from the ranking) and scores
it against data/ground_truth.json's graded relevance labels. This keeps
"retrieval accuracy" (how well the ranking orders candidates) separate from
"filter accuracy" (whether the hard must-have filter's pass/fail, computed
from extracted metadata, agrees with the same check run on the true
data/resume_specs.json values). Both are returned in the same dict, since
Step 6's experiments vary mode/weights/rerank and want every metric back
from one call.

CLI:
    python evaluate.py [--model bge-small] [--chunking section] \
        [--mode hybrid|semantic|keyword] [--rerank] [--warm-runs 20]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import config
import file_tools
from jd_parsing import parse_jd
from match_filtering import check_must_haves, filter_candidates
from match_models import JDRequirements
from match_scoring import score_candidate
from resume_embedding import Embedder, get_embedder
from resume_models import CandidateProfile
from resume_storage import collection_name, get_chroma_client, ingest, load_candidates
from retrieval import bm25_search, rrf_fuse, semantic_search


def load_ground_truth() -> dict[str, dict[str, int]]:
    return json.loads(config.GROUND_TRUTH_PATH.read_text())


def load_resume_specs() -> list[dict]:
    return json.loads(config.RESUME_SPECS_PATH.read_text())


def jd_id_for_path(jd_path: Path) -> str:
    return jd_path.stem.removeprefix("jd_")


def _profile_from_spec(spec: dict) -> CandidateProfile:
    """A `CandidateProfile` built from the true resume_specs.json values, so the same
    `check_must_haves` logic can be run against ground truth and against extraction."""
    return CandidateProfile(
        candidate_id=spec["candidate_id"],
        name=spec["name"],
        resume_path=spec["resume_path"],
        skills=spec["skills"],
        skill_years=spec["skill_years"],
        experience_years=spec["total_years"],
        education_level=spec["education"]["level"],
    )


# --- Retrieval accuracy -----------------------------------------------------


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


# --- Extraction and filter accuracy -----------------------------------------


def extraction_accuracy(candidates: list[CandidateProfile], specs_by_id: dict[str, dict]) -> dict:
    name_correct = years_correct = degree_correct = n = 0
    skill_precisions: list[float] = []
    skill_recalls: list[float] = []
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
    return {
        "n": n,
        "name_accuracy": name_correct / n if n else 0.0,
        "years_within_1yr_accuracy": years_correct / n if n else 0.0,
        "degree_accuracy": degree_correct / n if n else 0.0,
        "skills_precision": sum(skill_precisions) / n if n else 0.0,
        "skills_recall": sum(skill_recalls) / n if n else 0.0,
    }


def filter_confusion(
    jd: JDRequirements, candidates: list[CandidateProfile], specs_by_id: dict[str, dict]
) -> dict[str, int]:
    """TP/FP/FN/TN of the extracted-metadata must-have pass/fail against the same check
    run on the true resume_specs.json values ("positive" = passes the must-haves)."""
    tp = fp = fn = tn = 0
    for profile in candidates:
        spec = specs_by_id.get(profile.candidate_id)
        if spec is None:
            continue
        extracted_pass = not check_must_haves(jd, profile)
        true_pass = not check_must_haves(jd, _profile_from_spec(spec))
        if extracted_pass and true_pass:
            tp += 1
        elif extracted_pass and not true_pass:
            fp += 1
        elif true_pass:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _confusion_accuracy(confusion: dict[str, int]) -> float:
    total = sum(confusion.values())
    return (confusion["tp"] + confusion["tn"]) / total if total else 0.0


# --- Ranking (retrieval + fusion + soft filter + scoring), timed per stage -


def _rank_candidates(
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
    """One full ranking pass over every candidate, mirroring job_matcher.match_jd's pipeline
    but always `soft`-filtering (so the ranking covers everyone) and skipping evidence/output."""
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    jd_embedding = embedder.embed_query(jd.raw_text)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = (
        semantic_search(collection, jd_embedding, jd, config.SEMANTIC_POOL_SIZE)
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


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return quantiles[max(0, min(98, pct - 1))]


def _macro_average(per_jd: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = next(iter(per_jd.values()))
    return {key: sum(d[key] for d in per_jd.values()) / len(per_jd) for key in keys}


def _macro_average_latency(
    per_jd_latency: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    stages = next(iter(per_jd_latency.values()))
    n = len(per_jd_latency)
    return {
        stage: {
            "p50_ms": sum(d[stage]["p50_ms"] for d in per_jd_latency.values()) / n,
            "p95_ms": sum(d[stage]["p95_ms"] for d in per_jd_latency.values()) / n,
        }
        for stage in stages
    }


def _sum_confusion(per_jd_filter: dict[str, dict[str, int]]) -> dict[str, int]:
    return {key: sum(d[key] for d in per_jd_filter.values()) for key in ("tp", "fp", "fn", "tn")}


# --- Orchestration -----------------------------------------------------------


def run_eval(
    model: str = config.DEFAULT_MODEL,
    chunking: str = config.DEFAULT_CHUNKING,
    mode: str = "hybrid",
    weights: dict[str, float] | None = None,
    rerank: bool = False,
    warm_runs: int = 20,
) -> dict:
    t0 = time.perf_counter()
    embedder = get_embedder(model)
    model_cold_load_s = time.perf_counter() - t0

    ingest_stats = ingest(model=model, chunking=chunking, embedder=embedder)

    client = get_chroma_client()
    collection = client.get_collection(collection_name(model, chunking))
    candidates = load_candidates(model, chunking)
    candidates_by_id = {c.candidate_id: c for c in candidates}
    calibration = config.MODEL_REGISTRY[model]

    ground_truth = load_ground_truth()
    specs_by_id = {s["candidate_id"]: s for s in load_resume_specs()}

    per_jd_retrieval: dict[str, dict[str, float]] = {}
    per_jd_filter: dict[str, dict[str, int]] = {}
    per_jd_latency: dict[str, dict[str, dict[str, float]]] = {}

    for jd_path in sorted(config.JOB_DESCRIPTIONS_DIR.glob("*.txt")):
        jd_id = jd_id_for_path(jd_path)
        jd = parse_jd(jd_path.read_text())
        grades = {Path(filename).stem: grade for filename, grade in ground_truth[jd_id].items()}

        ranking: list[str] = []
        timed_runs: dict[str, list[float]] = {}
        for run_index in range(warm_runs + 1):
            result = _rank_candidates(
                jd,
                embedder,
                collection,
                candidates,
                candidates_by_id,
                calibration,
                mode,
                rerank,
                weights,
            )
            ranking = result["ranking"]
            if run_index == 0:
                continue  # discard the cold first run
            for stage, seconds in result["timings_s"].items():
                timed_runs.setdefault(stage, []).append(seconds * 1000)

        per_jd_retrieval[jd_id] = retrieval_metrics(ranking, grades)
        per_jd_filter[jd_id] = filter_confusion(jd, candidates, specs_by_id)
        per_jd_latency[jd_id] = {
            stage: {"p50_ms": _percentile(values, 50), "p95_ms": _percentile(values, 95)}
            for stage, values in timed_runs.items()
        }

    overall_confusion = _sum_confusion(per_jd_filter)
    return {
        "model": model,
        "chunking": chunking,
        "mode": mode,
        "rerank": rerank,
        "weights": weights or config.SCORE_WEIGHTS,
        "retrieval_accuracy": {
            "per_jd": per_jd_retrieval,
            "macro_avg": _macro_average(per_jd_retrieval),
        },
        "extraction_accuracy": extraction_accuracy(candidates, specs_by_id),
        "filter_accuracy": {
            "per_jd": per_jd_filter,
            "overall": {**overall_confusion, "accuracy": _confusion_accuracy(overall_confusion)},
        },
        "latency": {
            "model_cold_load_s": round(model_cold_load_s, 3),
            "ingest": asdict(ingest_stats),
            "query": {
                "per_jd": per_jd_latency,
                "macro_avg": _macro_average_latency(per_jd_latency),
            },
        },
    }


def _print_report(results: dict) -> None:
    print(
        f"\n## Retrieval accuracy ({results['model']}/{results['chunking']}/{results['mode']}"
        f"{', rerank' if results['rerank'] else ''})\n"
    )
    metrics = ["precision_at_5", "precision_at_10", "recall_at_10", "mrr", "ndcg_at_10", "hit_at_1"]
    print("| JD | " + " | ".join(metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    for jd_id, values in results["retrieval_accuracy"]["per_jd"].items():
        print(f"| {jd_id} | " + " | ".join(f"{values[m]:.3f}" for m in metrics) + " |")
    macro = results["retrieval_accuracy"]["macro_avg"]
    print("| **macro avg** | " + " | ".join(f"{macro[m]:.3f}" for m in metrics) + " |")

    print("\n## Extraction accuracy\n")
    extraction = results["extraction_accuracy"]
    print("| metric | value |")
    print("|---|---|")
    for key, value in extraction.items():
        if key != "n":
            print(f"| {key} | {value:.3f} |")
    print(f"| n | {extraction['n']} |")

    print("\n## Filter accuracy (extracted vs. true must-have pass/fail)\n")
    overall = results["filter_accuracy"]["overall"]
    print("| | predicted pass | predicted fail |\n|---|---|---|")
    print(f"| **true pass** | TP={overall['tp']} | FN={overall['fn']} |")
    print(f"| **true fail** | FP={overall['fp']} | TN={overall['tn']} |")
    print(f"\naccuracy: {overall['accuracy']:.3f}")

    print("\n## Latency (ms, macro-average p50/p95 across JDs)\n")
    print("| stage | p50 | p95 |")
    print("|---|---|---|")
    for stage, values in results["latency"]["query"]["macro_avg"].items():
        print(f"| {stage} | {values['p50_ms']:.2f} | {values['p95_ms']:.2f} |")
    print(f"\nmodel cold load: {results['latency']['model_cold_load_s']:.2f}s")
    print(f"ingest total: {results['latency']['ingest']['timings_s']['total']:.2f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval, extraction and filter accuracy, and latency."
    )
    parser.add_argument(
        "--model", default=config.DEFAULT_MODEL, choices=list(config.MODEL_REGISTRY)
    )
    parser.add_argument(
        "--chunking", default=config.DEFAULT_CHUNKING, choices=["section", "fixed", "whole"]
    )
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "semantic", "keyword"])
    parser.add_argument("--rerank", action="store_true", help="Cross-encoder rerank the top 20.")
    parser.add_argument("--warm-runs", type=int, default=20)
    args = parser.parse_args(argv)

    results = run_eval(
        model=args.model,
        chunking=args.chunking,
        mode=args.mode,
        rerank=args.rerank,
        warm_runs=args.warm_runs,
    )

    write_result = file_tools.write_file(str(config.METRICS_PATH), json.dumps(results, indent=2))
    if not write_result.get("success"):
        raise RuntimeError(f"Failed to write {config.METRICS_PATH}: {write_result.get('error')}")
    print(f"Wrote {config.METRICS_PATH}")
    _print_report(results)


if __name__ == "__main__":
    main()
