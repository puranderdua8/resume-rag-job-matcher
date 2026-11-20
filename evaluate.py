"""Part B CLI: retrieval accuracy, extraction accuracy, filter accuracy and latency.

The metrics themselves live across eval_data.py, eval_retrieval_metrics.py,
eval_extraction.py, eval_filtering.py, eval_ranking.py and eval_latency.py; this
module wires them together and owns the CLI and outputs/metrics.json.

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
import time
from dataclasses import asdict
from pathlib import Path

import config
import file_tools
from eval_data import jd_id_for_path, load_ground_truth, load_resume_specs
from eval_extraction import extraction_accuracy
from eval_filtering import confusion_accuracy, filter_confusion, sum_confusion
from eval_latency import macro_average_latency, percentile
from eval_ranking import rank_candidates
from eval_retrieval_metrics import macro_average, retrieval_metrics
from jd_parsing import parse_jd
from resume_embedding import get_embedder
from resume_storage import collection_name, get_chroma_client, ingest, load_candidates


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
    # R9 changed candidate_id to the file name *with* its extension (so same-stem
    # resumes in different formats don't collide); resume_specs.json's own
    # candidate_id field predates that and is still the bare stem, so key specs by
    # the file name derived from resume_path instead, which matches both the new
    # candidate_id convention and data/ground_truth.json's existing filenames.
    specs_by_id = {Path(s["resume_path"]).name: s for s in load_resume_specs()}

    per_jd_retrieval: dict[str, dict[str, float]] = {}
    per_jd_filter: dict[str, dict[str, int]] = {}
    per_jd_latency: dict[str, dict[str, dict[str, float]]] = {}

    for jd_path in sorted(config.JOB_DESCRIPTIONS_DIR.glob("*.txt")):
        jd_id = jd_id_for_path(jd_path)
        jd = parse_jd(jd_path.read_text())
        # ground_truth.json's filenames already include the extension, matching
        # candidate_id directly (see the specs_by_id note above).
        grades = dict(ground_truth[jd_id].items())

        ranking: list[str] = []
        timed_runs: dict[str, list[float]] = {}
        for run_index in range(warm_runs + 1):
            result = rank_candidates(
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
            stage: {"p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95)}
            for stage, values in timed_runs.items()
        }

    overall_confusion = sum_confusion(per_jd_filter)
    return {
        "model": model,
        "chunking": chunking,
        "mode": mode,
        "rerank": rerank,
        "weights": weights or config.SCORE_WEIGHTS,
        "retrieval_accuracy": {
            "per_jd": per_jd_retrieval,
            "macro_avg": macro_average(per_jd_retrieval),
        },
        "extraction_accuracy": extraction_accuracy(candidates, specs_by_id),
        "filter_accuracy": {
            "per_jd": per_jd_filter,
            "overall": {**overall_confusion, "accuracy": confusion_accuracy(overall_confusion)},
        },
        "latency": {
            "model_cold_load_s": round(model_cold_load_s, 3),
            "ingest": asdict(ingest_stats),
            "query": {
                "per_jd": per_jd_latency,
                "macro_avg": macro_average_latency(per_jd_latency),
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
