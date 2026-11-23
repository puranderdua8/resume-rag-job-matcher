"""Part B CLI: parse a JD, hybrid-retrieve, filter on must-haves, score and explain matches.

The pipeline itself lives across jd_parsing.py, retrieval.py, match_filtering.py,
match_scoring.py and match_evidence.py; this module wires them together and
owns the CLI and the output JSON.

CLI:
    python job_matcher.py --jd <file> | --text "..." | --all \
        [--k 10] [--must-have "5+ years Python"] [--soft] [--rerank] [--llm] \
        [--mode hybrid|semantic|keyword] [--model bge-small] [--chunking section]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import chromadb

import config
import file_tools
from jd_parsing import UnparsableMustHaveError, parse_jd, parse_must_have_clause
from match_evidence import llm_reasoning as llm_reasoning_fn
from match_evidence import matched_skills, must_have_check, relevant_excerpts, template_reasoning
from match_models import JDRequirements, MatchResult
from ranking import RankingOptions, rank
from resume_embedding import Embedder, get_embedder
from resume_models import CandidateProfile
from resume_storage import collection_name, get_chroma_client, load_candidates


def apply_must_have_overrides(jd: JDRequirements, clauses: list[str]) -> JDRequirements:
    """`--must-have` clauses add to or override the JD's regex-extracted must-haves."""
    for clause in clauses:
        skill_minimums, alternatives, total_minimum, degree_minimum = parse_must_have_clause(clause)
        jd.skill_year_minimums.update(skill_minimums)
        jd.skill_year_alternatives.extend(alternatives)
        if total_minimum is not None:
            jd.total_years_minimum = total_minimum
        if degree_minimum:
            jd.degree_minimum = degree_minimum
    return jd


def slugify(text: str) -> str:
    """Filesystem-safe slug for an output filename. Falls back to a hash of `text`
    when nothing alphanumeric survives (e.g. a title of only punctuation), so the
    result is never an empty stem (R23)."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if slug:
        return slug
    return f"jd_{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def match_jd(
    jd: JDRequirements,
    embedder: Embedder,
    collection,
    candidates: list[CandidateProfile],
    *,
    model: str = config.DEFAULT_MODEL,
    k: int = config.TOP_K,
    soft: bool = False,
    rerank: bool = False,
    llm: bool = False,
    mode: str = "hybrid",
) -> dict:
    """`candidates` is loaded once per process by the caller (A3), not once per JD."""
    t_start = time.perf_counter()
    candidates_by_id = {c.candidate_id: c for c in candidates}
    calibration = config.MODEL_REGISTRY[model]

    result = rank(
        jd,
        embedder,
        collection,
        candidates,
        calibration,
        RankingOptions(mode=mode, soft=soft, rerank=rerank),
    )
    top_ids = result.order[:k]

    top_matches = []
    for candidate_id in top_ids:
        sc = result.scored[candidate_id]
        profile = candidates_by_id[candidate_id]
        hit = result.hits.get(candidate_id)
        filter_result = result.filter_results[candidate_id]
        section_scores = hit.section_scores if hit else {}
        skills_matched = matched_skills(profile, jd)
        required_matched = [s for s in jd.required_skills if s in profile.skills]
        check = must_have_check(filter_result)
        reasoning = template_reasoning(
            jd, profile, sc.score, section_scores, sc.breakdown["required_coverage"], filter_result
        )
        if llm:
            evidence = {
                "candidate_name": profile.name,
                "score": sc.score,
                "score_breakdown": sc.breakdown,
                "section_scores": section_scores,
                "matched_skills": skills_matched,
                "missing_required_skills": [
                    s for s in jd.required_skills if s not in profile.skills
                ],
                "must_have_check": check,
            }
            reasoning = llm_reasoning_fn(reasoning, evidence)
        top_matches.append(
            MatchResult(
                candidate_name=profile.name,
                resume_path=profile.resume_path,
                match_score=sc.score,
                matched_skills=skills_matched,
                relevant_excerpts=relevant_excerpts(profile, hit, required_matched),
                reasoning=reasoning,
                score_breakdown=sc.breakdown,
                section_scores=section_scores,
                must_have_check=check,
            ).to_dict()
        )

    filtered_out = [
        {
            "candidate_name": candidates_by_id[cid].name,
            "resume_path": candidates_by_id[cid].resume_path,
            "reasons": fr.reasons,
        }
        for cid, fr in result.filter_results.items()
        if not fr.passed
    ]

    total_ms = (time.perf_counter() - t_start) * 1000
    latency_ms = {stage: round(seconds * 1000, 1) for stage, seconds in result.timings_s.items()}
    latency_ms["total"] = round(total_ms, 1)
    return {
        "job_description": jd.title,
        "top_matches": top_matches,
        "returned": len(top_matches),
        "requested": k,
        "filtered_out": filtered_out,
        "latency_ms": latency_ms,
    }


def _run_and_write(
    jd_text: str,
    stem: str,
    embedder: Embedder,
    collection,
    candidates: list[CandidateProfile],
    args: argparse.Namespace,
) -> dict:
    jd = parse_jd(jd_text)
    apply_must_have_overrides(jd, args.must_haves)
    result = match_jd(
        jd,
        embedder,
        collection,
        candidates,
        model=args.model,
        k=args.k,
        soft=args.soft,
        rerank=args.rerank,
        llm=args.llm,
        mode=args.mode,
    )
    out_path = config.MATCHES_DIR / f"{stem}.json"
    write_result = file_tools.write_file(str(out_path), json.dumps(result, indent=2))
    if not write_result.get("success"):
        raise RuntimeError(f"Failed to write {out_path}: {write_result.get('error')}")
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Match job descriptions against indexed resumes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jd", help="Path to a job description text file.")
    source.add_argument("--text", help="Job description text, given directly.")
    source.add_argument(
        "--all", action="store_true", help="Match every JD in data/job_descriptions/."
    )

    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--must-have",
        dest="must_haves",
        action="append",
        default=[],
        help='Add or override a must-have, e.g. "5+ years Python". Repeatable.',
    )
    parser.add_argument(
        "--soft", action="store_true", help="Penalize must-have failures instead of dropping them."
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Cross-encoder rerank the whole filtered pool."
    )
    parser.add_argument("--llm", action="store_true", help="Paraphrase reasoning with Gemini.")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "semantic", "keyword"])
    parser.add_argument(
        "--model", default=config.DEFAULT_MODEL, choices=list(config.MODEL_REGISTRY)
    )
    parser.add_argument(
        "--chunking", default=config.DEFAULT_CHUNKING, choices=["section", "fixed", "whole"]
    )
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error(f"--k must be >= 1, got {args.k}")

    for clause in args.must_haves:
        try:
            parse_must_have_clause(clause)
        except UnparsableMustHaveError as exc:
            parser.error(str(exc))

    client = get_chroma_client()
    name = collection_name(args.model, args.chunking)
    try:
        collection = client.get_collection(name)
    except chromadb.errors.NotFoundError:
        parser.error(
            f"No index found for collection {name!r}. Run "
            f"`python resume_rag.py ingest --model {args.model} --chunking {args.chunking}` first."
        )
    embedder = get_embedder(args.model)
    # A3: load candidates once per process, whether matching one JD or all of them.
    candidates = load_candidates(model=args.model, chunking=args.chunking)

    if args.text:
        _run_and_write(
            args.text, slugify(parse_jd(args.text).title), embedder, collection, candidates, args
        )
    elif args.jd:
        _run_and_write(
            Path(args.jd).read_text(), Path(args.jd).stem, embedder, collection, candidates, args
        )
    else:
        for jd_path in sorted(config.JOB_DESCRIPTIONS_DIR.glob("*.txt")):
            _run_and_write(
                jd_path.read_text(), jd_path.stem, embedder, collection, candidates, args
            )


if __name__ == "__main__":
    main()
