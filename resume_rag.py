"""Part A CLI: load resumes, chunk by section, extract metadata, embed and store in ChromaDB.

The pipeline itself lives across resume_loading.py, resume_sections.py,
resume_chunking.py, resume_metadata.py, resume_embedding.py and resume_storage.py;
this module just wires up the `ingest`/`stats` commands.

CLI:
    python resume_rag.py ingest [--model bge-small] [--chunking section]
    python resume_rag.py stats [--model bge-small] [--chunking section]
"""

from __future__ import annotations

import argparse
import json
import logging

import config
from resume_models import IngestStats
from resume_storage import collection_dir, ingest, load_candidates


def _print_ingest_report(stats: IngestStats) -> None:
    print(f"Ingested collection: {stats.collection}")
    print(f"Resumes loaded: {stats.resumes_loaded}")
    if stats.resumes_skipped:
        print(f"Resumes skipped: {len(stats.resumes_skipped)}")
        for reason in stats.resumes_skipped:
            print(f"  - {reason}")
    print("Chunks per section:")
    for section, n in sorted(stats.chunk_counts.items()):
        print(f"  {section:15s} {n}")
    print("Timings (s):")
    for stage, seconds in stats.timings_s.items():
        print(f"  {stage:20s} {seconds:.2f}")


def _print_stats_report(model: str, chunking: str) -> None:
    stats_path = collection_dir(model, chunking) / "ingest_stats.json"
    if stats_path.exists():
        stats_data = json.loads(stats_path.read_text())
        print(f"Collection: {stats_data['collection']}")
        print("Chunks per section:")
        for section, n in sorted(stats_data["chunk_counts"].items()):
            print(f"  {section:15s} {n}")
        print("Last ingest timings (s):")
        for stage, seconds in stats_data["timings_s"].items():
            print(f"  {stage:20s} {seconds:.2f}")
        print()

    candidates = load_candidates(model, chunking)
    print(f"Profiles: {len(candidates)}")
    header = f"{'name':22s} {'years':>6s} {'edu':>4s} {'skills':>7s} {'chunks':>7s}"
    print(header)
    for profile in sorted(candidates, key=lambda p: p.name):
        print(
            f"{profile.name[:22]:22s} {profile.experience_years:6.1f} "
            f"{profile.education_level:4d} {len(profile.skills):7d} {len(profile.chunks):7d}"
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Resume RAG: load, chunk, embed and store resumes."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Load, chunk, embed and store every resume.")
    ingest_parser.add_argument(
        "--model", default=config.DEFAULT_MODEL, choices=list(config.MODEL_REGISTRY)
    )
    ingest_parser.add_argument(
        "--chunking", default=config.DEFAULT_CHUNKING, choices=["section", "fixed", "whole"]
    )

    stats_parser = sub.add_parser("stats", help="Print chunk counts, profiles and timings.")
    stats_parser.add_argument(
        "--model", default=config.DEFAULT_MODEL, choices=list(config.MODEL_REGISTRY)
    )
    stats_parser.add_argument(
        "--chunking", default=config.DEFAULT_CHUNKING, choices=["section", "fixed", "whole"]
    )

    args = parser.parse_args(argv)
    if args.command == "ingest":
        stats = ingest(model=args.model, chunking=args.chunking)
        _print_ingest_report(stats)
    elif args.command == "stats":
        _print_stats_report(args.model, args.chunking)


if __name__ == "__main__":
    main()
