"""Ingest orchestration and ChromaDB storage: wires loading, chunking, metadata and embedding
together, and persists the result to a per-(model, chunking) collection."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import chromadb

import config
from resume_chunking import build_chunks
from resume_embedding import Embedder, get_embedder
from resume_loading import load_resumes
from resume_metadata import extract_metadata
from resume_models import CandidateProfile, IngestStats
from resume_sections import detect_sections


def collection_name(model: str, chunking: str) -> str:
    return f"resumes__{model}__{chunking}"


def collection_dir(model: str, chunking: str) -> Path:
    return config.INDEX_DIR / collection_name(model, chunking)


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def ingest(
    model: str = config.DEFAULT_MODEL,
    chunking: str = config.DEFAULT_CHUNKING,
    embedder: Embedder | None = None,
    resumes_dir: Path = config.RESUMES_DIR,
    as_of: date | None = None,
) -> IngestStats:
    """Load, chunk, embed and store every resume. Idempotent: recreates the collection."""
    name = collection_name(model, chunking)
    stats = IngestStats(model=model, chunking=chunking, collection=name)

    t0 = time.perf_counter()
    loaded, skipped = load_resumes(resumes_dir)
    stats.resumes_loaded = len(loaded)
    stats.resumes_skipped = skipped
    t1 = time.perf_counter()
    stats.timings_s["load"] = t1 - t0

    profiles: list[CandidateProfile] = []
    for item in loaded:
        sections = detect_sections(item["text"])
        profile = extract_metadata(
            item["candidate_id"], item["resume_path"], item["text"], sections, as_of=as_of
        )
        profile.chunks = build_chunks(
            item["candidate_id"], profile.name, sections, item["text"], strategy=chunking
        )
        profiles.append(profile)
    t2 = time.perf_counter()
    stats.timings_s["chunk_and_extract"] = t2 - t1

    all_chunks = [chunk for profile in profiles for chunk in profile.chunks]
    for chunk in all_chunks:
        stats.chunk_counts[chunk.section] = stats.chunk_counts.get(chunk.section, 0) + 1

    if embedder is None:
        embedder = get_embedder(model)
    embeddings = (
        embedder.embed_documents([chunk.embed_text for chunk in all_chunks]) if all_chunks else []
    )
    t3 = time.perf_counter()
    stats.timings_s["embed"] = t3 - t2

    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )

    profile_by_candidate = {profile.candidate_id: profile for profile in profiles}
    if all_chunks:
        collection.add(
            ids=[chunk.chunk_id for chunk in all_chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in all_chunks],
            metadatas=[
                {
                    "candidate_id": chunk.candidate_id,
                    "name": profile_by_candidate[chunk.candidate_id].name,
                    "resume_path": profile_by_candidate[chunk.candidate_id].resume_path,
                    "section": chunk.section,
                    "experience_years": profile_by_candidate[chunk.candidate_id].experience_years,
                    "education_level": profile_by_candidate[chunk.candidate_id].education_level,
                    "skills": profile_by_candidate[chunk.candidate_id].skills or [""],
                }
                for chunk in all_chunks
            ],
        )

    out_dir = collection_dir(model, chunking)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates.json").write_text(json.dumps([p.to_dict() for p in profiles], indent=2))

    t4 = time.perf_counter()
    stats.timings_s["store"] = t4 - t3
    stats.timings_s["total"] = t4 - t0

    (out_dir / "ingest_stats.json").write_text(json.dumps(asdict(stats), indent=2))

    return stats


def load_candidates(
    model: str = config.DEFAULT_MODEL, chunking: str = config.DEFAULT_CHUNKING
) -> list[CandidateProfile]:
    candidates_path = collection_dir(model, chunking) / "candidates.json"
    data = json.loads(candidates_path.read_text())
    return [CandidateProfile.from_dict(d) for d in data]
