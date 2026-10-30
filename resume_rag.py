"""Part A: load resumes, chunk by section, extract metadata, embed and store in ChromaDB.

CLI:
    python resume_rag.py ingest [--model bge-small] [--chunking section]
    python resume_rag.py stats [--model bge-small] [--chunking section]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import chromadb

import config
import file_tools
import skills

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

SECTION_SYNONYMS: dict[str, str] = {
    "summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "experience": "experience",
    "work history": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "education": "education",
    "skills": "skills",
    "technical skills": "skills",
    "core competencies": "skills",
    "projects": "projects",
    "certifications": "certifications",
}
MAX_HEADER_WORDS = 5

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
NAME_RE = re.compile(r"^(?:[A-Z][a-zA-Z'.-]*\s+){1,3}[A-Z][a-zA-Z'.-]*$")
EXPLICIT_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*years?", re.IGNORECASE)
DATE_TOKEN = r"[A-Za-z]{3,9}\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}"
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*[-–—]\s*(?P<end>Present|{DATE_TOKEN})",
    re.IGNORECASE,
)
EDU_RE = re.compile(
    r"^(?P<degree>Ph\.?D\.?|M\.?S\.?|M\.?A\.?|B\.?S\.?|B\.?A\.?|Associate'?s?|Bachelor'?s?|Master'?s?)"
    r"\s+(?:degree\s+)?in\s+(?P<field>[^,]+),\s*(?P<institution>.+)$",
    re.IGNORECASE,
)
DEGREE_LEVELS: dict[str, int] = {
    "phd": 4,
    "doctorate": 4,
    "ms": 3,
    "ma": 3,
    "master": 3,
    "masters": 3,
    "bs": 2,
    "ba": 2,
    "bachelor": 2,
    "bachelors": 2,
    "as": 1,
    "aa": 1,
    "associate": 1,
    "associates": 1,
}

CONTACT_SECTION = "contact"


@dataclass
class Chunk:
    chunk_id: str
    candidate_id: str
    section: str
    text: str
    embed_text: str


@dataclass
class CandidateProfile:
    candidate_id: str
    name: str
    resume_path: str
    email: str | None = None
    phone: str | None = None
    skills: list[str] = field(default_factory=list)
    skill_sections: dict[str, list[str]] = field(default_factory=dict)
    skill_years: dict[str, float] = field(default_factory=dict)
    experience_years: float = 0.0
    education_level: int = 0
    education_degree: str | None = None
    education_field: str | None = None
    education_institution: str | None = None
    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ""
    chunks: list[Chunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateProfile:
        chunk_dicts = data.pop("chunks", [])
        profile = cls(**data)
        profile.chunks = [Chunk(**c) for c in chunk_dicts]
        return profile


@dataclass
class IngestStats:
    model: str
    chunking: str
    collection: str
    resumes_loaded: int = 0
    resumes_skipped: list[str] = field(default_factory=list)
    chunk_counts: dict[str, int] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)


# --- Loading -----------------------------------------------------------------


def load_resumes(directory: Path = config.RESUMES_DIR) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every supported resume in `directory` via the M1 tools.

    Returns (loaded, skipped) where `loaded` items have candidate_id, resume_path
    and text, and `skipped` holds a human-readable reason per failed file.
    """
    loaded: list[dict[str, Any]] = []
    skipped: list[str] = []
    for ext in SUPPORTED_EXTENSIONS:
        for entry in file_tools.list_files(str(directory), ext):
            if not entry.get("success", True):
                skipped.append(f"list_files({ext}): {entry.get('error')}")
                continue
            path = entry["path"]
            result = file_tools.read_file(path)
            if not result.get("success"):
                reason = f"{path}: {result.get('error')}"
                logger.warning("Skipping resume: %s", reason)
                skipped.append(reason)
                continue
            text = result["content"]
            if not text.strip():
                reason = f"{path}: no text extracted"
                logger.warning("Skipping resume: %s", reason)
                skipped.append(reason)
                continue
            candidate_id = Path(path).stem
            loaded.append(
                {
                    "candidate_id": candidate_id,
                    "resume_path": str(Path(path).resolve().relative_to(config.ROOT_DIR)),
                    "text": text,
                }
            )
    loaded.sort(key=lambda item: item["candidate_id"])
    return loaded, skipped


# --- Section detection ---------------------------------------------------------


def _header_canonical(line: str) -> str | None:
    stripped = line.strip().rstrip(":")
    if not stripped or len(stripped.split()) > MAX_HEADER_WORDS:
        return None
    return SECTION_SYNONYMS.get(stripped.lower())


def _inline_header(line: str) -> tuple[str, str] | None:
    """Match `"Skills: Python, SQL"` style headers: label, colon, content on one line."""
    if ":" not in line:
        return None
    label, _, rest = line.partition(":")
    if not rest.strip():
        return None
    canonical = SECTION_SYNONYMS.get(label.strip().lower())
    if canonical and len(label.strip().split()) <= MAX_HEADER_WORDS:
        return canonical, rest.strip()
    return None


def detect_sections(text: str) -> dict[str, list[str]]:
    """Split resume text into canonical sections, preserving line order within each."""
    sections: dict[str, list[str]] = {CONTACT_SECTION: []}
    current = CONTACT_SECTION
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        canonical = _header_canonical(line)
        if canonical:
            current = canonical
            sections.setdefault(current, [])
            continue
        inline = _inline_header(line)
        if inline:
            canonical, content = inline
            current = canonical
            sections.setdefault(current, []).append(content)
            continue
        sections.setdefault(current, []).append(line)
    return {section: lines for section, lines in sections.items() if lines}


# --- Date parsing ---------------------------------------------------------


def _parse_date_token(token: str) -> date:
    token = token.strip()
    if token.lower() == "present":
        return date.today()
    if re.fullmatch(r"\d{4}", token):
        # A bare year loses sub-year precision; the middle of the year is the
        # unbiased point estimate given no other info.
        return date(int(token), 7, 2)
    if re.fullmatch(r"\d{1,2}/\d{4}", token):
        month, year = token.split("/")
        return date(int(year), int(month), 1)
    for fmt in ("%b %Y", "%B %Y"):
        try:
            return datetime.strptime(token.replace(".", ""), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date token: {token!r}")


def _parse_date_range(line: str) -> tuple[date, date] | None:
    match = DATE_RANGE_RE.search(line)
    if not match:
        return None
    try:
        start = _parse_date_token(match.group("start"))
        end = _parse_date_token(match.group("end"))
    except ValueError:
        return None
    if end <= start:
        end = start + timedelta(days=182)
    return start, end


def merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def years_between(intervals: list[tuple[date, date]]) -> float:
    merged = merge_intervals(intervals)
    days = sum((end - start).days for start, end in merged)
    return round(days / 365.25, 1)


# --- Experience entry splitting -------------------------------------------


def _split_experience_entries(lines: list[str]) -> list[list[str]]:
    """Split Experience-section lines into per-job entries at date-range lines.

    A date-range line's preceding title line (if any) is pulled in with it, so
    each entry reads as [title, date line, bullets...].
    """
    boundaries: list[int] = []
    for i, line in enumerate(lines):
        if not DATE_RANGE_RE.search(line):
            continue
        start = i
        if i > 0 and lines[i - 1].strip() and not DATE_RANGE_RE.search(lines[i - 1]):
            start = i - 1
        if not boundaries or start > boundaries[-1]:
            boundaries.append(start)
    if not boundaries:
        return [lines]
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    entries = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        entry = lines[start:end]
        if entry:
            entries.append(entry)
    return entries


# --- Chunking ---------------------------------------------------------


def _split_long_chunk(text: str, max_chars: int = config.MAX_CHUNK_CHARS) -> list[str]:
    """Split an over-long chunk at bullet boundaries, with a one-bullet overlap."""
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    bullets: list[list[str]] = [[]]
    for line in lines:
        if line.strip().startswith("-") and bullets[-1]:
            bullets.append([])
        bullets[-1].append(line)

    pieces: list[str] = []
    current: list[list[str]] = []
    current_len = 0
    for i, bullet in enumerate(bullets):
        bullet_text = "\n".join(bullet)
        if current and current_len + len(bullet_text) > max_chars:
            pieces.append("\n".join("\n".join(b) for b in current))
            current = [bullets[i - 1]] if i > 0 else []
            current_len = sum(len("\n".join(b)) for b in current)
        current.append(bullet)
        current_len += len(bullet_text)
    if current:
        pieces.append("\n".join("\n".join(b) for b in current))
    return pieces or [text]


def _chunk_fixed(text: str) -> list[str]:
    size, overlap = config.FIXED_CHUNK_SIZE, config.FIXED_CHUNK_OVERLAP
    step = size - overlap
    pieces = []
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if piece.strip():
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces or [text]


def build_chunks(
    candidate_id: str,
    name: str,
    sections: dict[str, list[str]],
    full_text: str,
    strategy: str = config.DEFAULT_CHUNKING,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    if strategy == "whole":
        chunks.append(
            Chunk(
                chunk_id=f"{candidate_id}::whole::0",
                candidate_id=candidate_id,
                section="whole",
                text=full_text,
                embed_text=f"{name} | whole\n{full_text}",
            )
        )
        return chunks

    if strategy == "fixed":
        for n, piece in enumerate(_chunk_fixed(full_text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{candidate_id}::fixed::{n}",
                    candidate_id=candidate_id,
                    section="fixed",
                    text=piece,
                    embed_text=f"{name} | fixed\n{piece}",
                )
            )
        return chunks

    for section, lines in sections.items():
        if section == "experience":
            entries = _split_experience_entries(lines)
            texts = ["\n".join(entry) for entry in entries]
        else:
            texts = ["\n".join(lines)]

        n = 0
        for text in texts:
            for piece in _split_long_chunk(text):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{candidate_id}::{section}::{n}",
                        candidate_id=candidate_id,
                        section=section,
                        text=piece,
                        embed_text=f"{name} | {section}\n{piece}",
                    )
                )
                n += 1
    return chunks


# --- Metadata extraction ---------------------------------------------------


def _extract_name(sections: dict[str, list[str]], candidate_id: str) -> str:
    for line in sections.get(CONTACT_SECTION, []):
        candidate = line.strip()
        if "@" in candidate or "|" in candidate or "http" in candidate.lower():
            continue
        if any(ch.isdigit() for ch in candidate):
            continue
        words = candidate.split()
        if 2 <= len(words) <= 4 and NAME_RE.match(candidate):
            return candidate
    return candidate_id.replace("_", " ").replace("-", " ").title()


def _extract_explicit_years(sections: dict[str, list[str]]) -> float | None:
    for section in ("summary", CONTACT_SECTION):
        for line in sections.get(section, []):
            match = EXPLICIT_YEARS_RE.search(line)
            if match:
                return float(match.group(1))
    return None


def _extract_skills(sections: dict[str, list[str]]) -> tuple[list[str], dict[str, list[str]]]:
    matched: list[str] = []
    skill_sections: dict[str, list[str]] = {}
    for canonical, aliases in skills.SKILL_ALIASES.items():
        found_in: list[str] = []
        for section, lines in sections.items():
            section_text = "\n".join(lines)
            for alias in aliases:
                if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", section_text, re.IGNORECASE):
                    found_in.append(section)
                    break
        if found_in:
            matched.append(canonical)
            skill_sections[canonical] = found_in
    return matched, skill_sections


def _extract_education(
    sections: dict[str, list[str]],
) -> tuple[str | None, str | None, str | None, int]:
    for line in sections.get("education", []):
        match = EDU_RE.match(line.strip())
        if not match:
            continue
        degree = match.group("degree")
        degree_key = re.sub(r"[.'\s]", "", degree).lower()
        level = DEGREE_LEVELS.get(degree_key, 0)
        return degree, match.group("field").strip(), match.group("institution").strip(), level
    return None, None, None, 0


def _experience_entries_with_dates(
    sections: dict[str, list[str]],
) -> list[tuple[tuple[date, date], str]]:
    entries = _split_experience_entries(sections.get("experience", []))
    result = []
    for entry in entries:
        entry_text = "\n".join(entry)
        for line in entry:
            date_range = _parse_date_range(line)
            if date_range:
                result.append((date_range, entry_text))
                break
    return result


def extract_metadata(
    candidate_id: str,
    resume_path: str,
    full_text: str,
    sections: dict[str, list[str]],
) -> CandidateProfile:
    name = _extract_name(sections, candidate_id)
    email_match = EMAIL_RE.search(full_text)
    phone_match = PHONE_RE.search(full_text)
    matched_skills, skill_sections = _extract_skills(sections)
    dated_entries = _experience_entries_with_dates(sections)

    explicit_years = _extract_explicit_years(sections)
    if explicit_years is not None:
        experience_years = explicit_years
    else:
        experience_years = years_between([dr for dr, _ in dated_entries])

    skill_years: dict[str, float] = {}
    for canonical_skill, aliases in skills.SKILL_ALIASES.items():
        if canonical_skill not in matched_skills:
            continue
        intervals = []
        for date_range, entry_text in dated_entries:
            if any(
                re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", entry_text, re.IGNORECASE)
                for alias in aliases
            ):
                intervals.append(date_range)
        if intervals:
            skill_years[canonical_skill] = years_between(intervals)
        elif explicit_years is not None and not dated_entries:
            skill_years[canonical_skill] = explicit_years

    degree, edu_field, institution, level = _extract_education(sections)

    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        resume_path=resume_path,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        skills=matched_skills,
        skill_sections=skill_sections,
        skill_years=skill_years,
        experience_years=experience_years,
        education_level=level,
        education_degree=degree,
        education_field=edu_field,
        education_institution=institution,
        sections={section: "\n".join(lines) for section, lines in sections.items()},
        full_text=full_text,
    )


# --- Embedding ---------------------------------------------------------


class Embedder:
    """Wraps a SentenceTransformer model with this project's query/document conventions."""

    def __init__(self, model: Any, spec: config.ModelSpec):
        self.model = model
        self.spec = spec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            prompt=self.spec.query_prompt,
            prompt_name=self.spec.query_prompt_name,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return embedding.tolist()


def get_embedder(model_name: str = config.DEFAULT_MODEL) -> Embedder:
    from sentence_transformers import SentenceTransformer

    spec = config.MODEL_REGISTRY[model_name]
    return Embedder(SentenceTransformer(spec.hf_id), spec)


# --- Storage ---------------------------------------------------------


def collection_name(model: str, chunking: str) -> str:
    return f"resumes__{model}__{chunking}"


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _collection_dir(model: str, chunking: str) -> Path:
    return config.INDEX_DIR / collection_name(model, chunking)


def ingest(
    model: str = config.DEFAULT_MODEL,
    chunking: str = config.DEFAULT_CHUNKING,
    embedder: Embedder | None = None,
    resumes_dir: Path = config.RESUMES_DIR,
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
            item["candidate_id"], item["resume_path"], item["text"], sections
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

    collection_dir = _collection_dir(model, chunking)
    collection_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = collection_dir / "candidates.json"
    candidates_path.write_text(json.dumps([p.to_dict() for p in profiles], indent=2))

    t4 = time.perf_counter()
    stats.timings_s["store"] = t4 - t3
    stats.timings_s["total"] = t4 - t0

    stats_path = collection_dir / "ingest_stats.json"
    stats_path.write_text(json.dumps(asdict(stats), indent=2))

    return stats


def load_candidates(
    model: str = config.DEFAULT_MODEL, chunking: str = config.DEFAULT_CHUNKING
) -> list[CandidateProfile]:
    candidates_path = _collection_dir(model, chunking) / "candidates.json"
    data = json.loads(candidates_path.read_text())
    return [CandidateProfile.from_dict(d) for d in data]


# --- CLI ---------------------------------------------------------


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
    collection_dir = _collection_dir(model, chunking)
    stats_path = collection_dir / "ingest_stats.json"
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
