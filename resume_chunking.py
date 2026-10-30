"""Build embeddable chunks from a resume's sections (`section`/`fixed`/`whole` strategies)."""

from __future__ import annotations

import config
from resume_dates import DATE_RANGE_RE
from resume_models import Chunk


def split_experience_entries(lines: list[str]) -> list[list[str]]:
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
            entries = split_experience_entries(lines)
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
