"""Build embeddable chunks from a resume's sections (`section`/`fixed`/`whole` strategies)."""

from __future__ import annotations

import logging

import config
from resume_dates import DATE_RANGE_RE
from resume_models import Chunk

logger = logging.getLogger(__name__)

BULLET_GLYPHS = "-*•–▪"


def split_experience_entries(lines: list[str]) -> list[list[str]]:
    """Split Experience-section lines into per-job entries at date-range lines.

    A date-range line's preceding title line (if any) is pulled in with it, so
    each entry reads as [title, date line, bullets...]. That pull only happens
    when the date-range line contains *nothing but* the date range; when the
    title and dates share one line ("Eng, A | Jan 2020 - Present"), pulling in
    the actual previous line would instead misattribute the prior job's last
    bullet to this entry (R15).
    """
    boundaries: list[int] = []
    for i, line in enumerate(lines):
        match = DATE_RANGE_RE.search(line)
        if not match:
            continue
        start = i
        remainder = (line[: match.start()] + line[match.end() :]).strip(" |,-–—")
        if (
            not remainder
            and i > 0
            and lines[i - 1].strip()
            and not DATE_RANGE_RE.search(lines[i - 1])
        ):
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


def _starts_with_bullet(line: str) -> bool:
    return line.strip()[:1] in BULLET_GLYPHS


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split a single over-long bullet by character count, as a last resort (R17)."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [text]


def _split_long_chunk(text: str, max_chars: int = config.MAX_CHUNK_CHARS) -> list[str]:
    """Split an over-long chunk at bullet boundaries, with a one-bullet overlap.

    Recognises any common bullet glyph, not just "-" (R17): a resume using
    "•" exclusively used to produce one chunk per section regardless of
    length. A single bullet that is still over `max_chars` on its own is
    hard-split by character count rather than left oversized.
    """
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    bullets: list[list[str]] = [[]]
    for line in lines:
        if _starts_with_bullet(line) and bullets[-1]:
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
    pieces = pieces or [text]

    result: list[str] = []
    for piece in pieces:
        if len(piece) > max_chars:
            result.extend(_hard_split(piece, max_chars))
        else:
            result.append(piece)
    return result


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


def _make_chunk(
    chunk_id: str, candidate_id: str, section: str, text: str, embed_text: str
) -> Chunk:
    if len(embed_text) > config.EMBEDDING_MAX_SEQ_CHARS:
        logger.warning(
            "Chunk %s's embed text is %d chars, over the ~%d-char budget for the model's "
            "max sequence length; it will be truncated at embedding time.",
            chunk_id,
            len(embed_text),
            config.EMBEDDING_MAX_SEQ_CHARS,
        )
    return Chunk(
        chunk_id=chunk_id,
        candidate_id=candidate_id,
        section=section,
        text=text,
        embed_text=embed_text,
    )


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
            _make_chunk(
                f"{candidate_id}::whole::0",
                candidate_id,
                "whole",
                full_text,
                f"{name} | whole\n{full_text}",
            )
        )
        return chunks

    if strategy == "fixed":
        for n, piece in enumerate(_chunk_fixed(full_text)):
            chunks.append(
                _make_chunk(
                    f"{candidate_id}::fixed::{n}",
                    candidate_id,
                    "fixed",
                    piece,
                    f"{name} | fixed\n{piece}",
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
                    _make_chunk(
                        f"{candidate_id}::{section}::{n}",
                        candidate_id,
                        section,
                        piece,
                        f"{name} | {section}\n{piece}",
                    )
                )
                n += 1
    return chunks
