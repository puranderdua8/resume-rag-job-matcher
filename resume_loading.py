"""Read every supported resume via the Milestone 1 file-system tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import config
import file_tools

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


def load_resumes(directory: Path = config.RESUMES_DIR) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every supported resume in `directory` via the M1 tools.

    Returns (loaded, skipped) where `loaded` items have candidate_id, resume_path,
    source_path and text, and `skipped` holds a human-readable reason per failed file.

    `candidate_id` is the file name *with* its extension (R9), so same-stem resumes
    in different formats (`jane_doe.pdf` and `jane_doe.docx`) don't collide; if two
    files still produce the same id, this raises instead of silently overwriting one.
    `resume_path` is relative to `config.ROOT_DIR` when the file is inside it and
    absolute otherwise (R10); `source_path` is always absolute, for M1 tools that
    resolve relative paths against the current working directory (R8).
    """
    loaded: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen_ids: dict[str, str] = {}
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

            resolved = Path(path).resolve()
            candidate_id = resolved.name
            if candidate_id in seen_ids:
                raise ValueError(
                    f"Duplicate candidate_id {candidate_id!r} from both "
                    f"{seen_ids[candidate_id]} and {resolved}"
                )
            seen_ids[candidate_id] = str(resolved)

            try:
                resume_path = str(resolved.relative_to(config.ROOT_DIR))
            except ValueError:
                resume_path = str(resolved)

            loaded.append(
                {
                    "candidate_id": candidate_id,
                    "resume_path": resume_path,
                    "source_path": str(resolved),
                    "text": text,
                }
            )
    loaded.sort(key=lambda item: item["candidate_id"])
    return loaded, skipped
