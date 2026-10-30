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
