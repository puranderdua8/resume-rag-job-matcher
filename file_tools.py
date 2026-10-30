"""Shim that re-exports Milestone 1's file-system tools for this project.

M1's modules (`fs_tools`, `parsers`, `sandbox`, `errors`, `utils`) are
top-level files that import each other by bare name, so the submodule
directory must be on `sys.path` before they're imported. `FS_ROOT` is left
unset, so M1's sandbox stays off and the tools accept normal paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

_M1_DIR = Path(__file__).resolve().parent / "vendor" / "file-system-assistant"
if str(_M1_DIR) not in sys.path:
    sys.path.insert(0, str(_M1_DIR))

from fs_tools import list_files, read_file, search_in_file, write_file  # noqa: E402

__all__ = ["list_files", "read_file", "search_in_file", "write_file"]
