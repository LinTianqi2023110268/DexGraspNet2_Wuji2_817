"""Resolve the active case without duplicating algorithm code.

The default case is selected by ``active_case.json``.  For a one-off run, set
the environment variable ``DGN2_CASE_ID`` to another directory name under
``01_cases/active``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
SHARED_ROOT = PIPELINE_ROOT / "00_shared"
CASES_ROOT = PIPELINE_ROOT / "01_cases/active"
ACTIVE_CASE_FILE = PIPELINE_ROOT / "active_case.json"


def active_case_id() -> str:
    if os.environ.get("DGN2_CASE_ID"):
        return os.environ["DGN2_CASE_ID"].strip()
    data = json.loads(ACTIVE_CASE_FILE.read_text(encoding="utf-8"))
    return str(data["active_case_id"])


def active_case_root() -> Path:
    case_root = (CASES_ROOT / active_case_id()).resolve()
    if case_root.parent != CASES_ROOT.resolve():
        raise RuntimeError(f"invalid case path: {case_root}")
    manifest = case_root / "case.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if str(data.get("case_id")) != case_root.name:
        raise RuntimeError(
            f"case_id mismatch: directory={case_root.name}, json={data.get('case_id')}"
        )
    return case_root
