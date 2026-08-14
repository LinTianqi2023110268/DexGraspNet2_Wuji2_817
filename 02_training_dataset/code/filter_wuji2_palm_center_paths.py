#!/usr/bin/env python3
"""Run the canonical non-destructive Wuji2 palm-centerline filter.

The implementation lives in ``src/wuji2_dgn2/palm_path.py`` so training-data
generation and inference share one audited path definition and one mask schema.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.palm_path import main  # noqa: E402


if __name__ == "__main__":
    main()
