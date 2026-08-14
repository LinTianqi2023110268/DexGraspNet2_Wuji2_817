#!/usr/bin/env python3
"""Run the canonical Wuji2 scene-collision filter.

The implementation lives in ``src/wuji2_dgn2/collision.py`` so dataset
generation and inference cannot silently use different PREGRASP directions.
The canonical Wuji2 policy is the reviewed tiger-mouth direction: semantic
palm centre toward the midpoint of the current-GRASP thumb and index tips.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.collision import main  # noqa: E402


if __name__ == "__main__":
    main()
