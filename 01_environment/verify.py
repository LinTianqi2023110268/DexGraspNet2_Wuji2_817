#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/project.json").read_text())
sys.path.insert(0, str(ROOT / "src"))
from wuji2_dgn2.official_asset import verify_canonical_assets  # noqa: E402

print("official_wuji2_assets", verify_canonical_assets())

for name in ("network_python", "simulation_python"):
    python = Path(CFG["runtime"][name])
    command = [str(python), "-c", "import sys,torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available())"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    print(name, "OK" if result.returncode == 0 else "FAIL", result.stdout.strip() or result.stderr.strip())
