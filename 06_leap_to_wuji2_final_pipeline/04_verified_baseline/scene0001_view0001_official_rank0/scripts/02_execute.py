"""Stage 02: execute COVER, GRASP, dense SQUEEZE and LIFT."""

import builtins
import runpy
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = BASELINE_ROOT.parents[1]
CASE_ROOT = PIPELINE_ROOT / "01_cases/scene0001_view0001_official_rank0"
JOB = BASELINE_ROOT / "task/final_waypoints.npz"
RESULT = BASELINE_ROOT / "result/replay_result.json"
COMMON = BASELINE_ROOT / "runtime/common_execute.py"
for path in (COMMON, JOB):
    if not path.is_file():
        raise FileNotFoundError(path)

keys = ("DGN2_AB_BRANCH", "DGN2_AB_JOB_PATH", "DGN2_AB_RESULT_PATH", "DGN2_CASE_ROOT")
old = {key: getattr(builtins, key, None) for key in keys}
try:
    setattr(builtins, "DGN2_AB_BRANCH", "wuji2_leap_root_drive")
    setattr(builtins, "DGN2_AB_JOB_PATH", JOB)
    setattr(builtins, "DGN2_AB_RESULT_PATH", RESULT)
    setattr(builtins, "DGN2_CASE_ROOT", CASE_ROOT)
    runpy.run_path(str(COMMON), run_name="__main__")
finally:
    for key in keys:
        if old[key] is None:
            if hasattr(builtins, key):
                delattr(builtins, key)
        else:
            setattr(builtins, key, old[key])
