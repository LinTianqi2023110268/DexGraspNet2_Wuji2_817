"""Isaac Sim Script Editor stage 01: import native Wuji2 case."""
import builtins
import runpy
from pathlib import Path

PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
PIPELINE_ROOT = PROJECT_ROOT / "07_wuji2_network_3p3r_sim"
CASE_ROOT = PIPELINE_ROOT / "01_cases/selected_native_case"
JOB = CASE_ROOT / "03_waypoints/native_wuji2_3p3r_waypoints.npz"
RESULT = CASE_ROOT / "05_isaacsim/final_result.json"
settings = {
    "DGN2_NATIVE_CASE_ROOT": CASE_ROOT,
    "DGN2_NATIVE_JOB_PATH": JOB,
    "DGN2_NATIVE_RESULT_PATH": RESULT,
}
old = {key: getattr(builtins, key, None) for key in settings}
try:
    for key, value in settings.items(): setattr(builtins, key, value)
    runpy.run_path(str(PIPELINE_ROOT / "03_runtime/import_scene_with_3p3r.py"), run_name="__main__")
finally:
    for key, value in old.items():
        if value is None and hasattr(builtins, key): delattr(builtins, key)
        elif value is not None: setattr(builtins, key, value)
