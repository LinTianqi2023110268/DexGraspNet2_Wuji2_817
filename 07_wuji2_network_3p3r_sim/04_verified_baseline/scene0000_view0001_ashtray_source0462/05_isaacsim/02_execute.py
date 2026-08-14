"""Isaac Sim Script Editor stage 02: execute native Wuji2 case."""
import builtins
import runpy
from pathlib import Path

PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
PIPELINE_ROOT = PROJECT_ROOT / "07_wuji2_network_3p3r_sim"
CASE_ROOT = PIPELINE_ROOT / "01_cases/selected_native_case"
JOB = CASE_ROOT / "03_waypoints/native_wuji2_3p3r_waypoints.npz"
import asyncio
from omni.kit.async_engine import run_coroutine

CONTEXT_KEY = "DGN2_NATIVE_WUJI2_3P3R_CONTEXT"

async def wait_for_import_then_execute():
    # Stage 01 loads assets asynchronously. Wait for its final context instead
    # of failing when the user clicks 02 a few seconds too early.
    for _ in range(1800):
        if hasattr(builtins, CONTEXT_KEY):
            print("[02] Stage 01 context is ready; starting execution.")
            runpy.run_path(
                str(PIPELINE_ROOT / "03_runtime/execute_native_grasp.py"),
                run_name="__main__",
            )
            return
        await asyncio.sleep(0.1)
    raise RuntimeError(
        "Stage 01 did not finish within 180 s. Re-run 01_import.py and inspect "
        "its first error before running 02 again."
    )

print("[02] Waiting for [01 IMPORT COMPLETE] if necessary...")
run_coroutine(wait_for_import_then_execute())
