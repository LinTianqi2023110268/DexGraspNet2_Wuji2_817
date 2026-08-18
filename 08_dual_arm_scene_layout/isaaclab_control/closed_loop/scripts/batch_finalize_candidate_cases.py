#!/usr/bin/env python3
"""Batch wrapper for existing final-waypoint and arm-target builders."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PIPELINE = PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline/02_scripts"
FINAL_WAYPOINTS = PIPELINE / "05_build_isaacsim_validation.py"
ARM_TARGETS = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/tools/03_build_arm_execution_targets.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = json.loads(args.items_json.read_text(encoding="utf-8"))
    started = time.perf_counter()
    results = []
    old_case_root = os.environ.get("DGN2_CASE_ROOT")
    old_argv = list(sys.argv)
    try:
        for item in items:
            case_root = Path(item["case_root"]).expanduser().resolve()
            os.environ["DGN2_CASE_ROOT"] = str(case_root)
            case_started = time.perf_counter()
            result = {
                "case_id": item["case_id"],
                "case_root": str(case_root),
                "candidate_index": int(item["candidate_index"]),
                "target_rank": int(item["target_rank"]) if "target_rank" in item else None,
            }
            try:
                sys.argv = [str(FINAL_WAYPOINTS)]
                failure_stage = "final_waypoints"
                runpy.run_path(str(FINAL_WAYPOINTS), run_name="__main__")
                sys.argv = [str(ARM_TARGETS), "--case-root", str(case_root)]
                failure_stage = "arm_targets"
                runpy.run_path(str(ARM_TARGETS), run_name="__main__")
            except Exception as exc:
                result.update({
                    "status": "REJECT",
                    "failure_stage": failure_stage,
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                    "wall_time_s": time.perf_counter() - case_started,
                })
                print(json.dumps({
                    "status": "REJECT",
                    "case_id": result["case_id"],
                    "candidate_index": result["candidate_index"],
                    "target_rank": result["target_rank"],
                    "failure_reason": result["failure_reason"],
                }, ensure_ascii=False), flush=True)
            else:
                result.update({
                    "status": "PASS",
                    "wall_time_s": time.perf_counter() - case_started,
                })
            results.append(result)
    finally:
        sys.argv = old_argv
        if old_case_root is None:
            os.environ.pop("DGN2_CASE_ROOT", None)
        else:
            os.environ["DGN2_CASE_ROOT"] = old_case_root

    out = {
        "status": "PASS",
        "candidate_count": len(items),
        "pass_count": sum(1 for row in results if row["status"] == "PASS"),
        "reject_count": sum(1 for row in results if row["status"] == "REJECT"),
        "wall_time_s": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "candidate_count": out["candidate_count"],
        "pass_count": out["pass_count"],
        "reject_count": out["reject_count"],
        "wall_time_s": out["wall_time_s"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
