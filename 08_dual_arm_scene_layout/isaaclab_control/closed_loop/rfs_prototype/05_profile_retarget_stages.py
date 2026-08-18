#!/usr/bin/env python3
"""
Stage-wise timing wrapper for the existing LEAP->Wuji2 retarget pipeline.

It intentionally implements NO retargeting math.  It mirrors the production
batch_retarget_cases.py behavior (one retarget interpreter stays alive) while
timing the existing 01/02/03 scripts separately for every candidate.
"""

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
STAGES = [
    PIPELINE / "01_retarget_grasp_official.py",
    PIPELINE / "02_align_root6d.py",
    PIPELINE / "03_retarget_squeeze_official.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for stage in STAGES:
        if not stage.is_file():
            raise FileNotFoundError(stage)

    items = json.loads(args.items_json.read_text(encoding="utf-8"))
    old_case_root = os.environ.get("DGN2_CASE_ROOT")
    old_argv = list(sys.argv)

    batch_t0 = time.perf_counter()
    results = []
    stage_totals = {stage.name: 0.0 for stage in STAGES}
    stage_pass_counts = {stage.name: 0 for stage in STAGES}

    try:
        for n, item in enumerate(items, start=1):
            case_root = Path(item["case_root"]).expanduser().resolve()
            os.environ["DGN2_CASE_ROOT"] = str(case_root)
            case_t0 = time.perf_counter()
            stage_rows = []
            status = "PASS"
            failure = None

            for stage in STAGES:
                sys.argv = [str(stage)]
                t0 = time.perf_counter()
                try:
                    runpy.run_path(str(stage), run_name="__main__")
                except Exception as exc:
                    dt = time.perf_counter() - t0
                    stage_totals[stage.name] += dt
                    stage_rows.append(
                        {
                            "stage": stage.name,
                            "status": "FAIL",
                            "wall_time_s": dt,
                            "failure_type": type(exc).__name__,
                            "failure_reason": str(exc),
                        }
                    )
                    status = "FAIL"
                    failure = {
                        "stage": stage.name,
                        "failure_type": type(exc).__name__,
                        "failure_reason": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-12:],
                    }
                    break
                else:
                    dt = time.perf_counter() - t0
                    stage_totals[stage.name] += dt
                    stage_pass_counts[stage.name] += 1
                    stage_rows.append(
                        {"stage": stage.name, "status": "PASS", "wall_time_s": dt}
                    )

            case_dt = time.perf_counter() - case_t0
            results.append(
                {
                    "filtered_position": item.get("filtered_position"),
                    "target_rank": int(item["target_rank"]),
                    "candidate_index": int(item["candidate_index"]),
                    "case_id": item["case_id"],
                    "case_root": str(case_root),
                    "status": status,
                    "wall_time_s": case_dt,
                    "stages": stage_rows,
                    "failure": failure,
                }
            )
            stage_str = " ".join(
                f"{row['stage'][:2]}={row['wall_time_s']:.3f}s" for row in stage_rows
            )
            print(
                f"[retarget {n:02d}/{len(items):02d}] "
                f"rank={int(item['target_rank']):4d} cand={int(item['candidate_index']):4d} "
                f"{status} total={case_dt:.3f}s | {stage_str}",
                flush=True,
            )
    finally:
        sys.argv = old_argv
        if old_case_root is None:
            os.environ.pop("DGN2_CASE_ROOT", None)
        else:
            os.environ["DGN2_CASE_ROOT"] = old_case_root

    wall = time.perf_counter() - batch_t0
    n = max(1, len(items))
    out = {
        "status": "PASS" if all(r["status"] == "PASS" for r in results) else "PARTIAL_FAIL",
        "candidate_count": len(items),
        "pass_count": sum(r["status"] == "PASS" for r in results),
        "fail_count": sum(r["status"] != "PASS" for r in results),
        "wall_time_s": wall,
        "stage_totals_s": stage_totals,
        "stage_mean_per_candidate_s": {k: v / n for k, v in stage_totals.items()},
        "stage_pass_counts": stage_pass_counts,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": out["status"],
                "candidate_count": out["candidate_count"],
                "pass_count": out["pass_count"],
                "fail_count": out["fail_count"],
                "wall_time_s": out["wall_time_s"],
                "stage_totals_s": stage_totals,
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
