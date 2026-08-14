#!/usr/bin/env python3
"""One-click test scene/view -> LEAP -> Wuji2 -> Trimesh -> Isaac scripts.

Usually you only edit these three values, then press Run or execute this file:
``SCENE_INDEX``, ``VIEW_INDEX`` and ``CASE_ID``.

The program intentionally calls the existing reviewed stage scripts.  It does
not duplicate inference, retargeting or Isaac Sim logic.  Completed products
are reused, so an interrupted run can be started again safely.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ======================== USER SETTINGS ========================
SCENE_INDEX = 1
VIEW_INDEX = 1
CASE_ID = "scene0001_view0001_official_rank0"
# ===============================================================

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
SCRIPTS = PIPELINE_ROOT / "02_scripts"
ACTIVE_CASES = PIPELINE_ROOT / "01_cases/active"
ACTIVE_CASE = PIPELINE_ROOT / "active_case.json"
RETARGET_PYTHON = PROJECT_ROOT / "01_environment/conda/wuji_retargeting/bin/python"
NETWORK_PYTHON = Path("/home/lin/miniconda3/envs/graspnet2.0/bin/python")


def run_step(label: str, command: list[str], env: dict[str, str], log) -> None:
    banner = f"\n{'=' * 18} {label} {'=' * 18}\n"
    print(banner, end="", flush=True)
    log.write(banner)
    log.flush()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{label} failed with exit code {return_code}")


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-index", type=int, default=SCENE_INDEX)
    parser.add_argument("--view-index", type=int, default=VIEW_INDEX)
    parser.add_argument("--case-id", default=CASE_ID)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rerun completed stages; never deletes the case directory",
    )
    args = parser.parse_args()
    case_root = ACTIVE_CASES / args.case_id
    case_json = case_root / "case.json"
    expected_scene = f"scene_{args.scene_index:04d}"
    expected_view = f"view_{args.view_index:04d}"

    for executable in (RETARGET_PYTHON, NETWORK_PYTHON):
        require(executable)
    env = os.environ.copy()
    env["DGN2_CASE_ID"] = args.case_id

    other_active = [
        path for path in ACTIVE_CASES.iterdir()
        if path.is_dir() and path.name != args.case_id
    ]
    if other_active:
        raise RuntimeError(
            "01_cases/active may contain only one working case. Move the current "
            f"case to 99_archive/regenerable_cases first: {other_active}"
        )

    if case_json.is_file():
        recorded = json.loads(case_json.read_text(encoding="utf-8"))
        if recorded.get("scene_id") != expected_scene or recorded.get("view_id") != expected_view:
            raise RuntimeError(
                f"case id already belongs to {recorded.get('scene_id')}/"
                f"{recorded.get('view_id')}, not {expected_scene}/{expected_view}"
            )
    else:
        subprocess.run(
            [
                str(RETARGET_PYTHON), str(SCRIPTS / "00a_prepare_test_case.py"),
                "--scene-index", str(args.scene_index),
                "--view-index", str(args.view_index),
                "--case-id", args.case_id,
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )

    log_path = case_root / "pipeline.log"
    with log_path.open("a", encoding="utf-8") as log:
        stages = [
            (
                "00 point-cloud visualization",
                RETARGET_PYTHON,
                SCRIPTS / "00d_view_case_point_cloud.py",
                case_root / "01_input/single_view_point_cloud.glb",
            ),
            (
                "01 official DexGraspNet2 LEAP inference",
                NETWORK_PYTHON,
                SCRIPTS / "00b_predict_official_leap.py",
                case_root / "01_input/official_leap_1024.npz",
            ),
            (
                "02 official LEAP waypoint construction",
                NETWORK_PYTHON,
                SCRIPTS / "00c_build_leap_waypoints.py",
                case_root / "01_input/leap_official_waypoints.npz",
            ),
            (
                "03 official Wuji retargeting at GRASP",
                RETARGET_PYTHON,
                SCRIPTS / "01_retarget_grasp_official.py",
                case_root / "02_retargeting/grasp_official.npz",
            ),
            (
                "04 four-fingertip root 6D alignment",
                RETARGET_PYTHON,
                SCRIPTS / "02_align_root6d.py",
                case_root / "03_root_alignment/root_alignment.npz",
            ),
            (
                "05 official Wuji retargeting at SQUEEZE",
                RETARGET_PYTHON,
                SCRIPTS / "03_retarget_squeeze_official.py",
                case_root / "04_squeeze/squeeze_official.npz",
            ),
            (
                "06 four-hand Trimesh export",
                RETARGET_PYTHON,
                SCRIPTS / "04_visualize_final.py",
                case_root / "05_visualization/four_hand_final.glb",
            ),
            (
                "07 Isaac Sim two-stage script generation",
                NETWORK_PYTHON,
                SCRIPTS / "05_build_isaacsim_validation.py",
                case_root / "06_isaacsim/final_waypoints.npz",
            ),
        ]
        for label, python, script, product in stages:
            if product.is_file() and not args.rebuild:
                line = f"[REUSE] {label}: {product}\n"
                print(line, end="")
                log.write(line)
                continue
            run_step(label, [str(python), str(script)], env, log)
            require(product)

    # Switch the default only after every required stage succeeds.
    ACTIVE_CASE.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_case_id": args.case_id,
                "note": "Written only after the one-click pipeline completed.",
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    final_case = json.loads(case_json.read_text(encoding="utf-8"))
    final_case["pipeline_status"] = "offline_pipeline_complete"
    case_json.write_text(
        json.dumps(final_case, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n[ALL PASS] one-click case generation completed")
    print(f"[CASE] {case_root}")
    print(f"[POINT CLOUD] {case_root / '01_input/single_view_point_cloud.glb'}")
    print(f"[FOUR HANDS] {case_root / '05_visualization/four_hand_final.glb'}")
    print(f"[ISAAC 01] {case_root / '06_isaacsim/01_import.py'}")
    print(f"[ISAAC 02] {case_root / '06_isaacsim/02_execute.py'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[STOP] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
