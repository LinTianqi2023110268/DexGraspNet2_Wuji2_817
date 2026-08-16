#!/usr/bin/env python3
"""Screen settled-scene target candidates through retargeting and arm IK.

This is an orchestration layer only.  It reuses the reviewed stage programs;
it does not implement a second retargeter or IK solver.  Every child process is
run synchronously and its output is streamed to this terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
LAYOUT = PROJECT / "08_dual_arm_scene_layout"
PIPELINE = PROJECT / "06_leap_to_wuji2_final_pipeline"
GRASPNET_PY = Path("/home/lin/miniconda3/envs/graspnet2.0/bin/python")
FACTORY_PY = Path("/home/lin/miniconda3/envs/wuji2_factory/bin/python")
RETARGET_PY = PROJECT / "01_environment/conda/wuji_retargeting/bin/python"


def run(command: list[str], env: dict[str, str] | None = None) -> bool:
    print("\n$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT, env=env, check=False)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--target", default="ashtray")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument(
        "--start-rank",
        type=int,
        default=0,
        help="Resume from this zero-based official-score rank without repeating earlier candidates.",
    )
    parser.add_argument("--position-mm", type=float, default=5.0)
    parser.add_argument("--orientation-deg", type=float, default=5.0)
    args = parser.parse_args()

    capture = args.capture_root.resolve()
    scene_manifest = args.scene_manifest.resolve()
    input_root = capture / "dgn2" / args.target
    collision_path = input_root / "official_leap_target_collision_filtered.npz"
    with np.load(collision_path, allow_pickle=False) as archive:
        candidate_order = np.asarray(
            archive["valid_score_descending_candidate_index"], dtype=np.int64
        ).tolist()

    output = capture / "pipeline_candidate_screen.json"
    previous_summary = []
    if args.start_rank > 0 and output.is_file():
        previous_summary = json.loads(output.read_text(encoding="utf-8")).get("candidates", [])
        previous_summary = [row for row in previous_summary if int(row["rank"]) < args.start_rank]

    summary = list(previous_summary)
    stop_rank = min(len(candidate_order), args.start_rank + args.max_candidates)
    for rank in range(args.start_rank, stop_rank):
        candidate = candidate_order[rank]
        case_id = f"live_dynamic_scene0000_{args.target}_candidate{candidate}"
        case_root = PIPELINE / "01_cases" / case_id
        print(f"\n{'=' * 72}\n[CANDIDATE {rank:02d}] source={candidate}\n{'=' * 72}", flush=True)

        prepare = [
            str(GRASPNET_PY), str(LAYOUT / "scripts/11_prepare_ashtray_retarget_case.py"),
            "--input-root", str(input_root), "--capture-root", str(capture),
            "--scene-manifest", str(scene_manifest), "--case-id", case_id,
            "--candidate-index", str(candidate), "--replace",
        ]
        if not run(prepare):
            summary.append({"rank": rank, "candidate": candidate, "gate": "prepare", "pass": False})
            continue

        env = dict(os.environ, DGN2_CASE_ID=case_id)
        stages = [
            [str(RETARGET_PY), str(PIPELINE / "02_scripts/01_retarget_grasp_official.py")],
            [str(RETARGET_PY), str(PIPELINE / "02_scripts/02_align_root6d.py")],
            [str(RETARGET_PY), str(PIPELINE / "02_scripts/03_retarget_squeeze_official.py")],
            [str(GRASPNET_PY), str(PIPELINE / "02_scripts/05_build_isaacsim_validation.py")],
            [str(GRASPNET_PY), str(LAYOUT / "isaaclab_control/tools/03_build_arm_execution_targets.py"), "--case-root", str(case_root)],
            [str(FACTORY_PY), str(LAYOUT / "isaaclab_control/tools/08_solve_full_arm_waypoints.py"),
             "--case-root", str(case_root), "--position-mm", str(args.position_mm),
             "--orientation-deg", str(args.orientation_deg), "--grasp-only"],
        ]
        passed = True
        failed_gate = ""
        gate_names = ("retarget_grasp", "root6d", "retarget_squeeze", "hand_job", "flange_targets", "arm_ik")
        for gate, command in zip(gate_names, stages):
            if not run(command, env):
                passed = False
                failed_gate = gate
                break
        summary.append(
            {"rank": rank, "candidate": candidate, "gate": "arm_ik" if passed else failed_gate, "pass": passed}
        )
        if passed:
            print(f"\n[REACHABLE CANDIDATE] rank={rank}; source={candidate}; case={case_root}", flush=True)
            break

    output.write_text(json.dumps({"candidates": summary}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[SCREEN REPORT] {output}", flush=True)
    if not summary or not any(bool(row["pass"]) for row in summary):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
