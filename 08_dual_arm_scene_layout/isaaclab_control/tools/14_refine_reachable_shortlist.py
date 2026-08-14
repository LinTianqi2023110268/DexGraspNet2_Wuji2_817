#!/usr/bin/env python3
"""Refine coarse arm-reachable candidates through the complete audited gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
LAYOUT = PROJECT / "08_dual_arm_scene_layout"
PIPELINE = PROJECT / "06_leap_to_wuji2_final_pipeline"
GRASPNET_PY = Path("/home/lin/miniconda3/envs/graspnet2.0/bin/python")
FACTORY_PY = Path("/home/lin/miniconda3/envs/wuji2_factory/bin/python")
RETARGET_PY = PROJECT / "01_environment/conda/wuji_retargeting/bin/python"


def run(command: list[str], env: dict[str, str] | None = None) -> bool:
    print("\n$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=PROJECT, env=env, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--position-mm", type=float, default=5.0)
    parser.add_argument("--orientation-deg", type=float, default=5.0)
    args = parser.parse_args()

    capture = args.capture_root.resolve()
    input_root = args.input_root.resolve()
    scene_manifest = args.scene_manifest.resolve()
    shortlist_path = input_root / "arm_approach_reachability_shortlist.json"
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))["candidates"]
    records = []
    selected = None
    for row in shortlist[: args.max_candidates]:
        rank = int(row["official_score_rank"])
        candidate = int(row["candidate_index"])
        case_id = f"live_dynamic_scene0000_{args.target}_candidate{candidate}"
        case_root = PIPELINE / "01_cases" / case_id
        print(f"\n{'=' * 76}\n[FULL REFINE] coarse rank={rank}; candidate={candidate}\n{'=' * 76}", flush=True)
        prepare = [
            str(GRASPNET_PY), str(LAYOUT / "scripts/11_prepare_ashtray_retarget_case.py"),
            "--input-root", str(input_root), "--capture-root", str(capture),
            "--scene-manifest", str(scene_manifest), "--target", args.target,
            "--case-id", case_id, "--candidate-index", str(candidate), "--replace",
        ]
        env = dict(os.environ, DGN2_CASE_ID=case_id)
        gates = [
            ("prepare", prepare, None),
            ("retarget_grasp", [str(RETARGET_PY), str(PIPELINE / "02_scripts/01_retarget_grasp_official.py")], env),
            ("root6d", [str(RETARGET_PY), str(PIPELINE / "02_scripts/02_align_root6d.py")], env),
            ("retarget_squeeze", [str(RETARGET_PY), str(PIPELINE / "02_scripts/03_retarget_squeeze_official.py")], env),
            ("hand_job", [str(GRASPNET_PY), str(PIPELINE / "02_scripts/05_build_isaacsim_validation.py")], env),
            ("flange_targets", [str(GRASPNET_PY), str(LAYOUT / "isaaclab_control/tools/03_build_arm_execution_targets.py"), "--case-root", str(case_root)], None),
            ("full_ik", [str(FACTORY_PY), str(LAYOUT / "isaaclab_control/tools/08_solve_full_arm_waypoints.py"), "--case-root", str(case_root), "--position-mm", str(args.position_mm), "--orientation-deg", str(args.orientation_deg)], None),
            ("joint_path", [str(FACTORY_PY), str(LAYOUT / "isaaclab_control/tools/09_audit_full_joint_path.py"), "--case-root", str(case_root)], None),
        ]
        failed_gate = None
        for gate, command, command_env in gates:
            if not run(command, command_env):
                failed_gate = gate
                break
        record = {
            "coarse_rank": rank, "candidate": candidate,
            "status": "PASS" if failed_gate is None else "FAIL",
            "failed_gate": failed_gate, "case_root": str(case_root),
        }
        records.append(record)
        if failed_gate is None:
            selected = record
            print(f"\n[FULL EXECUTABLE CASE] {case_root}", flush=True)
            break

    report = {
        "schema_version": 1,
        "status": "FULL_EXECUTABLE_FOUND" if selected else "NO_FULL_EXECUTABLE_CASE",
        "selected": selected,
        "candidates": records,
    }
    output = input_root / "full_candidate_refinement.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[{report['status']}] {output}", flush=True)
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
