#!/usr/bin/env python3
"""Resample the official diffusion model until a collision-safe arm-reachable grasp exists.

Each seed is isolated in its own directory.  The program reuses the reviewed
official inference, official PREGRASP collision checker and read-only coarse
right-arm IK screen.  It never starts Isaac Sim and never commands the robot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
LAYOUT = PROJECT / "08_dual_arm_scene_layout"
GRASPNET_PY = Path("/home/lin/miniconda3/envs/graspnet2.0/bin/python")
FACTORY_PY = Path("/home/lin/miniconda3/envs/wuji2_factory/bin/python")
DEFAULT_REFERENCE = (
    PROJECT / "06_leap_to_wuji2_final_pipeline/01_cases"
    / "live_scene0000_ashtray_isaaclab_candidate0274"
)


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--target", default="ashtray")
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--position-mm", type=float, default=8.0)
    parser.add_argument("--orientation-deg", type=float, default=8.0)
    parser.add_argument("--reference-case", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()

    capture = args.capture_root.resolve()
    source_input = capture / "dgn2" / args.target / "network_input.npz"
    if not source_input.is_file():
        raise FileNotFoundError(source_input)
    scene_manifest = args.scene_manifest.resolve()
    if not scene_manifest.is_file():
        raise FileNotFoundError(scene_manifest)

    records = []
    selected = None
    for seed in range(args.first_seed, args.first_seed + args.seed_count):
        seed_root = capture / "dgn2" / args.target / "resamples" / f"diffusion_seed_{seed:02d}"
        seed_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_input, seed_root / "network_input.npz")
        print(f"\n{'=' * 76}\n[DIFFUSION RESAMPLE] seed={seed}\n{'=' * 76}", flush=True)
        try:
            run([
                str(GRASPNET_PY), str(LAYOUT / "scripts/09_predict_official_leap_target.py"),
                "--input-root", str(seed_root), "--target", args.target, "--seed", str(seed),
            ])
            run([
                str(GRASPNET_PY), str(LAYOUT / "scripts/10_filter_target_pregrasp_collision.py"),
                "--input-root", str(seed_root), "--target", args.target,
                "--scene-manifest", str(scene_manifest),
            ])
            result = subprocess.run(
                [
                    str(FACTORY_PY),
                    str(LAYOUT / "isaaclab_control/tools/07_screen_approach_reachable_candidates.py"),
                    "--capture-target", str(seed_root),
                    "--reference-case", str(args.reference_case.resolve()),
                    "--keep", "20", "--position-mm", str(args.position_mm),
                    "--orientation-deg", str(args.orientation_deg),
                ],
                cwd=PROJECT,
                check=False,
            )
            report_path = seed_root / "arm_approach_reachability_shortlist.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            kept = int(report["kept_candidates"])
            records.append({"seed": seed, "status": report["status"], "kept": kept, "root": str(seed_root)})
            print(f"[SEED RESULT] seed={seed}; arm-reachable coarse candidates={kept}", flush=True)
            if result.returncode == 0 and kept > 0:
                selected = records[-1]
                break
        except subprocess.CalledProcessError as error:
            records.append({"seed": seed, "status": "stage_failed", "returncode": error.returncode, "root": str(seed_root)})

    summary = {
        "schema_version": 1,
        "status": "REACHABLE_SEED_FOUND" if selected else "NO_REACHABLE_SEED",
        "thresholds": {"position_mm": args.position_mm, "orientation_deg": args.orientation_deg},
        "selected": selected,
        "seeds": records,
    }
    output = capture / "dgn2" / args.target / "diffusion_resample_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[{summary['status']}] {output}", flush=True)
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
