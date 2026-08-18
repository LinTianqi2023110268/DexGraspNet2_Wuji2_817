#!/usr/bin/env python3
"""Calibrate the coarse LEAP-root -> final Wuji2-wrist bridge from completed cases.

This is a standalone offline diagnostic.  It does not start Isaac Sim and does not
modify the closed-loop pipeline.  It uses already-retargeted candidate cases to
measure how much the final Wuji2 wrist pose differs from the raw DexGraspNet2 LEAP
root pose.  The resulting mean transform and residual percentiles are consumed by
02_build_rfs_prototype.py for a conservative *pre-retarget* RFS coarse filter.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def safe_slug(text: str) -> str:
    import re
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rotation_angle_rad(R: np.ndarray) -> float:
    x = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(x))


def chordal_rotation_mean(rotations: np.ndarray) -> np.ndarray:
    M = np.asarray(rotations, dtype=np.float64).mean(axis=0)
    U, _S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def percentile_dict(values: np.ndarray, scale: float = 1.0) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1) * float(scale)
    qs = [0, 5, 50, 90, 95, 99, 100]
    p = np.percentile(values, qs)
    return {
        "min": float(p[0]),
        "p05": float(p[1]),
        "p50": float(p[2]),
        "p90": float(p[3]),
        "p95": float(p[4]),
        "p99": float(p[5]),
        "max": float(p[6]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def discover_cases(cycle_root: Path) -> list[Path]:
    cases = sorted(
        cycle_root.glob("scratch/final_planning/rank_*/*/case.json"),
        key=lambda p: p.parent.parent.name,
    )
    return [p for p in cases if (p.parent / "07_arm_execution/arm_flange_targets.npz").is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home() / "Projects/DexGraspNet2_Wuji2")
    parser.add_argument("--cycle-root", type=Path, required=True)
    parser.add_argument("--query", default="bottle")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = use all completed cases")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    cycle_root = args.cycle_root.expanduser().resolve()
    query_slug = safe_slug(args.query)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else cycle_root / "rfs_prototype"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dgn_path = cycle_root / "capture/dgn2" / query_slug / "official_leap_1024_target_ranked.npz"
    if not dgn_path.is_file():
        raise FileNotFoundError(dgn_path)

    with np.load(dgn_path, allow_pickle=False) as z:
        rotation_world = np.asarray(z["rotation_world"], dtype=np.float64)
        translation_world = np.asarray(z["translation_world"], dtype=np.float64)
        score = np.asarray(z["score"], dtype=np.float64)

    cases = discover_cases(cycle_root)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError(f"No completed retarget/finalize cases found under {cycle_root}")

    bridges: list[np.ndarray] = []
    flange_from_wrist_all: list[np.ndarray] = []
    rows: list[dict] = []

    for case_path in cases:
        case = load_json(case_path)
        source_index = int(case["source_candidate_index"])
        if source_index < 0 or source_index >= len(rotation_world):
            raise IndexError(f"{case_path}: source_candidate_index={source_index} out of range")

        T_world_leap = np.eye(4, dtype=np.float64)
        T_world_leap[:3, :3] = rotation_world[source_index]
        T_world_leap[:3, 3] = translation_world[source_index]

        arm_path = case_path.parent / "07_arm_execution/arm_flange_targets.npz"
        with np.load(arm_path, allow_pickle=False) as z:
            names = [str(x) for x in z["waypoint_names"].tolist()]
            wrist = np.asarray(z["world_from_wuji2_wrist"], dtype=np.float64)
            flange_from_wrist = np.asarray(z["flange_from_wuji2_wrist"], dtype=np.float64)
        if "grasp" not in names:
            raise RuntimeError(f"{arm_path}: no grasp waypoint")
        T_world_wuji = wrist[names.index("grasp")]
        T_leap_wuji = np.linalg.inv(T_world_leap) @ T_world_wuji

        bridges.append(T_leap_wuji)
        flange_from_wrist_all.append(flange_from_wrist)
        rows.append(
            {
                "case_id": str(case.get("case_id", case_path.parent.name)),
                "source_candidate_index": source_index,
                "official_score": float(case.get("official_score", score[source_index])),
                "T_leap_from_wuji2_wrist": T_leap_wuji.tolist(),
            }
        )

    bridges_a = np.stack(bridges)
    flange_a = np.stack(flange_from_wrist_all)

    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, :3] = chordal_rotation_mean(bridges_a[:, :3, :3])
    T_mean[:3, 3] = bridges_a[:, :3, 3].mean(axis=0)

    trans_abs = np.linalg.norm(bridges_a[:, :3, 3], axis=1)
    rot_abs = np.asarray([rotation_angle_rad(T[:3, :3]) for T in bridges_a])

    trans_res = []
    rot_res = []
    residual_rows = []
    for meta, T in zip(rows, bridges_a):
        D = np.linalg.inv(T_mean) @ T
        dt = float(np.linalg.norm(D[:3, 3]))
        dr = rotation_angle_rad(D[:3, :3])
        trans_res.append(dt)
        rot_res.append(dr)
        residual_rows.append(
            {
                **meta,
                "translation_residual_m": dt,
                "rotation_residual_deg": float(np.rad2deg(dr)),
            }
        )

    trans_res_a = np.asarray(trans_res)
    rot_res_a = np.asarray(rot_res)

    flange_reference = flange_a[0]
    flange_max_abs_delta = float(np.max(np.abs(flange_a - flange_reference[None, ...])))

    p99_t = float(np.percentile(trans_res_a, 99))
    p99_r = float(np.percentile(rot_res_a, 99))
    # Conservative first-prototype inflation: p99 residual plus a modest numerical/model buffer.
    recommended_position_inflation_m = float(math.ceil((p99_t + 0.010) / 0.005) * 0.005)
    recommended_orientation_inflation_deg = float(
        math.ceil((math.degrees(p99_r) + 2.0) / 1.0) * 1.0
    )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "calibrate approximate pre-retarget LEAP-root -> final Wuji2-wrist bridge",
        "project_root": str(project_root),
        "cycle_root": str(cycle_root),
        "query": args.query,
        "sample_count": len(bridges_a),
        "T_leap_from_wuji2_wrist_mean": T_mean.tolist(),
        "flange_from_wuji2_wrist": flange_reference.tolist(),
        "flange_transform_max_abs_delta_across_cases": flange_max_abs_delta,
        "absolute_bridge_translation_m": percentile_dict(trans_abs),
        "absolute_bridge_rotation_deg": percentile_dict(rot_abs, scale=180.0 / math.pi),
        "residual_about_mean_translation_m": percentile_dict(trans_res_a),
        "residual_about_mean_rotation_deg": percentile_dict(rot_res_a, scale=180.0 / math.pi),
        "recommended_first_prototype": {
            "position_inflation_m": recommended_position_inflation_m,
            "orientation_inflation_deg": recommended_orientation_inflation_deg,
            "policy": "coarse filter only; do not replace post-retarget exact COVER IK",
        },
        "rows": residual_rows,
    }

    json_path = output_dir / "bridge_calibration.json"
    npz_path = output_dir / "bridge_calibration.npz"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        npz_path,
        T_leap_from_wuji2_wrist_mean=T_mean,
        flange_from_wuji2_wrist=flange_reference,
        translation_residual_m=trans_res_a,
        rotation_residual_rad=rot_res_a,
        source_candidate_index=np.asarray([int(x["source_candidate_index"]) for x in rows], dtype=np.int64),
        official_score=np.asarray([float(x["official_score"]) for x in rows], dtype=np.float64),
        recommended_position_inflation_m=np.asarray(recommended_position_inflation_m, dtype=np.float64),
        recommended_orientation_inflation_deg=np.asarray(recommended_orientation_inflation_deg, dtype=np.float64),
    )

    print("=" * 78)
    print("LEAP -> Wuji2 bridge calibration")
    print(f"cases             : {len(bridges_a)}")
    print(
        "translation residual: "
        f"p50={1000*np.percentile(trans_res_a,50):.1f} mm | "
        f"p95={1000*np.percentile(trans_res_a,95):.1f} mm | "
        f"p99={1000*np.percentile(trans_res_a,99):.1f} mm | "
        f"max={1000*np.max(trans_res_a):.1f} mm"
    )
    print(
        "rotation residual   : "
        f"p50={np.degrees(np.percentile(rot_res_a,50)):.2f} deg | "
        f"p95={np.degrees(np.percentile(rot_res_a,95)):.2f} deg | "
        f"p99={np.degrees(np.percentile(rot_res_a,99)):.2f} deg | "
        f"max={np.degrees(np.max(rot_res_a)):.2f} deg"
    )
    print(f"recommended position inflation: {1000*recommended_position_inflation_m:.0f} mm")
    print(f"recommended orientation buffer : {recommended_orientation_inflation_deg:.0f} deg")
    print(f"JSON: {json_path}")
    print(f"NPZ : {npz_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
