#!/usr/bin/env python3
"""All-candidate GPU coarse prefilter before LEAP->Wuji2 retargeting.

Level 1 uses only DGN2 candidate GRASP root poses and cuRobo right-arm IK.
Level 2 checks the selected right-arm solution against measured-baseline
self-collision plus arm-only non-target observed ESDF.  It intentionally does
not use unretargeted Wuji2 finger geometry as a final hand collision decision.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
ASSEMBLY_SPEC = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/config/assembly_spec.json"
)
sys.path.insert(0, str(CONTROL_ROOT))

from core.bridge import CuroboWorkerClient  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_xyz_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def transform_from_xyz_rpy(xyz: list[float], rpy: list[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_xyz_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def nvidia_memory_mib() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        values = [int(x.strip()) for x in out.splitlines() if x.strip()]
        return max(values) if values else None
    except Exception:
        return None


def load_targets(
    project_root: Path,
    prediction: Path,
    pregrasp_offset_m: float,
) -> tuple[list[dict], np.ndarray, np.ndarray, int]:
    with np.load(prediction, allow_pickle=False) as z:
        order = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)
        score = np.asarray(z["score"], dtype=np.float64)
        graspness = np.asarray(z["graspness"], dtype=np.float64)
        log_prob = np.asarray(z["log_prob"], dtype=np.float64)
        rotation = np.asarray(z["rotation_world"], dtype=np.float64)
        translation = np.asarray(z["translation_world"], dtype=np.float64)
        total = int(len(score))
    assembly = load_json(ASSEMBLY_SPEC)
    flange_from_wrist = transform_from_xyz_rpy(
        assembly["mount_transform_parent_to_child"]["xyz_m"],
        assembly["mount_transform_parent_to_child"]["rpy_rad"],
    )
    wrist_from_flange = np.linalg.inv(flange_from_wrist)
    T_base_from_world = np.linalg.inv(world_from_base(project_root))
    rows = []
    targets = []
    pregrasp_targets = []
    for rank, idx in enumerate(order):
        i = int(idx)
        T_world_wrist_approx = np.eye(4, dtype=np.float64)
        T_world_wrist_approx[:3, :3] = rotation[i]
        T_world_wrist_approx[:3, 3] = translation[i]
        T_world_wrist_pre = T_world_wrist_approx.copy()
        T_world_wrist_pre[:3, 3] += rotation[i][:, 2] * float(pregrasp_offset_m)
        targets.append(T_base_from_world @ T_world_wrist_approx @ wrist_from_flange)
        pregrasp_targets.append(T_base_from_world @ T_world_wrist_pre @ wrist_from_flange)
        rows.append({
            "target_rank": int(rank),
            "candidate_index": i,
            "score": float(score[i]),
            "graspness": float(graspness[i]),
            "log_prob": float(log_prob[i]),
        })
    return rows, np.stack(targets), np.stack(pregrasp_targets), total


def _selected_q(report: dict, index: int) -> np.ndarray | None:
    selected = report.get("selected") or []
    if index < 0 or index >= len(selected) or selected[index] is None:
        return None
    return np.asarray(selected[index]["q_rad"], dtype=np.float64)


def _check_path_survivors(
    *,
    client: CuroboWorkerClient,
    candidate_indices: list[int],
    q_start_by_index: dict[int, np.ndarray],
    q_goal_by_index: dict[int, np.ndarray],
    measured: dict,
    T_world_base: np.ndarray,
    phase: str,
    label: str,
    path_batch_size: int,
    max_step_rad: float,
) -> tuple[list[int], float]:
    if not candidate_indices:
        return [], 0.0
    survivors: list[int] = []
    started = time.perf_counter()
    total_batches = int(math.ceil(len(candidate_indices) / max(1, path_batch_size)))
    for batch_i, start in enumerate(range(0, len(candidate_indices), max(1, path_batch_size)), start=1):
        batch = candidate_indices[start:start + max(1, path_batch_size)]
        elapsed = time.perf_counter() - started
        rate = start / elapsed if elapsed > 0 and start > 0 else 0.0
        eta = (len(candidate_indices) - start) / rate if rate > 0 else None
        eta_text = "unknown" if eta is None else f"{eta:.1f}s"
        print(
            f"[APPROACH {batch_i}/{total_batches}] {label} candidates={len(batch)} "
            f"elapsed={elapsed:.1f}s ETA={eta_text}",
            flush=True,
        )
        for idx in batch:
            result = client.check_joint_path(
                np.stack([q_start_by_index[idx], q_goal_by_index[idx]]),
                measured,
                joint_positions_by_node=[measured, measured],
                T_world_base=T_world_base,
                phases=[phase],
                margin_m=0.0,
                path_max_joint_step_rad=max_step_rad,
                check_observed_map=True,
            )
            if bool(result.get("path_pass")):
                survivors.append(idx)
    return survivors, time.perf_counter() - started


def run_strict_ordered_prefilter(
    *,
    client: CuroboWorkerClient,
    candidates: list[dict],
    grasp_targets: np.ndarray,
    pregrasp_targets: np.ndarray,
    q_current: np.ndarray,
    measured: dict,
    T_world_base: np.ndarray,
    path_batch_size: int = 256,
    path_max_joint_step_rad: float = math.radians(3.0),
    progress: bool = True,
) -> dict:
    """Run the cheap-to-expensive coarse funnel without forwarding failed rows.

    Order:
    GRASP IK -> GRASP thresholds/joint-limits -> GRASP scene ESDF ->
    PREGRASP IK -> PREGRASP thresholds/joint-limits -> PREGRASP scene ESDF ->
    q_current->PREGRASP path -> PREGRASP->GRASP path.
    """
    total = len(candidates)
    if progress:
        print(f"[PREFILTER] total_target_candidates = {total}", flush=True)

    grasp_started = time.perf_counter()
    grasp = client.coarse_prefilter(
        grasp_targets,
        q_current,
        joint_positions_by_name=measured,
        T_world_base=T_world_base,
        phase="pregrasp",
        margin_m=0.0,
        arm_link_prefixes=["arm_base_link", "arm_r_link"],
    )
    grasp_wall_s = time.perf_counter() - grasp_started
    grasp_raw = set(int(x) for x in grasp.get("raw_reachable_indices", []))
    grasp_threshold = set(int(x) for x in grasp.get("threshold_accepted_indices", []))
    grasp_scene = [int(x) for x in grasp.get("coarse_collision_pass_indices", [])]
    if progress:
        print(
            "[PREFILTER] GRASP: "
            f"raw_ik_reachable={len(grasp_raw)} "
            f"threshold_accepted={len(grasp_threshold)} "
            f"scene_esdf_pass={len(grasp_scene)} "
            f"ik_time_s={float(grasp.get('solve_time_s', 0.0)):.3f} "
            f"collision_time_s={max(0.0, grasp_wall_s - float(grasp.get('solve_time_s', 0.0))):.3f}",
            flush=True,
        )

    pregrasp_started = time.perf_counter()
    pregrasp_subset = pregrasp_targets[grasp_scene] if grasp_scene else np.empty((0, 4, 4), dtype=np.float64)
    if len(pregrasp_subset):
        pregrasp = client.coarse_prefilter(
            pregrasp_subset,
            q_current,
            joint_positions_by_name=measured,
            T_world_base=T_world_base,
            phase="pregrasp",
            margin_m=0.0,
            arm_link_prefixes=["arm_base_link", "arm_r_link"],
        )
    else:
        pregrasp = {
            "raw_reachable_indices": [],
            "threshold_accepted_indices": [],
            "coarse_collision_pass_indices": [],
            "selected": [],
            "solve_time_s": 0.0,
        }
    pregrasp_wall_s = time.perf_counter() - pregrasp_started
    pre_raw_local = set(int(x) for x in pregrasp.get("raw_reachable_indices", []))
    pre_threshold_local = set(int(x) for x in pregrasp.get("threshold_accepted_indices", []))
    pre_scene_local = [int(x) for x in pregrasp.get("coarse_collision_pass_indices", [])]
    pre_scene = [grasp_scene[i] for i in pre_scene_local]
    if progress:
        print(
            "[PREFILTER] PREGRASP: "
            f"raw_ik_reachable={len(pre_raw_local)} "
            f"threshold_accepted={len(pre_threshold_local)} "
            f"scene_esdf_pass={len(pre_scene)} "
            f"ik_time_s={float(pregrasp.get('solve_time_s', 0.0)):.3f} "
            f"collision_time_s={max(0.0, pregrasp_wall_s - float(pregrasp.get('solve_time_s', 0.0))):.3f}",
            flush=True,
        )

    q_grasp = {
        idx: _selected_q(grasp, idx)
        for idx in grasp_scene
    }
    q_pregrasp = {
        grasp_scene[local]: _selected_q(pregrasp, local)
        for local in pre_scene_local
    }
    q_grasp = {idx: q for idx, q in q_grasp.items() if q is not None}
    q_pregrasp = {idx: q for idx, q in q_pregrasp.items() if q is not None}
    path_input_1 = [idx for idx in pre_scene if idx in q_pregrasp]
    q_current_by_index = {idx: np.asarray(q_current, dtype=np.float64) for idx in path_input_1}
    path1, path1_time = _check_path_survivors(
        client=client,
        candidate_indices=path_input_1,
        q_start_by_index=q_current_by_index,
        q_goal_by_index=q_pregrasp,
        measured=measured,
        T_world_base=T_world_base,
        phase="pregrasp",
        label="q_current_to_pregrasp",
        path_batch_size=path_batch_size,
        max_step_rad=path_max_joint_step_rad,
    )
    path_input_2 = [idx for idx in path1 if idx in q_pregrasp and idx in q_grasp]
    path2, path2_time = _check_path_survivors(
        client=client,
        candidate_indices=path_input_2,
        q_start_by_index=q_pregrasp,
        q_goal_by_index=q_grasp,
        measured=measured,
        T_world_base=T_world_base,
        phase="approach",
        label="pregrasp_to_grasp",
        path_batch_size=path_batch_size,
        max_step_rad=path_max_joint_step_rad,
    )
    if progress:
        print(
            "[PREFILTER] APPROACH PATH: "
            f"q_current_to_pregrasp_input={len(path_input_1)} "
            f"q_current_to_pregrasp_pass={len(path1)} "
            f"pregrasp_to_grasp_input={len(path_input_2)} "
            f"pregrasp_to_grasp_pass={len(path2)} "
            f"path_collision_time_s={path1_time + path2_time:.3f}",
            flush=True,
        )
        print(f"[PREFILTER] final_coarse_survivors = {len(path2)}", flush=True)

    survivors = [
        {
            **candidates[idx],
            "local_target_index": int(idx),
            "grasp_raw_success_count": int(grasp["raw_success_per_target"][idx]),
            "grasp_threshold_accepted_count": int(grasp["threshold_accepted_per_target"][idx]),
        }
        for idx in path2
    ]
    return {
        "SELF_COLLISION_POLICY": "REPORT_ONLY_UNRESOLVED",
        "candidate_count": total,
        "survivor_indices": [int(x) for x in path2],
        "survivors": survivors,
        "grasp": {
            "raw_ik_reachable_indices": sorted(grasp_raw),
            "threshold_accepted_indices": sorted(grasp_threshold),
            "scene_esdf_pass_indices": [int(x) for x in grasp_scene],
            "raw_ik_reachable": len(grasp_raw),
            "threshold_accepted": len(grasp_threshold),
            "scene_esdf_pass": len(grasp_scene),
            "ik_time_s": float(grasp.get("solve_time_s", 0.0)),
            "collision_time_s": max(0.0, grasp_wall_s - float(grasp.get("solve_time_s", 0.0))),
        },
        "pregrasp": {
            "input_indices": [int(x) for x in grasp_scene],
            "raw_ik_reachable": len(pre_raw_local),
            "threshold_accepted": len(pre_threshold_local),
            "scene_esdf_pass": len(pre_scene),
            "scene_esdf_pass_indices": [int(x) for x in pre_scene],
            "ik_time_s": float(pregrasp.get("solve_time_s", 0.0)),
            "collision_time_s": max(0.0, pregrasp_wall_s - float(pregrasp.get("solve_time_s", 0.0))),
        },
        "approach_path": {
            "q_current_to_pregrasp_input": len(path_input_1),
            "q_current_to_pregrasp_pass": len(path1),
            "pregrasp_to_grasp_input": len(path_input_2),
            "pregrasp_to_grasp_pass": len(path2),
            "q_current_to_pregrasp_pass_indices": [int(x) for x in path1],
            "pregrasp_to_grasp_pass_indices": [int(x) for x in path2],
            "path_collision_time_s": path1_time + path2_time,
        },
        "raw_reports": {
            "grasp": grasp,
            "pregrasp": pregrasp,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-batch-size", type=int, default=512)
    parser.add_argument("--pregrasp-offset-m", type=float, default=0.10)
    parser.add_argument("--skip-map", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    prediction = args.prediction.expanduser().resolve()
    capture_root = args.capture_root.expanduser().resolve()
    state = load_json(args.robot_state.expanduser().resolve())
    measured = {str(k): float(v) for k, v in state["joint_positions_by_name"].items()}
    q_current = np.asarray(state["right_arm_q_current_rad"], dtype=np.float64)
    T_world_base = world_from_base(project_root)
    candidates, targets, pregrasp_targets, total_proposals = load_targets(
        project_root, prediction, args.pregrasp_offset_m
    )

    mem_before = nvidia_memory_mib()
    wall_start = time.perf_counter()
    worker_start_count = 1
    map_build_count = 0
    with CuroboWorkerClient(
        project_root,
        seeds=48,
        batch_size=max(1, int(args.gpu_batch_size)),
    ) as client:
        if not args.skip_map:
            map_start = time.perf_counter()
            map_report = client.build_map(
                capture_root / "depth_m.npy",
                capture_root / "intrinsics.npy",
                capture_root / "T_world_camera.npy",
                args.mask.expanduser().resolve(),
            )
            map_wall_s = time.perf_counter() - map_start
            map_build_count = 1
        else:
            map_report = None
            map_wall_s = 0.0
        ordered = run_strict_ordered_prefilter(
            client=client,
            candidates=candidates,
            grasp_targets=targets,
            pregrasp_targets=pregrasp_targets,
            q_current=q_current,
            measured=measured,
            T_world_base=T_world_base,
            path_batch_size=max(1, int(args.gpu_batch_size)),
            path_max_joint_step_rad=math.radians(3.0),
            progress=True,
        )
    total_wall_s = time.perf_counter() - wall_start
    mem_after = nvidia_memory_mib()
    mem_peak = None
    if mem_before is not None or mem_after is not None:
        mem_peak = max(x for x in (mem_before, mem_after) if x is not None)
    grasp = ordered["grasp"]
    pregrasp = ordered["pregrasp"]
    approach = ordered["approach_path"]
    survivors = ordered["survivors"]
    out = {
        "schema_version": 1,
        "scope": "all-candidate GPU coarse prefilter; no retarget/full-route/physical",
        "execution_order": [
            "GRASP GPU IK",
            "GRASP threshold / joint-limit filtering",
            "GRASP single-state observed-scene ESDF",
            "PREGRASP GPU IK on GRASP scene survivors only",
            "PREGRASP threshold / joint-limit filtering",
            "PREGRASP single-state observed-scene ESDF",
            "q_current -> PREGRASP continuous observed-scene path",
            "PREGRASP -> GRASP continuous observed-scene path",
        ],
        "approximation_contract": (
            "DGN2 GRASP root pose is treated as approximate Wuji2 wrist pose; "
            "right flange target = world_from_wrist_approx @ inverse(flange_from_wuji2_wrist); "
            "approximate PREGRASP offsets wrist along DGN2 local +Z"
        ),
        "SELF_COLLISION_POLICY": "REPORT_ONLY_UNRESOLVED",
        "total_proposals": total_proposals,
        "target_proposals": len(candidates),
        "gpu_ik_batch_size": int(args.gpu_batch_size),
        "batch_count": int(math.ceil(len(candidates) / max(1, int(args.gpu_batch_size)))),
        "worker_start_count": worker_start_count,
        "map_build_count": map_build_count,
        "grasp_raw_ik_reachable": grasp["raw_ik_reachable"],
        "threshold_accepted": grasp["threshold_accepted"],
        "arm_coarse_collision_survivors": len(ordered["survivor_indices"]),
        "survivors_without_self_collision": len(ordered["survivor_indices"]),
        "self_collision_pass": None,
        "scene_collision_pass": grasp["scene_esdf_pass"],
        "pregrasp_offset_m": float(args.pregrasp_offset_m),
        "pregrasp_raw_reachable": pregrasp["raw_ik_reachable"],
        "both_pregrasp_grasp_threshold_accepted": pregrasp["threshold_accepted"],
        "pregrasp_scene_pass": pregrasp["scene_esdf_pass"],
        "approach_path_survivors_without_self_collision": len(ordered["survivor_indices"]),
        "ik_time_s": grasp["ik_time_s"] + pregrasp["ik_time_s"],
        "collision_time_s": grasp["collision_time_s"] + pregrasp["collision_time_s"],
        "approach_time_s": approach["path_collision_time_s"],
        "map_time_s": map_wall_s,
        "total_wall_time_s": total_wall_s,
        "candidates_per_s": len(candidates) / max(total_wall_s, 1.0e-9),
        "peak_vram_mib": mem_peak,
        "survivor_ratio": len(ordered["survivor_indices"]) / max(len(candidates), 1),
        "raw_reachable_ratio": grasp["raw_ik_reachable"] / max(len(candidates), 1),
        "threshold_accepted_ratio": grasp["threshold_accepted"] / max(len(candidates), 1),
        "survivors": survivors,
        "map": map_report,
        "strict_funnel": {
            "grasp": grasp,
            "pregrasp": pregrasp,
            "approach_path": approach,
        },
        "approach_raw_report": {
            "pregrasp_scene_pass_indices": pregrasp["scene_esdf_pass_indices"],
            "approach_path_pass_indices": approach["pregrasp_to_grasp_pass_indices"],
            "survivor_indices": ordered["survivor_indices"],
        },
        "raw_report": {
            "grasp": ordered["raw_reports"]["grasp"],
            "pregrasp": ordered["raw_reports"]["pregrasp"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "total_proposals": out["total_proposals"],
        "target_proposals": out["target_proposals"],
        "gpu_ik_batch_size": out["gpu_ik_batch_size"],
        "batch_count": out["batch_count"],
        "grasp_raw_ik_reachable": out["grasp_raw_ik_reachable"],
        "threshold_accepted": out["threshold_accepted"],
        "arm_coarse_collision_survivors": out["arm_coarse_collision_survivors"],
        "survivors_without_self_collision": out["survivors_without_self_collision"],
        "self_collision_pass": out["self_collision_pass"],
        "scene_collision_pass": out["scene_collision_pass"],
        "pregrasp_raw_reachable": out["pregrasp_raw_reachable"],
        "both_pregrasp_grasp_threshold_accepted": out["both_pregrasp_grasp_threshold_accepted"],
        "pregrasp_scene_pass": out["pregrasp_scene_pass"],
        "approach_path_survivors_without_self_collision": out["approach_path_survivors_without_self_collision"],
        "ik_time_s": out["ik_time_s"],
        "collision_time_s": out["collision_time_s"],
        "approach_time_s": out["approach_time_s"],
        "total_wall_time_s": out["total_wall_time_s"],
        "candidates_per_s": out["candidates_per_s"],
        "peak_vram_mib": out["peak_vram_mib"],
        "survivor_ratio": out["survivor_ratio"],
        "worker_start_count": worker_start_count,
        "map_build_count": map_build_count,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
