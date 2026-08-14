#!/usr/bin/env python3
"""Freeze the selected live-camera ashtray grasp as a retargeting case.

This adapter writes only case data.  All LEAP -> Wuji2 algorithms remain in
``06_leap_to_wuji2_final_pipeline/02_scripts`` and are reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout"
PIPELINE_ROOT = PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline"
DATASET_ROOT = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/wuji2_test60_10upright_10view_v1"
)
TARGET = "ashtray"
CASE_ID = "live_scene0000_ashtray_official_best"
ROOT_JOINT_NAMES = np.asarray(
    ["x_joint", "y_joint", "z_joint", "x_rotation_joint", "y_rotation_joint", "z_rotation_joint"]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--case-id", default=CASE_ID)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Directory containing network input, prediction and collision-filter outputs.",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=None,
        help="Capture directory recorded in the case audit metadata.",
    )
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        default=None,
        help="Exact post-settle manifest used for capture and collision filtering.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help="Use this collision-valid official candidate instead of the stored score-only selection.",
    )
    args = parser.parse_args()

    live_root = (
        args.input_root.resolve()
        if args.input_root is not None
        else LAYOUT_ROOT / "captures/latest/dgn2" / args.target
    )
    capture_root = (
        args.capture_root.resolve()
        if args.capture_root is not None
        else LAYOUT_ROOT / "captures/latest"
    )
    collision_path = live_root / "official_leap_target_collision_filtered.npz"
    prediction_path = live_root / "official_leap_1024_target_ranked.npz"
    network_input_path = live_root / "network_input.npz"
    source_scene_path = (
        args.scene_manifest.resolve()
        if args.scene_manifest is not None
        else DATASET_ROOT / "scenes/scene_0000/scene_manifest.json"
    )
    source_stage03 = DATASET_ROOT / (
        "grasp_label_stages/03_reference_points_and_surface_graspness/scene_0000"
    )
    case_root = PIPELINE_ROOT / "01_cases" / args.case_id
    for path in (collision_path, prediction_path, network_input_path, source_scene_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if case_root.exists():
        if not args.replace:
            raise FileExistsError(f"case exists; pass --replace to refresh generated data: {case_root}")
        shutil.rmtree(case_root)

    for relative in (
        "00_config", "01_input", "02_retargeting", "03_root_alignment",
        "04_squeeze", "05_visualization", "06_isaacsim",
    ):
        (case_root / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PIPELINE_ROOT / "00_shared/config/wuji2_native_width_mapper.json",
        case_root / "00_config/wuji2_native_width_mapper.json",
    )
    frozen_input = case_root / "01_input/live_top_camera_network_input.npz"
    shutil.copy2(network_input_path, frozen_input)

    with np.load(collision_path, allow_pickle=False) as archive:
        collision = {key: archive[key] for key in archive.files}
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    selected = (
        int(args.candidate_index)
        if args.candidate_index is not None
        else int(np.asarray(collision["selected_candidate_index"]).item())
    )
    target_id = int(np.asarray(prediction["target_segmentation_id"]).item())
    target_indices = np.asarray(collision["target_candidate_index"], dtype=np.int64)
    selected_matches = np.flatnonzero(target_indices == selected)
    if selected_matches.size != 1:
        raise RuntimeError(f"candidate {selected} is not a target-seed candidate")
    local = int(selected_matches[0])
    if not bool(collision["collision_valid"][local]):
        raise RuntimeError("selected collision record is internally inconsistent")

    selected_waypoint_pose_source = np.asarray(
        collision["waypoint_pose_source"][local:local + 1], dtype=np.float32
    )
    selected_waypoint_qpos = np.asarray(
        collision["waypoint_joint_positions"][local:local + 1], dtype=np.float32
    )
    source_from_world = np.asarray(collision["source_from_world"], dtype=np.float64)
    seed_world = np.asarray(prediction["seed_point_world"][selected], dtype=np.float64)
    seed_source = seed_world @ source_from_world[:3, :3].T + source_from_world[:3, 3]
    leap_job = case_root / "01_input/leap_official_waypoints.npz"
    np.savez_compressed(
        leap_job,
        waypoint_pose_world=selected_waypoint_pose_source,
        waypoint_joint_positions=selected_waypoint_qpos,
        waypoint_names=np.asarray(["pregrasp", "cover", "grasp", "squeeze", "lift"]),
        waypoint_steps=np.asarray([40, 20, 20, 60], dtype=np.int64),
        finger_joint_names=np.asarray(prediction["leap_joint_order"]),
        root_joint_names=ROOT_JOINT_NAMES,
        pregrasp_valid=np.asarray([True]),
        scene_penetration=np.asarray([collision["scene_penetration_m"][local]], dtype=np.float32),
        table_penetration=np.asarray([collision["table_penetration_m"][local]], dtype=np.float32),
        source_candidate_index=np.asarray([selected], dtype=np.int64),
        score=np.asarray([prediction["score"][selected]], dtype=np.float32),
        graspness=np.asarray([prediction["graspness"][selected]], dtype=np.float32),
        log_prob=np.asarray([prediction["log_prob"][selected]], dtype=np.float32),
        seed_point_world=seed_source[None].astype(np.float32),
        target_segmentation_id=np.asarray([target_id], dtype=np.int64),
    )

    source_scene = json.loads(source_scene_path.read_text(encoding="utf-8"))
    objects = []
    for record in source_scene["objects"]:
        seg_id = int(record["segmentation_id"])
        pool_index = int(record["object_pool_index"])
        surface_matches = sorted(
            source_stage03.glob(f"object_{seg_id:03d}_*_surface_graspness.npz")
        )
        if len(surface_matches) != 1:
            raise RuntimeError(f"segmentation {seg_id}: surface files={surface_matches}")
        with np.load(surface_matches[0], allow_pickle=False) as archive:
            surface_local = np.asarray(
                archive["surface_points_centered_object"], dtype=np.float32
            )
        surface_path = case_root / "01_input" / f"object_{seg_id:03d}_surface_points.npy"
        np.save(surface_path, surface_local)
        simulation_usd = (
            DATASET_ROOT / f"usd_cache/object_{pool_index:03d}/flat/"
            f"object_{pool_index:03d}_editable.usd"
        )
        if not simulation_usd.is_file():
            raise FileNotFoundError(simulation_usd)
        objects.append(
            {
                "segmentation_id": seg_id,
                "object_pool_index": pool_index,
                "code": str(record["object_code"]),
                "pose_world_object": record["T_world_centered_object"],
                "surface_points": str(surface_path.resolve()),
                "visual_mesh": str(Path(record["asset"]["centered_combined_obj"]).resolve()),
                "simulation_usd": str(simulation_usd.resolve()),
            }
        )
    scene_manifest = {
        "schema_version": 1,
        "experiment": "post-settle live RGB-D ashtray selection and full arm execution",
        "scene_index": 0,
        "view_index": -1,
        "source_scene_manifest": str(source_scene_path.resolve()),
        "source_network_input": str(network_input_path.resolve()),
        "frozen_view_input": str(frozen_input.resolve()),
        "coordinate_contract": {
            "world": "SourceZone frame: tabletop z=0, +z upward",
            "live_layout_bridge": "captured in calibrated layout world then rigidly returned to SourceZone frame",
            "pose_source": "exact post-physics settled scene manifest",
        },
        "table": source_scene["table"],
        "camera": {"source": "live TopD435iVirtual single RGB-D capture"},
        "objects": objects,
    }
    manifest_path = case_root / "01_input/scene_0000_manifest.json"
    manifest_path.write_text(
        json.dumps(scene_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target_matches = [item for item in objects if item["segmentation_id"] == target_id]
    if len(target_matches) != 1:
        raise RuntimeError(f"target segmentation {target_id} is ambiguous")
    case = {
        "schema_version": 1,
        "case_id": args.case_id,
        "scene_id": "scene_0000",
        "view_id": "live_top_camera",
        "target_segmentation_id": target_id,
        "target_object_code": target_matches[0]["code"],
        "source_candidate_index": selected,
        "selection_policy": (
            "GroundedSAM target seed then official PREGRASP collision then arm-reachability shortlist"
            if args.candidate_index is not None
            else "GroundedSAM target seed then official score then official PREGRASP collision"
        ),
        "source_hand": "LEAP Hand",
        "target_hand": "Wuji Hand 2 Beta1 right",
        "pipeline_status": "official_leap_waypoints_ready",
        "physics_status": "not_tested",
        "point_cloud_shape": [1, 40000, 3],
        "live_capture_root": str(capture_root),
        "official_rank0_pregrasp_valid": True,
    }
    (case_root / "case.json").write_text(
        json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (case_root / "README.md").write_text(
        f"# Live scene 0000 ashtray\n\n"
        f"- Target: segmentation `{target_id}`, `{target_matches[0]['code']}`\n"
        f"- Official candidate: `{selected}`\n"
        f"- Input: one calibrated live RGB-D frame, 40000 full-scene points\n"
        f"- Selection: GroundedSAM membership -> official score -> official PREGRASP collision\n"
        f"- Isaac Sim: run `06_isaacsim/01_import.py`, then `02_execute.py`\n",
        encoding="utf-8",
    )
    print(f"[PASS] selected candidate={selected}; target={target_id}")
    print(f"[PASS] LEAP waypoint job={leap_job}")
    print(f"[OK] case={case_root}")


if __name__ == "__main__":
    main()
