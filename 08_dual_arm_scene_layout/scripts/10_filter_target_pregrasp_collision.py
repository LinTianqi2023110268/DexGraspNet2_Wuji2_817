#!/usr/bin/env python3
"""Apply the official LEAP PREGRASP collision test to target proposals.

The official checker assumes that the table plane is ``z=0``.  The calibrated
robot layout places SourceZone at another world pose, so all grasp poses are
rigidly converted into SourceZone coordinates for checking.  This is a frame
change only; candidates, scores, hand joints and collision thresholds remain
untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout"
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network/official_core"
DATASET_ROOT = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/wuji2_test60_10upright_10view_v1"
)
SCENE_INDEX = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="ashtray")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Collision-check batch size; chunking preserves the official per-grasp test.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Directory containing network input and ranked predictions.",
    )
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        default=None,
        help="Exact post-settle scene manifest; defaults to the legacy scene-0000 manifest.",
    )
    args = parser.parse_args()
    root = (
        args.input_root.resolve()
        if args.input_root is not None
        else LAYOUT_ROOT / "captures/latest/dgn2" / args.target
    )
    prediction_path = root / "official_leap_1024_target_ranked.npz"
    input_path = root / "network_input.npz"
    output_path = root / "official_leap_target_collision_filtered.npz"
    scene_manifest_path = (
        args.scene_manifest.resolve()
        if args.scene_manifest is not None
        else DATASET_ROOT / f"scenes/scene_{SCENE_INDEX:04d}/scene_manifest.json"
    )
    for path in (prediction_path, input_path, scene_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official torchprimitivesdf checker")

    with np.load(prediction_path, allow_pickle=False) as archive:
        pred = {key: archive[key] for key in archive.files}
    with np.load(input_path, allow_pickle=False) as archive:
        source_from_world = np.asarray(archive["source_from_world"], dtype=np.float32)
        world_from_source = np.asarray(archive["world_from_source"], dtype=np.float32)
    target_indices = np.asarray(pred["target_candidate_index"], dtype=np.int64)
    joint_names = [str(value) for value in pred["leap_joint_order"].tolist()]

    r_sw = source_from_world[:3, :3]
    t_sw = source_from_world[:3, 3]
    rotation_source = np.einsum("ij,njk->nik", r_sw, pred["rotation_world"][target_indices])
    translation_source = pred["translation_world"][target_indices] @ r_sw.T + t_sw
    qpos = np.asarray(pred["leap_qpos_rad"][target_indices], dtype=np.float32)

    original_cwd = Path.cwd()
    os.chdir(OFFICIAL_ROOT)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from src.eval.prepare_isaacsim5_job import compose_waypoints
        from src.utils.collision_checker import CollisionChecker
        from src.utils.robot_model import RobotModel
        from src.utils.width_mapper import WidthMapper

        robot = RobotModel(
            "robot_models/urdf/leap_hand_simplified.urdf",
            "robot_models/meta/leap_hand/meta.yaml",
        )
        if list(robot.joint_names) != joint_names:
            raise RuntimeError("LEAP joint order mismatch")
        mapper = WidthMapper(robot, "robot_models/meta/leap_hand/width_mapper_meta.yaml")
        grasps = {
            "rotation": rotation_source.astype(np.float32),
            "translation": translation_source.astype(np.float32),
            **{
                name: qpos[:, index]
                for index, name in enumerate(joint_names)
            },
        }
        waypoint_pose_source, waypoint_qpos, named_pregrasp = compose_waypoints(
            grasps, robot, mapper
        )

        config_path = OFFICIAL_ROOT / "configs/collision_checker/leap_hand/CollisionChecker.yaml"
        checker = CollisionChecker(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            torch.device("cuda:0"),
        )
        tensors = {
            key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
            for key, value in named_pregrasp.items()
        }

        # Reuse the audited object-local 1000 surface samples, but transform
        # them with the exact scene manifest passed to this run.  This is
        # essential after physics settling: the old ``surface_points_world``
        # encode the pre-settle pose and are no longer valid.
        scene = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
        stage03 = DATASET_ROOT / (
            f"grasp_label_stages/03_reference_points_and_surface_graspness/"
            f"scene_{SCENE_INDEX:04d}"
        )
        world_points = []
        surface_files = []
        for obj in scene["objects"]:
            segmentation_id = int(obj["segmentation_id"])
            matches = sorted(stage03.glob(f"object_{segmentation_id:03d}_*_surface_graspness.npz"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"segmentation {segmentation_id}: expected one surface file, got {matches}"
                )
            with np.load(matches[0], allow_pickle=False) as archive:
                local_points = np.asarray(
                    archive["surface_points_centered_object"], dtype=np.float32
                )
            source_from_object = np.asarray(
                obj["T_world_centered_object"], dtype=np.float32
            )
            source_points = (
                local_points @ source_from_object[:3, :3].T
                + source_from_object[:3, 3]
            )
            world_points.append(source_points.astype(np.float32))
            surface_files.append(str(matches[0]))
        points = torch.as_tensor(
            np.concatenate(world_points), dtype=torch.float32, device="cuda:0"
        )
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        scene_chunks = []
        table_chunks = []
        batch_count = (len(target_indices) + args.batch_size - 1) // args.batch_size
        for batch_index, start in enumerate(range(0, len(target_indices), args.batch_size)):
            stop = min(len(target_indices), start + args.batch_size)
            batch = {
                key: value[start:stop] if isinstance(value, torch.Tensor) else value
                for key, value in tensors.items()
            }
            scene_batch, table_batch = checker.check_collision_batch(batch, points)
            scene_chunks.append(scene_batch.detach().cpu().numpy())
            table_chunks.append(table_batch.detach().cpu().numpy())
            print(
                f"[COLLISION] batch {batch_index + 1}/{batch_count}; "
                f"grasps {start}:{stop}",
                flush=True,
            )
        scene_pen = np.concatenate(scene_chunks)
        table_pen = np.concatenate(table_chunks)
    finally:
        os.chdir(original_cwd)

    valid = (scene_pen < 0.0) & (table_pen < 0.0)
    valid_local = np.flatnonzero(valid)
    if len(valid_local) == 0:
        raise RuntimeError("No target PREGRASP passed the official scene/table collision test")
    valid_candidate_indices = target_indices[valid_local]
    scores = np.asarray(pred["score"], dtype=np.float32)
    valid_order = valid_candidate_indices[
        np.argsort(-scores[valid_candidate_indices], kind="stable")
    ]
    best = int(valid_order[0])
    best_local = int(np.flatnonzero(target_indices == best)[0])

    # Convert all official waypoints from SourceZone coordinates back into the
    # calibrated robot-layout world frame.
    r_ws = world_from_source[:3, :3]
    t_ws = world_from_source[:3, 3]
    waypoint_pose_world = waypoint_pose_source.copy()
    waypoint_pose_world[:, :, :3, :3] = np.einsum(
        "ij,nwjk->nwik", r_ws, waypoint_pose_source[:, :, :3, :3]
    )
    waypoint_pose_world[:, :, :3, 3] = (
        waypoint_pose_source[:, :, :3, 3] @ r_ws.T + t_ws
    )
    np.savez_compressed(
        output_path,
        target_candidate_index=target_indices,
        collision_valid=valid,
        scene_penetration_m=scene_pen.astype(np.float32),
        table_penetration_m=table_pen.astype(np.float32),
        valid_candidate_index=valid_candidate_indices,
        valid_score_descending_candidate_index=valid_order,
        selected_candidate_index=np.asarray(best, dtype=np.int64),
        selected_target_local_index=np.asarray(best_local, dtype=np.int64),
        waypoint_pose_world=waypoint_pose_world.astype(np.float32),
        waypoint_pose_source=waypoint_pose_source.astype(np.float32),
        waypoint_joint_positions=waypoint_qpos.astype(np.float32),
        waypoint_names=np.asarray(["pregrasp", "cover", "grasp", "squeeze", "lift"]),
        leap_joint_order=np.asarray(joint_names),
        source_from_world=source_from_world,
        world_from_source=world_from_source,
    )
    report = {
        "schema_version": 1,
        "status": "official_target_pregrasp_collision_filtered",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frame_note": "collision in SourceZone frame where table z=0; outputs restored to layout world",
        "target_proposal_count": int(len(target_indices)),
        "pregrasp_valid_count": int(valid.sum()),
        "selected_candidate_index": best,
        "selected_score": float(scores[best]),
        "selected_scene_penetration_m": float(scene_pen[best_local]),
        "selected_table_penetration_m": float(table_pen[best_local]),
        "surface_point_count": int(sum(len(points_) for points_ in world_points)),
        "surface_files": surface_files,
        "scene_manifest": str(scene_manifest_path),
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] target PREGRASP valid: {valid.sum()}/{len(valid)}")
    print(f"[BEST EXECUTABLE] candidate={best}; score={scores[best]:.6f}")
    print(
        f"[CLEARANCE] scene={scene_pen[best_local]:+.6f} m; "
        f"table={table_pen[best_local]:+.6f} m"
    )
    print(f"[OK] {output_path}")


if __name__ == "__main__":
    main()
