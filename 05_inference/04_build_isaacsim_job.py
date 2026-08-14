#!/usr/bin/env python3
"""Build one clean five-stage Isaac Sim job from filtered network candidates.

This production baseline intentionally exposes one squeeze policy only:
DexGraspNet2-style 30 mm per-fingertip surface-normal IK with keep_z=True.
It does not contain the historical force-delta/contact-controller experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402
from wuji2_dgn2.collision import (  # noqa: E402
    load_wuji2_module,
    shift_fingertips_like_official_width_mapper,
    tiger_mouth_pregrasp_pose,
)
from wuji2_dgn2.project import output_path  # noqa: E402


DEFAULT_INPUT = SCRIPT_DIR / "outputs/scene_0036_view_0000_collision_filtered.npz"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "02_training_dataset/config/wuji2_train60_100seminal_256view_v1.json"
)
DEFAULT_OUTPUT = output_path("isaac_jobs") / "scene_0036_view_0000_best_waypoints.npz"
WAYPOINT_NAMES = ("pregrasp", "cover", "grasp", "squeeze", "lift")
WAYPOINT_STEPS = (40, 20, 20, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Rank inside the already collision-filtered file; zero is best.",
    )
    parser.add_argument("--squeeze-width-m", type=float, default=0.03)
    parser.add_argument("--approach-retreat-m", type=float, default=0.10)
    parser.add_argument("--lift-height-m", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.rank < 0:
        raise ValueError("--rank must be non-negative")
    if (
        args.squeeze_width_m <= 0.0
        or args.approach_retreat_m <= 0.0
        or args.lift_height_m <= 0.0
    ):
        raise ValueError("Approach, squeeze and lift distances must be positive")
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    allowed = output_path("isaac_jobs")
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(f"Isaac jobs must stay under {allowed}") from exc

    with np.load(input_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    required = {
        "T_world_r_base_link", "pregrasp_T_world_r_base_link",
        "qpos", "pregrasp_qpos", "joint_order", "score",
        "graspness", "log_prob", "target_segmentation_id",
        "source_candidate_index", "scene_index", "view_index",
    }
    missing = sorted(required.difference(source))
    if missing:
        raise KeyError(f"Filtered prediction is missing: {missing}")
    candidate_count = int(source["qpos"].shape[0])
    if args.rank >= candidate_count:
        raise IndexError(f"rank {args.rank} outside [0, {candidate_count})")

    index = args.rank
    joint_order = [str(value) for value in source["joint_order"].tolist()]
    module = load_wuji2_module()
    if joint_order != list(module.RIGHT_HAND_JOINT_ORDER):
        raise RuntimeError("Prediction joint order differs from Wuji2 contract")
    device = torch.device(args.device)
    model = module.Wuji2HandKinematics(
        module.ORIGINAL_HAND_URDF, device=device, dtype=torch.float32
    )
    config = load_config(args.config.resolve())
    q_grasp_tensor = torch.as_tensor(
        source["qpos"][[index]], dtype=torch.float32, device=device
    )
    q_squeeze = shift_fingertips_like_official_width_mapper(
        model=model,
        module=module,
        qpos=q_grasp_tensor,
        label_cfg=config["grasp_label_generation"],
        delta_width_m=float(args.squeeze_width_m),
        keep_z=True,
        direction_mode="surface_normal",
    ).cpu().numpy()[0]

    grasp_pose = np.asarray(source["T_world_r_base_link"][index], np.float32)
    pre_qpos = np.asarray(source["pregrasp_qpos"][index], np.float32)
    grasp_qpos = np.asarray(source["qpos"][index], np.float32)
    grasp_pose_tensor = torch.as_tensor(
        grasp_pose[None], dtype=torch.float32, device=device
    )
    tiger_pre_pose, approach_base, gap_midpoint = tiger_mouth_pregrasp_pose(
        model,
        grasp_pose_tensor,
        q_grasp_tensor,
        float(args.approach_retreat_m),
    )
    pre_pose = tiger_pre_pose.cpu().numpy()[0].astype(np.float32)
    approach_base_np = approach_base.cpu().numpy()[0].astype(np.float32)
    approach_world = (
        grasp_pose[:3, :3] @ approach_base_np
    ).astype(np.float32)
    approach_world /= np.linalg.norm(approach_world)
    gap_midpoint_np = gap_midpoint.cpu().numpy()[0].astype(np.float32)

    filtered_pre_pose = np.asarray(
        source["pregrasp_T_world_r_base_link"][index], np.float32
    )
    if not np.allclose(pre_pose, filtered_pre_pose, atol=1.0e-5):
        raise RuntimeError(
            "Filtered prediction was not checked with the Wuji2 tiger-mouth "
            "PREGRASP. Re-run 02_filter_scene_collisions.py before building a job."
        )
    lift_pose = grasp_pose.copy()
    lift_pose[2, 3] += float(args.lift_height_m)
    poses = np.stack(
        [pre_pose, grasp_pose, grasp_pose, grasp_pose, lift_pose], axis=0
    )[None]
    joints = np.stack(
        [pre_qpos, pre_qpos, grasp_qpos, q_squeeze, q_squeeze], axis=0
    )[None]

    atomic_savez(
        output,
        waypoint_pose_world=poses.astype(np.float32),
        waypoint_joint_positions=joints.astype(np.float32),
        waypoint_names=np.asarray(WAYPOINT_NAMES),
        waypoint_steps=np.asarray(WAYPOINT_STEPS, dtype=np.int64),
        # Current prediction labels retain the legacy r_base_link convention.
        # The official-USD importer performs the audited conversion to the
        # Hand2 Beta1 r_wrist root.  Never infer this frame from array shape.
        hand_root_pose_frame=np.asarray("legacy_r_base_link"),
        joint_order=np.asarray(joint_order),
        target_segmentation_id=np.asarray(
            [source["target_segmentation_id"][index]], dtype=np.int64
        ),
        score=np.asarray([source["score"][index]], dtype=np.float32),
        graspness=np.asarray([source["graspness"][index]], dtype=np.float32),
        log_prob=np.asarray([source["log_prob"][index]], dtype=np.float32),
        source_candidate_index=np.asarray(
            [source["source_candidate_index"][index]], dtype=np.int64
        ),
        scene_index=np.asarray(source["scene_index"], dtype=np.int64),
        view_index=np.asarray(source["view_index"], dtype=np.int64),
        squeeze_width_m=np.asarray(args.squeeze_width_m, dtype=np.float32),
        squeeze_keep_z=np.asarray(True),
        squeeze_direction_mode=np.asarray("surface_normal"),
        approach_policy=np.asarray("palm_to_thumb_index_gap_center"),
        approach_retreat_m=np.asarray(args.approach_retreat_m, dtype=np.float32),
        approach_direction_base=approach_base_np[None],
        approach_direction_world=approach_world[None],
        tiger_mouth_center_base=gap_midpoint_np[None],
        lift_height_m=np.asarray(args.lift_height_m, dtype=np.float32),
    )
    manifest = {
        "schema_version": 1,
        "status": "ready_for_isaacsim_not_yet_physically_validated",
        "source_filtered_prediction": str(input_path),
        "selected_filtered_rank": index,
        "source_candidate_index": int(source["source_candidate_index"][index]),
        "scene_index": int(np.asarray(source["scene_index"]).item()),
        "view_index": int(np.asarray(source["view_index"]).item()),
        "waypoints": list(WAYPOINT_NAMES),
        "hand_runtime_asset": "official Wuji Hand 2 Beta1 wujihand2.usd",
        "hand_root_pose_frame_in_npz": "legacy_r_base_link",
        "hand_root_runtime_frame": "official_r_wrist (converted by importer)",
        "approach": {
            "policy": "palm_to_thumb_index_gap_center",
            "retreat_m": float(args.approach_retreat_m),
            "direction_in_r_base_link": approach_base_np.tolist(),
            "direction_in_world": approach_world.tolist()
        },
        "squeeze": {
            "method": "DexGraspNet2 WidthMapper-style 20-step IK",
            "per_fingertip_m": float(args.squeeze_width_m),
            "direction": "Wuji2 configured local inward normals",
            "keep_z": True
        },
        "lift_world_z_m": float(args.lift_height_m),
        "output_npz": str(output)
    }
    write_json_atomic(output.with_suffix(".json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
