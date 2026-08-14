#!/usr/bin/env python3
"""Convert DexGraspNet2 ``grasps.npz`` into a simulator-neutral waypoint job.

Run this with the ``graspnet2.0`` environment.  It performs LEAP kinematics and
the paper evaluator's pregrasp/grasp/squeeze/lift waypoint construction, but it
does not import or start any simulator.  Isaac Sim 5.0 consumes the resulting
NPZ/JSON pair in a separate process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch3d.transforms import matrix_to_euler_angles


REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.collision_checker import CollisionChecker
from src.utils.robot_model import RobotModel
from src.utils.width_mapper import WidthMapper


ROOT_JOINT_NAMES = (
    "x_joint",
    "y_joint",
    "z_joint",
    "x_rotation_joint",
    "y_rotation_joint",
    "z_rotation_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasps", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-device", default="cuda:0")
    parser.add_argument(
        "--skip-collision-check",
        action="store_true",
        help="Teaching/debug only; official evaluator rejects colliding pregrasps",
    )
    return parser.parse_args()


def compose_waypoints(
    grasps: dict,
    robot_model: RobotModel,
    width_mapper: WidthMapper,
) -> tuple[np.ndarray, np.ndarray, dict]:
    count = len(grasps["translation"])
    device = torch.device("cpu")
    grasp_pose = torch.eye(4, dtype=torch.float32, device=device)[None].repeat(count, 1, 1)
    grasp_pose[:, :3, :3] = torch.as_tensor(grasps["rotation"], dtype=torch.float32)
    grasp_pose[:, :3, 3] = torch.as_tensor(grasps["translation"], dtype=torch.float32)
    grasp_qpos = {
        name: torch.as_tensor(grasps[name], dtype=torch.float32) for name in robot_model.joint_names
    }

    canonical = torch.tensor(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float32
    )
    pre_qpos = width_mapper.squeeze_fingers(grasp_qpos, -0.025, -0.025)[0]
    squeeze_qpos = width_mapper.squeeze_fingers(grasp_qpos, 0.03, 0.03, keep_z=True)[0]

    pre_local = torch.eye(4, dtype=torch.float32)[None].repeat(count, 1, 1)
    pre_local[:, :3, 3] = canonical.T @ torch.tensor([-0.1, 0.0, 0.0])
    pre_pose = grasp_pose @ pre_local

    lift_top_local = torch.eye(4, dtype=torch.float32)[None].repeat(count, 1, 1)
    lift_top_local[:, :3, 3] = canonical.T @ torch.tensor([-0.2, 0.0, 0.0])
    lift_top = grasp_pose @ lift_top_local
    lift_side = grasp_pose.clone()
    lift_side[:, :3, 3] += torch.tensor([0.0, 0.0, 0.2])
    gripper_x = (grasp_pose[:, :3, :3] @ canonical.T)[:, :, 0]
    top_mask = (gripper_x * torch.tensor([0.0, 0.0, -1.0])).sum(dim=1) > np.cos(np.pi / 3)
    lift_pose = torch.where(top_mask[:, None, None], lift_top, lift_side)

    pose_list = [pre_pose, grasp_pose, grasp_pose, grasp_pose, lift_pose]
    qpos_list = [pre_qpos, pre_qpos, grasp_qpos, squeeze_qpos, squeeze_qpos]
    pose_array = torch.stack(pose_list, dim=1).numpy()
    qpos_array = torch.stack(
        [torch.stack([values[name] for name in robot_model.joint_names], dim=1) for values in qpos_list],
        dim=1,
    ).numpy()
    named_pregrasp = dict(pre_qpos)
    named_pregrasp["translation"] = pre_pose[:, :3, 3]
    named_pregrasp["rotation"] = pre_pose[:, :3, :3]
    return pose_array, qpos_array, named_pregrasp


def compute_pregrasp_valid(
    named_pregrasp: dict,
    scene: dict,
    collision_device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if torch.device(collision_device).type != "cuda":
        raise ValueError(
            "The official torchprimitivesdf pregrasp checker requires CUDA. "
            "Use --collision-device cuda:0 after the driver is healthy, or use "
            "--skip-collision-check only for a teaching smoke test."
        )
    config_path = REPO_ROOT / "configs" / "collision_checker" / "leap_hand" / "CollisionChecker.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checker = CollisionChecker(config, torch.device(collision_device))
    tensors = {
        key: value.to(collision_device) if isinstance(value, torch.Tensor) else value
        for key, value in named_pregrasp.items()
    }
    world_points = []
    for obj in scene["objects"]:
        local = np.load(obj["surface_points"])
        pose = np.asarray(obj["pose_world_object"], dtype=np.float32)
        world_points.append(local @ pose[:3, :3].T + pose[:3, 3])
    points = torch.as_tensor(np.concatenate(world_points), dtype=torch.float32, device=collision_device)
    scene_pen, table_pen = checker.check_collision_batch(tensors, points)
    valid = (scene_pen < 0.0) & (table_pen < 0.0)
    return valid.cpu().numpy(), scene_pen.cpu().numpy(), table_pen.cpu().numpy()


def main() -> None:
    args = parse_args()
    grasps = dict(np.load(args.grasps))
    scene = json.loads(args.scene_manifest.read_text(encoding="utf-8"))
    required = {"rotation", "translation"}
    if required.difference(grasps):
        raise KeyError("grasps file must contain rotation and translation")

    robot_model = RobotModel(
        "robot_models/urdf/leap_hand_simplified.urdf",
        "robot_models/meta/leap_hand/meta.yaml",
    )
    missing = [name for name in robot_model.joint_names if name not in grasps]
    if missing:
        raise KeyError("grasps file is missing LEAP joints: {}".format(missing))
    width_mapper = WidthMapper(robot_model, "robot_models/meta/leap_hand/width_mapper_meta.yaml")
    waypoint_pose, waypoint_qpos, named_pregrasp = compose_waypoints(
        grasps, robot_model, width_mapper
    )

    count = len(grasps["translation"])
    if args.skip_collision_check:
        pregrasp_valid = np.ones(count, dtype=bool)
        scene_pen = np.full(count, np.nan, dtype=np.float32)
        table_pen = np.full(count, np.nan, dtype=np.float32)
    else:
        pregrasp_valid, scene_pen, table_pen = compute_pregrasp_valid(
            named_pregrasp, scene, args.collision_device
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        waypoint_pose_world=waypoint_pose.astype(np.float32),
        waypoint_root_dofs=np.concatenate(
            [
                waypoint_pose[:, :, :3, 3],
                matrix_to_euler_angles(
                    torch.as_tensor(waypoint_pose[:, :, :3, :3]), "XYZ"
                ).numpy(),
            ],
            axis=2,
        ).astype(np.float32),
        waypoint_joint_positions=waypoint_qpos.astype(np.float32),
        pregrasp_valid=pregrasp_valid,
        scene_penetration=scene_pen.astype(np.float32),
        table_penetration=table_pen.astype(np.float32),
        finger_joint_names=np.asarray(robot_model.joint_names),
        root_joint_names=np.asarray(ROOT_JOINT_NAMES),
        source_grasp_indices=np.arange(count, dtype=np.int64),
    )
    job_manifest = {
        "schema_version": 1,
        "backend": "Isaac Sim 5.0 + Isaac Lab 2.2",
        "paper_robot": "LEAP Hand, 16 finger joints",
        "source_grasps": str(args.grasps.resolve()),
        "waypoint_job": str(args.output.resolve()),
        "scene_manifest": str(args.scene_manifest.resolve()),
        "objects": scene["objects"],
        "waypoints": ["pregrasp", "cover", "grasp", "squeeze", "lift"],
        "waypoint_steps": [40, 20, 20, 60],
        "success_rule": "any scene object rises more than 0.03 m",
        "pregrasp_collision_checked": not args.skip_collision_check,
        "coordinate_contract": "all waypoint poses and object poses are T_world_local in metres",
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(job_manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote {}".format(args.output.resolve()))
    print("wrote {}".format(manifest_path.resolve()))
    print("grasps={}, pregrasp_valid={}".format(count, int(pregrasp_valid.sum())))


if __name__ == "__main__":
    main()
