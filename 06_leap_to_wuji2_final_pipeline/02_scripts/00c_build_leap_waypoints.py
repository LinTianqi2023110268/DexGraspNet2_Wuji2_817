#!/usr/bin/env python3
"""Select official raw-score rank 0 and build official LEAP waypoints."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_euler_angles

from case_paths import PROJECT_ROOT, active_case_root


CASE_ROOT = active_case_root()
CASE_PATH = CASE_ROOT / "case.json"
CASE = json.loads(CASE_PATH.read_text(encoding="utf-8"))
INPUT = CASE_ROOT / "01_input" / "official_leap_1024.npz"
SCENE_MANIFEST = CASE_ROOT / "01_input" / f"{CASE['scene_id']}_manifest.json"
OUTPUT = CASE_ROOT / "01_input" / "leap_official_waypoints.npz"
SELECTED = CASE_ROOT / "01_input" / "leap_selected_rank0.npz"
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network" / "official_core"
WAYPOINT_NAMES = ["pregrasp", "cover", "grasp", "squeeze", "lift"]
OFFICIAL_CONTROL_STEPS = [40, 20, 20, 60]


def main() -> None:
    for path in (INPUT, SCENE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("Official PREGRASP collision checking requires visible CUDA.")

    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    scene = json.loads(SCENE_MANIFEST.read_text(encoding="utf-8"))
    candidate = int(np.argmax(data["score"]))
    if candidate != int(data["score_descending_candidate_index"][0]):
        raise RuntimeError("recorded rank order differs from argmax(score)")
    joint_order = [str(value) for value in data["leap_joint_order"].tolist()]
    qpos = np.asarray(data["leap_qpos_rad"][candidate], dtype=np.float32)
    grasp = {
        "rotation": data["rotation_world"][[candidate]].astype(np.float32),
        "translation": data["translation_world"][[candidate]].astype(np.float32),
        **{
            name: np.asarray([qpos[index]], dtype=np.float32)
            for index, name in enumerate(joint_order)
        },
    }
    np.savez_compressed(
        SELECTED,
        **grasp,
        source_candidate_index=np.asarray([candidate], dtype=np.int64),
        score=np.asarray([data["score"][candidate]], dtype=np.float32),
        graspness=np.asarray([data["graspness"][candidate]], dtype=np.float32),
        log_prob=np.asarray([data["log_prob"][candidate]], dtype=np.float32),
        seed_point_world=data["seed_point_world"][[candidate]].astype(np.float32),
        target_segmentation_id=np.asarray(
            [data["target_segmentation_id"][candidate]], dtype=np.int64
        ),
        leap_joint_order=np.asarray(joint_order),
        leap_qpos_rad=qpos[None],
    )

    original_cwd = Path.cwd()
    os.chdir(OFFICIAL_ROOT)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from src.eval.prepare_isaacsim5_job import (
            ROOT_JOINT_NAMES,
            compose_waypoints,
            compute_pregrasp_valid,
        )
        from src.utils.robot_model import RobotModel
        from src.utils.width_mapper import WidthMapper

        robot = RobotModel(
            "robot_models/urdf/leap_hand_simplified.urdf",
            "robot_models/meta/leap_hand/meta.yaml",
        )
        if list(robot.joint_names) != joint_order:
            raise RuntimeError("LEAP joint order mismatch")
        mapper = WidthMapper(robot, "robot_models/meta/leap_hand/width_mapper_meta.yaml")
        waypoint_pose, waypoint_qpos, named_pregrasp = compose_waypoints(
            grasp, robot, mapper
        )
        valid, scene_pen, table_pen = compute_pregrasp_valid(
            named_pregrasp, scene, "cuda:0"
        )
    finally:
        os.chdir(original_cwd)

    pose = waypoint_pose[0]
    root_dofs = np.concatenate(
        [
            pose[:, :3, 3],
            matrix_to_euler_angles(torch.as_tensor(pose[:, :3, :3]), "XYZ").numpy(),
        ],
        axis=1,
    ).astype(np.float32)
    target_seg = int(data["target_segmentation_id"][candidate])
    matching = [
        item for item in scene["objects"]
        if int(item["segmentation_id"]) == target_seg
    ]
    if len(matching) != 1:
        raise RuntimeError(f"target segmentation {target_seg}: {len(matching)} matches")
    target_code = str(matching[0]["code"])
    np.savez_compressed(
        OUTPUT,
        waypoint_pose_world=waypoint_pose.astype(np.float32),
        waypoint_root_dofs=root_dofs[None],
        waypoint_joint_positions=waypoint_qpos.astype(np.float32),
        waypoint_names=np.asarray(WAYPOINT_NAMES),
        waypoint_steps=np.asarray(OFFICIAL_CONTROL_STEPS, dtype=np.int64),
        finger_joint_names=np.asarray(joint_order),
        root_joint_names=np.asarray(ROOT_JOINT_NAMES),
        pregrasp_valid=np.asarray(valid, dtype=bool),
        scene_penetration=np.asarray(scene_pen, dtype=np.float32),
        table_penetration=np.asarray(table_pen, dtype=np.float32),
        source_candidate_index=np.asarray([candidate], dtype=np.int64),
        score=np.asarray([data["score"][candidate]], dtype=np.float32),
        graspness=np.asarray([data["graspness"][candidate]], dtype=np.float32),
        log_prob=np.asarray([data["log_prob"][candidate]], dtype=np.float32),
        seed_point_world=data["seed_point_world"][[candidate]].astype(np.float32),
        target_segmentation_id=np.asarray([target_seg], dtype=np.int64),
    )
    report = {
        "schema_version": 1,
        "status": "official_raw_rank0_leap_waypoints_ready",
        "case_id": CASE_ROOT.name,
        "selection_order": "argmax(score), then report PREGRASP collision",
        "selected_candidate_index": candidate,
        "target_segmentation_id": target_seg,
        "target_object_code": target_code,
        "score": float(data["score"][candidate]),
        "graspness": float(data["graspness"][candidate]),
        "log_prob": float(data["log_prob"][candidate]),
        "pregrasp_valid": bool(valid[0]),
        "scene_penetration_m": float(scene_pen[0]),
        "table_penetration_m": float(table_pen[0]),
        "official_control_steps": OFFICIAL_CONTROL_STEPS,
        "official_action": {
            "pregrasp_finger_relaxation_m": 0.025,
            "pregrasp_root_retreat_m": 0.1,
            "squeeze_per_fingertip_m": 0.03,
            "squeeze_keep_z": True,
            "lift_m": 0.2,
        },
        "output": str(OUTPUT),
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    CASE.update(
        {
            "target_segmentation_id": target_seg,
            "target_object_code": target_code,
            "source_candidate_index": candidate,
            "pipeline_status": "official_leap_waypoints_ready",
            "official_rank0_pregrasp_valid": bool(valid[0]),
        }
    )
    CASE_PATH.write_text(
        json.dumps(CASE, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    status = "PASS" if bool(valid[0]) else "FAIL"
    print(f"[PASS] rank0={candidate}; target={target_seg} {target_code}")
    print(
        f"[{status}] PREGRASP scene={scene_pen[0]:+.6f} m; "
        f"table={table_pen[0]:+.6f} m"
    )
    print(f"[OK] {OUTPUT}")


if __name__ == "__main__":
    main()
