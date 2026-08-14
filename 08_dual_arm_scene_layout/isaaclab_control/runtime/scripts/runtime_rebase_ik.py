"""Pure-math runtime helpers for a scene that settles before perception.

The cached DexGraspNet2 grasp is expressed relative to the object pose visible
in the cached RGB-D frame.  If gravity changes that object pose before the arm
moves, this module rigidly carries PREGRASP/COVER/GRASP/SQUEEZE with the object,
then re-solves the seven right-arm joints.  It contains no Isaac Sim imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]


def pose_from_position_quaternion_wxyz(position, quaternion_wxyz) -> np.ndarray:
    """Build a row-major ``T_world_body`` from Isaac Lab tensors."""
    position = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    pose[:3, 3] = position
    return pose


def rebase_pick_waypoints(
    wrist_targets_world: np.ndarray,
    object_pose_before_settle: np.ndarray,
    object_pose_after_settle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the cached hand/object relation after the object settles."""
    targets = np.asarray(wrist_targets_world, dtype=np.float64).copy()
    object_delta = object_pose_after_settle @ np.linalg.inv(object_pose_before_settle)
    for index in range(4):
        targets[index] = object_delta @ targets[index]

    # The official validation lifts along world +Z after SQUEEZE.  Preserve
    # that policy instead of rotating the lift vector with the toppled object.
    lift_world = wrist_targets_world[4, :3, 3] - wrist_targets_world[3, :3, 3]
    targets[4] = targets[3].copy()
    targets[4, :3, 3] += lift_world
    return targets, object_delta


def solve_right_arm_targets(
    robot_urdf: Path,
    layout_json: Path,
    flange_targets_world: np.ndarray,
    seed_joint_positions: np.ndarray,
    initial_joint_position: np.ndarray,
    random_starts: int = 16,
) -> tuple[np.ndarray, list[dict]]:
    """Solve bounded seven-joint IK for five exact flange targets."""
    layout = json.loads(layout_json.read_text(encoding="utf-8"))
    world_from_base = np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T
    base_from_world = np.linalg.inv(world_from_base)

    model = pin.buildModelFromUrdf(str(robot_urdf))
    data = model.createData()
    flange_frame = model.getFrameId("arm_r_link_tf")
    joint_ids = [model.getJointId(name) for name in RIGHT_ARM_NAMES]
    q_indices = np.asarray([model.joints[joint_id].idx_q for joint_id in joint_ids])
    q_template = pin.neutral(model)
    lower = model.lowerPositionLimit[q_indices] + 0.01
    upper = model.upperPositionLimit[q_indices] - 0.01

    def frame_at(right_q: np.ndarray) -> pin.SE3:
        q = q_template.copy()
        q[q_indices] = right_q
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[flange_frame].copy()

    rng = np.random.default_rng(20260813)
    solved = []
    reports = []
    previous = np.asarray(initial_joint_position, dtype=np.float64).copy()
    for index, target_world in enumerate(np.asarray(flange_targets_world)):
        target_matrix = base_from_world @ target_world
        target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])

        def residual(right_q: np.ndarray) -> np.ndarray:
            current = frame_at(right_q)
            position = (current.translation - target.translation) / 0.01
            orientation = pin.log3(current.rotation.T @ target.rotation) / 0.10
            return np.concatenate((position, orientation))

        starts = [previous, seed_joint_positions[index], initial_joint_position]
        starts.extend(rng.uniform(lower, upper) for _ in range(random_starts))
        candidates = []
        for start in starts:
            result = least_squares(
                residual, np.clip(start, lower, upper), bounds=(lower, upper),
                max_nfev=800, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11,
            )
            achieved = frame_at(result.x)
            position_mm = 1000.0 * float(
                np.linalg.norm(achieved.translation - target.translation)
            )
            orientation_deg = float(np.degrees(
                np.linalg.norm(pin.log3(achieved.rotation.T @ target.rotation))
            ))
            margin_deg = float(np.degrees(np.min(
                np.minimum(result.x - lower, upper - result.x)
            )))
            cost = position_mm + 2.0 * orientation_deg + 20.0 * max(0.0, 3.0 - margin_deg)
            candidates.append((cost, result.x, position_mm, orientation_deg, margin_deg))

        _, best, position_mm, orientation_deg, margin_deg = min(candidates, key=lambda x: x[0])
        solved.append(best.copy())
        previous = best.copy()
        reports.append({
            "position_error_mm": position_mm,
            "orientation_error_deg": orientation_deg,
            "minimum_limit_margin_deg": margin_deg,
        })
    return np.asarray(solved), reports

