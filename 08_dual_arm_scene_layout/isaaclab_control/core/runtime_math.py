"""Pure NumPy transforms shared by the Isaac Lab Route-C V2 runtime."""

from __future__ import annotations

import numpy as np


def pose_from_position_quaternion_wxyz(position, quaternion_wxyz) -> np.ndarray:
    """Build ``T_world_body`` from an Isaac Lab position and wxyz quaternion."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def rebase_pick_waypoints(
    wrist_targets_world: np.ndarray,
    object_pose_before_settle: np.ndarray,
    object_pose_after_settle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Carry PREGRASP..SQUEEZE with the settled object and keep LIFT world +Z."""
    targets = np.asarray(wrist_targets_world, dtype=np.float64).copy()
    if targets.shape != (5, 4, 4):
        raise ValueError(f"expected five 4x4 wrist targets, got {targets.shape}")
    object_delta = (
        np.asarray(object_pose_after_settle, dtype=np.float64)
        @ np.linalg.inv(np.asarray(object_pose_before_settle, dtype=np.float64))
    )
    for index in range(4):
        targets[index] = object_delta @ targets[index]
    lift_world = np.asarray(wrist_targets_world, dtype=np.float64)[4, :3, 3] - np.asarray(
        wrist_targets_world, dtype=np.float64
    )[3, :3, 3]
    targets[4] = targets[3].copy()
    targets[4, :3, 3] += lift_world
    return targets, object_delta
