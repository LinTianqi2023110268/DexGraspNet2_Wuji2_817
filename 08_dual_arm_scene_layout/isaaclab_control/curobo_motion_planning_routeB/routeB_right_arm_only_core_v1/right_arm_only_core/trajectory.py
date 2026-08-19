from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import RIGHT_ARM_JOINTS, LockedJointContract


def _to_numpy(x: Any, name: str) -> np.ndarray:
    if x is None:
        raise RuntimeError(f"{name} is missing")
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    a = np.asarray(x)
    if not np.all(np.isfinite(a)):
        raise RuntimeError(f"{name} contains non-finite values")
    return a


def _trajectory_matrix(x: Any, name: str) -> np.ndarray:
    """
    Convert cuRobo trajectory field to [N, D].
    Accepts leading singleton batch/seed dimensions.
    """
    a = _to_numpy(x, name)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise RuntimeError(f"{name} must reduce to [N,D], got {a.shape}")
    return np.asarray(a, dtype=np.float64)


def _joint_names_from_state(state: Any, width: int, planner_joint_names: Sequence[str]) -> Tuple[str, ...]:
    state_names = getattr(state, "joint_names", None)
    if state_names is not None and len(state_names) == width:
        return tuple(str(n) for n in state_names)
    if len(planner_joint_names) == width:
        return tuple(str(n) for n in planner_joint_names)
    raise RuntimeError(
        "Cannot resolve dense trajectory joint_names: "
        f"width={width}, state_joint_names={state_names}, planner_joint_names={planner_joint_names}"
    )


def _select_columns(a: np.ndarray, names: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    idx = {str(n): i for i, n in enumerate(names)}
    missing = [n for n in selected if n not in idx]
    if missing:
        raise RuntimeError(f"Dense trajectory missing joints: {missing}")
    return a[:, [idx[n] for n in selected]]


def _dt_and_time(state: Any, n: int) -> Tuple[float, np.ndarray, str]:
    dt_raw = getattr(state, "dt", None)
    if dt_raw is None:
        raise RuntimeError("Dense cuRobo trajectory has no dt; refusing to guess timing")
    dt_arr = _to_numpy(dt_raw, "dense.dt").astype(np.float64).reshape(-1)
    if dt_arr.size == 1:
        dt = float(dt_arr[0])
        if not (dt > 0.0):
            raise RuntimeError(f"Invalid dt={dt}")
        return dt, np.arange(n, dtype=np.float64) * dt, "curobo_dense_joint_state.dt"

    # Some JointState variants may expose per-step dt.
    if dt_arr.size == n:
        if np.any(dt_arr <= 0):
            raise RuntimeError("Per-point dt contains non-positive values")
        time_s = np.zeros(n, dtype=np.float64)
        if n > 1:
            time_s[1:] = np.cumsum(dt_arr[:-1])
        return float(np.median(dt_arr)), time_s, "curobo_dense_joint_state.per_point_dt"

    if dt_arr.size == n - 1:
        if np.any(dt_arr <= 0):
            raise RuntimeError("Per-segment dt contains non-positive values")
        time_s = np.concatenate([[0.0], np.cumsum(dt_arr)])
        return float(np.median(dt_arr)), time_s, "curobo_dense_joint_state.per_segment_dt"

    raise RuntimeError(f"Unsupported dense.dt shape/length: {dt_arr.shape}, N={n}")


@dataclass
class DenseTrajectory:
    joint_names: Tuple[str, ...]
    q_rad: np.ndarray
    qd_rad_s: np.ndarray
    qdd_rad_s2: np.ndarray
    jerk_rad_s3: np.ndarray
    time_s: np.ndarray
    dt_s: float
    dt_source: str
    full_joint_names: Optional[Tuple[str, ...]] = None
    full_q_rad: Optional[np.ndarray] = None

    @property
    def point_count(self) -> int:
        return int(self.q_rad.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1]) if self.time_s.size else 0.0

    def to_report(self) -> Dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "shape": list(self.q_rad.shape),
            "point_count": self.point_count,
            "dt_s": self.dt_s,
            "dt_source": self.dt_source,
            "duration_s": self.duration_s,
            "max_abs_velocity_rad_s": float(np.max(np.abs(self.qd_rad_s))),
            "max_abs_acceleration_rad_s2": float(np.max(np.abs(self.qdd_rad_s2))),
            "max_abs_jerk_rad_s3": float(np.max(np.abs(self.jerk_rad_s3))),
        }


def extract_dense_right_arm_trajectory(
    result: Any,
    planner_joint_names: Sequence[str],
    contract: LockedJointContract,
) -> DenseTrajectory:
    """
    Extract the final cuRobo dense/interpolated trajectory.

    Crucial contract:
      - result.solution must be active 7DOF. This proves the optimizer itself was 7DOF.
      - A full-state dense trajectory is allowed, but right-arm channels are selected by name.
      - No new interpolation/resampling is performed.
    """
    solution = _to_numpy(getattr(result, "solution", None), "result.solution")
    if solution.shape[-1] != len(RIGHT_ARM_JOINTS):
        raise RuntimeError(
            "Optimizer output is not true 7DOF. Refusing post-hoc slicing of a full-DOF plan: "
            f"result.solution.shape={solution.shape}"
        )

    if hasattr(result, "get_interpolated_plan"):
        dense = result.get_interpolated_plan()
    else:
        dense = getattr(result, "interpolated_trajectory", None)
    if dense is None:
        raise RuntimeError("cuRobo result exposes no dense/interpolated trajectory")

    q = _trajectory_matrix(getattr(dense, "position", None), "dense.position")
    qd = _trajectory_matrix(getattr(dense, "velocity", None), "dense.velocity")
    qdd = _trajectory_matrix(getattr(dense, "acceleration", None), "dense.acceleration")
    jerk = _trajectory_matrix(getattr(dense, "jerk", None), "dense.jerk")
    if not (q.shape == qd.shape == qdd.shape == jerk.shape):
        raise RuntimeError(
            f"Dense state field shapes disagree: q={q.shape}, qd={qd.shape}, "
            f"qdd={qdd.shape}, jerk={jerk.shape}"
        )

    dense_names = _joint_names_from_state(dense, q.shape[1], planner_joint_names)
    right = tuple(RIGHT_ARM_JOINTS)
    q_r = _select_columns(q, dense_names, right)
    qd_r = _select_columns(qd, dense_names, right)
    qdd_r = _select_columns(qdd, dense_names, right)
    jerk_r = _select_columns(jerk, dense_names, right)

    dt_s, time_s, dt_source = _dt_and_time(dense, q_r.shape[0])

    full_q = None
    full_names = None
    if set(contract.locked_joint_names).issubset(set(dense_names)):
        full_q = q.copy()
        full_names = tuple(dense_names)

    return DenseTrajectory(
        joint_names=right,
        q_rad=q_r,
        qd_rad_s=qd_r,
        qdd_rad_s2=qdd_r,
        jerk_rad_s3=jerk_r,
        time_s=time_s,
        dt_s=dt_s,
        dt_source=dt_source,
        full_joint_names=full_names,
        full_q_rad=full_q,
    )


def validate_dense_trajectory(
    traj: DenseTrajectory,
    contract: LockedJointContract,
    *,
    start_tolerance_rad: float = 1e-6,
    goal_tolerance_rad: float = 1e-4,
    locked_tolerance_rad: float = 1e-7,
) -> Dict[str, Any]:
    if traj.q_rad.ndim != 2 or traj.q_rad.shape[1] != 7 or traj.q_rad.shape[0] <= 1:
        raise RuntimeError(f"Right-arm trajectory must be [N>1,7], got {traj.q_rad.shape}")
    if tuple(traj.joint_names) != RIGHT_ARM_JOINTS:
        raise RuntimeError(f"Unexpected right-arm joint order: {traj.joint_names}")

    start_err = float(np.max(np.abs(traj.q_rad[0] - contract.q_current_active)))
    goal_err = float(np.max(np.abs(traj.q_rad[-1] - contract.q_goal_active)))
    if start_err > start_tolerance_rad:
        raise RuntimeError(
            f"Trajectory start error {start_err:.3e} > {start_tolerance_rad:.3e} rad"
        )
    if goal_err > goal_tolerance_rad:
        raise RuntimeError(
            f"Trajectory goal error {goal_err:.3e} > {goal_tolerance_rad:.3e} rad"
        )

    locked_max_dev = 0.0
    locked_mode = "planner_kinematic_lock"
    if traj.full_q_rad is not None and traj.full_joint_names is not None:
        idx = {n: i for i, n in enumerate(traj.full_joint_names)}
        for n in contract.locked_joint_names:
            j = idx[n]
            v = contract.lock_joints[n]
            locked_max_dev = max(
                locked_max_dev,
                float(np.max(np.abs(traj.full_q_rad[:, j] - v))),
            )
        locked_mode = "full_dense_state"
        if locked_max_dev > locked_tolerance_rad:
            raise RuntimeError(
                f"Locked DOFs moved by {locked_max_dev:.3e} rad, "
                f"limit={locked_tolerance_rad:.3e}"
            )

    return {
        "start_error_rad": start_err,
        "goal_error_rad": goal_err,
        "locked_joint_validation_mode": locked_mode,
        "max_locked_trajectory_deviation_rad": locked_max_dev,
        "locked_tolerance_rad": locked_tolerance_rad,
    }


def save_right_arm_npz(path: str | Path, traj: DenseTrajectory) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        joint_names=np.asarray(traj.joint_names, dtype="U"),
        q_rad=traj.q_rad.astype(np.float32),
        qd_rad_s=traj.qd_rad_s.astype(np.float32),
        qdd_rad_s2=traj.qdd_rad_s2.astype(np.float32),
        jerk_rad_s3=traj.jerk_rad_s3.astype(np.float32),
        time_s=traj.time_s.astype(np.float64),
        dt_s=np.asarray(traj.dt_s, dtype=np.float64),
        dt_source=np.asarray(traj.dt_source, dtype="U"),
    )
    return p
