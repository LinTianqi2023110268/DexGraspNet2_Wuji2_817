from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


RIGHT_ARM_JOINTS: Tuple[str, ...] = (
    "arm_r_joint_1",
    "arm_r_joint_2",
    "arm_r_joint_3",
    "arm_r_joint_4",
    "arm_r_joint_5",
    "arm_r_joint_6",
    "arm_r_joint_7",
)


@dataclass(frozen=True)
class LockedJointContract:
    full_joint_names: Tuple[str, ...]
    active_joint_names: Tuple[str, ...]
    locked_joint_names: Tuple[str, ...]
    active_indices: Tuple[int, ...]
    locked_indices: Tuple[int, ...]
    lock_joints: Dict[str, float]
    q_current_active: np.ndarray
    q_goal_active: np.ndarray
    max_locked_goal_difference_rad: float

    @property
    def action_dim(self) -> int:
        return len(self.active_joint_names)

    @property
    def locked_joint_count(self) -> int:
        return len(self.locked_joint_names)

    def to_report(self) -> Dict[str, Any]:
        return {
            "active_joint_names": list(self.active_joint_names),
            "action_dim": self.action_dim,
            "locked_joint_names": list(self.locked_joint_names),
            "locked_joint_count": self.locked_joint_count,
            "max_locked_goal_difference_rad": self.max_locked_goal_difference_rad,
            "lock_joints": {k: float(v) for k, v in self.lock_joints.items()},
        }


def _as_1d_float64(x: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def build_locked_joint_contract(
    full_joint_names: Sequence[str],
    q_current_planning: Sequence[float],
    q_pregrasp_planning: Sequence[float],
    *,
    active_joint_names: Sequence[str] = RIGHT_ARM_JOINTS,
    locked_goal_tolerance_rad: float = 1e-5,
) -> LockedJointContract:
    """
    Build the right-arm-only planning contract.

    Source of truth:
      - full_joint_names defines how full q vectors are indexed.
      - only active_joint_names may move.
      - every other joint is locked at q_current_planning.
      - q_pregrasp_planning must agree with those locked values to numerical tolerance.

    This function never clips, edits, or silently changes either q vector.
    """
    names = tuple(str(n) for n in full_joint_names)
    if len(names) != len(set(names)):
        raise ValueError("full_joint_names contains duplicates")

    active = tuple(str(n) for n in active_joint_names)
    if active != RIGHT_ARM_JOINTS:
        # Deliberately strict for Route B phase-1.
        raise ValueError(
            "Route B phase-1 active joints must exactly equal RIGHT_ARM_JOINTS "
            f"in canonical order; got {active}"
        )

    missing = [n for n in active if n not in names]
    if missing:
        raise KeyError(f"Active right-arm joints missing from full model: {missing}")

    q0 = _as_1d_float64(q_current_planning, "q_current_planning")
    qg = _as_1d_float64(q_pregrasp_planning, "q_pregrasp_planning")
    if len(q0) != len(names) or len(qg) != len(names):
        raise ValueError(
            f"Full state lengths must equal len(full_joint_names)={len(names)}; "
            f"got current={len(q0)}, goal={len(qg)}"
        )

    index = {n: i for i, n in enumerate(names)}
    active_idx = tuple(index[n] for n in active)
    locked = tuple(n for n in names if n not in set(active))
    locked_idx = tuple(index[n] for n in locked)

    if locked_idx:
        locked_delta = np.abs(qg[list(locked_idx)] - q0[list(locked_idx)])
        max_locked_delta = float(np.max(locked_delta))
    else:
        max_locked_delta = 0.0

    if max_locked_delta > float(locked_goal_tolerance_rad):
        offenders = []
        for n, i in zip(locked, locked_idx):
            d = abs(float(qg[i] - q0[i]))
            if d > float(locked_goal_tolerance_rad):
                offenders.append(
                    {"joint_name": n, "current": float(q0[i]), "goal": float(qg[i]), "diff": d}
                )
        raise RuntimeError(
            "Locked DOFs differ between q_current_planning and q_pregrasp_planning "
            f"by more than {locked_goal_tolerance_rad} rad: {offenders}"
        )

    lock_joints = {n: float(q0[index[n]]) for n in locked}
    return LockedJointContract(
        full_joint_names=names,
        active_joint_names=active,
        locked_joint_names=locked,
        active_indices=active_idx,
        locked_indices=locked_idx,
        lock_joints=lock_joints,
        q_current_active=q0[list(active_idx)].copy(),
        q_goal_active=qg[list(active_idx)].copy(),
        max_locked_goal_difference_rad=max_locked_delta,
    )


def _locate_kinematics_dict(robot_dict: Mapping[str, Any]) -> Dict[str, Any]:
    data = deepcopy(dict(robot_dict))
    if "robot_cfg" in data:
        robot_cfg = data["robot_cfg"]
        if not isinstance(robot_cfg, dict):
            raise TypeError("robot_cfg must be a dict")
        if "kinematics" not in robot_cfg or not isinstance(robot_cfg["kinematics"], dict):
            raise KeyError("robot_cfg.kinematics dict not found")
        robot_cfg["kinematics"]["lock_joints"] = {}
        return data
    if "kinematics" not in data or not isinstance(data["kinematics"], dict):
        raise KeyError("kinematics dict not found")
    data["kinematics"]["lock_joints"] = {}
    return data


def robot_dict_with_lock_joints(
    robot_dict: Mapping[str, Any],
    lock_joints: Mapping[str, float],
) -> Dict[str, Any]:
    """
    Deep-copy a cuRobo robot YAML/dict and set kinematics.lock_joints.
    Never mutates the caller's robot dictionary.
    """
    data = _locate_kinematics_dict(robot_dict)
    target = data["robot_cfg"]["kinematics"] if "robot_cfg" in data else data["kinematics"]
    target["lock_joints"] = {str(k): float(v) for k, v in lock_joints.items()}
    return data


def rebuild_robot_cfg_with_lock_joints(
    robot_source: Any,
    lock_joints: Mapping[str, float],
    *,
    device_cfg: Any = None,
) -> Any:
    """
    Rebuild a cuRobo RobotCfg with all non-right-arm joints converted to fixed joints.

    Supported robot_source:
      1. Robot YAML/dict (preferred): deep-copy, inject kinematics.lock_joints,
         then RobotCfg.create(...).
      2. Existing RobotCfg: rebuild through its KinematicsCfg.generator_config.

    This intentionally does NOT mutate a prebuilt KinematicsCfg in place. cuRobo's
    locked-joint topology is created by KinematicsLoader during model construction.
    """
    try:
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.types.robot import RobotCfg
    except Exception as exc:
        raise RuntimeError(
            "cuRobo imports are required to rebuild a locked-joint robot config"
        ) from exc

    if isinstance(robot_source, Mapping):
        locked_dict = robot_dict_with_lock_joints(robot_source, lock_joints)
        if device_cfg is None:
            return RobotCfg.create(locked_dict)
        return RobotCfg.create(locked_dict, device_cfg=device_cfg)

    if isinstance(robot_source, RobotCfg):
        gen = getattr(robot_source.kinematics, "generator_config", None)
        if gen is None:
            raise RuntimeError(
                "Existing RobotCfg has no kinematics.generator_config; "
                "pass the original robot YAML/dict to Route B instead."
            )
        gen_locked = deepcopy(gen)
        gen_locked.lock_joints = {str(k): float(v) for k, v in lock_joints.items()}
        if device_cfg is not None:
            gen_locked.device_cfg = device_cfg
        locked_kinematics = KinematicsCfg.from_config(gen_locked)
        return RobotCfg(
            kinematics=locked_kinematics,
            dynamics=None,
            device_cfg=device_cfg if device_cfg is not None else robot_source.device_cfg,
        )

    raise TypeError(
        "robot_source must be a robot config dict or cuRobo RobotCfg. "
        f"Got {type(robot_source)!r}"
    )
