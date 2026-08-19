from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from .contract import (
    RIGHT_ARM_JOINTS,
    LockedJointContract,
    build_locked_joint_contract,
    rebuild_robot_cfg_with_lock_joints,
)
from .trajectory import (
    DenseTrajectory,
    extract_dense_right_arm_trajectory,
    save_right_arm_npz,
    validate_dense_trajectory,
)


PlannerFactory = Callable[[Any], Any]
PlannerFixup = Callable[[Any], Mapping[str, Any]]
TrajectoryPostcheck = Callable[[DenseTrajectory, LockedJointContract, Any, Any], Mapping[str, Any]]


@dataclass
class RightArmPlanResult:
    planner: Any
    raw_result: Any
    contract: LockedJointContract
    trajectory: DenseTrajectory
    report: Dict[str, Any]
    trajectory_path: Path
    report_path: Path


def _tensor_bool_success(x: Any) -> bool:
    if x is None:
        return False
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    a = np.asarray(x)
    return bool(np.any(a))


def _make_joint_state(q: Sequence[float], planner: Any) -> Any:
    try:
        import torch
        from curobo._src.state.state_joint import JointState
    except Exception as exc:
        raise RuntimeError("cuRobo + torch are required for planning") from exc

    device_cfg = getattr(planner, "device_cfg", None)
    if device_cfg is None:
        raise RuntimeError("planner.device_cfg is unavailable")
    t = torch.as_tensor(
        np.asarray(q, dtype=np.float32),
        device=device_cfg.device,
        dtype=device_cfg.dtype,
    ).view(1, -1)
    return JointState.from_position(t)


def _canonical_joint_list(planner: Any) -> Sequence[str]:
    names = getattr(planner, "joint_names", None)
    if names is None:
        raise RuntimeError("planner.joint_names is unavailable")
    return [str(n) for n in names]


def plan_right_arm_only(
    *,
    robot_source: Any,
    full_joint_names: Sequence[str],
    q_current_planning: Sequence[float],
    q_pregrasp_planning: Sequence[float],
    planner_factory: PlannerFactory,
    output_dir: str | Path,
    planner_fixup: Optional[PlannerFixup] = None,
    trajectory_postcheck: Optional[TrajectoryPostcheck] = None,
    max_attempts: int = 2,
    enable_graph_attempt: int = 1_000_000,
    locked_goal_tolerance_rad: float = 1e-5,
    start_tolerance_rad: float = 1e-6,
    goal_tolerance_rad: float = 1e-4,
    locked_trajectory_tolerance_rad: float = 1e-7,
) -> RightArmPlanResult:
    """
    Execute a true right-arm-only current->PREGRASP cuRobo plan.

    Integration boundary:
      - This core owns the lock-joint contract, true 7DOF requirement,
        plan_cspace call, dense trajectory extraction, validation, and artifacts.
      - The local Route B adapter supplies planner_factory so it can preserve the
        already-validated local MotionPlannerCfg parameters and scene.
      - planner_fixup is where the adapter reuses its existing collision-policy
        and exact-VoxelData-shape hooks on the NEW locked planner.
      - trajectory_postcheck reuses the project's existing ESDF / constraint
        audit helpers. This core does not duplicate them.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    contract = build_locked_joint_contract(
        full_joint_names,
        q_current_planning,
        q_pregrasp_planning,
        active_joint_names=RIGHT_ARM_JOINTS,
        locked_goal_tolerance_rad=locked_goal_tolerance_rad,
    )

    # Rebuild topology with 28 fixed joints. This is NOT a mask on a 35DOF optimizer.
    locked_robot_cfg = rebuild_robot_cfg_with_lock_joints(
        robot_source,
        contract.lock_joints,
    )

    planner = planner_factory(locked_robot_cfg)
    fixup_report: Dict[str, Any] = {}
    if planner_fixup is not None:
        fixup_report = dict(planner_fixup(planner))

    action_dim = int(getattr(planner, "action_dim"))
    active_names = tuple(_canonical_joint_list(planner))
    if action_dim != 7:
        raise RuntimeError(f"Expected planner.action_dim=7, got {action_dim}")
    if active_names != RIGHT_ARM_JOINTS:
        raise RuntimeError(
            "Locked planner active joint order mismatch. "
            f"Expected {RIGHT_ARM_JOINTS}, got {active_names}"
        )

    current_state = _make_joint_state(contract.q_current_active, planner)
    goal_state = _make_joint_state(contract.q_goal_active, planner)

    raw = planner.plan_cspace(
        goal_state=goal_state,
        current_state=current_state,
        max_attempts=int(max_attempts),
        enable_graph_attempt=int(enable_graph_attempt),
    )
    if raw is None:
        raise RuntimeError("MotionPlanner.plan_cspace returned None")
    if not _tensor_bool_success(getattr(raw, "success", None)):
        raise RuntimeError("MotionPlanner.plan_cspace returned success=false")

    solution = getattr(raw, "solution", None)
    if solution is None or int(solution.shape[-1]) != 7:
        raise RuntimeError(
            "Planner reported success but optimizer solution is not 7DOF: "
            f"shape={getattr(solution, 'shape', None)}"
        )

    traj = extract_dense_right_arm_trajectory(raw, active_names, contract)
    validation = validate_dense_trajectory(
        traj,
        contract,
        start_tolerance_rad=start_tolerance_rad,
        goal_tolerance_rad=goal_tolerance_rad,
        locked_tolerance_rad=locked_trajectory_tolerance_rad,
    )

    postcheck: Dict[str, Any] = {}
    if trajectory_postcheck is not None:
        postcheck = dict(trajectory_postcheck(traj, contract, planner, raw))

    # Hard-fail on explicitly returned bad project checks.
    if postcheck.get("environment_collision") is True:
        raise RuntimeError("Project postcheck reports environment collision")
    if postcheck.get("joint_limit_pass") is False:
        raise RuntimeError("Project postcheck reports joint-limit failure")
    if postcheck.get("acceleration_pass") is False:
        raise RuntimeError("Project postcheck reports acceleration failure")
    if postcheck.get("jerk_pass") is False:
        raise RuntimeError("Project postcheck reports jerk failure")

    trajectory_path = save_right_arm_npz(out / "trajectory_right_arm.npz", traj)

    report: Dict[str, Any] = {
        "schema_version": 1,
        "route": "RouteB",
        "stage": "current_to_pregrasp_right_arm_only",
        "success": True,
        "planner": {
            "action_dim": action_dim,
            "active_joint_names": list(active_names),
            "max_attempts": int(max_attempts),
            "enable_graph_attempt": int(enable_graph_attempt),
        },
        "locked_joint_contract": contract.to_report(),
        "planner_fixup": fixup_report,
        "trajectory": {
            **traj.to_report(),
            **validation,
            "artifact": str(trajectory_path),
        },
        "postcheck": postcheck,
        "invariants": {
            "true_7dof_optimizer": True,
            "posthoc_35dof_slicing_used_as_planner": False,
            "extra_quintic_interpolation": False,
            "route_a_modified": False,
        },
    }
    report_path = out / "report_right_arm.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return RightArmPlanResult(
        planner=planner,
        raw_result=raw,
        contract=contract,
        trajectory=traj,
        report=report,
        trajectory_path=trajectory_path,
        report_path=report_path,
    )
