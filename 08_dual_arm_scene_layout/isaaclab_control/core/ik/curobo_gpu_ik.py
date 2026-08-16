from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

from ..config import IKConfig, RIGHT_ARM_NAMES


class IKSolverUnavailable(RuntimeError):
    pass


@dataclass
class BatchedIKResult:
    """All returned cuRobo seeds for a batch of 6-D targets."""

    q_rad: np.ndarray                 # [B, R, 7]
    raw_success: np.ndarray           # [B, R]
    accepted: np.ndarray              # [B, R]
    position_error_m: np.ndarray      # [B, R]
    orientation_error_rad: np.ndarray # [B, R]
    inner_limit_margin_rad: np.ndarray# [B, R]
    lower_inner_rad: np.ndarray       # [7]
    upper_inner_rad: np.ndarray       # [7]
    joint_names: tuple[str, ...]
    solve_time_s: float

    @property
    def batch_size(self) -> int:
        return int(self.q_rad.shape[0])

    @property
    def returned_solutions(self) -> int:
        return int(self.q_rad.shape[1])

    def to_jsonable(self) -> dict:
        return {
            "q_rad": self.q_rad.tolist(),
            "raw_success": self.raw_success.tolist(),
            "accepted": self.accepted.tolist(),
            "position_error_m": self.position_error_m.tolist(),
            "orientation_error_rad": self.orientation_error_rad.tolist(),
            "inner_limit_margin_rad": self.inner_limit_margin_rad.tolist(),
            "lower_inner_rad": self.lower_inner_rad.tolist(),
            "upper_inner_rad": self.upper_inner_rad.tolist(),
            "joint_names": list(self.joint_names),
            "solve_time_s": float(self.solve_time_s),
        }


def _root_link(urdf_path: Path) -> str:
    root = ET.parse(urdf_path).getroot()
    links = {x.attrib["name"] for x in root.findall("link")}
    children = {
        j.find("child").attrib["link"]
        for j in root.findall("joint")
        if j.find("child") is not None
    }
    roots = sorted(links - children)
    if len(roots) != 1:
        raise RuntimeError(f"URDF root link is not unique: {roots}")
    return roots[0]


def _right_arm_limits(urdf_path: Path, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name not in RIGHT_ARM_NAMES:
            continue
        lim = joint.find("limit")
        if lim is None or "lower" not in lim.attrib or "upper" not in lim.attrib:
            raise RuntimeError(f"{name} has no finite lower/upper limits")
        limits[name] = (float(lim.attrib["lower"]), float(lim.attrib["upper"]))
    missing = [name for name in RIGHT_ARM_NAMES if name not in limits]
    if missing:
        raise RuntimeError(f"URDF missing right-arm joints: {missing}")
    lo = np.asarray([limits[n][0] for n in RIGHT_ARM_NAMES], dtype=np.float64) + shrink
    hi = np.asarray([limits[n][1] for n in RIGHT_ARM_NAMES], dtype=np.float64) - shrink
    if np.any(lo >= hi):
        raise RuntimeError("Inner joint limits are invalid after shrink")
    return lo, hi


def _normalize_solution(a: np.ndarray, batch: int, returned: int) -> np.ndarray:
    a = np.asarray(a)
    while a.ndim > 3 and 1 in a.shape[1:-1]:
        for axis in range(1, a.ndim - 1):
            if a.shape[axis] == 1:
                a = np.squeeze(a, axis=axis)
                break
    if a.ndim == 2 and returned == 1:
        a = a[:, None, :]
    if a.ndim != 3 or a.shape[0] != batch:
        raise RuntimeError(f"Unexpected cuRobo solution shape {a.shape}; batch={batch}")
    return a


def _normalize_metric(a: np.ndarray, batch: int, returned: int, name: str) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim == 0 and batch == 1 and returned == 1:
        a = a.reshape(1, 1)
    elif a.ndim == 1 and batch == 1:
        a = a[None, :]
    elif a.ndim == 1 and returned == 1:
        a = a[:, None]
    try:
        a = a.reshape(batch, returned)
    except Exception as exc:
        raise RuntimeError(f"Unexpected cuRobo {name} shape {np.asarray(a).shape}") from exc
    return a


class CuroboGpuIK:
    """Persistent cuRobo V2 batched IK wrapper.

    The wrapper intentionally keeps scene/self collision disabled by default so the
    validated geometric IK contract remains stable.  Observed-scene collision is a
    separate gate in ``perception_collision`` and can later be moved into cuRobo's
    motion planner without changing IK selection semantics.
    """

    def __init__(self, robot_urdf: Path | str, config: IKConfig | None = None):
        self.robot_urdf = Path(robot_urdf).expanduser().resolve()
        if not self.robot_urdf.is_file():
            raise FileNotFoundError(self.robot_urdf)
        self.config = config or IKConfig()
        self.lower_inner_rad, self.upper_inner_rad = _right_arm_limits(
            self.robot_urdf, self.config.inner_limit_shrink_rad
        )
        self._setup_solver()

    def _setup_solver(self) -> None:
        try:
            import torch
            import curobo
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
            from curobo.types import DeviceCfg, GoalToolPose, Pose
            from curobo._src.types.robot import RobotCfg
        except Exception as exc:
            raise IKSolverUnavailable(
                "cuRobo V2 import failed. Run this module inside the dedicated "
                "curobo_v2 conda environment; do not install cuRobo into Isaac Lab. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise IKSolverUnavailable("CUDA is not visible to PyTorch in curobo_v2")

        self.torch = torch
        self.curobo = curobo
        self.GoalToolPose = GoalToolPose
        self.Pose = Pose
        device_cfg = DeviceCfg(device=self.config.device, dtype=torch.float32)
        robot_cfg = RobotCfg.from_basic(
            urdf_path=str(self.robot_urdf),
            base_link=_root_link(self.robot_urdf),
            tool_frames=[self.config.tool_frame],
            device_cfg=device_cfg,
            load_dynamics=False,
        )
        cfg = InverseKinematicsCfg.create(
            robot=robot_cfg,
            self_collision_check=self.config.self_collision_check,
            load_collision_spheres=self.config.load_collision_spheres,
            num_seeds=self.config.num_seeds,
            seed_solver_num_seeds=self.config.num_seeds,
            position_tolerance=self.config.position_tolerance_m,
            orientation_tolerance=self.config.orientation_tolerance_rad,
            use_cuda_graph=self.config.use_cuda_graph,
            random_seed=self.config.random_seed,
            success_requires_convergence=True,
            max_batch_size=self.config.batch_size,
        )
        self.ik = InverseKinematics(cfg)
        names = tuple(getattr(self.ik, "joint_names", self.ik.kinematics.joint_names))
        if names != RIGHT_ARM_NAMES:
            raise RuntimeError(
                "cuRobo active-joint order does not match the project contract: "
                f"expected={RIGHT_ARM_NAMES}, actual={names}"
            )
        self.joint_names = names

    def _solve_chunk(self, targets: np.ndarray, returned: int) -> BatchedIKResult:
        torch = self.torch
        B = int(len(targets))
        mats = torch.as_tensor(targets, device=self.config.device, dtype=torch.float32)
        poses = self.Pose.from_matrix(mats)
        goal = self.GoalToolPose.from_poses(
            {self.config.tool_frame: poses},
            ordered_tool_frames=[self.config.tool_frame],
            num_goalset=1,
        )
        torch.cuda.synchronize()
        result = self.ik.solve_pose(goal_tool_poses=goal, return_seeds=returned)
        torch.cuda.synchronize()

        to_np = lambda x: x.detach().cpu().numpy()
        q = _normalize_solution(to_np(result.solution), B, returned).astype(np.float64)
        success = _normalize_metric(to_np(result.success).astype(bool), B, returned, "success")
        pos = _normalize_metric(to_np(result.position_error), B, returned, "position_error").astype(np.float64)
        rot = _normalize_metric(to_np(result.rotation_error), B, returned, "rotation_error").astype(np.float64)
        margin = np.minimum(
            q - self.lower_inner_rad[None, None, :],
            self.upper_inner_rad[None, None, :] - q,
        ).min(axis=-1)
        accepted = (
            success
            & (pos <= self.config.position_tolerance_m)
            & (rot <= self.config.orientation_tolerance_rad)
            & (margin >= self.config.minimum_inner_limit_margin_rad)
        )
        return BatchedIKResult(
            q_rad=q,
            raw_success=success,
            accepted=accepted,
            position_error_m=pos,
            orientation_error_rad=rot,
            inner_limit_margin_rad=margin,
            lower_inner_rad=self.lower_inner_rad.copy(),
            upper_inner_rad=self.upper_inner_rad.copy(),
            joint_names=self.joint_names,
            solve_time_s=float(getattr(result, "solve_time", 0.0)),
        )

    def solve(self, target_matrices_base: np.ndarray, return_seeds: int | None = None) -> BatchedIKResult:
        targets = np.asarray(target_matrices_base, dtype=np.float64)
        if targets.ndim == 2:
            targets = targets[None, ...]
        if targets.ndim != 3 or targets.shape[1:] != (4, 4):
            raise ValueError(f"Expected [B,4,4] target matrices, got {targets.shape}")
        returned = int(return_seeds or self.config.return_seeds)
        returned = max(1, min(returned, self.config.num_seeds))

        parts: list[BatchedIKResult] = []
        for start in range(0, len(targets), self.config.batch_size):
            parts.append(self._solve_chunk(targets[start:start + self.config.batch_size], returned))
        if len(parts) == 1:
            return parts[0]
        return BatchedIKResult(
            q_rad=np.concatenate([x.q_rad for x in parts], axis=0),
            raw_success=np.concatenate([x.raw_success for x in parts], axis=0),
            accepted=np.concatenate([x.accepted for x in parts], axis=0),
            position_error_m=np.concatenate([x.position_error_m for x in parts], axis=0),
            orientation_error_rad=np.concatenate([x.orientation_error_rad for x in parts], axis=0),
            inner_limit_margin_rad=np.concatenate([x.inner_limit_margin_rad for x in parts], axis=0),
            lower_inner_rad=self.lower_inner_rad.copy(),
            upper_inner_rad=self.upper_inner_rad.copy(),
            joint_names=self.joint_names,
            solve_time_s=float(sum(x.solve_time_s for x in parts)),
        )

    def warmup(self, target_matrix_base: np.ndarray, runs: int = 2) -> None:
        target = np.asarray(target_matrix_base, dtype=np.float64).reshape(1, 4, 4)
        for _ in range(max(0, int(runs))):
            self.solve(target, return_seeds=min(self.config.return_seeds, self.config.num_seeds))
        if hasattr(self.ik, "reset_seed"):
            self.ik.reset_seed()
        self.torch.cuda.empty_cache()

    @property
    def version(self) -> str:
        return str(getattr(self.curobo, "__version__", "unknown"))
