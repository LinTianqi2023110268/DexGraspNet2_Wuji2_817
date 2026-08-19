from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from core.config import MapperConfig, RIGHT_ARM_NAMES
from core.perception_collision import CuroboRGBDMapper, RGBDFrame


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_FILE = (
    DEFAULT_PROJECT_ROOT
    / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
)
DEFAULT_LAYOUT_JSON = (
    DEFAULT_PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
)
DEFAULT_ENABLE_GRAPH_ATTEMPT = 1_000_000


@dataclass
class RouteBPlanResult:
    success: bool
    trajectory_q_rad: np.ndarray
    joint_names: list[str]
    start_q_rad: np.ndarray
    goal_q_rad: np.ndarray
    planning_time_s: float
    solve_time_s: float | None
    total_time_s: float | None
    waypoint_count: int
    self_collision_check: bool
    esdf_collision_constraint_enabled: bool
    message: str
    joint_state_sanitization: dict[str, Any] | None = None

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "route": "RouteB",
            "stage": "current_to_pregrasp",
            "success": bool(self.success),
            "planning_time_s": float(self.planning_time_s),
            "solve_time_s": self.solve_time_s,
            "total_time_s": self.total_time_s,
            "trajectory_point_count": int(self.waypoint_count),
            "joint_names": self.joint_names,
            "start_q_rad": self.start_q_rad.tolist(),
            "goal_q_rad": self.goal_q_rad.tolist(),
            "joint_limit_check": "delegated_to_curobo_motion_planner",
            "self_collision_check": bool(self.self_collision_check),
            "esdf_collision_constraint_enabled": bool(self.esdf_collision_constraint_enabled),
            "collision_policy": {
                "environment_collision": bool(self.esdf_collision_constraint_enabled),
                "self_collision": bool(self.self_collision_check),
            },
            "motion_planner_graph_seed": {
                "enable_graph_attempt": None,
                "note": "set by RouteBMotionPlannerAdapter report wrapper",
            },
            "message": self.message,
            "joint_state_sanitization": self.joint_state_sanitization,
        }


class RouteBMotionPlannerAdapter:
    """Thin Route B backend for cuRobo MotionPlanner.

    Phase 1 only owns the arm trajectory from the current state to an already
    solved PREGRASP q.  Existing Route A code remains responsible for PREGRASP
    sampling/IK, strict COVER IK, Wuji2 fingers, and Isaac execution.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        route_cfg = dict(self.config.get("routeB", self.config))
        self.route_cfg = route_cfg
        self.device = str(route_cfg.get("device", "cuda:0"))
        self.robot_file = Path(route_cfg.get("robot_file", DEFAULT_ROBOT_FILE)).expanduser()
        if not self.robot_file.is_absolute():
            self.robot_file = (DEFAULT_PROJECT_ROOT / self.robot_file).resolve()
        self.layout_json = Path(route_cfg.get("layout_json", DEFAULT_LAYOUT_JSON)).expanduser()
        if not self.layout_json.is_absolute():
            self.layout_json = (DEFAULT_PROJECT_ROOT / self.layout_json).resolve()
        self.use_cuda_graph = bool(route_cfg.get("use_cuda_graph", False))
        collision_cfg = dict(route_cfg.get("collision", {}))
        self.environment_collision = bool(collision_cfg.get("environment_collision", True))
        self.self_collision_check = bool(
            collision_cfg.get("self_collision", route_cfg.get("self_collision_check", False))
        )
        if not self.environment_collision:
            raise ValueError("Route B contract requires collision.environment_collision=true")
        self.num_ik_seeds = int(route_cfg.get("num_ik_seeds", 32))
        self.num_trajopt_seeds = int(route_cfg.get("num_trajopt_seeds", 4))
        self.max_attempts = int(route_cfg.get("max_attempts", 2))
        self.enable_graph_attempt = int(route_cfg.get("enable_graph_attempt", DEFAULT_ENABLE_GRAPH_ATTEMPT))
        self.interpolation_dt_s = float(route_cfg.get("interpolation_dt_s", 0.025))
        self.warmup_iterations = int(route_cfg.get("warmup_iterations", 1))
        self.planner = None
        self.scene = None
        self.last_scene_report: dict[str, Any] = {}
        self.last_collision_policy_report: dict[str, Any] = {}
        self.last_voxel_shape_contract_report: dict[str, Any] = {}
        self.last_joint_state_sanitization_report: dict[str, Any] = {}
        self.joint_names: list[str] = []

    def build_pick_scene(
        self,
        filtered_depth_path: Path | str,
        intrinsics_path: Path | str,
        camera_pose_path: Path | str,
    ):
        """Build a cuRobo SceneCfg from RobotSegmenter-filtered depth."""
        mapper_cfg = self._mapper_config()
        frame = RGBDFrame.from_npy(
            filtered_depth_path,
            intrinsics_path,
            camera_pose_path,
        )
        valid_input = np.isfinite(frame.depth_m) & (frame.depth_m > 0.0)
        frame = RGBDFrame(
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            T_world_camera=self._base_from_world() @ frame.T_world_camera,
            target_mask=frame.target_mask,
        ).validated()
        observed = CuroboRGBDMapper(mapper_cfg).build(frame)
        SceneCfg = self._scene_cfg_type()
        self.scene = SceneCfg(voxel=[observed.scene_grid])
        self.last_scene_report = {
            "scene_frame": "arm_base_link",
            "input_filtered_depth": str(Path(filtered_depth_path).expanduser().resolve()),
            "valid_depth_pixels": int(np.count_nonzero(valid_input)),
            "grid_center_base_m": observed.grid_center_world.tolist(),
            "extent_meters_xyz": observed.extent_meters_xyz.tolist(),
            "voxel_size_m": float(mapper_cfg.voxel_size_m),
            "esdf_voxel_size_m": float(mapper_cfg.esdf_voxel_size_m),
            "T_base_camera_contract": "adapter computes T_base_camera = inv(T_world_base) @ T_world_camera",
        }
        return self.scene

    def create_planner(self, scene=None):
        if scene is None:
            scene = self.scene
        if scene is None:
            raise RuntimeError("Route B scene is not built; call build_pick_scene() first")
        if not self.robot_file.is_file():
            raise FileNotFoundError(self.robot_file)

        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        device_cfg = DeviceCfg(device=torch.device(self.device), dtype=torch.float32)
        cfg = MotionPlannerCfg.create(
            robot=str(self.robot_file),
            scene_model=scene,
            device_cfg=device_cfg,
            self_collision_check=self.self_collision_check,
            num_ik_seeds=self.num_ik_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            use_cuda_graph=self.use_cuda_graph,
            interpolation_dt=self.interpolation_dt_s,
        )
        self.last_collision_policy_report = self._apply_collision_policy_to_motion_cfg(cfg)
        self.planner = MotionPlanner(cfg)
        self.last_voxel_shape_contract_report = self._normalize_voxel_shape_contract(
            self.planner,
            scene,
        )
        self.planner.warmup(
            enable_graph=self.use_cuda_graph,
            num_warmup_iterations=self.warmup_iterations,
        )
        self.joint_names = [str(name) for name in self.planner.joint_names]
        return self.planner

    def plan_current_to_pregrasp(
        self,
        q_current,
        q_pregrasp,
        scene=None,
    ) -> RouteBPlanResult:
        """Plan a cuRobo MotionPlanner C-space path to PREGRASP."""
        if self.planner is None:
            self.create_planner(scene)
        if self.planner is None:
            raise RuntimeError("MotionPlanner was not created")

        current_q_raw = self._coerce_q(q_current)
        goal_q_raw = self._coerce_q(q_pregrasp, base_q=current_q_raw)
        current_q, goal_q, sanitization_report = self.sanitize_planning_joint_states(
            current_q_raw,
            goal_q_raw,
        )
        self.last_joint_state_sanitization_report = sanitization_report

        import torch
        from curobo.types import JointState

        current_state = JointState.from_position(
            torch.as_tensor(current_q, device=self.device, dtype=torch.float32).reshape(1, -1),
            joint_names=self.joint_names,
        )
        goal_state = JointState.from_position(
            torch.as_tensor(goal_q, device=self.device, dtype=torch.float32).reshape(1, -1),
            joint_names=self.joint_names,
        )

        t0 = time.time()
        result = self.planner.plan_cspace(
            goal_state=goal_state,
            current_state=current_state,
            max_attempts=self.max_attempts,
            enable_graph_attempt=self.enable_graph_attempt,
        )
        planning_time_s = time.time() - t0

        if result is None:
            return RouteBPlanResult(
                success=False,
                trajectory_q_rad=np.empty((0, len(self.joint_names)), dtype=np.float32),
                joint_names=self.joint_names,
                start_q_rad=current_q,
                goal_q_rad=goal_q,
                planning_time_s=planning_time_s,
                solve_time_s=None,
                total_time_s=None,
                waypoint_count=0,
                self_collision_check=self.self_collision_check,
                esdf_collision_constraint_enabled=True,
                message="MotionPlanner returned None",
                joint_state_sanitization=sanitization_report,
            )

        success = bool(result.success.detach().cpu().bool().any().item())
        trajectory = np.empty((0, len(self.joint_names)), dtype=np.float32)
        if success:
            interpolated = result.get_interpolated_plan()
            pos = interpolated.position.detach().cpu().numpy()
            trajectory = np.asarray(pos, dtype=np.float32).reshape(-1, pos.shape[-1])

        return RouteBPlanResult(
            success=success,
            trajectory_q_rad=trajectory,
            joint_names=self.joint_names,
            start_q_rad=current_q,
            goal_q_rad=goal_q,
            planning_time_s=planning_time_s,
            solve_time_s=self._float_attr(result, "solve_time"),
            total_time_s=self._float_attr(result, "total_time"),
            waypoint_count=int(trajectory.shape[0]),
            self_collision_check=self.self_collision_check,
            esdf_collision_constraint_enabled=True,
            message="PASS" if success else "MotionPlanner did not find a successful trajectory",
            joint_state_sanitization=sanitization_report,
        )

    def build_carry_scene(self, *args, **kwargs):
        raise NotImplementedError("Route B carry scene is reserved for the next phase")

    def plan_grasp_to_place(self, *args, **kwargs):
        raise NotImplementedError("Route B grasp-to-place planning is reserved for the next phase")

    def _mapper_config(self) -> MapperConfig:
        mapper_cfg = dict(self.route_cfg.get("mapper", {}))
        return MapperConfig(
            device=self.device,
            voxel_size_m=float(mapper_cfg.get("voxel_size_m", 0.01)),
            esdf_voxel_size_m=float(mapper_cfg.get("esdf_voxel_size_m", 0.02)),
            depth_min_m=float(mapper_cfg.get("depth_min_m", 0.05)),
            depth_max_m=float(mapper_cfg.get("depth_max_m", 3.0)),
        )

    def _base_from_world(self) -> np.ndarray:
        import json

        layout = json.loads(self.layout_json.read_text(encoding="utf-8"))
        # Stored as OpenUSD row-vector Gf.Matrix4d. Project code uses
        # column-vector T_A_B, so transpose before inversion.
        T_world_base = np.asarray(
            layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
            dtype=np.float64,
        ).T
        return np.linalg.inv(T_world_base).astype(np.float32)

    @staticmethod
    def _scene_cfg_type():
        # cuRobo 0.8.x exposes SceneCfg only through _src.
        from curobo._src.geom.types import SceneCfg

        return SceneCfg

    def _apply_collision_policy_to_motion_cfg(self, cfg) -> dict[str, Any]:
        """Force Route B collision policy into all cuRobo rollout configs.

        In cuRobo 0.8.x, ``MotionPlannerCfg.create(self_collision_check=False)``
        disables optimizer rollout self-collision costs, but graph-planner
        start/end feasibility uses a separate metrics rollout. Route B therefore
        explicitly disables self-collision weights on IK, TrajOpt, and Graph
        rollout configs before constructing ``MotionPlanner``.

        This does not touch scene collision config, so robot-vs-ESDF collision
        remains enabled.
        """
        rollout_entries: list[dict[str, Any]] = []

        def visit_rollout(label: str, rollout_cfg) -> None:
            if rollout_cfg is None:
                return
            for i, cost_cfg in enumerate(rollout_cfg.get_cost_manager_configs()):
                before = self._self_collision_weight_snapshot(cost_cfg)
                if not self.self_collision_check:
                    cost_cfg.disable_self_collision()
                after = self._self_collision_weight_snapshot(cost_cfg)
                rollout_entries.append(
                    {
                        "label": f"{label}.cost_manager[{i}]",
                        "self_collision_weight_before": before,
                        "self_collision_weight_after": after,
                        "self_collision_disabled": bool(after in (None, 0.0)),
                    }
                )

        def visit_solver(label: str, solver_cfg) -> None:
            core = getattr(solver_cfg, "core_cfg", None)
            if core is None:
                return
            visit_rollout(f"{label}.metrics_rollout", getattr(core, "metrics_rollout_config", None))
            for j, rollout_cfg in enumerate(getattr(core, "optimizer_rollout_configs", []) or []):
                visit_rollout(f"{label}.optimizer_rollout[{j}]", rollout_cfg)

        visit_solver("ik", getattr(cfg, "ik_solver_config", None))
        visit_solver("trajopt", getattr(cfg, "trajopt_solver_config", None))
        graph_cfg = getattr(cfg, "graph_planner_config", None)
        visit_rollout("graph.metrics_rollout", getattr(graph_cfg, "rollout_config", None))

        return {
            "policy": {
                "environment_collision": bool(self.environment_collision),
                "self_collision": bool(self.self_collision_check),
            },
            "environment_scene_collision_cfg_present": bool(
                getattr(cfg, "scene_collision_cfg", None) is not None
            ),
            "self_collision_rollout_entries": rollout_entries,
            "all_self_collision_rollouts_disabled": bool(
                (not self.self_collision_check)
                and rollout_entries
                and all(entry["self_collision_disabled"] for entry in rollout_entries)
            ),
            "curobo_api_location": {
                "factory": "curobo._src.motion.motion_planner_cfg.MotionPlannerCfg.create",
                "optimizer_disable": "curobo._src.solver.solver_core_cfg.create_rollout_configs -> RobotCostManagerCfg.disable_self_collision",
                "graph_start_end_check": "curobo._src.graph_planner.graph_planner_prm.PRMGraphPlanner.check_samples_feasibility",
                "routeB_fix": "RouteBMotionPlannerAdapter._apply_collision_policy_to_motion_cfg",
            },
        }

    def _normalize_voxel_shape_contract(self, planner, scene) -> dict[str, Any]:
        """Force cuRobo VoxelData discrete counts to match feature tensor shape.

        cuRobo 0.8.x stores VoxelData dimensions in ``params[..., 0:3]`` as
        float32.  If the values are produced from continuous extents, a value
        such as ``26.999998`` is truncated by Warp's ``wp.int32()`` to ``26``,
        corrupting X-slowest/Z-fastest flatten indexing for a tensor whose
        authoritative shape is ``[nx, ny, nz] == [36, 55, 27]``.

        Route B does not change VoxelGrid pose, VoxelData inv_pose, VoxelData
        dims, voxel_size, features, ESDF values, thresholds, or planner params.
        It only writes exact discrete voxel counts into ``VoxelData.params``.
        """
        import torch

        voxel_grids = list(getattr(scene, "voxel", []) or [])
        if len(voxel_grids) != 1:
            raise RuntimeError(f"Route B expects exactly one scene voxel grid, got {len(voxel_grids)}")
        scene_grid = voxel_grids[0]
        feature = getattr(scene_grid, "feature_tensor", None)
        if feature is None or len(tuple(feature.shape)) != 3:
            raise RuntimeError(
                "Route B VoxelGrid feature_tensor must have shape [nx, ny, nz], "
                f"got {None if feature is None else tuple(feature.shape)}"
            )
        expected_shape = [int(x) for x in tuple(feature.shape)]
        expected_count = int(np.prod(expected_shape))

        entries = self._find_curobo_voxel_data_entries(planner)
        if not entries:
            raise RuntimeError("Route B could not find cuRobo SceneCollision VoxelData to normalize")

        reports: list[dict[str, Any]] = []
        seen: set[int] = set()
        for label, vox in entries:
            key = id(vox)
            if key in seen:
                continue
            seen.add(key)

            params = getattr(vox, "params", None)
            dims = getattr(vox, "dims", None)
            inv_pose = getattr(vox, "inv_pose", None)
            features = getattr(vox, "features", None)
            if params is None or dims is None or inv_pose is None or features is None:
                raise RuntimeError(f"VoxelData at {label} is missing params/dims/inv_pose/features")

            params_before_t = params.detach().clone()
            dims_before_t = dims.detach().clone()
            inv_pose_before_t = inv_pose.detach().clone()
            features_before_t = features.detach().clone()
            params_before = params_before_t.detach().cpu().numpy().tolist()
            dims_before = dims_before_t.detach().cpu().numpy().tolist()
            inv_pose_before = inv_pose_before_t.detach().cpu().numpy().tolist()

            if params.shape[-1] < 4:
                raise RuntimeError(f"VoxelData.params at {label} has invalid shape {tuple(params.shape)}")
            feature_count = int(features.shape[-2]) if features.ndim >= 3 else int(features.numel())
            if feature_count != expected_count:
                raise RuntimeError(
                    f"VoxelData feature count mismatch at {label}: "
                    f"features count {feature_count} != expected {expected_count}"
                )

            expected_t = torch.as_tensor(
                expected_shape,
                device=params.device,
                dtype=params.dtype,
            )
            params[..., 0:3] = expected_t

            params_after_t = params.detach().clone()
            params_after = params_after_t.detach().cpu().numpy().tolist()
            params_after_np = params_after_t.detach().cpu().numpy()
            if not np.all(params_after_np[..., 0].astype(np.int64) == expected_shape[0]):
                raise RuntimeError(f"VoxelData.params x count did not normalize to {expected_shape[0]}")
            if not np.all(params_after_np[..., 1].astype(np.int64) == expected_shape[1]):
                raise RuntimeError(f"VoxelData.params y count did not normalize to {expected_shape[1]}")
            if not np.all(params_after_np[..., 2].astype(np.int64) == expected_shape[2]):
                raise RuntimeError(f"VoxelData.params z count did not normalize to {expected_shape[2]}")
            if not torch.equal(dims, dims_before_t):
                raise RuntimeError("VoxelData.dims changed while normalizing params; aborting")
            if not torch.equal(inv_pose, inv_pose_before_t):
                raise RuntimeError("VoxelData.inv_pose changed while normalizing params; aborting")
            if not torch.equal(features, features_before_t):
                raise RuntimeError("VoxelData.features changed while normalizing params; aborting")

            reports.append(
                {
                    "label": label,
                    "feature_shape": expected_shape,
                    "params_before": params_before,
                    "params_after": params_after,
                    "feature_count": feature_count,
                    "expected_feature_count": expected_count,
                    "pose_unchanged": True,
                    "dims_unchanged": True,
                    "features_unchanged": True,
                    "dims_before": dims_before,
                    "dims_after": dims.detach().cpu().numpy().tolist(),
                    "inv_pose_before": inv_pose_before,
                    "inv_pose_after": inv_pose.detach().cpu().numpy().tolist(),
                    "applied": True,
                }
            )

        first = reports[0]
        return {
            "feature_shape": first["feature_shape"],
            "params_before": first["params_before"],
            "params_after": first["params_after"],
            "feature_count": first["feature_count"],
            "expected_feature_count": first["expected_feature_count"],
            "pose_unchanged": all(x["pose_unchanged"] for x in reports),
            "dims_unchanged": all(x["dims_unchanged"] for x in reports),
            "features_unchanged": all(x["features_unchanged"] for x in reports),
            "applied": True,
            "entries": reports,
        }

    @staticmethod
    def _find_curobo_voxel_data_entries(root) -> list[tuple[str, Any]]:
        entries: list[tuple[str, Any]] = []
        seen: set[int] = set()

        def visit(label: str, obj, depth: int) -> None:
            if obj is None or depth > 8:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)
            cfg = getattr(obj, "config", None)
            checker = getattr(cfg, "scene_collision_checker", None) if cfg is not None else None
            vox = getattr(getattr(checker, "data", None), "voxels", None)
            if vox is not None:
                entries.append((label, vox))

            if isinstance(obj, dict):
                for key, value in obj.items():
                    visit(f"{label}.{key}", value, depth + 1)
                return
            if isinstance(obj, (list, tuple)):
                for i, value in enumerate(obj):
                    visit(f"{label}[{i}]", value, depth + 1)
                return
            if hasattr(obj, "detach"):
                return
            if isinstance(obj, (str, bytes, int, float, bool, Path)):
                return
            data = getattr(obj, "__dict__", None)
            if not data:
                return
            for key, value in data.items():
                if key.startswith("_") and key not in {"_costs", "_cost_manager"}:
                    continue
                visit(f"{label}.{key}", value, depth + 1)

        visit("planner", root, 0)
        return entries

    def sanitize_planning_joint_states(
        self,
        q_current: np.ndarray,
        q_goal: np.ndarray,
        *,
        numerical_tolerance_rad: float = 1.0e-5,
        interior_margin_rad: float = 1.0e-6,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Correct only tiny numeric joint-bound residuals before Route B planning.

        This is deliberately not a general clamp.  Any violation larger than
        ``numerical_tolerance_rad`` raises immediately.
        """
        if self.planner is None:
            raise RuntimeError("planner must exist before joint-state sanitization")
        lower, upper = self._motion_planner_position_bounds()
        current = np.asarray(q_current, dtype=np.float32).reshape(-1).copy()
        goal = np.asarray(q_goal, dtype=np.float32).reshape(-1).copy()
        if current.shape[0] != len(self.joint_names) or goal.shape[0] != len(self.joint_names):
            raise RuntimeError(
                "Route B sanitization expects full joint vectors matching planner.joint_names"
            )
        corrections: list[dict[str, Any]] = []
        large_violations: list[dict[str, Any]] = []

        def group(name: str) -> str:
            return "active_right_arm" if name in RIGHT_ARM_NAMES else "static_other_dof"

        def apply_one(label: str, q: np.ndarray) -> None:
            for i, name in enumerate(self.joint_names):
                before = float(q[i])
                lo = float(lower[i])
                hi = float(upper[i])
                violation = 0.0
                reason = None
                after = before
                if before < lo:
                    violation = lo - before
                    reason = "numerical_lower_bound_residual"
                    if violation <= numerical_tolerance_rad:
                        after = lo + interior_margin_rad
                    else:
                        large_violations.append(
                            {
                                "state": label,
                                "joint_name": name,
                                "group": group(name),
                                "before": before,
                                "lower": lo,
                                "upper": hi,
                                "violation_before": violation,
                                "reason": "lower_bound_violation_exceeds_tolerance",
                            }
                        )
                        continue
                elif before > hi:
                    violation = before - hi
                    reason = "numerical_upper_bound_residual"
                    if violation <= numerical_tolerance_rad:
                        after = hi - interior_margin_rad
                    else:
                        large_violations.append(
                            {
                                "state": label,
                                "joint_name": name,
                                "group": group(name),
                                "before": before,
                                "lower": lo,
                                "upper": hi,
                                "violation_before": violation,
                                "reason": "upper_bound_violation_exceeds_tolerance",
                            }
                        )
                        continue
                if after != before:
                    q[i] = np.float32(after)
                    corrections.append(
                        {
                            "state": label,
                            "joint_name": name,
                            "group": group(name),
                            "before": before,
                            "after": float(q[i]),
                            "lower": lo,
                            "upper": hi,
                            "violation_before": float(violation),
                            "correction": reason,
                        }
                    )

        apply_one("q_current_planning", current)
        apply_one("q_pregrasp_planning", goal)

        report = {
            "tolerance_rad": float(numerical_tolerance_rad),
            "interior_margin_rad": float(interior_margin_rad),
            "correction_count": int(len(corrections)),
            "corrections": corrections,
            "large_violation_count": int(len(large_violations)),
            "large_violations": large_violations,
            "max_original_violation_rad": float(
                max([abs(x["violation_before"]) for x in corrections + large_violations] or [0.0])
            ),
            "q_current_planning_diff_max_abs_rad": float(np.max(np.abs(current - q_current))),
            "q_pregrasp_planning_diff_max_abs_rad": float(np.max(np.abs(goal - q_goal))),
        }
        if large_violations:
            raise RuntimeError(
                "Route B joint-state sanitization found violations above tolerance: "
                + repr(large_violations)
            )
        return current, goal, report

    def _motion_planner_position_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._motion_planner_position_bounds_for(self.planner)

    @staticmethod
    def _motion_planner_position_bounds_for(planner) -> tuple[np.ndarray, np.ndarray]:
        bounds = getattr(planner.trajopt_solver.metrics_rollout, "state_bounds", None)
        if bounds is None:
            bounds = getattr(planner.trajopt_solver.metrics_rollout, "action_bounds", None)
        if bounds is None:
            raise RuntimeError("Route B could not read MotionPlanner position bounds")
        if hasattr(bounds, "detach"):
            bounds_np = bounds.detach().cpu().numpy()
        else:
            bounds_np = np.asarray(bounds)
        bounds_np = np.asarray(bounds_np, dtype=np.float64)
        joint_names = [str(name) for name in getattr(planner, "joint_names", [])]
        if bounds_np.shape[0] != 2 or bounds_np.shape[1] < len(joint_names):
            raise RuntimeError(
                f"unexpected MotionPlanner bounds shape {bounds_np.shape}; "
                f"joint count is {len(joint_names)}"
            )
        return (
            bounds_np[0, : len(joint_names)],
            bounds_np[1, : len(joint_names)],
        )

    @staticmethod
    def _self_collision_weight_snapshot(cost_cfg) -> float | None:
        self_cfg = getattr(cost_cfg, "self_collision_cfg", None)
        if self_cfg is None:
            return None
        weight = getattr(self_cfg, "weight", None)
        if weight is None:
            return None
        try:
            if hasattr(weight, "detach"):
                return float(weight.detach().cpu().max().item())
            arr = np.asarray(weight, dtype=np.float64)
            return float(arr.max())
        except Exception:
            return None

    @staticmethod
    def _float_attr(obj: Any, name: str) -> float | None:
        value = getattr(obj, name, None)
        if value is None:
            return None
        try:
            return float(value)
        except TypeError:
            try:
                return float(value.detach().cpu().item())
            except Exception:
                return None

    def _coerce_q(self, q, *, base_q: np.ndarray | None = None) -> np.ndarray:
        if not self.joint_names:
            raise RuntimeError("planner joint names are not available")
        if isinstance(q, dict):
            missing = [name for name in self.joint_names if name not in q]
            if missing:
                raise KeyError("missing joint values for MotionPlanner: " + ", ".join(missing))
            return np.asarray([float(q[name]) for name in self.joint_names], dtype=np.float32)

        arr = np.asarray(q, dtype=np.float32).reshape(-1)
        if arr.shape[0] == len(self.joint_names):
            return arr.copy()
        if arr.shape[0] == len(RIGHT_ARM_NAMES):
            if base_q is None:
                base_q = np.asarray(
                    self.planner.default_joint_state.position.detach().cpu().numpy(),
                    dtype=np.float32,
                ).reshape(-1)
            out = np.asarray(base_q, dtype=np.float32).reshape(-1).copy()
            name_to_index = {name: i for i, name in enumerate(self.joint_names)}
            for value, name in zip(arr, RIGHT_ARM_NAMES):
                if name not in name_to_index:
                    raise KeyError(f"planner joint_names does not contain {name}")
                out[name_to_index[name]] = float(value)
            return out
        raise ValueError(
            f"q must have length {len(RIGHT_ARM_NAMES)} or {len(self.joint_names)}, got {arr.shape[0]}"
        )
