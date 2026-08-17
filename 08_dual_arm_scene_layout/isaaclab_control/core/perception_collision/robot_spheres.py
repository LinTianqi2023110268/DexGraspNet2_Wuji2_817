from __future__ import annotations

from pathlib import Path
import numpy as np


class CuroboRobotSphereModel:
    """GPU FK adapter that converts a named robot joint state to collision spheres.

    This loader deliberately requires a cuRobo robot YAML with fitted collision
    spheres.  A raw URDF is insufficient for cuRobo collision queries.  Generate the
    YAML once with ``core/tools/build_robot_collision_model.py``; the vendor URDF and
    meshes are never modified.
    """

    def __init__(self, robot_config: Path | str, device: str = "cuda:0"):
        self.robot_config = Path(robot_config).expanduser().resolve()
        if not self.robot_config.is_file():
            raise FileNotFoundError(
                f"cuRobo robot collision config not found: {self.robot_config}. "
                "Run core/tools/build_robot_collision_model.py in curobo_v2 first."
            )
        try:
            import torch
            from curobo._src.cost.cost_self_collision import SelfCollisionCost
            from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
            from curobo.kinematics import Kinematics, KinematicsCfg
            from curobo.types import DeviceCfg, JointState
        except Exception as exc:
            raise RuntimeError(
                "cuRobo kinematics import failed; run this inside curobo_v2. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to PyTorch in curobo_v2")
        self.torch = torch
        self.JointState = JointState
        self.device_cfg = DeviceCfg(device=device, dtype=torch.float32)
        cfg = KinematicsCfg.from_robot_yaml_file(
            str(self.robot_config),
            device_cfg=self.device_cfg,
        )
        self.robot = Kinematics(cfg, compute_spheres=True)
        self.self_collision_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                self_collision_kin_config=cfg.self_collision_config,
                store_pair_distance=True,
            )
        )
        self.joint_names = tuple(self.robot.joint_names)
        self._joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.sphere_link_names = self._load_sphere_link_names()

    def _load_sphere_link_names(self) -> tuple[str, ...]:
        try:
            import yaml
            data = yaml.safe_load(self.robot_config.read_text(encoding="utf-8"))
            kin = data.get("kinematics", data)
            spheres = kin.get("collision_spheres", {})
            links = []
            for link_name in kin.get("collision_link_names", spheres.keys()):
                for _sphere in spheres.get(link_name, []) or []:
                    links.append(str(link_name))
            return tuple(links)
        except Exception:
            return tuple()

    def _spheres_tensor_from_named_joints(
        self,
        joint_positions_rad: dict[str, float],
    ):
        unknown = sorted(set(joint_positions_rad) - set(self._joint_index))
        if unknown:
            raise KeyError(f"joint names not present in cuRobo robot config: {unknown}")
        q = self.robot.default_joint_position.detach().clone()
        for name, value in joint_positions_rad.items():
            q[self._joint_index[name]] = float(value)
        q = q.view(1, -1)
        state = self.robot.compute_kinematics(
            self.JointState.from_position(q, joint_names=list(self.joint_names))
        )
        spheres = state.robot_spheres
        if spheres is None:
            raise RuntimeError(
                "cuRobo robot model returned no collision spheres; regenerate the robot YAML"
            )
        return spheres

    def spheres_from_named_joints(
        self,
        joint_positions_rad: dict[str, float],
        T_world_base: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return world-frame collision spheres ``[N,4] = x,y,z,r``.

        Unspecified active joints keep cuRobo's configured default position.  For a
        production call the Isaac Lab integration should provide every measured active
        joint by name, so static/left-arm/hand geometry is represented consistently.
        """
        spheres = self._spheres_tensor_from_named_joints(joint_positions_rad)
        spheres = spheres.detach().cpu().numpy()
        while spheres.ndim > 2 and spheres.shape[0] == 1:
            spheres = spheres[0]
        if spheres.ndim != 2 or spheres.shape[-1] != 4:
            raise RuntimeError(f"unexpected robot_spheres shape: {spheres.shape}")
        spheres = np.asarray(spheres, dtype=np.float64).copy()

        if T_world_base is not None:
            T = np.asarray(T_world_base, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"T_world_base must be 4x4, got {T.shape}")
            homogeneous = np.concatenate(
                [spheres[:, :3], np.ones((len(spheres), 1), dtype=np.float64)], axis=1
            )
            spheres[:, :3] = (T @ homogeneous.T).T[:, :3]
        return spheres

    def check_self_collision(self, joint_positions_rad: dict[str, float]) -> dict:
        """Check cuRobo self-collision using the YAML ignore matrix and buffers."""
        spheres = self._spheres_tensor_from_named_joints(joint_positions_rad)
        if spheres.ndim == 3:
            spheres = spheres[:, None, :, :]
        elif spheres.ndim != 4:
            raise RuntimeError(f"unexpected robot_spheres tensor shape: {tuple(spheres.shape)}")
        batch, horizon = int(spheres.shape[0]), int(spheres.shape[1])
        self.self_collision_cost.setup_batch_tensors(batch, horizon)
        distance = self.self_collision_cost.forward(spheres)
        pair_distance = self.self_collision_cost._pair_distance
        pair_np = pair_distance.detach().cpu().numpy()
        penetration = pair_np.reshape(-1)
        colliding = penetration > 0.0
        max_penetration = float(np.max(penetration)) if penetration.size else 0.0
        top_pairs = []
        if penetration.size:
            order = np.argsort(-penetration)[:10]
            shape = tuple(int(x) for x in pair_np.shape)
            for linear in order:
                value = float(penetration[int(linear)])
                if value <= 0.0:
                    continue
                unraveled = tuple(int(x) for x in np.unravel_index(int(linear), pair_np.shape))
                sphere_indices = [x for x in unraveled if 0 <= x < len(self.sphere_link_names)]
                links = [self.sphere_link_names[x] for x in sphere_indices[:2]]
                top_pairs.append({
                    "pair_linear_index": int(linear),
                    "pair_unraveled_index": list(unraveled),
                    "pair_distance_shape": list(shape),
                    "sphere_indices_guess": sphere_indices[:2],
                    "link_names_guess": links,
                    "penetration_m": value,
                })
        return {
            "self_collision_pass": bool(not np.any(colliding)),
            "self_collision_pair_count": int(np.count_nonzero(colliding)),
            "self_collision_max_penetration_m": max(0.0, max_penetration),
            "self_collision_cost": float(distance.detach().cpu().max().item()),
            "self_collision_top_pairs": top_pairs,
        }
