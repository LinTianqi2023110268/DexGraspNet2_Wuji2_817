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
        self.joint_names = tuple(self.robot.joint_names)
        self._joint_index = {name: i for i, name in enumerate(self.joint_names)}

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
