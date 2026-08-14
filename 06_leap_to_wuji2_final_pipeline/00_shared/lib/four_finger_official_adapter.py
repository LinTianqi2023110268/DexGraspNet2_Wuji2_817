"""Minimal correspondence adapter around the official wuji-retargeting solver.

The official coordinate preprocessing, RobotWrapper, NLopt solver, Huber loss,
regularizers and analytical gradients remain in the vendor implementation.
This class changes only two contracts requested for the LEAP four-finger hand:

1. Wuji2 thumb target landmarks are proximal_abd, middle and tip (the distal
   box landmark is skipped).
2. During the official four-finger solve, Wuji2 pinky joints are fixed at their
   configured neutral values.  The pinky cannot affect the active solution.
   A separate audited post-step may then copy the ring flexion chain and add a
   small outward side-sway bias; that does not alter the official solve.
"""

from __future__ import annotations

import numpy as np

from wuji_retargeting.opt.adaptive_analytical import AdaptiveOptimizerAnalytical


class FourFingerOfficialAdaptiveOptimizer(AdaptiveOptimizerAnalytical):
    """Official analytical optimizer with an explicit four-finger contract."""

    def __init__(self, config: dict):
        self.adapter_config = dict(config.get("four_finger_adapter") or {})
        if self.adapter_config.get("active_fingers") != [
            "thumb", "index", "middle", "ring"
        ]:
            raise ValueError("active_fingers must be thumb/index/middle/ring")
        super().__init__(config)

        fixed_by_name = dict(
            self.adapter_config.get("fixed_pinky_joint_positions_rad") or {}
        )
        optimizer_order = list(self.robot.dof_joint_names)
        expected_pinky = {
            "r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip"
        }
        if set(fixed_by_name) != expected_pinky:
            raise ValueError(
                "fixed pinky contract mismatch: "
                f"expected={sorted(expected_pinky)}, got={sorted(fixed_by_name)}"
            )
        self.fixed_pinky_indices = np.asarray(
            [optimizer_order.index(name) for name in fixed_by_name], dtype=np.int64
        )
        self.fixed_pinky_values = np.asarray(
            [fixed_by_name[optimizer_order[index]] for index in self.fixed_pinky_indices],
            dtype=np.float64,
        )
        limits = np.asarray(self.robot.joint_limits, dtype=np.float64)
        lower = limits[:, 0].copy()
        upper = limits[:, 1].copy()
        for index, value in zip(self.fixed_pinky_indices, self.fixed_pinky_values):
            if not lower[index] <= value <= upper[index]:
                raise ValueError(
                    f"fixed pinky value {value} outside [{lower[index]}, {upper[index]}]"
                )
            lower[index] = value
            upper[index] = value
        self.opt.set_lower_bounds(lower.tolist())
        self.opt.set_upper_bounds(upper.tolist())

    def _resolve_link_names(self, config: dict):
        super()._resolve_link_names(config)
        thumb = dict(
            (config.get("four_finger_adapter") or {}).get("thumb_robot_landmarks")
            or {}
        )
        required = {"pip_role_link", "dip_role_link", "tip_role_link"}
        if set(thumb) != required:
            raise ValueError(f"thumb_robot_landmarks must contain {sorted(required)}")
        self.link3_names[0] = str(thumb["pip_role_link"])
        self.link4_names[0] = str(thumb["dip_role_link"])
        self.task_link_names[0] = str(thumb["tip_role_link"])

    def _resolve_flex_indices(self):
        """Keep official physical PIP/DIP penalties on the physical joints.

        Changing the thumb correspondence landmarks must not accidentally move
        the official hyperextension/coupling penalties onto CMC joints.
        """
        physical = dict(self.adapter_config.get("thumb_physical_constraint_links") or {})
        if set(physical) != {"pip", "dip"}:
            raise ValueError("thumb_physical_constraint_links must contain pip/dip")
        pip_names = list(self.link3_names)
        dip_names = list(self.link4_names)
        pip_names[0] = str(physical["pip"])
        dip_names[0] = str(physical["dip"])
        pip_idx = [self.robot.get_actuated_qpos_index(name) for name in pip_names]
        dip_idx = [self.robot.get_actuated_qpos_index(name) for name in dip_names]
        combined = pip_idx + dip_idx
        if len(set(combined)) != len(combined):
            raise RuntimeError(f"physical PIP/DIP indices are not unique: {combined}")
        self._pip_idx = np.asarray(pip_idx, dtype=np.int64)
        self._dip_idx = np.asarray(dip_idx, dtype=np.int64)
        self._flex_idx = np.asarray(sorted(combined), dtype=np.int64)

    def _compute_pinch_alpha(self, mediapipe_keypoints: np.ndarray) -> np.ndarray:
        """Use only the three real opposing LEAP fingers for pinch activation."""
        thumb_tip = mediapipe_keypoints[self.MP_TIP_INDICES[0]]
        real_tips = mediapipe_keypoints[self.MP_TIP_INDICES[1:4]]
        distances_cm = np.linalg.norm(real_tips - thumb_tip, axis=1) * 100.0
        alphas_real = np.clip(
            (self.d2[:3] - distances_cm) / (self.d2[:3] - self.d1[:3] + 1.0e-8),
            0.0,
            0.7,
        )
        return np.asarray(
            [float(np.max(alphas_real)), *alphas_real.tolist(), 0.0],
            dtype=np.float64,
        )

    def _get_init_qpos(self, last_qpos):
        qpos = np.asarray(super()._get_init_qpos(last_qpos), dtype=np.float64)
        qpos[self.fixed_pinky_indices] = self.fixed_pinky_values
        return qpos

    def _run_optimization(self, objective_fn, init_qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(super()._run_optimization(objective_fn, init_qpos), dtype=np.float64)
        if not np.allclose(
            qpos[self.fixed_pinky_indices], self.fixed_pinky_values, atol=1.0e-7
        ):
            raise RuntimeError(
                "official solver violated fixed pinky equality bounds: "
                f"actual={qpos[self.fixed_pinky_indices]}, expected={self.fixed_pinky_values}"
            )
        return qpos.astype(np.float32)


def install_four_finger_optimizer(retargeter, config: dict):
    """Replace only the optimizer inside an official Retargeter instance."""
    retargeter.optimizer = FourFingerOfficialAdaptiveOptimizer(config)
    return retargeter
