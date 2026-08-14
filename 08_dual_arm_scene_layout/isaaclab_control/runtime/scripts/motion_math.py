"""Small, simulator-independent helpers for smooth arm motion.

This file deliberately contains no Isaac Sim imports.  The interpolation and
command limiter can therefore be checked quickly with ordinary Python.
"""

from __future__ import annotations

import math

import torch


def quintic_time_scale(progress: float) -> float:
    """Return a zero-velocity, zero-acceleration time scale in ``[0, 1]``."""

    u = min(max(float(progress), 0.0), 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def quaternion_slerp(start: torch.Tensor, goal: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate ``(w, x, y, z)`` quaternions along the shortest arc."""

    q0 = start / torch.linalg.vector_norm(start, dim=-1, keepdim=True)
    q1 = goal / torch.linalg.vector_norm(goal, dim=-1, keepdim=True)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)

    if float(torch.min(1.0 - dot)) < 1.0e-6:
        result = q0 + float(alpha) * (q1 - q0)
        return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    weight0 = torch.sin((1.0 - float(alpha)) * theta) / sin_theta
    weight1 = torch.sin(float(alpha) * theta) / sin_theta
    return weight0 * q0 + weight1 * q1


def quaternion_error_deg(actual: torch.Tensor, target: torch.Tensor) -> float:
    """Return the unsigned shortest angular distance between two quaternions."""

    qa = actual / torch.linalg.vector_norm(actual, dim=-1, keepdim=True)
    qt = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    cosine = torch.sum(qa * qt, dim=-1).abs().clamp(-1.0, 1.0)
    return math.degrees(2.0 * math.acos(float(cosine[0])))


class JointCommandLimiter:
    """Apply joint-velocity and joint-acceleration limits to position commands."""

    def __init__(
        self,
        initial_position: torch.Tensor,
        physics_dt_s: float,
        max_velocity_rad_s: float,
        max_acceleration_rad_s2: float,
    ) -> None:
        self.position = initial_position.clone()
        self.velocity = torch.zeros_like(initial_position)
        self.dt = float(physics_dt_s)
        self.max_velocity = float(max_velocity_rad_s)
        self.max_acceleration = float(max_acceleration_rad_s2)

    def step(self, requested_position: torch.Tensor) -> torch.Tensor:
        """Return the next reachable position command without discontinuities."""

        requested_velocity = (requested_position - self.position) / self.dt
        requested_velocity = requested_velocity.clamp(-self.max_velocity, self.max_velocity)

        velocity_change = requested_velocity - self.velocity
        max_change = self.max_acceleration * self.dt
        velocity_change = velocity_change.clamp(-max_change, max_change)

        self.velocity = (self.velocity + velocity_change).clamp(-self.max_velocity, self.max_velocity)
        self.position = self.position + self.velocity * self.dt
        return self.position.clone()

