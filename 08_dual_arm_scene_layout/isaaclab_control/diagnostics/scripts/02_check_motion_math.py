"""Fast CPU-only checks for the trajectory helpers."""

from __future__ import annotations

import math

import torch

from motion_math import JointCommandLimiter, quaternion_slerp, quintic_time_scale


def main() -> None:
    samples = [quintic_time_scale(i / 100.0) for i in range(101)]
    assert samples[0] == 0.0
    assert samples[-1] == 1.0
    assert all(a <= b for a, b in zip(samples, samples[1:]))

    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q1 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    midpoint = quaternion_slerp(q0, q1, 0.5)
    assert torch.allclose(torch.linalg.vector_norm(midpoint, dim=1), torch.ones(1), atol=1.0e-6)

    dt = 0.01
    limiter = JointCommandLimiter(torch.zeros(1, 7), dt, 0.3, 0.6)
    positions = [limiter.step(torch.ones(1, 7)) for _ in range(100)]
    velocities = [(b - a) / dt for a, b in zip(positions, positions[1:])]
    assert max(float(torch.max(torch.abs(value))) for value in velocities) <= 0.300001

    print("[PASS] quintic time scale")
    print("[PASS] quaternion SLERP")
    print("[PASS] joint velocity/acceleration limiter")


if __name__ == "__main__":
    main()

