from __future__ import annotations

import unittest
import numpy as np

from core.ik.curobo_gpu_ik import BatchedIKResult
from core.ik.ik_solution_selector import select_solution, select_waypoint_chain


def synthetic_result() -> BatchedIKResult:
    q = np.array([
        [[0.9, -1.2, 0, 0.7, 0.6, 0, 0.4], [0.1, 0.1, 0, 0.1, 0.1, 0, 0.1]],
        [[1.0, -1.1, 0, 0.8, 0.7, 0, 0.5], [-0.2, 0.1, 0, -0.1, 0.1, 0, 0.1]],
    ], dtype=float)
    return BatchedIKResult(
        q_rad=q,
        raw_success=np.ones((2,2), dtype=bool),
        accepted=np.ones((2,2), dtype=bool),
        position_error_m=np.full((2,2), 1e-4),
        orientation_error_rad=np.full((2,2), 1e-4),
        inner_limit_margin_rad=np.array([[0.5,0.5],[0.5,0.5]]),
        lower_inner_rad=np.full(7, -2.0),
        upper_inner_rad=np.full(7, 2.0),
        joint_names=tuple(f"arm_r_joint_{i}" for i in range(1,8)),
        solve_time_s=0.0,
    )


class SelectorTest(unittest.TestCase):
    def test_nearest_reference_wins(self):
        result = synthetic_result()
        qref = np.array([0.87,-1.22,0,0.70,0.61,0,0.44])
        pick = select_solution(result, 0, qref)
        self.assertIsNotNone(pick)
        self.assertEqual(pick.solution_index, 0)

    def test_chain_updates_reference(self):
        result = synthetic_result()
        qref = np.array([0.87,-1.22,0,0.70,0.61,0,0.44])
        chain = select_waypoint_chain(result, qref)
        self.assertIsNotNone(chain)
        self.assertEqual([x.solution_index for x in chain], [0,0])


if __name__ == '__main__':
    unittest.main()
