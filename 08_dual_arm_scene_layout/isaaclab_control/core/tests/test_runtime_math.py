from __future__ import annotations

import unittest
import numpy as np

from core.runtime_math import pose_from_position_quaternion_wxyz, rebase_pick_waypoints


class RuntimeMathTests(unittest.TestCase):
    def test_wxyz_pose(self):
        pose = pose_from_position_quaternion_wxyz([1, 2, 3], [1, 0, 0, 0])
        np.testing.assert_allclose(pose, np.asarray([
            [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]
        ]))

    def test_rebase_preserves_world_lift_vector(self):
        targets = np.repeat(np.eye(4)[None], 5, axis=0)
        targets[4, 2, 3] = 0.1
        before = np.eye(4)
        after = np.eye(4)
        after[0, 3] = 0.2
        rebased, delta = rebase_pick_waypoints(targets, before, after)
        np.testing.assert_allclose(delta, after)
        np.testing.assert_allclose(rebased[3, :3, 3], [0.2, 0, 0])
        np.testing.assert_allclose(rebased[4, :3, 3], [0.2, 0, 0.1])


if __name__ == "__main__":
    unittest.main()
