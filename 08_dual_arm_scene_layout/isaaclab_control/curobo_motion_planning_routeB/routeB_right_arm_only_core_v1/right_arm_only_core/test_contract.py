import unittest

import numpy as np

from right_arm_only_core.contract import RIGHT_ARM_JOINTS, build_locked_joint_contract


class ContractTests(unittest.TestCase):
    def test_build_contract(self):
        names = list(RIGHT_ARM_JOINTS) + ["left_a", "finger", "hand_a"]
        q0 = np.arange(len(names), dtype=float) * 0.01
        qg = q0.copy()
        qg[:7] += 0.2
        c = build_locked_joint_contract(names, q0, qg)
        self.assertEqual(c.action_dim, 7)
        self.assertEqual(c.locked_joint_count, 3)
        self.assertEqual(tuple(c.active_joint_names), RIGHT_ARM_JOINTS)
        self.assertAlmostEqual(c.max_locked_goal_difference_rad, 0.0)

    def test_locked_goal_mismatch_fails(self):
        names = list(RIGHT_ARM_JOINTS) + ["left_a"]
        q0 = np.zeros(len(names))
        qg = q0.copy()
        qg[-1] = 1e-3
        with self.assertRaises(RuntimeError):
            build_locked_joint_contract(names, q0, qg)


if __name__ == "__main__":
    unittest.main()
