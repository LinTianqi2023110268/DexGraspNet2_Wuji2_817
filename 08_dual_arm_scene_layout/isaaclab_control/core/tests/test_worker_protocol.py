from __future__ import annotations

import unittest
import numpy as np

from core.bridge.worker_client import CuroboWorkerClient, _jsonable, _worker_subprocess_env
from core.bridge.curobo_worker import (
    selected_collision_records_for_independent_targets,
)


class WorkerProtocolTests(unittest.TestCase):
    def test_numpy_payload_is_jsonable(self):
        value = {
            "q": np.asarray([1.0, 2.0]),
            "n": np.int64(3),
            "ok": np.bool_(True),
        }
        converted = _jsonable(value)
        self.assertEqual(converted["q"], [1.0, 2.0])
        self.assertEqual(converted["n"], 3)
        self.assertIs(converted["ok"], True)

    def test_collision_context_is_forwarded(self):
        client = object.__new__(CuroboWorkerClient)
        captured = {}

        def request(payload, timeout=None):
            captured.update(payload)
            return {"ok": True}

        client.request = request
        context = {
            "phases": ["pregrasp"],
            "joint_positions_by_name": {"arm_r_joint_1": 0.0},
            "T_world_base": np.eye(4),
        }
        client.solve_ik(np.eye(4)[None], np.zeros(7), collision_context=context)
        self.assertEqual(captured["op"], "solve_ik")
        self.assertIs(captured["collision_context"], context)

    def test_worker_environment_drops_isaac_python_injection(self):
        env = _worker_subprocess_env({
            "PATH": "/bin",
            "PYTHONPATH": "/isaac/kit",
            "LD_LIBRARY_PATH": "/isaac/lib",
            "ISAAC_PATH": "/isaac",
        })
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["CONDA_NO_PLUGINS"], "true")
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("LD_LIBRARY_PATH", env)
        self.assertNotIn("ISAAC_PATH", env)

    def test_independent_selected_collision_allows_null_targets(self):
        class Pick:
            def __init__(self, target_index, solution_index):
                self.target_index = target_index
                self.solution_index = solution_index

        selected = [Pick(0, 1), None, Pick(2, 0)]
        feasible = [
            [
                {
                    "target_index": 0,
                    "solution_index": 1,
                    "self_collision_pass": True,
                    "observed_scene_collision_pass": True,
                }
            ],
            [],
            [
                {
                    "target_index": 2,
                    "solution_index": 0,
                    "self_collision_pass": True,
                    "observed_scene_collision_pass": True,
                }
            ],
        ]
        records = selected_collision_records_for_independent_targets(
            selected, feasible
        )
        self.assertEqual(records[0]["target_index"], 0)
        self.assertIsNone(records[1])
        self.assertEqual(records[2]["target_index"], 2)

    def test_independent_selected_collision_allows_empty_feasible_after_ik(self):
        class Pick:
            target_index = 0
            solution_index = 3

        records = selected_collision_records_for_independent_targets(
            [Pick()], [[]]
        )
        self.assertEqual(records, [None])


if __name__ == "__main__":
    unittest.main()
