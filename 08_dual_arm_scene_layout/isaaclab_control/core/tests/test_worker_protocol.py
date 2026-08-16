from __future__ import annotations

import unittest
import numpy as np

from core.bridge.worker_client import CuroboWorkerClient, _jsonable, _worker_subprocess_env


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


if __name__ == "__main__":
    unittest.main()
