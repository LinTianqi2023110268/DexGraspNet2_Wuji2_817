import unittest

from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter


class RouteBImportTest(unittest.TestCase):
    def test_import_route_b_adapter(self):
        adapter = RouteBMotionPlannerAdapter({"routeB": {"enabled": False}})
        self.assertFalse(adapter.route_cfg["enabled"])


if __name__ == "__main__":
    unittest.main()
