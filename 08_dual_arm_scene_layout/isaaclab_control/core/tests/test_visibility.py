from __future__ import annotations

import unittest
import numpy as np

from core.perception_collision.visibility import SingleViewVisibility, VisibilityClass


class VisibilityTest(unittest.TestCase):
    def setUp(self):
        self.depth = np.ones((101,101), dtype=float)
        self.K = np.array([[100,0,50],[0,100,50],[0,0,1]], dtype=float)
        self.T = np.eye(4)
        self.vis = SingleViewVisibility(self.depth, self.K, self.T, surface_band_m=0.01)

    def test_front_of_surface_is_known_free(self):
        cls = self.vis.classify_spheres([[0,0,0.5]], [0.05])[0]
        self.assertEqual(cls, VisibilityClass.OBSERVED_FREE)

    def test_behind_surface_is_unknown(self):
        cls = self.vis.classify_spheres([[0,0,1.5]], [0.05])[0]
        self.assertEqual(cls, VisibilityClass.OCCLUDED_UNKNOWN)

    def test_near_surface(self):
        cls = self.vis.classify_spheres([[0,0,1.0]], [0.01])[0]
        self.assertEqual(cls, VisibilityClass.NEAR_OBSERVED_SURFACE)


if __name__ == '__main__':
    unittest.main()
