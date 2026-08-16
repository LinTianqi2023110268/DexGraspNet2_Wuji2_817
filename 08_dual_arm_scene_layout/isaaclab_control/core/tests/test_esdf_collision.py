from __future__ import annotations

import unittest
import numpy as np
import torch

from core.perception_collision.esdf_collision import query_esdf_distance, query_spheres


class FakeGrid:
    def __init__(self):
        # 3x3x3 field whose zero surface is near the center; value grows with x index.
        self.feature_tensor = torch.tensor(
            [[[-1.0, -1.0, -1.0],[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]],
             [[ 0.0,  0.0,  0.0],[ 0.0, 0.0, 0.0],[ 0.0, 0.0, 0.0]],
             [[ 1.0,  1.0,  1.0],[ 1.0, 1.0, 1.0],[ 1.0, 1.0, 1.0]]],
            dtype=torch.float32,
        )
        self.pose = [0.0,0.0,0.0,1.0,0.0,0.0,0.0]
        self.voxel_size = 1.0


class EsdfQueryTest(unittest.TestCase):
    def test_center_query_is_zero(self):
        grid = FakeGrid()
        d, inside = query_esdf_distance(grid, [[0,0,0]])
        self.assertTrue(bool(inside[0]))
        self.assertAlmostEqual(float(d[0]), 0.0, places=5)

    def test_outside_grid_marked_unknown(self):
        grid = FakeGrid()
        _, inside = query_esdf_distance(grid, [[10,0,0]])
        self.assertFalse(bool(inside[0]))

    def test_sphere_collision_uses_radius(self):
        grid = FakeGrid()
        result = query_spheres(grid, [[0.5,0,0]], [0.6])
        self.assertTrue(bool(result.collision[0]))


if __name__ == '__main__':
    unittest.main()
