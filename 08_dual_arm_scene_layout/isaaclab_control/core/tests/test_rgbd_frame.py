from __future__ import annotations

import unittest
import numpy as np

from core.config import MapperConfig
from core.perception_collision.rgbd_mapper import RGBDFrame, _dilate_mask


class RGBDFrameTest(unittest.TestCase):
    def test_world_points(self):
        depth = np.ones((4,4), dtype=np.float32)
        K = np.array([[2,0,1.5],[0,2,1.5],[0,0,1]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        frame = RGBDFrame(depth, K, T).validated()
        pts = frame.world_points(MapperConfig(depth_max_m=2.0), stride=1)
        self.assertEqual(pts.shape, (16,3))
        self.assertTrue(np.allclose(pts[:,2], 1.0))

    def test_mask_dilation(self):
        m = np.zeros((5,5), dtype=bool)
        m[2,2] = True
        d = _dilate_mask(m, 1)
        self.assertEqual(int(d.sum()), 9)


if __name__ == '__main__':
    unittest.main()
