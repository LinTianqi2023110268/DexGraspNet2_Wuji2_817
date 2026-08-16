from __future__ import annotations

from enum import IntEnum
import numpy as np


class VisibilityClass(IntEnum):
    UNKNOWN = 0
    OBSERVED_FREE = 1
    NEAR_OBSERVED_SURFACE = 2
    OCCLUDED_UNKNOWN = 3


class SingleViewVisibility:
    """Conservative single-view known/free/unknown classifier.

    "No measured point" is never promoted to known free.  Sphere classification
    samples the center plus six radius-offset points and requires every sample to
    be observed free before the whole sphere is labelled observed-free.  This is a
    lightweight visibility gate, not a replacement for the ESDF collision query.
    """

    def __init__(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        T_world_camera: np.ndarray,
        surface_band_m: float = 0.015,
    ):
        self.depth = np.asarray(depth_m, dtype=np.float64)
        self.K = np.asarray(intrinsics, dtype=np.float64)
        self.Twc = np.asarray(T_world_camera, dtype=np.float64)
        if self.depth.ndim != 2 or self.K.shape != (3, 3) or self.Twc.shape != (4, 4):
            raise ValueError("Invalid depth/K/T_world_camera shapes")
        self.Tcw = np.linalg.inv(self.Twc)
        self.band = float(surface_band_m)

    def classify_points(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        p_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        pc = (self.Tcw @ p_h.T).T[:, :3]
        z = pc[:, 2]
        out = np.full(len(points), VisibilityClass.UNKNOWN, dtype=np.int64)
        forward = z > 1.0e-6
        if not np.any(forward):
            return out

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = np.rint(fx * pc[:, 0] / np.maximum(z, 1.0e-9) + cx).astype(np.int64)
        v = np.rint(fy * pc[:, 1] / np.maximum(z, 1.0e-9) + cy).astype(np.int64)
        H, W = self.depth.shape
        inside = forward & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            return out
        d = self.depth[v[idx], u[idx]]
        valid_depth = np.isfinite(d) & (d > 0)
        idx_valid = idx[valid_depth]
        d = d[valid_depth]
        if idx_valid.size == 0:
            return out

        dz = z[idx_valid] - d
        out[idx_valid[dz < -self.band]] = VisibilityClass.OBSERVED_FREE
        out[idx_valid[dz > self.band]] = VisibilityClass.OCCLUDED_UNKNOWN
        out[idx_valid[np.abs(dz) <= self.band]] = VisibilityClass.NEAR_OBSERVED_SURFACE
        return out

    def classify_spheres(self, centers_world: np.ndarray, radii_m: np.ndarray) -> np.ndarray:
        centers = np.asarray(centers_world, dtype=np.float64).reshape(-1, 3)
        radii = np.asarray(radii_m, dtype=np.float64).reshape(-1)
        if radii.size == 1:
            radii = np.repeat(radii, len(centers))
        if len(radii) != len(centers):
            raise ValueError("radii_m must be scalar or one value per sphere")
        # Seven representative points.  Requiring every one to be observed-free
        # is intentionally stricter than classifying only the sphere center.
        dirs = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        samples = centers[:, None, :] + radii[:, None, None] * dirs[None, :, :]
        classes = self.classify_points(samples.reshape(-1, 3)).reshape(len(centers), len(dirs))
        out = np.full(len(centers), VisibilityClass.OBSERVED_FREE, dtype=np.int64)
        has_free = np.any(classes == VisibilityClass.OBSERVED_FREE, axis=1)
        has_near = np.any(classes == VisibilityClass.NEAR_OBSERVED_SURFACE, axis=1)
        has_unknown = np.any(classes == VisibilityClass.UNKNOWN, axis=1)
        has_occluded = np.any(classes == VisibilityClass.OCCLUDED_UNKNOWN, axis=1)
        # If samples lie on both sides of the measured surface, the sphere spans
        # that surface and is classified as near-surface rather than merely behind it.
        spans_surface = has_near | (has_free & has_occluded)
        out[has_occluded & ~spans_surface] = VisibilityClass.OCCLUDED_UNKNOWN
        out[spans_surface] = VisibilityClass.NEAR_OBSERVED_SURFACE
        # Outside-FOV / invalid-depth samples mean the complete sphere is not observed.
        out[has_unknown] = VisibilityClass.UNKNOWN
        return out
