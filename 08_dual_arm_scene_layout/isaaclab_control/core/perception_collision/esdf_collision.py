from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SphereCollisionBatch:
    distance_m: np.ndarray
    collision: np.ndarray
    inside_grid: np.ndarray

    def to_jsonable(self) -> dict:
        return {
            "distance_m": self.distance_m.tolist(),
            "collision": self.collision.tolist(),
            "inside_grid": self.inside_grid.tolist(),
        }


def _as_numpy3(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)[:3]


def query_esdf_distance(voxel_grid, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear ESDF query following cuRobo's official volumetric-mapping example."""
    import torch
    import torch.nn.functional as F

    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    feature = voxel_grid.feature_tensor
    if feature is None:
        raise RuntimeError("VoxelGrid has no feature_tensor; compute_esdf() was not completed")
    if not isinstance(feature, torch.Tensor):
        feature = torch.as_tensor(feature)
    device = feature.device
    grid = feature.float()
    if grid.ndim != 3:
        raise RuntimeError(f"Expected a 3-D ESDF feature tensor, got {tuple(grid.shape)}")
    nx, ny, nz = [int(x) for x in grid.shape]
    center = _as_numpy3(voxel_grid.pose)
    voxel_size = float(voxel_grid.voxel_size)
    half = np.asarray(
        [(nx - 1) * voxel_size / 2, (ny - 1) * voxel_size / 2, (nz - 1) * voxel_size / 2],
        dtype=np.float32,
    )
    half_safe = np.maximum(half, 1.0e-9)
    normalized_np = (pts - center.astype(np.float32)) / half_safe
    inside = np.all(np.abs(normalized_np) <= (1.0 + 1.0e-6), axis=1)

    normalized = torch.as_tensor(normalized_np, device=device, dtype=torch.float32)
    # Official example: 5-D grid_sample expects (z,y,x) for a tensor treated as D,H,W.
    coords = normalized[:, [2, 1, 0]].view(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        grid.unsqueeze(0).unsqueeze(0),
        coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1)
    distance = sampled.detach().cpu().numpy().astype(np.float64)
    return distance, inside


def query_spheres(voxel_grid, centers_world: np.ndarray, radii_m: np.ndarray, margin_m: float = 0.0) -> SphereCollisionBatch:
    centers = np.asarray(centers_world, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii_m, dtype=np.float64).reshape(-1)
    if radii.size == 1:
        radii = np.repeat(radii, len(centers))
    if len(radii) != len(centers):
        raise ValueError("radii_m must be scalar or one value per sphere")
    distance, inside = query_esdf_distance(voxel_grid, centers)
    collision = inside & (distance <= (radii + float(margin_m)))
    return SphereCollisionBatch(distance_m=distance, collision=collision, inside_grid=inside)
