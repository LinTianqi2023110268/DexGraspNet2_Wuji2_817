from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np

from ..config import MapperConfig
from .esdf_collision import query_spheres, SphereCollisionBatch
from .phase_policy import CollisionPhase, PhaseCollisionPolicy
from .visibility import SingleViewVisibility, VisibilityClass


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    H, W = mask.shape
    out = np.zeros_like(mask)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            out |= padded[radius + dv:radius + dv + H, radius + du:radius + du + W]
    return out


@dataclass(frozen=True)
class RGBDFrame:
    depth_m: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    target_mask: np.ndarray | None = None

    @staticmethod
    def from_npy(
        depth_path: Path | str,
        intrinsics_path: Path | str,
        T_world_camera_path: Path | str,
        target_mask_path: Path | str | None = None,
    ) -> "RGBDFrame":
        depth = np.asarray(np.load(depth_path), dtype=np.float32)
        K = np.asarray(np.load(intrinsics_path), dtype=np.float32)
        Twc = np.asarray(np.load(T_world_camera_path), dtype=np.float32)
        mask = None if target_mask_path is None else np.asarray(np.load(target_mask_path), dtype=bool)
        return RGBDFrame(depth, K, Twc, mask).validated()

    def validated(self) -> "RGBDFrame":
        if self.depth_m.ndim != 2:
            raise ValueError(f"depth_m must be HxW, got {self.depth_m.shape}")
        if self.intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must be 3x3, got {self.intrinsics.shape}")
        if self.T_world_camera.shape != (4, 4):
            raise ValueError(f"T_world_camera must be 4x4, got {self.T_world_camera.shape}")
        if self.target_mask is not None and self.target_mask.shape != self.depth_m.shape:
            raise ValueError("target_mask must match depth image shape")
        return self

    def valid_depth_mask(self, cfg: MapperConfig) -> np.ndarray:
        d = self.depth_m
        return np.isfinite(d) & (d >= cfg.depth_min_m) & (d <= cfg.depth_max_m)

    def world_points(self, cfg: MapperConfig, stride: int = 8) -> np.ndarray:
        valid = self.valid_depth_mask(cfg)
        v, u = np.nonzero(valid)
        if len(u) == 0:
            raise RuntimeError("RGB-D frame contains no valid depth in configured range")
        take = np.arange(0, len(u), max(1, int(stride)))
        u = u[take]
        v = v[take]
        z = self.depth_m[v, u].astype(np.float64)
        fx, fy = float(self.intrinsics[0, 0]), float(self.intrinsics[1, 1])
        cx, cy = float(self.intrinsics[0, 2]), float(self.intrinsics[1, 2])
        x = (u.astype(np.float64) - cx) * z / fx
        y = (v.astype(np.float64) - cy) * z / fy
        pc = np.stack([x, y, z, np.ones_like(z)], axis=1)
        return (np.asarray(self.T_world_camera, dtype=np.float64) @ pc.T).T[:, :3]


@dataclass
class ObservedSceneMap:
    map_id: str
    scene_grid: object
    target_grid: object | None
    frame: RGBDFrame
    grid_center_world: np.ndarray
    extent_meters_xyz: np.ndarray
    config: MapperConfig

    def check_spheres(
        self,
        centers_world: np.ndarray,
        radii_m: np.ndarray,
        phase: CollisionPhase | str,
        margin_m: float = 0.0,
    ) -> dict:
        phase = CollisionPhase(phase)
        policy = PhaseCollisionPolicy(phase)
        scene = query_spheres(self.scene_grid, centers_world, radii_m, margin_m)
        target: SphereCollisionBatch | None = None
        if self.target_grid is not None:
            target = query_spheres(self.target_grid, centers_world, radii_m, margin_m)
        visibility = SingleViewVisibility(
            self.frame.depth_m,
            self.frame.intrinsics,
            self.frame.T_world_camera,
            self.config.visibility_surface_band_m,
        ).classify_spheres(centers_world, radii_m)
        target_collision = (
            np.zeros_like(scene.collision)
            if target is None
            else target.collision
        )
        blocking_collision = scene.collision | (target_collision if policy.target_is_obstacle else False)
        unknown = (
            (~scene.inside_grid)
            | (visibility == VisibilityClass.UNKNOWN)
            | (visibility == VisibilityClass.OCCLUDED_UNKNOWN)
        )
        return {
            "phase": phase.value,
            "blocking_collision": blocking_collision,
            "scene_collision": scene.collision,
            "target_collision": target_collision,
            "target_contact_allowed": not policy.target_is_obstacle,
            "visibility_class": visibility,
            "unknown": unknown,
            "scene_distance_m": scene.distance_m,
            "target_distance_m": None if target is None else target.distance_m,
        }


class CuroboRGBDMapper:
    """Build separate non-target and target ESDF layers from one RGB-D frame."""

    def __init__(self, config: MapperConfig | None = None):
        self.config = config or MapperConfig()
        self._import_curobo()

    def _import_curobo(self) -> None:
        try:
            import torch
            from curobo.perception import Mapper, MapperCfg
            from curobo.types import CameraObservation, Pose
        except Exception as exc:
            raise RuntimeError(
                "cuRobo Mapper import failed. Run mapping inside curobo_v2. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to PyTorch in curobo_v2")
        self.torch = torch
        self.Mapper = Mapper
        self.MapperCfg = MapperCfg
        self.CameraObservation = CameraObservation
        self.Pose = Pose

    def _bounds(self, frame: RGBDFrame) -> tuple[np.ndarray, np.ndarray]:
        pts = frame.world_points(self.config, stride=8)
        lo = pts.min(axis=0) - self.config.workspace_padding_m
        hi = pts.max(axis=0) + self.config.workspace_padding_m
        extent = hi - lo
        min_e = np.asarray(self.config.minimum_extent_m, dtype=np.float64)
        max_e = np.asarray(self.config.maximum_extent_m, dtype=np.float64)
        extent = np.maximum(extent, min_e)
        if np.any(extent > max_e):
            raise RuntimeError(
                f"Auto-fitted mapper extent {extent.tolist()} exceeds safety cap {max_e.tolist()}; "
                "check depth scale/extrinsics or increase maximum_extent_m deliberately."
            )
        center = 0.5 * (lo + hi)
        # If an axis was expanded by minimum_extent_m, keep the observed center.
        observed_center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        center = observed_center
        return center, extent

    def _new_mapper(self, image_shape: tuple[int, int], center: np.ndarray, extent: np.ndarray):
        torch = self.torch
        H, W = image_shape
        cfg = self.MapperCfg(
            extent_meters_xyz=tuple(float(x) for x in extent),
            voxel_size=float(self.config.voxel_size_m),
            esdf_voxel_size=float(self.config.esdf_voxel_size_m),
            extent_esdf_meters_xyz=tuple(float(x) for x in extent),
            grid_center=torch.as_tensor(center, device=self.config.device, dtype=torch.float32),
            truncation_distance=float(self.config.truncation_distance_m),
            minimum_tsdf_weight=float(self.config.minimum_tsdf_weight),
            depth_minimum_distance=float(self.config.depth_min_m),
            depth_maximum_distance=float(self.config.depth_max_m),
            decay_factor=1.0,
            frustum_decay_factor=1.0,
            num_cameras=1,
            image_height=int(H),
            image_width=int(W),
            device=self.config.device,
        )
        return self.Mapper(cfg)

    def _observation(self, depth: np.ndarray, frame: RGBDFrame):
        torch = self.torch
        depth_t = torch.as_tensor(depth, device=self.config.device, dtype=torch.float32).unsqueeze(0)
        # cuRobo 0.8.x's camera-project TSDF integrator validates an RGB tensor
        # even for a depth-only map.  Supply the documented uint8 BxHxWx3
        # placeholder; it carries no geometry or target semantics.
        rgb_t = torch.zeros(
            (*depth_t.shape, 3),
            device=self.config.device,
            dtype=torch.uint8,
        )
        K_t = torch.as_tensor(frame.intrinsics, device=self.config.device, dtype=torch.float32).unsqueeze(0)
        T_t = torch.as_tensor(frame.T_world_camera, device=self.config.device, dtype=torch.float32).unsqueeze(0)
        return self.CameraObservation(
            rgb_image=rgb_t,
            depth_image=depth_t,
            pose=self.Pose.from_matrix(T_t),
            intrinsics=K_t,
            resolution=list(frame.depth_m.shape),
        )

    def build(self, frame: RGBDFrame) -> ObservedSceneMap:
        frame = frame.validated()
        center, extent = self._bounds(frame)
        valid = frame.valid_depth_mask(self.config)
        scene_depth = np.where(valid, frame.depth_m, 0.0).astype(np.float32)
        target_depth = None
        if frame.target_mask is not None:
            target_mask = np.asarray(frame.target_mask, dtype=bool) & valid
            exclusion = _dilate_mask(
                target_mask,
                self.config.target_scene_exclusion_dilation_px,
            )
            scene_depth[exclusion] = 0.0
            target_depth = np.where(target_mask, frame.depth_m, 0.0).astype(np.float32)

        scene_mapper = self._new_mapper(frame.depth_m.shape, center, extent)
        scene_mapper.integrate(self._observation(scene_depth, frame))
        scene_grid = scene_mapper.compute_esdf()

        target_grid = None
        if target_depth is not None and np.count_nonzero(target_depth) > 0:
            target_mapper = self._new_mapper(frame.depth_m.shape, center, extent)
            target_mapper.integrate(self._observation(target_depth, frame))
            target_grid = target_mapper.compute_esdf()

        return ObservedSceneMap(
            map_id=str(uuid.uuid4()),
            scene_grid=scene_grid,
            target_grid=target_grid,
            frame=frame,
            grid_center_world=np.asarray(center, dtype=np.float64),
            extent_meters_xyz=np.asarray(extent, dtype=np.float64),
            config=self.config,
        )
