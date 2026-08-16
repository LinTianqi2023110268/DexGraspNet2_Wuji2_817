from __future__ import annotations

from dataclasses import dataclass
import math

RIGHT_ARM_NAMES = tuple(f"arm_r_joint_{i}" for i in range(1, 8))
DEFAULT_INITIAL_RIGHT_ARM_DEG = (50.0, -70.0, 0.0, 40.0, 35.0, 0.0, 25.0)


@dataclass(frozen=True)
class IKConfig:
    """cuRobo IK settings matching the validated project acceptance contract."""

    device: str = "cuda:0"
    num_seeds: int = 48
    batch_size: int = 64
    return_seeds: int = 48
    use_cuda_graph: bool = True
    random_seed: int = 20260815
    position_tolerance_m: float = 0.005
    orientation_tolerance_rad: float = math.radians(5.0)
    minimum_inner_limit_margin_rad: float = math.radians(3.0)
    inner_limit_shrink_rad: float = 0.01
    tool_frame: str = "arm_r_link_tf"
    self_collision_check: bool = False
    load_collision_spheres: bool = False
    dedupe_round_decimals: int = 3


@dataclass(frozen=True)
class MapperConfig:
    """Single-view RGB-D mapping defaults for a tabletop workcell.

    Bounds are auto-fitted to the current valid depth frame and then padded.  This
    avoids hard-coding the current simulated camera/world placement into the mapper.
    """

    device: str = "cuda:0"
    voxel_size_m: float = 0.01
    esdf_voxel_size_m: float = 0.02
    truncation_distance_m: float = 0.06
    minimum_tsdf_weight: float = 0.1
    depth_min_m: float = 0.05
    depth_max_m: float = 3.0
    workspace_padding_m: float = 0.15
    minimum_extent_m: tuple[float, float, float] = (0.50, 0.50, 0.40)
    maximum_extent_m: tuple[float, float, float] = (3.00, 3.00, 2.00)
    target_scene_exclusion_dilation_px: int = 2
    visibility_surface_band_m: float = 0.015


@dataclass(frozen=True)
class WorkerConfig:
    """Cross-conda bridge settings.

    The Isaac Lab runtime must not import cuRobo directly.  It starts one persistent
    worker in ``curobo_v2`` and communicates over a line-oriented JSON protocol.
    """

    conda_env: str = "curobo_v2"
    conda_exe: str | None = None
    startup_timeout_s: float = 120.0
    request_timeout_s: float = 120.0
