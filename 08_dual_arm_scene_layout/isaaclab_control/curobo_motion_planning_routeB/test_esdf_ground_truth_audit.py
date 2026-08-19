#!/usr/bin/env python3
"""Route B ESDF ground-truth audit using filtered_depth surface points.

Diagnostic only:
- no planner changes
- no VoxelGrid/VoxelData/feature mutation
- no Isaac launch/execution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
if str(ISAACLAB_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_CONTROL_ROOT))

from core.perception_collision.esdf_collision import query_esdf_distance
from core.perception_collision.rgbd_mapper import RGBDFrame
from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
from test_trajopt_feasibility_audit import _json_default, _to_numpy


def _load_surface_points_base(capture_dir: Path, adapter: RouteBMotionPlannerAdapter, count: int) -> np.ndarray:
    frame = RGBDFrame.from_npy(
        capture_dir / "planning/filtered_depth.npy",
        capture_dir / "intrinsics.npy",
        capture_dir / "T_world_camera.npy",
    )
    T_base_camera = adapter._base_from_world() @ frame.T_world_camera
    depth = np.asarray(frame.depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise RuntimeError("filtered_depth contains no valid pixels")
    take_n = min(int(count), len(u))
    order = np.linspace(0, len(u) - 1, take_n, dtype=np.int64)
    u = u[order]
    v = v[order]
    z = depth[v, u]
    K = np.asarray(frame.intrinsics, dtype=np.float64)
    x = (u.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (v.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (T_base_camera @ pc.T).T[:, :3]


def _project_sdf(voxel_grid: Any, points_base: np.ndarray) -> np.ndarray:
    sdf, _inside = query_esdf_distance(voxel_grid, points_base)
    return np.asarray(sdf, dtype=np.float64)


def _feature3_project(voxel_grid: Any) -> np.ndarray:
    feat = _to_numpy(getattr(voxel_grid, "feature_tensor", None))
    if feat is None:
        raise RuntimeError("project VoxelGrid has no feature_tensor")
    if feat.ndim != 3:
        raise RuntimeError(f"expected project feature_tensor [nx,ny,nz], got {feat.shape}")
    return np.asarray(feat, dtype=np.float64)


def _internal_voxel_data(scene_cost: Any) -> dict[str, Any]:
    vox = scene_cost.config.scene_collision_checker.data.voxels
    features = _to_numpy(vox.features)
    params = _to_numpy(vox.params)
    inv_pose = _to_numpy(vox.inv_pose)
    if features is None or params is None or inv_pose is None:
        raise RuntimeError("failed to read cuRobo VoxelData features/params/inv_pose")
    dims_float = params.reshape(-1, 4)[0, :3].astype(np.float64)
    dims = dims_float.astype(np.int64)
    voxel_size = float(params.reshape(-1, 4)[0, 3])
    # VoxelData stores inverse pose. For pure translation, center = -inv_pose xyz.
    inv = inv_pose.reshape(-1, inv_pose.shape[-1])[0]
    center = -np.asarray(inv[:3], dtype=np.float64)
    flat = np.asarray(features, dtype=np.float64).reshape(-1)
    expected = int(np.prod(dims))
    if flat.size < expected:
        raise RuntimeError(f"VoxelData feature length {flat.size} < dims product {expected}")
    return {
        "dims": dims,
        "dims_float": dims_float,
        "voxel_size": voxel_size,
        "center_base": center,
        "feature3": flat[:expected].reshape(tuple(dims.tolist())),
        "raw_params": params.tolist(),
        "raw_inv_pose": inv_pose.tolist(),
        "names": getattr(vox, "names", None),
    }


def _trilinear_sample_align_corners(feature: np.ndarray, points_base: np.ndarray, center: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    nx, ny, nz = feature.shape
    dims = np.asarray([nx, ny, nz], dtype=np.float64)
    v = (pts - center.reshape(1, 3)) / float(voxel_size) + dims.reshape(1, 3) / 2.0 - 0.5
    inside = np.all((v >= 0.0) & (v <= (dims.reshape(1, 3) - 1.0)), axis=1)
    out = np.empty(len(pts), dtype=np.float64)
    for i, coord in enumerate(v):
        x, y, z = coord
        x0 = int(np.floor(x)); y0 = int(np.floor(y)); z0 = int(np.floor(z))
        x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
        fx = x - x0; fy = y - y0; fz = z - z0
        def val(ix: int, iy: int, iz: int) -> float:
            ix = min(max(ix, 0), nx - 1)
            iy = min(max(iy, 0), ny - 1)
            iz = min(max(iz, 0), nz - 1)
            return float(feature[ix, iy, iz])
        c000 = val(x0, y0, z0); c001 = val(x0, y0, z1)
        c010 = val(x0, y1, z0); c011 = val(x0, y1, z1)
        c100 = val(x1, y0, z0); c101 = val(x1, y0, z1)
        c110 = val(x1, y1, z0); c111 = val(x1, y1, z1)
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        out[i] = c0 * (1 - fz) + c1 * fz
    return out, inside


def _curobo_semantic_sdf(scene_cost: Any, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = _internal_voxel_data(scene_cost)
    sdf, inside = _trilinear_sample_align_corners(
        data["feature3"],
        points_base,
        data["center_base"],
        data["voxel_size"],
    )
    meta = {
        "method": "CPU reproduction of cuRobo VoxelData data_voxel.py sample_voxel_sdf_with_grad convention",
        "tensor_shape_order": "[nx, ny, nz], X slowest, Z fastest",
        "coordinate_rule": "voxel = (point_base - center_base) / voxel_size + dims/2 - 0.5",
        "dims": data["dims"].tolist(),
        "dims_float_from_params": data["dims_float"].tolist(),
        "voxel_size_m": data["voxel_size"],
        "center_base_m": data["center_base"].tolist(),
        "names": data["names"],
    }
    return sdf, inside, meta


def _dimension_diagnostic(scene_grid: Any, scene_cost: Any) -> dict[str, Any]:
    data = _internal_voxel_data(scene_cost)
    feature_shape = list(_feature3_project(scene_grid).shape)
    dims_float = data["dims_float"]
    dims_int32_trunc = data["dims"]
    dims_round = np.rint(dims_float).astype(np.int64)
    return {
        "project_feature_shape": feature_shape,
        "curobo_params_dims_float": dims_float.tolist(),
        "curobo_kernel_int32_trunc_dims": dims_int32_trunc.tolist(),
        "rounded_dims": dims_round.tolist(),
        "trunc_matches_feature_shape": bool(np.array_equal(dims_int32_trunc, np.asarray(feature_shape))),
        "round_matches_feature_shape": bool(np.array_equal(dims_round, np.asarray(feature_shape))),
        "issue": (
            "VoxelData params store dimensions as float and cuRobo Warp kernel casts with int32(). "
            "Here z dimension is 26.999998, so truncation gives 26 while feature_tensor shape is 27."
        )
        if not np.array_equal(dims_int32_trunc, np.asarray(feature_shape))
        else "none",
    }


def _raw_collision_probe(scene_cost: Any, points_base: np.ndarray, device: str) -> np.ndarray:
    """cuRobo raw checker with radius=0, weight=1.

    This is not a signed SDF for outside points; it only returns max(-sdf, 0).
    It is included to confirm inside/negative cases against cuRobo's CUDA path.
    """
    import torch
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    spheres = np.zeros((len(points_base), 4), dtype=np.float32)
    spheres[:, :3] = np.asarray(points_base, dtype=np.float32)
    sph = torch.as_tensor(spheres, device=device, dtype=torch.float32).reshape(1, len(points_base), 1, 4)
    buf = CollisionBuffer.from_shape(tuple(sph.shape), scene_cost.device_cfg)
    unit_weight = torch.ones_like(scene_cost._weight)
    out = scene_cost.config.scene_collision_checker.get_sphere_distance_raw(
        sph,
        buf,
        unit_weight,
        scene_cost.config.activation_distance,
        return_loss=scene_cost.config.use_grad_input,
    )
    return out.detach().cpu().numpy().reshape(-1)


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    av = np.abs(v)
    return {
        "count": int(v.size),
        "median_abs_sdf": float(np.median(av)),
        "p90_abs_sdf": float(np.percentile(av, 90)),
        "max_abs_sdf": float(np.max(av)),
        "mean_abs_sdf": float(np.mean(av)),
    }


def _voxel_center_checks(voxel_grid: Any, scene_cost: Any, count: int = 20) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    project_feature = _feature3_project(voxel_grid)
    cu_sdf_meta = _internal_voxel_data(scene_cost)
    nx, ny, nz = project_feature.shape
    # Avoid only testing borders; include deterministic interior spread.
    xs = np.linspace(1, max(1, nx - 2), count, dtype=np.int64)
    ys = np.linspace(1, max(1, ny - 2), count, dtype=np.int64)
    zs = np.linspace(1, max(1, nz - 2), count, dtype=np.int64)
    indices = [(int(xs[i]), int(ys[(i * 7) % count]), int(zs[(i * 11) % count])) for i in range(count)]
    center = np.asarray(voxel_grid.pose, dtype=np.float64).reshape(-1)[:3]
    voxel_size = float(voxel_grid.voxel_size)
    dims = np.asarray(project_feature.shape, dtype=np.float64)
    rows = []
    project_err = []
    curobo_err = []
    for ix, iy, iz in indices:
        direct = float(project_feature[ix, iy, iz])
        xyz = center + (np.asarray([ix, iy, iz], dtype=np.float64) - dims / 2.0 + 0.5) * voxel_size
        p_sdf = float(_project_sdf(voxel_grid, xyz.reshape(1, 3))[0])
        c_sdf = float(_trilinear_sample_align_corners(
            cu_sdf_meta["feature3"],
            xyz.reshape(1, 3),
            cu_sdf_meta["center_base"],
            cu_sdf_meta["voxel_size"],
        )[0][0])
        project_err.append(abs(p_sdf - direct))
        curobo_err.append(abs(c_sdf - direct))
        rows.append(
            {
                "index_xyz": [ix, iy, iz],
                "center_base_m": xyz.tolist(),
                "direct_feature": direct,
                "project_query": p_sdf,
                "curobo_semantic_query": c_sdf,
                "abs_error_project": abs(p_sdf - direct),
                "abs_error_curobo": abs(c_sdf - direct),
            }
        )
    return rows, _stats(np.asarray(project_err)), _stats(np.asarray(curobo_err))


def _project_query_contract() -> dict[str, Any]:
    path = ISAACLAB_CONTROL_ROOT / "core/perception_collision/esdf_collision.py"
    text = path.read_text(encoding="utf-8")
    if "normalized[:, [2, 1, 0]]" in text:
        order = "zyx"
    elif "normalized[:, [0, 1, 2]]" in text:
        order = "xyz"
    else:
        order = "unknown"
    return {
        "source_file": str(path),
        "tensor_shape_order": "[nx, ny, nz]",
        "grid_sample_coordinate_order": order,
        "align_corners": "True" if "align_corners=True" in text else "unknown",
        "half_extent_formula": "(n - 1) * voxel_size / 2",
        "voxel_center_convention": "center index maps to VoxelGrid.pose; align_corners=True",
    }


def _verdict(project_surface: dict[str, float], curobo_surface: dict[str, float]) -> str:
    p = float(project_surface["median_abs_sdf"])
    c = float(curobo_surface["median_abs_sdf"])
    # With voxel_size 0.02, near-surface should be O(0.01~0.02). Use relative
    # comparison instead of a hard physical threshold because Mapper ESDF may
    # represent voxelized surfaces with nonzero residuals.
    if p > 2.0 * c:
        return "PROJECT_QUERY_IMPLEMENTATION_WRONG"
    if c > 2.0 * p:
        return "CUROBO_SCENE_REPRESENTATION_WRONG"
    if p > 0.05 and c > 0.05:
        return "MAPPER_OR_FRAME_CONTRACT_WRONG"
    return "PROJECT_AND_CUROBO_QUERY_SURFACE_AGREE"


def _root_cause(verdict: str, dim_diag: dict[str, Any]) -> tuple[str, str]:
    if not dim_diag["trunc_matches_feature_shape"] and dim_diag["round_matches_feature_shape"]:
        return (
            "CUROBO_SCENE_REPRESENTATION_WRONG: VoxelData params dimensions are stored as floats "
            f"{dim_diag['curobo_params_dims_float']} and cuRobo kernel truncates them to "
            f"{dim_diag['curobo_kernel_int32_trunc_dims']}, while the feature tensor shape is "
            f"{dim_diag['project_feature_shape']}. This corrupts X-slowest/Z-fastest flat indexing.",
            "Ensure Route B hands cuRobo VoxelData exact integer dimensions matching feature_tensor shape, "
            "or patch the adapter-side SceneCfg/VoxelGrid construction so params dims cannot become 26.999998. "
            "Do not change voxel pose, ESDF values, thresholds, or planner parameters.",
        )
    if verdict == "CUROBO_SCENE_REPRESENTATION_WRONG":
        return (
            "CUROBO_SCENE_REPRESENTATION_WRONG: project query matches surface points and direct voxel reads, "
            "but cuRobo VoxelData semantics do not.",
            "Audit Route B SceneCfg/VoxelData creation for feature layout, dimensions, and pose consistency.",
        )
    if verdict == "PROJECT_QUERY_IMPLEMENTATION_WRONG":
        return (
            "PROJECT_QUERY_IMPLEMENTATION_WRONG: cuRobo semantic query is closer to filtered_depth surface points.",
            "Fix project query_spheres/grid_sample semantics only.",
        )
    if verdict == "MAPPER_OR_FRAME_CONTRACT_WRONG":
        return (
            "MAPPER_OR_FRAME_CONTRACT_WRONG: neither query is close to filtered_depth surface points.",
            "Audit CameraObservation pose and depth-to-base transform.",
        )
    return (
        "PROJECT_AND_CUROBO_QUERY_SURFACE_AGREE: both are close to filtered_depth surface points.",
        "No ESDF query semantic fix indicated by this audit.",
    )


def run(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    output_dir = capture_dir / "curobo_test_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = RouteBMotionPlannerAdapter(
        {
            "routeB": {
                "device": args.device,
                "collision": {"environment_collision": True, "self_collision": False},
            }
        }
    )
    scene = adapter.build_pick_scene(
        capture_dir / "planning/filtered_depth.npy",
        capture_dir / "intrinsics.npy",
        capture_dir / "T_world_camera.npy",
    )
    planner = adapter.create_planner(scene)
    scene_cost = planner.trajopt_solver.metrics_rollout.metrics_constraint_manager.costs["scene_collision"]

    surface_points = _load_surface_points_base(capture_dir, adapter, args.surface_count)
    project_sdf = _project_sdf(scene.voxel[0], surface_points)
    curobo_sdf, curobo_inside, curobo_meta = _curobo_semantic_sdf(scene_cost, surface_points)
    raw_collision = _raw_collision_probe(scene_cost, surface_points, args.device)
    voxel_rows, voxel_project_err, voxel_curobo_err = _voxel_center_checks(scene.voxel[0], scene_cost, args.voxel_count)
    dim_diag = _dimension_diagnostic(scene.voxel[0], scene_cost)

    project_stats = _stats(project_sdf)
    curobo_stats = _stats(curobo_sdf)
    verdict = _verdict(project_stats, curobo_stats)
    root_cause, minimal_fix = _root_cause(verdict, dim_diag)
    report = {
        "schema_version": 1,
        "route": "RouteB",
        "audit": "esdf_ground_truth_surface_points",
        "inputs": {
            "capture_dir": str(capture_dir),
            "filtered_depth": str(capture_dir / "planning/filtered_depth.npy"),
            "intrinsics": str(capture_dir / "intrinsics.npy"),
            "T_world_camera": str(capture_dir / "T_world_camera.npy"),
        },
        "surface_points": {
            "count": int(len(surface_points)),
            "frame": "arm_base_link",
            "project": project_stats,
            "curobo": curobo_stats,
            "curobo_inside_grid_count": int(np.count_nonzero(curobo_inside)),
            "raw_cuda_collision_radius0_weight1": {
                "note": "cuRobo raw checker with radius=0 returns max(-sdf,0), not full signed positive distance",
                "positive_count": int(np.count_nonzero(raw_collision > 0.0)),
                "max": float(np.max(raw_collision)) if raw_collision.size else 0.0,
                "median": float(np.median(raw_collision)) if raw_collision.size else 0.0,
            },
        },
        "project_query_contract": _project_query_contract(),
        "curobo_query_contract": curobo_meta,
        "dimension_diagnostic": dim_diag,
        "voxel_center_checks": voxel_rows,
        "voxel_center_error_summary": {
            "direct_read_vs_project_query": voxel_project_err,
            "direct_read_vs_curobo_semantic_query": voxel_curobo_err,
        },
        "verdict": verdict,
        "root_cause": root_cause,
        "minimal_fix": minimal_fix,
    }
    out = output_dir / "esdf_ground_truth_audit.json"
    out.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    print("[Route B ESDF ground-truth audit]")
    print(f"project_axis_order={report['project_query_contract']['grid_sample_coordinate_order']}")
    print(f"surface_count={len(surface_points)}")
    print(
        "project surface abs sdf: "
        f"median={project_stats['median_abs_sdf']:.6f} p90={project_stats['p90_abs_sdf']:.6f}"
    )
    print(
        "curobo surface abs sdf: "
        f"median={curobo_stats['median_abs_sdf']:.6f} p90={curobo_stats['p90_abs_sdf']:.6f}"
    )
    print(
        "voxel direct-read error project/curobo median: "
        f"{voxel_project_err['median_abs_sdf']:.9f} / {voxel_curobo_err['median_abs_sdf']:.9f}"
    )
    print(f"dimension_issue={dim_diag['issue']}")
    print(f"verdict={report['verdict']}")
    print(f"root_cause={root_cause}")
    print(f"report={out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--surface-count", type=int, default=500)
    parser.add_argument("--voxel-count", type=int, default=20)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
