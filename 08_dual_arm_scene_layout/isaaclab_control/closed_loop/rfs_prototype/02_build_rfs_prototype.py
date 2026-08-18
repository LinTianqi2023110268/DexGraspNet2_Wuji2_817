#!/usr/bin/env python3
"""Standalone RFS v1 prototype for one captured closed-loop cycle.

Goal
----
Build a *coarse* Reachable Free-Space (RFS) volume before LEAP->Wuji2 retargeting:

  right-arm reachability  ∩  RGB-D/ESDF collision-free  ∩  HOME-connected

The result is a spatial region, not one final trajectory.  It is intentionally a
coarse front-end filter and must NOT replace post-retarget exact COVER IK or final
motion/collision verification.

This script:
  1. loads the existing RGB-D capture + SAM target mask;
  2. reuses the project's cuRobo V2 IK, RGB-D Mapper, and robot-sphere model;
  3. creates a world-space voxel grid spanning HOME to the SAM target neighborhood;
  4. tests several task-relevant flange orientations per voxel;
  5. keeps one collision-free right-arm IK state per voxel;
  6. finds the HOME-connected component with local q-continuity + coarse edge checks;
  7. projects the RFS volume and DGN2 coarse-filter decisions back to the RGB camera;
  8. optionally scores retention/rejection against the previous exact-COVER diagnosis.

It does NOT start Isaac Sim and does NOT modify the production pipeline.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


STATUS_UNREACHABLE = 0
STATUS_REACHABLE_BLOCKED = 1
STATUS_FREE_UNCONNECTED = 2
STATUS_HOME_CONNECTED = 3
STATUS_NAME = {
    STATUS_UNREACHABLE: "UNREACHABLE",
    STATUS_REACHABLE_BLOCKED: "REACHABLE_BUT_POINTCLOUD_BLOCKED",
    STATUS_FREE_UNCONNECTED: "FREE_BUT_NOT_HOME_CONNECTED",
    STATUS_HOME_CONNECTED: "HOME_CONNECTED_RFS",
}


def safe_slug(text: str) -> str:
    import re
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rotation_angle_rad(R: np.ndarray) -> float:
    x = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(x))


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = [float(x) for x in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(float(angle)), math.sin(float(angle))
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None or not text.strip():
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in text.split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3-vector, got {text!r}")
    return np.asarray(vals, dtype=np.float64)


def urdf_fk_to_link(urdf_path: Path, joint_positions: dict[str, float], target_link: str) -> np.ndarray:
    """Minimal standard-URDF FK used only to locate the captured HOME flange pose."""
    root = ET.parse(urdf_path).getroot()
    links = {x.attrib["name"] for x in root.findall("link")}
    joints = []
    children = set()
    by_child = {}
    for j in root.findall("joint"):
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue
        row = {
            "name": j.attrib.get("name", ""),
            "type": j.attrib.get("type", "fixed"),
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "origin": j.find("origin"),
            "axis": j.find("axis"),
        }
        joints.append(row)
        children.add(row["child"])
        by_child[row["child"]] = row

    roots = sorted(links - children)
    if len(roots) != 1:
        raise RuntimeError(f"URDF root link is not unique: {roots}")
    if target_link not in links:
        raise KeyError(f"URDF has no target link {target_link!r}")

    chain = []
    link = target_link
    while link != roots[0]:
        if link not in by_child:
            raise RuntimeError(f"Cannot trace {target_link} back to root {roots[0]}; stuck at {link}")
        joint = by_child[link]
        chain.append(joint)
        link = joint["parent"]
    chain.reverse()

    T = np.eye(4, dtype=np.float64)
    for joint in chain:
        origin = joint["origin"]
        xyz = parse_vec(None if origin is None else origin.attrib.get("xyz"), (0.0, 0.0, 0.0))
        rpy = parse_vec(None if origin is None else origin.attrib.get("rpy"), (0.0, 0.0, 0.0))
        To = np.eye(4, dtype=np.float64)
        To[:3, :3] = rpy_matrix(rpy)
        To[:3, 3] = xyz
        T = T @ To

        jt = joint["type"]
        q = float(joint_positions.get(joint["name"], 0.0))
        axis = parse_vec(None if joint["axis"] is None else joint["axis"].attrib.get("xyz"), (1.0, 0.0, 0.0))
        Tv = np.eye(4, dtype=np.float64)
        if jt in ("revolute", "continuous"):
            Tv[:3, :3] = axis_angle_matrix(axis, q)
        elif jt == "prismatic":
            axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
            Tv[:3, 3] = axis * q
        elif jt == "fixed":
            pass
        else:
            raise RuntimeError(f"Unsupported URDF joint type {jt!r} for {joint['name']}")
        T = T @ Tv
    return T


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def target_world_points(depth: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool) & np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise RuntimeError("SAM mask contains no valid depth pixels")
    z = depth[v, u].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    pc = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (T_world_camera @ pc.T).T[:, :3]


def make_axes(lo: np.ndarray, hi: np.ndarray, step: float, max_points: int) -> tuple[list[np.ndarray], float]:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    effective = float(step)
    for _ in range(20):
        axes = [np.arange(lo[i], hi[i] + 0.5 * effective, effective, dtype=np.float64) for i in range(3)]
        count = int(np.prod([len(x) for x in axes]))
        if count <= int(max_points):
            return axes, effective
        effective *= 1.10
    raise RuntimeError("Could not fit RFS grid below max_grid_points")


def orientation_distance_matrix(candidates: np.ndarray, selected: list[np.ndarray]) -> np.ndarray:
    if not selected:
        return np.full(len(candidates), np.inf, dtype=np.float64)
    out = np.full(len(candidates), np.inf, dtype=np.float64)
    for S in selected:
        for i, R in enumerate(candidates):
            out[i] = min(out[i], rotation_angle_rad(S.T @ R))
    return out


def select_orientation_bins(home_R: np.ndarray, candidate_R: np.ndarray, count: int) -> np.ndarray:
    selected = [np.asarray(home_R, dtype=np.float64)]
    if count <= 1 or len(candidate_R) == 0:
        return np.stack(selected)
    pool = np.asarray(candidate_R, dtype=np.float64)
    used = np.zeros(len(pool), dtype=bool)
    for _ in range(count - 1):
        d = orientation_distance_matrix(pool, selected)
        d[used] = -1.0
        idx = int(np.argmax(d))
        if d[idx] < math.radians(5.0):
            break
        selected.append(pool[idx])
        used[idx] = True
    return np.stack(selected)


def project_world(points_world: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    T_camera_world = np.linalg.inv(T_world_camera)
    ph = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    pc = (T_camera_world @ ph.T).T[:, :3]
    z = pc[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = z > 1.0e-6
    if np.any(valid):
        proj = (K @ pc[valid].T).T
        uv[valid] = proj[:, :2] / proj[:, 2:3]
    return uv, z


def load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    grid_points: np.ndarray,
    status: np.ndarray,
    home_point: np.ndarray,
    candidate_points: np.ndarray,
    candidate_pass: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    output: Path,
    candidate_draw_topk: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("RFS overlay requires Pillow in curobo_v2") from exc

    base = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    tint = np.zeros((*mask.shape, 4), dtype=np.uint8)
    tint[mask.astype(bool)] = np.array([255, 255, 255, 45], dtype=np.uint8)
    base = Image.alpha_composite(base, Image.fromarray(tint, mode="RGBA"))

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    H, W = mask.shape

    uv, z = project_world(grid_points, K, T_world_camera)
    colors = {
        STATUS_UNREACHABLE: (220, 45, 45, 95),
        STATUS_REACHABLE_BLOCKED: (255, 140, 20, 125),
        STATUS_FREE_UNCONNECTED: (245, 205, 40, 115),
        STATUS_HOME_CONNECTED: (35, 210, 70, 145),
    }
    order = np.argsort(-z)  # far -> near
    for i in order:
        if not np.isfinite(uv[i]).all() or z[i] <= 0.0:
            continue
        u, v = float(uv[i, 0]), float(uv[i, 1])
        if u < 0 or u >= W or v < 0 or v >= H:
            continue
        r = 2 if int(status[i]) != STATUS_HOME_CONNECTED else 3
        c = colors[int(status[i])]
        draw.ellipse((u - r, v - r, u + r, v + r), fill=c)

    huv, hz = project_world(np.asarray(home_point)[None, :], K, T_world_camera)
    if hz[0] > 0 and np.isfinite(huv[0]).all():
        u, v = [float(x) for x in huv[0]]
        if 0 <= u < W and 0 <= v < H:
            draw.ellipse((u - 7, v - 7, u + 7, v + 7), fill=(40, 120, 255, 230), outline=(255, 255, 255, 255), width=2)

    topk = min(int(candidate_draw_topk), len(candidate_points))
    if topk > 0:
        cuv, cz = project_world(candidate_points[:topk], K, T_world_camera)
        for i in range(topk):
            if cz[i] <= 0 or not np.isfinite(cuv[i]).all():
                continue
            u, v = [float(x) for x in cuv[i]]
            if not (0 <= u < W and 0 <= v < H):
                continue
            if bool(candidate_pass[i]):
                draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(0, 255, 130, 240), outline=(0, 0, 0, 220))
            else:
                draw.line((u - 3, v - 3, u + 3, v + 3), fill=(255, 30, 30, 230), width=2)
                draw.line((u - 3, v + 3, u + 3, v - 3), fill=(255, 30, 30, 230), width=2)

    result = Image.alpha_composite(base, layer).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def nearest_distances(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    if len(reference) == 0:
        return np.full(len(query), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(reference)
        d, _ = tree.query(query, k=1)
        return np.asarray(d, dtype=np.float64)
    except Exception:
        out = np.empty(len(query), dtype=np.float64)
        block = 512
        for start in range(0, len(query), block):
            q = query[start:start + block]
            d2 = np.sum((q[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
            out[start:start + block] = np.sqrt(np.min(d2, axis=1))
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home() / "Projects/DexGraspNet2_Wuji2")
    parser.add_argument("--cycle-root", type=Path, required=True)
    parser.add_argument("--query", default="bottle")
    parser.add_argument("--bridge-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cover-diagnostic-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ik-seeds", type=int, default=24)
    parser.add_argument("--ik-batch-size", type=int, default=512)
    parser.add_argument("--coarse-joint-margin-deg", type=float, default=0.0)
    parser.add_argument("--orientation-bins", type=int, default=8)
    parser.add_argument("--orientation-source-topk", type=int, default=256)
    parser.add_argument("--grid-step-m", type=float, default=0.05)
    parser.add_argument("--max-grid-points", type=int, default=2200)
    parser.add_argument("--target-padding-xy-m", type=float, default=0.20)
    parser.add_argument("--target-padding-z-below-m", type=float, default=0.08)
    parser.add_argument("--target-padding-z-above-m", type=float, default=0.25)
    parser.add_argument("--corridor-side-padding-m", type=float, default=0.12)
    parser.add_argument("--collision-margin-m", type=float, default=0.005)
    parser.add_argument("--moving-link-prefix", action="append", default=["arm_r_"], help="Can be repeated")
    parser.add_argument("--seeds-per-pose-to-check", type=int, default=4)
    parser.add_argument("--neighbor-max-joint-delta-deg", type=float, default=28.0)
    parser.add_argument("--home-max-joint-delta-deg", type=float, default=45.0)
    parser.add_argument("--edge-step-deg", type=float, default=10.0)
    parser.add_argument("--block-unknown", action="store_true")
    parser.add_argument("--check-self-collision", action="store_true")
    parser.add_argument("--candidate-draw-topk", type=int, default=300)
    args = parser.parse_args()

    started = time.perf_counter()
    project_root = args.project_root.expanduser().resolve()
    cycle_root = args.cycle_root.expanduser().resolve()
    query_slug = safe_slug(args.query)
    capture = cycle_root / "capture"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else cycle_root / "rfs_prototype"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_npz = (
        args.bridge_npz.expanduser().resolve()
        if args.bridge_npz is not None
        else output_dir / "bridge_calibration.npz"
    )
    if not bridge_npz.is_file():
        raise FileNotFoundError(f"Run 01_calibrate_leap_wuji_bridge.py first: {bridge_npz}")

    rgb_path = capture / "rgb.png"
    depth_path = capture / "depth_m.npy"
    K_path = capture / "intrinsics.npy"
    Twc_path = capture / "T_world_camera.npy"
    mask_path = capture / "grounded_sam" / query_slug / "mask.npy"
    dgn_path = capture / "dgn2" / query_slug / "official_leap_1024_target_ranked.npz"
    robot_state_path = capture / "robot_state.json"
    robot_urdf = project_root / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
    collision_yaml = project_root / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
    for path in [rgb_path, depth_path, K_path, Twc_path, mask_path, dgn_path, robot_state_path, robot_urdf, collision_yaml]:
        if not path.is_file():
            raise FileNotFoundError(path)

    control_root = project_root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(control_root))
    from core.config import IKConfig, MapperConfig, RIGHT_ARM_NAMES  # noqa: E402
    from core.ik import CuroboGpuIK  # noqa: E402
    from core.perception_collision import RGBDFrame, CuroboRGBDMapper, CuroboRobotSphereModel  # noqa: E402

    print("[RFS 1/8] loading capture + SAM + DGN2 ...", flush=True)
    rgb = load_rgb(rgb_path)
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    K = np.asarray(np.load(K_path), dtype=np.float64)
    T_world_camera = np.asarray(np.load(Twc_path), dtype=np.float64)
    mask = np.asarray(np.load(mask_path), dtype=bool)
    robot_state = load_json(robot_state_path)
    measured = {str(k): float(v) for k, v in robot_state["joint_positions_by_name"].items()}
    q_home = np.asarray([measured[name] for name in RIGHT_ARM_NAMES], dtype=np.float64)
    T_world_base = world_from_base(project_root)

    with np.load(bridge_npz, allow_pickle=False) as z:
        T_leap_wuji_mean = np.asarray(z["T_leap_from_wuji2_wrist_mean"], dtype=np.float64)
        flange_from_wrist = np.asarray(z["flange_from_wuji2_wrist"], dtype=np.float64)
        bridge_inflation_m = float(np.asarray(z["recommended_position_inflation_m"]).reshape(()))
    wrist_from_flange = np.linalg.inv(flange_from_wrist)

    with np.load(dgn_path, allow_pickle=False) as z:
        dgn_R = np.asarray(z["rotation_world"], dtype=np.float64)
        dgn_t = np.asarray(z["translation_world"], dtype=np.float64)
        dgn_score = np.asarray(z["score"], dtype=np.float64)
        target_sorted = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)

    target_pts = target_world_points(depth, K, T_world_camera, mask)
    target_lo = np.percentile(target_pts, 2.0, axis=0)
    target_hi = np.percentile(target_pts, 98.0, axis=0)
    target_center = np.median(target_pts, axis=0)

    # HOME flange from measured q, using the same arm_r_link_tf tool frame as project IK.
    T_base_home_flange = urdf_fk_to_link(robot_urdf, measured, "arm_r_link_tf")
    T_world_home_flange = T_world_base @ T_base_home_flange
    home_xyz = T_world_home_flange[:3, 3]
    home_R = T_world_home_flange[:3, :3]

    print(
        f"    target center={np.round(target_center,4).tolist()} | HOME flange={np.round(home_xyz,4).tolist()}",
        flush=True,
    )

    print("[RFS 2/8] building corridor grid + task-relevant orientation bins ...", flush=True)
    target_lo_pad = target_lo - np.array([args.target_padding_xy_m, args.target_padding_xy_m, args.target_padding_z_below_m])
    target_hi_pad = target_hi + np.array([args.target_padding_xy_m, args.target_padding_xy_m, args.target_padding_z_above_m])
    lo = np.minimum(target_lo_pad, home_xyz - args.corridor_side_padding_m)
    hi = np.maximum(target_hi_pad, home_xyz + args.corridor_side_padding_m)
    axes, grid_step = make_axes(lo, hi, args.grid_step_m, args.max_grid_points)
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    grid_shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    print(f"    grid={grid_shape} -> {len(grid_points)} voxels | step={1000*grid_step:.1f} mm", flush=True)

    top_source = target_sorted[: min(args.orientation_source_topk, len(target_sorted))]
    approx_flange_top = []
    for idx in top_source:
        T_world_leap = np.eye(4, dtype=np.float64)
        T_world_leap[:3, :3] = dgn_R[int(idx)]
        T_world_leap[:3, 3] = dgn_t[int(idx)]
        approx_flange_top.append(T_world_leap @ T_leap_wuji_mean @ wrist_from_flange)
    approx_flange_top = np.stack(approx_flange_top)
    orientation_bins = select_orientation_bins(home_R, approx_flange_top[:, :3, :3], args.orientation_bins)
    print(f"    orientation bins={len(orientation_bins)} (includes measured HOME orientation)", flush=True)

    print("[RFS 3/8] cuRobo coarse batched IK over spatial volume ...", flush=True)
    T_base_world = np.linalg.inv(T_world_base)
    pose_count = len(grid_points) * len(orientation_bins)
    targets_base = np.empty((pose_count, 4, 4), dtype=np.float64)
    pose_voxel_index = np.empty(pose_count, dtype=np.int64)
    pose_orientation_index = np.empty(pose_count, dtype=np.int16)
    k = 0
    for vi, p in enumerate(grid_points):
        for oi, Rw in enumerate(orientation_bins):
            Tw = np.eye(4, dtype=np.float64)
            Tw[:3, :3] = Rw
            Tw[:3, 3] = p
            targets_base[k] = T_base_world @ Tw
            pose_voxel_index[k] = vi
            pose_orientation_index[k] = oi
            k += 1

    ik_cfg = IKConfig(
        device=args.device,
        num_seeds=args.ik_seeds,
        batch_size=args.ik_batch_size,
        return_seeds=args.ik_seeds,
        minimum_inner_limit_margin_rad=math.radians(args.coarse_joint_margin_deg),
    )
    ik_solver = CuroboGpuIK(robot_urdf, ik_cfg)
    ik_started = time.perf_counter()
    ik_result = ik_solver.solve(targets_base, return_seeds=args.ik_seeds)
    print(
        f"    poses={pose_count} | accepted-pose-count={int(np.count_nonzero(np.any(ik_result.accepted,axis=1)))} "
        f"| wall={time.perf_counter()-ik_started:.1f}s",
        flush=True,
    )

    print("[RFS 4/8] RGB-D -> cuRobo ESDF + right-arm endpoint collision screening ...", flush=True)
    frame = RGBDFrame.from_npy(depth_path, K_path, Twc_path, mask_path)
    mapper = CuroboRGBDMapper(MapperConfig(device=args.device))
    observed_map = mapper.build(frame)
    sphere_model = CuroboRobotSphereModel(collision_yaml, device=args.device)

    sphere_names = list(sphere_model.sphere_link_names)
    moving_mask = None
    if sphere_names:
        moving_mask = np.asarray(
            [any(str(name).startswith(prefix) for prefix in args.moving_link_prefix) for name in sphere_names],
            dtype=bool,
        )
        if not np.any(moving_mask):
            raise RuntimeError(
                f"moving-link prefixes {args.moving_link_prefix} match zero collision spheres; "
                f"sample links={sphere_names[:20]}"
            )
    else:
        print("    [WARN] sphere link labels unavailable; using all robot collision spheres", flush=True)

    status = np.full(len(grid_points), STATUS_UNREACHABLE, dtype=np.int8)
    q_selected = np.full((len(grid_points), 7), np.nan, dtype=np.float64)
    selected_orientation = np.full(len(grid_points), -1, dtype=np.int16)
    selected_clearance = np.full(len(grid_points), np.nan, dtype=np.float64)
    selected_unknown_count = np.full(len(grid_points), -1, dtype=np.int32)

    def named_with_q(q: np.ndarray) -> dict:
        named = dict(measured)
        for name, value in zip(RIGHT_ARM_NAMES, q):
            named[name] = float(value)
        return named

    collision_cache: dict[tuple[float, ...], tuple[bool, float, int]] = {}

    def q_collision_free(q: np.ndarray) -> tuple[bool, float, int]:
        key = tuple(np.round(np.asarray(q, dtype=np.float64), 4).tolist())
        if key in collision_cache:
            return collision_cache[key]
        named = named_with_q(q)
        if args.check_self_collision:
            self_report = sphere_model.check_self_collision(named)
            if not bool(self_report["self_collision_pass"]):
                collision_cache[key] = (False, -math.inf, 0)
                return collision_cache[key]
        spheres = sphere_model.spheres_from_named_joints(named, T_world_base)
        if moving_mask is not None and len(moving_mask) == len(spheres):
            spheres = spheres[moving_mask]
        c = observed_map.check_spheres(
            spheres[:, :3], spheres[:, 3], "pregrasp", args.collision_margin_m
        )
        blocking = np.asarray(c["blocking_collision"], dtype=bool)
        unknown = np.asarray(c["unknown"], dtype=bool)
        blocked = bool(np.any(blocking) or (args.block_unknown and np.any(unknown)))
        scene_d = np.asarray(c["scene_distance_m"], dtype=np.float64)
        clearance = float(np.min(scene_d - spheres[:, 3])) if len(scene_d) else math.inf
        result = (not blocked, clearance, int(np.count_nonzero(unknown)))
        collision_cache[key] = result
        return result

    voxel_pose_indices: list[list[int]] = [[] for _ in range(len(grid_points))]
    for pi, vi in enumerate(pose_voxel_index):
        voxel_pose_indices[int(vi)].append(pi)

    for vi, pose_indices in enumerate(voxel_pose_indices):
        candidates = []
        any_ik = False
        for pi in pose_indices:
            seeds = np.flatnonzero(ik_result.accepted[pi])
            if len(seeds) == 0:
                continue
            any_ik = True
            order = sorted(
                [int(s) for s in seeds],
                key=lambda s: (
                    float(np.max(np.abs(ik_result.q_rad[pi, s] - q_home))),
                    float(np.linalg.norm(ik_result.q_rad[pi, s] - q_home)),
                    -float(ik_result.inner_limit_margin_rad[pi, s]),
                ),
            )[: max(1, args.seeds_per_pose_to_check)]
            for s in order:
                q = np.asarray(ik_result.q_rad[pi, s], dtype=np.float64)
                ok, clearance, unknown_count = q_collision_free(q)
                if ok:
                    candidates.append(
                        (
                            float(np.linalg.norm(q - q_home)),
                            -float(ik_result.inner_limit_margin_rad[pi, s]),
                            q,
                            int(pose_orientation_index[pi]),
                            clearance,
                            unknown_count,
                        )
                    )
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            best = candidates[0]
            status[vi] = STATUS_FREE_UNCONNECTED
            q_selected[vi] = best[2]
            selected_orientation[vi] = best[3]
            selected_clearance[vi] = best[4]
            selected_unknown_count[vi] = best[5]
        elif any_ik:
            status[vi] = STATUS_REACHABLE_BLOCKED
        if vi % 100 == 0 or vi + 1 == len(grid_points):
            print(
                f"    endpoint screen {vi+1:4d}/{len(grid_points)} | "
                f"free={int(np.count_nonzero(status==STATUS_FREE_UNCONNECTED))} | "
                f"blocked={int(np.count_nonzero(status==STATUS_REACHABLE_BLOCKED))}",
                flush=True,
            )

    print("[RFS 5/8] extracting HOME-connected coarse path-space component ...", flush=True)
    nx, ny, nz = grid_shape

    def flat(ix: int, iy: int, iz: int) -> int:
        return (ix * ny + iy) * nz + iz

    def unflat(index: int) -> tuple[int, int, int]:
        ix = index // (ny * nz)
        rem = index % (ny * nz)
        iy = rem // nz
        iz = rem % nz
        return int(ix), int(iy), int(iz)

    neighbors6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    edge_cache: dict[tuple[int, int], bool] = {}
    edge_checks = 0

    def q_edge_free(q0: np.ndarray, q1: np.ndarray) -> bool:
        max_delta = float(np.max(np.abs(q1 - q0)))
        sample_count = max(2, int(math.ceil(max_delta / math.radians(args.edge_step_deg))) + 1)
        for si in range(1, sample_count - 1):
            a = si / (sample_count - 1)
            q = (1.0 - a) * q0 + a * q1
            ok, _clear, _unknown = q_collision_free(q)
            if not ok:
                return False
        return True

    free_indices = np.flatnonzero(status == STATUS_FREE_UNCONNECTED)
    connected = np.zeros(len(grid_points), dtype=bool)
    queue: deque[int] = deque()

    # Seed the spatial graph from free voxels close to the exact measured HOME flange.
    if len(free_indices):
        dxyz = np.linalg.norm(grid_points[free_indices] - home_xyz[None, :], axis=1)
        seed_order = free_indices[np.argsort(dxyz)]
        seed_radius = max(2.25 * grid_step, args.corridor_side_padding_m)
        for vi in seed_order[: min(40, len(seed_order))]:
            if float(np.linalg.norm(grid_points[vi] - home_xyz)) > seed_radius:
                break
            q = q_selected[vi]
            if np.max(np.abs(q - q_home)) > math.radians(args.home_max_joint_delta_deg):
                continue
            if q_edge_free(q_home, q):
                connected[vi] = True
                queue.append(int(vi))

    while queue:
        vi = queue.popleft()
        ix, iy, iz = unflat(vi)
        for dx, dy, dz in neighbors6:
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            if not (0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz):
                continue
            vj = flat(jx, jy, jz)
            if connected[vj] or status[vj] != STATUS_FREE_UNCONNECTED:
                continue
            q0, q1 = q_selected[vi], q_selected[vj]
            if np.max(np.abs(q1 - q0)) > math.radians(args.neighbor_max_joint_delta_deg):
                continue
            key = (min(vi, vj), max(vi, vj))
            if key not in edge_cache:
                edge_checks += 1
                edge_cache[key] = q_edge_free(q0, q1)
            if edge_cache[key]:
                connected[vj] = True
                queue.append(vj)

    status[connected] = STATUS_HOME_CONNECTED
    print(
        f"    HOME-connected={int(np.count_nonzero(connected))} / "
        f"free-endpoint={int(np.count_nonzero((status==STATUS_HOME_CONNECTED)|(status==STATUS_FREE_UNCONNECTED)))} "
        f"| edge checks={edge_checks}",
        flush=True,
    )

    print("[RFS 6/8] coarse-filtering DGN2 LEAP candidates against inflated HOME-connected RFS ...", flush=True)
    all_candidate_pose = []
    for idx in target_sorted:
        T_world_leap = np.eye(4, dtype=np.float64)
        T_world_leap[:3, :3] = dgn_R[int(idx)]
        T_world_leap[:3, 3] = dgn_t[int(idx)]
        all_candidate_pose.append(T_world_leap @ T_leap_wuji_mean @ wrist_from_flange)
    all_candidate_pose = np.stack(all_candidate_pose)
    candidate_points = all_candidate_pose[:, :3, 3]
    rfs_points = grid_points[status == STATUS_HOME_CONNECTED]
    nearest_rfs_m = nearest_distances(rfs_points, candidate_points)
    spatial_half_diagonal = 0.5 * math.sqrt(3.0) * grid_step
    coarse_keep_radius_m = bridge_inflation_m + spatial_half_diagonal
    candidate_pass = nearest_rfs_m <= coarse_keep_radius_m

    filter_rows = []
    for rank, (idx, p, d, keep) in enumerate(zip(target_sorted, candidate_points, nearest_rfs_m, candidate_pass)):
        filter_rows.append(
            {
                "target_rank": int(rank),
                "candidate_index": int(idx),
                "official_score": float(dgn_score[int(idx)]),
                "approx_flange_position_world_m": p.tolist(),
                "nearest_home_connected_rfs_m": None if not np.isfinite(d) else float(d),
                "status": "PASS" if bool(keep) else "REJECT",
                "reason": "WITHIN_INFLATED_HOME_CONNECTED_RFS" if bool(keep) else "OUTSIDE_INFLATED_HOME_CONNECTED_RFS",
            }
        )

    diag_validation = None
    diag_path = args.cover_diagnostic_json
    if diag_path is not None:
        diag_path = diag_path.expanduser().resolve()
    elif (Path.home() / "下载/bottle_cover_ik_diag_first8.json").is_file():
        diag_path = Path.home() / "下载/bottle_cover_ik_diag_first8.json"
    if diag_path is not None and diag_path.is_file():
        diag = load_json(diag_path)
        by_idx = {int(row["candidate_index"]): row for row in diag.get("records", [])}
        rows = []
        for fr in filter_rows:
            idx = int(fr["candidate_index"])
            if idx in by_idx:
                rows.append((by_idx[idx], fr))
        exact_accepted = [(d, f) for d, f in rows if d.get("classification") == "ACCEPTED"]
        no_raw = [(d, f) for d, f in rows if d.get("classification") == "NO_RAW_CUROBO_SUCCESS"]
        margin_only = [(d, f) for d, f in rows if str(d.get("classification", "")).startswith("RAW_SUCCESS_REJECTED_JOINT_MARGIN")]
        diag_validation = {
            "diagnostic_json": str(diag_path),
            "matched_count": len(rows),
            "exact_accepted_count": len(exact_accepted),
            "exact_accepted_retained_count": int(sum(1 for _d, f in exact_accepted if f["status"] == "PASS")),
            "no_raw_count": len(no_raw),
            "no_raw_rejected_count": int(sum(1 for _d, f in no_raw if f["status"] == "REJECT")),
            "joint_margin_only_count": len(margin_only),
            "joint_margin_only_retained_count": int(sum(1 for _d, f in margin_only if f["status"] == "PASS")),
            "accepted_details": [
                {
                    "target_rank": int(d["target_rank"]),
                    "candidate_index": int(d["candidate_index"]),
                    "rfs_status": f["status"],
                    "nearest_rfs_m": f["nearest_home_connected_rfs_m"],
                }
                for d, f in exact_accepted
            ],
        }
        print(
            "    exact-COVER validation: "
            f"accepted retained={diag_validation['exact_accepted_retained_count']}/{diag_validation['exact_accepted_count']} | "
            f"NO_RAW rejected={diag_validation['no_raw_rejected_count']}/{diag_validation['no_raw_count']}",
            flush=True,
        )
        if diag_validation["exact_accepted_retained_count"] < diag_validation["exact_accepted_count"]:
            print("    [WARN] RFS false-rejected at least one known exact-COVER PASS. Increase inflation / refine grid before integration.", flush=True)

    print("[RFS 7/8] saving map/report + camera-view overlay ...", flush=True)
    npz_path = output_dir / "rfs_map.npz"
    filter_path = output_dir / "dgn2_rfs_filter.json"
    report_path = output_dir / "rfs_report.json"
    overlay_path = output_dir / "rfs_overlay.png"

    np.savez_compressed(
        npz_path,
        grid_points_world_m=grid_points,
        grid_status=status,
        q_selected_rad=q_selected,
        selected_orientation_index=selected_orientation,
        selected_clearance_m=selected_clearance,
        selected_unknown_sphere_count=selected_unknown_count,
        grid_shape=np.asarray(grid_shape, dtype=np.int32),
        grid_axes_x_m=axes[0],
        grid_axes_y_m=axes[1],
        grid_axes_z_m=axes[2],
        grid_step_m=np.asarray(grid_step, dtype=np.float64),
        orientation_bins_world=orientation_bins,
        home_flange_world=T_world_home_flange,
        target_center_world_m=target_center,
        target_bbox_lo_world_m=target_lo,
        target_bbox_hi_world_m=target_hi,
        bridge_position_inflation_m=np.asarray(bridge_inflation_m),
        coarse_keep_radius_m=np.asarray(coarse_keep_radius_m),
    )

    filter_payload = {
        "schema_version": 1,
        "status": "PASS",
        "policy": "pre-retarget coarse position filter only; keep DGN2 original score order among PASS candidates",
        "bridge_position_inflation_m": bridge_inflation_m,
        "grid_half_diagonal_m": spatial_half_diagonal,
        "coarse_keep_radius_m": coarse_keep_radius_m,
        "candidate_count": len(filter_rows),
        "pass_count": int(np.count_nonzero(candidate_pass)),
        "reject_count": int(np.count_nonzero(~candidate_pass)),
        "rows": filter_rows,
    }
    filter_path.write_text(json.dumps(filter_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = {STATUS_NAME[s]: int(np.count_nonzero(status == s)) for s in STATUS_NAME}
    report = {
        "schema_version": 1,
        "status": "PASS",
        "architecture": "coarse right-arm reachability + RGB-D ESDF + HOME-connected local q graph",
        "project_root": str(project_root),
        "cycle_root": str(cycle_root),
        "query": args.query,
        "does_not_start_isaac": True,
        "does_not_modify_production_pipeline": True,
        "safety_contract": "RFS is a coarse front-end filter; exact post-retarget COVER IK and final path verification remain mandatory",
        "home_flange_world": T_world_home_flange.tolist(),
        "target_center_world_m": target_center.tolist(),
        "target_bbox_lo_world_m": target_lo.tolist(),
        "target_bbox_hi_world_m": target_hi.tolist(),
        "grid_shape": list(grid_shape),
        "grid_step_m": grid_step,
        "grid_point_count": len(grid_points),
        "orientation_bin_count": len(orientation_bins),
        "ik_pose_count": pose_count,
        "coarse_ik": {
            "seeds": args.ik_seeds,
            "position_tolerance_m": ik_cfg.position_tolerance_m,
            "orientation_tolerance_deg": math.degrees(ik_cfg.orientation_tolerance_rad),
            "minimum_inner_limit_margin_deg": args.coarse_joint_margin_deg,
        },
        "collision": {
            "moving_link_prefixes": args.moving_link_prefix,
            "margin_m": args.collision_margin_m,
            "block_unknown": bool(args.block_unknown),
            "check_self_collision": bool(args.check_self_collision),
            "map_id": observed_map.map_id,
            "map_center_world_m": observed_map.grid_center_world.tolist(),
            "map_extent_m": observed_map.extent_meters_xyz.tolist(),
        },
        "voxel_counts": counts,
        "edge_checks": edge_checks,
        "candidate_filter": {
            "bridge_position_inflation_m": bridge_inflation_m,
            "coarse_keep_radius_m": coarse_keep_radius_m,
            "input_candidates": len(candidate_pass),
            "pass": int(np.count_nonzero(candidate_pass)),
            "reject": int(np.count_nonzero(~candidate_pass)),
        },
        "exact_cover_diagnostic_validation": diag_validation,
        "outputs": {
            "rfs_map_npz": str(npz_path),
            "rfs_filter_json": str(filter_path),
            "rfs_overlay_png": str(overlay_path),
        },
        "wall_time_s": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_overlay(
        rgb=rgb,
        mask=mask,
        grid_points=grid_points,
        status=status,
        home_point=home_xyz,
        candidate_points=candidate_points,
        candidate_pass=candidate_pass,
        K=K,
        T_world_camera=T_world_camera,
        output=overlay_path,
        candidate_draw_topk=args.candidate_draw_topk,
    )

    print("[RFS 8/8] DONE", flush=True)
    print("=" * 84)
    print(f"UNREACHABLE                 : {counts['UNREACHABLE']}")
    print(f"REACHABLE_BUT_BLOCKED       : {counts['REACHABLE_BUT_POINTCLOUD_BLOCKED']}")
    print(f"FREE_BUT_NOT_HOME_CONNECTED : {counts['FREE_BUT_NOT_HOME_CONNECTED']}")
    print(f"HOME_CONNECTED_RFS          : {counts['HOME_CONNECTED_RFS']}")
    print(f"DGN2 coarse PASS            : {int(np.count_nonzero(candidate_pass))}/{len(candidate_pass)}")
    print(f"overlay : {overlay_path}")
    print(f"report  : {report_path}")
    print(f"filter  : {filter_path}")
    print(f"map     : {npz_path}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
