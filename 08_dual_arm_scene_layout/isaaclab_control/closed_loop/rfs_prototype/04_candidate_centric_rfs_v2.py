#!/usr/bin/env python3
"""Candidate-centric RFS V2 standalone prototype.

This prototype implements the contract requested for the *first* pre-retarget filter:

1) TARGET REACH REGION
   The objects being classified are DexGraspNet2 LEAP-hand grasp candidates, not
   RGB-D points.  Each LEAP root 6-D pose is mapped through the calibrated mean
   LEAP->Wuji2-wrist bridge to an approximate right-arm flange target only for the
   purpose of coarse arm IK.  The reachable set is represented back in candidate
   space as an inflated union around reachable LEAP grasp poses.

2) HOME -> PREGRASP -> GRASP ROUGH TRAJECTORY SPACE
   The official LEAP PREGRASP is constructed before retargeting using DexGraspNet2's
   0.1 m root retreat.  A set of diverse LEAP candidates supplies corridor anchors.
   Around every anchor, the script samples multiple arm-flange route tubes from HOME
   to PREGRASP (direct and lifted alternatives) and a local PREGRASP->GRASP approach
   tube.  cuRobo IK + full right-arm collision spheres + the observed non-target
   RGB-D ESDF are used to keep a HOME-connected layered graph.  The union of all
   successful branches is the rough trajectory *space*; it is intentionally not a
   single execution trajectory.

3) FIRST FILTER
   A DGN2 LEAP candidate passes only when:
     (a) its grasp belongs to the inflated target reach region, and
     (b) its PREGRASP/GRASP pair lies inside at least one successful rough trajectory
         branch family.

The RGB-D point cloud is used only as an obstacle field for the arm trajectory
space.  The script never labels RGB-D/point-cloud points as "reachable" or
"unreachable".

Safety contract
---------------
This is a conservative front-end coarse filter.  It does NOT start Isaac Sim, does
NOT modify the production pipeline, and must NOT replace post-retarget exact COVER
IK, final motion planning, or final collision/physics verification.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Iterable

import numpy as np


# -----------------------------------------------------------------------------
# Generic geometry / IO
# -----------------------------------------------------------------------------


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rotation_angle_rad(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64)
    x = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(x))


def parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None or not str(text).strip():
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in str(text).split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3-vector, got {text!r}")
    return np.asarray(vals, dtype=np.float64)


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
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= 1.0e-12 or abs(float(angle)) <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / n
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


def matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Robust rotation-matrix -> unit quaternion [w,x,y,z]."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s],
            dtype=np.float64,
        )
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-16)) * 2.0
            q = np.array(
                [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s],
                dtype=np.float64,
            )
        elif i == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-16)) * 2.0
            q = np.array(
                [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-16)) * 2.0
            q = np.array(
                [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s],
                dtype=np.float64,
            )
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    if q[0] < 0.0:
        q = -q
    return q


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_rotation(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    q0 = matrix_to_quaternion_wxyz(R0)
    q1 = matrix_to_quaternion_wxyz(R1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(t) * (q1 - q0)
        q /= max(float(np.linalg.norm(q)), 1.0e-12)
        return quaternion_wxyz_to_matrix(q)
    theta = math.acos(dot)
    s = math.sin(theta)
    a = math.sin((1.0 - float(t)) * theta) / s
    b = math.sin(float(t) * theta) / s
    return quaternion_wxyz_to_matrix(a * q0 + b * q1)


def urdf_fk_to_link(urdf_path: Path, joint_positions: dict[str, float], target_link: str) -> np.ndarray:
    """Minimal standard-URDF FK used to locate the measured HOME flange pose."""
    root = ET.parse(urdf_path).getroot()
    links = {x.attrib["name"] for x in root.findall("link")}
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
        elif jt != "fixed":
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


# -----------------------------------------------------------------------------
# LEAP candidate construction
# -----------------------------------------------------------------------------


def official_leap_pregrasp_local(retreat_m: float) -> np.ndarray:
    """DexGraspNet2 official root retreat used by compose_waypoints()."""
    canonical = np.asarray(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = canonical.T @ np.asarray([-float(retreat_m), 0.0, 0.0], dtype=np.float64)
    return T


def build_candidate_poses(
    dgn_R: np.ndarray,
    dgn_t: np.ndarray,
    target_sorted: np.ndarray,
    T_leap_from_wuji_mean: np.ndarray,
    wrist_from_flange: np.ndarray,
    pregrasp_retreat_m: float,
) -> dict[str, np.ndarray]:
    pre_local = official_leap_pregrasp_local(pregrasp_retreat_m)
    n = len(target_sorted)
    leap_grasp = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], n, axis=0)
    leap_pre = leap_grasp.copy()
    approx_flange_grasp = leap_grasp.copy()
    approx_flange_pre = leap_grasp.copy()
    source_indices = np.asarray(target_sorted, dtype=np.int64)
    for rank, idx in enumerate(source_indices):
        Tg = np.eye(4, dtype=np.float64)
        Tg[:3, :3] = dgn_R[int(idx)]
        Tg[:3, 3] = dgn_t[int(idx)]
        Tp = Tg @ pre_local
        leap_grasp[rank] = Tg
        leap_pre[rank] = Tp
        approx_flange_grasp[rank] = Tg @ T_leap_from_wuji_mean @ wrist_from_flange
        approx_flange_pre[rank] = Tp @ T_leap_from_wuji_mean @ wrist_from_flange
    approach = approx_flange_grasp[:, :3, 3] - approx_flange_pre[:, :3, 3]
    norm = np.linalg.norm(approach, axis=1, keepdims=True)
    approach = approach / np.maximum(norm, 1.0e-12)
    return {
        "source_candidate_index": source_indices,
        "leap_grasp_world": leap_grasp,
        "leap_pregrasp_world": leap_pre,
        "approx_flange_grasp_world": approx_flange_grasp,
        "approx_flange_pregrasp_world": approx_flange_pre,
        "approx_approach_axis_world": approach,
    }


# -----------------------------------------------------------------------------
# Target reach-region membership
# -----------------------------------------------------------------------------


def pose_region_membership(
    query_T: np.ndarray,
    reference_T: np.ndarray,
    position_radius_m: float,
    orientation_radius_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return membership in an inflated union of reference SE(3) samples.

    Distances are evaluated in the approximate arm-flange frame because the inflation
    comes from the measured LEAP->final-Wuji2 bridge residual.  The candidate identity
    and visualization remain in LEAP space.
    """
    query_T = np.asarray(query_T, dtype=np.float64)
    reference_T = np.asarray(reference_T, dtype=np.float64)
    n = len(query_T)
    keep = np.zeros(n, dtype=bool)
    best_pos = np.full(n, np.inf, dtype=np.float64)
    best_rot = np.full(n, np.inf, dtype=np.float64)
    best_ref = np.full(n, -1, dtype=np.int64)
    if len(reference_T) == 0:
        return keep, best_pos, best_rot, best_ref

    qpos = query_T[:, :3, 3]
    rpos = reference_T[:, :3, 3]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(rpos)
        neighborhoods = tree.query_ball_point(qpos, r=float(position_radius_m))
        for i, nbrs in enumerate(neighborhoods):
            if not nbrs:
                d, j = tree.query(qpos[i], k=1)
                best_pos[i] = float(d)
                best_ref[i] = int(j)
                best_rot[i] = rotation_angle_rad(reference_T[int(j), :3, :3].T @ query_T[i, :3, :3])
                continue
            local = []
            for j in nbrs:
                dp = float(np.linalg.norm(qpos[i] - rpos[int(j)]))
                dr = rotation_angle_rad(reference_T[int(j), :3, :3].T @ query_T[i, :3, :3])
                local.append((max(dp / max(position_radius_m, 1.0e-9), dr / max(orientation_radius_rad, 1.0e-9)), dp, dr, int(j)))
            local.sort(key=lambda x: (x[0], x[1], x[2]))
            _score, dp, dr, j = local[0]
            best_pos[i], best_rot[i], best_ref[i] = dp, dr, j
            keep[i] = bool(dp <= position_radius_m and dr <= orientation_radius_rad)
    except Exception:
        # Small-N fallback with bounded chunks.
        block = 256
        for start in range(0, n, block):
            stop = min(n, start + block)
            for i in range(start, stop):
                dp = np.linalg.norm(rpos - qpos[i][None, :], axis=1)
                close = np.flatnonzero(dp <= position_radius_m)
                pool = close if len(close) else np.asarray([int(np.argmin(dp))], dtype=np.int64)
                local = []
                for j in pool:
                    dr = rotation_angle_rad(reference_T[int(j), :3, :3].T @ query_T[i, :3, :3])
                    local.append((max(float(dp[j]) / max(position_radius_m, 1.0e-9), dr / max(orientation_radius_rad, 1.0e-9)), float(dp[j]), dr, int(j)))
                local.sort(key=lambda x: (x[0], x[1], x[2]))
                _score, dpp, drr, j = local[0]
                best_pos[i], best_rot[i], best_ref[i] = dpp, drr, j
                keep[i] = bool(dpp <= position_radius_m and drr <= orientation_radius_rad)
    return keep, best_pos, best_rot, best_ref


# -----------------------------------------------------------------------------
# Corridor sampling
# -----------------------------------------------------------------------------


def orthogonal_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tangent, dtype=np.float64).reshape(3)
    t /= max(float(np.linalg.norm(t)), 1.0e-12)
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(t, ref))) > 0.90:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(t, ref)
    u /= max(float(np.linalg.norm(u)), 1.0e-12)
    v = np.cross(t, u)
    v /= max(float(np.linalg.norm(v)), 1.0e-12)
    return u, v


def cross_section_offsets(count: int) -> np.ndarray:
    base = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    if count >= 9:
        d = 1.0 / math.sqrt(2.0)
        base += [[d, d], [d, -d], [-d, d], [-d, -d]]
    if count > len(base):
        extra = count - len(base)
        for i in range(extra):
            a = 2.0 * math.pi * i / max(extra, 1)
            base.append([math.cos(a), math.sin(a)])
    return np.asarray(base[: max(1, count)], dtype=np.float64)


def sample_polyline(points: list[np.ndarray], layers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = [np.asarray(x, dtype=np.float64).reshape(3) for x in points]
    seg_len = np.asarray([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)], dtype=np.float64)
    total = float(np.sum(seg_len))
    if total <= 1.0e-9:
        centers = np.repeat(pts[0][None, :], max(2, layers), axis=0)
        tangents = np.repeat(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), len(centers), axis=0)
        frac = np.linspace(0.0, 1.0, len(centers))
        return centers, tangents, frac
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    distances = np.linspace(0.0, total, max(2, int(layers)))
    centers = []
    tangents = []
    for d in distances:
        si = min(int(np.searchsorted(cum, d, side="right") - 1), len(seg_len) - 1)
        si = max(0, si)
        local = (d - cum[si]) / max(float(seg_len[si]), 1.0e-12)
        p = (1.0 - local) * pts[si] + local * pts[si + 1]
        t = pts[si + 1] - pts[si]
        t /= max(float(np.linalg.norm(t)), 1.0e-12)
        centers.append(p)
        tangents.append(t)
    return np.stack(centers), np.stack(tangents), distances / total


@dataclass
class SupportPose:
    branch_id: int
    family: str
    stage: str
    layer: int
    offset_id: int
    world_from_flange: np.ndarray


@dataclass
class SupportSolution:
    q_rad: np.ndarray
    clearance_m: float
    unknown_count: int
    ik_margin_rad: float


def make_home_family_support(
    branch_id: int,
    family: str,
    home_T: np.ndarray,
    pre_T: np.ndarray,
    target_hi_z: float,
    layers: int,
    cross_count: int,
    radius_m: float,
    overhead_clearance_m: float,
) -> list[SupportPose]:
    home_p = np.asarray(home_T[:3, 3], dtype=np.float64)
    pre_p = np.asarray(pre_T[:3, 3], dtype=np.float64)
    if family == "home_direct":
        poly = [home_p, pre_p]
    elif family == "home_lifted":
        via = 0.5 * (home_p + pre_p)
        via[2] = max(float(home_p[2]), float(pre_p[2]), float(target_hi_z)) + float(overhead_clearance_m)
        poly = [home_p, via, pre_p]
    else:
        raise ValueError(family)
    centers, tangents, frac = sample_polyline(poly, layers)
    offsets = cross_section_offsets(cross_count)
    out = []
    for li, (center, tangent, t) in enumerate(zip(centers, tangents, frac)):
        u, v = orthogonal_basis(tangent)
        # Narrow at endpoints, wide in the middle.  Keep nonzero endpoint width so
        # bridge uncertainty can still enter the trajectory space.
        radius = float(radius_m) * (0.35 + 0.65 * math.sin(math.pi * float(t)))
        R = slerp_rotation(home_T[:3, :3], pre_T[:3, :3], float(t))
        for oi, uv in enumerate(offsets):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = center + radius * (uv[0] * u + uv[1] * v)
            out.append(SupportPose(branch_id, family, "HOME_TO_PREGRASP", li, oi, T))
    return out


def make_approach_support(
    branch_id: int,
    pre_T: np.ndarray,
    grasp_T: np.ndarray,
    layers: int,
    cross_count: int,
    radius_m: float,
) -> list[SupportPose]:
    pre_p = np.asarray(pre_T[:3, 3], dtype=np.float64)
    grasp_p = np.asarray(grasp_T[:3, 3], dtype=np.float64)
    centers, tangents, frac = sample_polyline([pre_p, grasp_p], layers)
    offsets = cross_section_offsets(cross_count)
    out = []
    for li, (center, tangent, t) in enumerate(zip(centers, tangents, frac)):
        u, v = orthogonal_basis(tangent)
        radius = float(radius_m) * (1.0 - 0.35 * float(t))
        R = slerp_rotation(pre_T[:3, :3], grasp_T[:3, :3], float(t))
        for oi, uv in enumerate(offsets):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = center + radius * (uv[0] * u + uv[1] * v)
            out.append(SupportPose(branch_id, "approach", "PREGRASP_TO_GRASP", li, oi, T))
    return out


# -----------------------------------------------------------------------------
# Anchor selection
# -----------------------------------------------------------------------------


def select_anchor_ranks(
    eligible_ranks: np.ndarray,
    grasp_T: np.ndarray,
    count: int,
    source_topk: int,
    min_position_sep_m: float,
    min_orientation_sep_rad: float,
) -> list[int]:
    eligible = [int(r) for r in np.asarray(eligible_ranks, dtype=np.int64) if int(r) < int(source_topk)]
    if not eligible:
        return []
    selected: list[int] = []
    deferred: list[int] = []
    for rank in eligible:
        if not selected:
            selected.append(rank)
            if len(selected) >= count:
                break
            continue
        pos = grasp_T[rank, :3, 3]
        R = grasp_T[rank, :3, :3]
        nearest_pos = min(float(np.linalg.norm(pos - grasp_T[s, :3, 3])) for s in selected)
        nearest_rot = min(rotation_angle_rad(grasp_T[s, :3, :3].T @ R) for s in selected)
        if nearest_pos >= min_position_sep_m or nearest_rot >= min_orientation_sep_rad:
            selected.append(rank)
            if len(selected) >= count:
                break
        else:
            deferred.append(rank)
    if len(selected) < count:
        for rank in deferred + eligible:
            if rank not in selected:
                selected.append(rank)
                if len(selected) >= count:
                    break
    return selected


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def _base_with_mask(rgb: np.ndarray, mask: np.ndarray):
    from PIL import Image
    base = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    tint = np.zeros((*mask.shape, 4), dtype=np.uint8)
    tint[np.asarray(mask, dtype=bool)] = np.array([255, 255, 255, 55], dtype=np.uint8)
    return Image.alpha_composite(base, Image.fromarray(tint, mode="RGBA"))


def _projected_radius_px(position_world: np.ndarray, radius_m: float, K: np.ndarray, T_world_camera: np.ndarray) -> float:
    uv, z = project_world(np.asarray(position_world)[None, :], K, T_world_camera)
    if z[0] <= 1.0e-6:
        return 2.0
    f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
    return float(np.clip(f * float(radius_m) / float(z[0]), 2.0, 45.0))


def save_target_reach_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    leap_grasp_world: np.ndarray,
    target_direct: np.ndarray,
    target_region: np.ndarray,
    overall_pass: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    region_radius_m: float,
    output: Path,
    draw_topk: int,
) -> None:
    from PIL import Image, ImageDraw
    base = _base_with_mask(rgb, mask)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    H, W = mask.shape
    topk = min(int(draw_topk), len(leap_grasp_world))
    pts = leap_grasp_world[:topk, :3, 3]
    uv, z = project_world(pts, K, T_world_camera)
    # Draw the union-like target region first as soft discs around reachable LEAP roots.
    for i in range(topk - 1, -1, -1):
        if not target_region[i] or z[i] <= 0 or not np.isfinite(uv[i]).all():
            continue
        u, v = map(float, uv[i])
        if not (0 <= u < W and 0 <= v < H):
            continue
        r = _projected_radius_px(pts[i], region_radius_m, K, T_world_camera)
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(30, 210, 80, 20))
    # Candidate markers: target-stage direct/rescued and final first-filter result.
    for i in range(topk):
        if z[i] <= 0 or not np.isfinite(uv[i]).all():
            continue
        u, v = map(float, uv[i])
        if not (0 <= u < W and 0 <= v < H):
            continue
        if target_direct[i]:
            c = (0, 245, 120, 235)
            rr = 3
            draw.ellipse((u - rr, v - rr, u + rr, v + rr), fill=c, outline=(0, 0, 0, 180))
        elif target_region[i]:
            c = (50, 220, 220, 230)
            rr = 3
            draw.ellipse((u - rr, v - rr, u + rr, v + rr), fill=c, outline=(0, 0, 0, 180))
        else:
            draw.line((u - 3, v - 3, u + 3, v + 3), fill=(255, 50, 50, 235), width=2)
            draw.line((u - 3, v + 3, u + 3, v - 3), fill=(255, 50, 50, 235), width=2)
        if overall_pass[i]:
            draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(255, 255, 255, 210), width=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, layer).convert("RGB").save(output)


def save_trajectory_space_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    home_xyz: np.ndarray,
    support_world: np.ndarray,
    connected_support: np.ndarray,
    graph_edges: np.ndarray,
    branch_success: np.ndarray,
    support_branch_id: np.ndarray,
    anchor_pre_world: np.ndarray,
    anchor_grasp_world: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    output: Path,
) -> None:
    from PIL import Image, ImageDraw
    base = _base_with_mask(rgb, mask)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    H, W = mask.shape
    uv, z = project_world(support_world, K, T_world_camera)
    # Draw only graph edges that belong to successful HOME->PRE->GRASP branches.
    for a, b in np.asarray(graph_edges, dtype=np.int64).reshape(-1, 2):
        if a < 0 or b < 0 or a >= len(support_world) or b >= len(support_world):
            continue
        bid = int(support_branch_id[b])
        if bid < 0 or bid >= len(branch_success) or not bool(branch_success[bid]):
            continue
        if not connected_support[a] or not connected_support[b]:
            continue
        if z[a] <= 0 or z[b] <= 0 or not np.isfinite(uv[a]).all() or not np.isfinite(uv[b]).all():
            continue
        ua, va = map(float, uv[a])
        ub, vb = map(float, uv[b])
        if max(ua, ub) < 0 or min(ua, ub) >= W or max(va, vb) < 0 or min(va, vb) >= H:
            continue
        draw.line((ua, va, ub, vb), fill=(25, 220, 170, 48), width=7)
        draw.line((ua, va, ub, vb), fill=(20, 180, 130, 120), width=2)
    # Connected support nodes are arm-state samples, not point-cloud points.
    for i in np.flatnonzero(connected_support):
        bid = int(support_branch_id[i])
        if bid < 0 or bid >= len(branch_success) or not bool(branch_success[bid]):
            continue
        if z[i] <= 0 or not np.isfinite(uv[i]).all():
            continue
        u, v = map(float, uv[i])
        if 0 <= u < W and 0 <= v < H:
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(30, 235, 170, 100))
    huv, hz = project_world(np.asarray(home_xyz)[None, :], K, T_world_camera)
    if hz[0] > 0 and np.isfinite(huv[0]).all():
        u, v = map(float, huv[0])
        draw.ellipse((u - 7, v - 7, u + 7, v + 7), fill=(40, 120, 255, 235), outline=(255, 255, 255, 255), width=2)
    if len(anchor_pre_world):
        puv, pz = project_world(anchor_pre_world[:, :3, 3], K, T_world_camera)
        guv, gz = project_world(anchor_grasp_world[:, :3, 3], K, T_world_camera)
        for i in range(len(anchor_pre_world)):
            if not branch_success[i]:
                continue
            if pz[i] > 0 and np.isfinite(puv[i]).all():
                u, v = map(float, puv[i])
                if 0 <= u < W and 0 <= v < H:
                    draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(255, 210, 40, 230), width=2)
            if gz[i] > 0 and np.isfinite(guv[i]).all():
                u, v = map(float, guv[i])
                if 0 <= u < W and 0 <= v < H:
                    draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(40, 255, 100, 235), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, layer).convert("RGB").save(output)


def save_candidate_filter_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    leap_pre: np.ndarray,
    leap_grasp: np.ndarray,
    target_pass: np.ndarray,
    traj_pass: np.ndarray,
    overall_pass: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    output: Path,
    draw_topk: int,
    exact_accepted_ranks: set[int],
) -> None:
    from PIL import Image, ImageDraw
    base = _base_with_mask(rgb, mask)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    H, W = mask.shape
    topk = min(int(draw_topk), len(leap_grasp))
    puv, pz = project_world(leap_pre[:topk, :3, 3], K, T_world_camera)
    guv, gz = project_world(leap_grasp[:topk, :3, 3], K, T_world_camera)
    for i in range(topk - 1, -1, -1):
        if pz[i] <= 0 or gz[i] <= 0 or not np.isfinite(puv[i]).all() or not np.isfinite(guv[i]).all():
            continue
        up, vp = map(float, puv[i])
        ug, vg = map(float, guv[i])
        if max(up, ug) < 0 or min(up, ug) >= W or max(vp, vg) < 0 or min(vp, vg) >= H:
            continue
        if overall_pass[i]:
            color = (30, 235, 100, 200)
        elif not target_pass[i]:
            color = (255, 45, 45, 205)
        else:
            color = (255, 150, 20, 210)
        draw.line((up, vp, ug, vg), fill=color, width=2)
        r = 3
        draw.ellipse((ug - r, vg - r, ug + r, vg + r), fill=color)
        if i in exact_accepted_ranks:
            draw.ellipse((ug - 7, vg - 7, ug + 7, vg + 7), outline=(30, 240, 255, 255), width=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, layer).convert("RGB").save(output)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home() / "Projects/DexGraspNet2_Wuji2")
    parser.add_argument("--cycle-root", type=Path, required=True)
    parser.add_argument("--query", default="bottle")
    parser.add_argument("--bridge-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cover-diagnostic-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")

    # Candidate endpoint IK: intentionally coarse, no 3-deg production margin.
    parser.add_argument("--endpoint-ik-seeds", type=int, default=24)
    parser.add_argument("--endpoint-ik-batch-size", type=int, default=512)
    parser.add_argument("--coarse-joint-margin-deg", type=float, default=0.0)
    parser.add_argument("--pregrasp-retreat-m", type=float, default=0.10)

    # Candidate reach-region inflation comes primarily from bridge calibration.
    parser.add_argument("--target-extra-position-inflation-m", type=float, default=0.0)
    parser.add_argument("--target-extra-orientation-inflation-deg", type=float, default=0.0)

    # Diverse candidate anchors for shared HOME->PRE->GRASP trajectory space.
    parser.add_argument("--anchor-count", type=int, default=32)
    parser.add_argument("--anchor-source-topk", type=int, default=2048)
    parser.add_argument("--anchor-min-position-sep-m", type=float, default=0.04)
    parser.add_argument("--anchor-min-orientation-sep-deg", type=float, default=15.0)

    # HOME->PREGRASP corridor tubes.
    parser.add_argument("--home-direct-layers", type=int, default=11)
    parser.add_argument("--home-lifted-layers", type=int, default=15)
    parser.add_argument("--home-cross-section-points", type=int, default=5)
    parser.add_argument("--home-corridor-radius-m", type=float, default=0.08)
    parser.add_argument("--overhead-clearance-m", type=float, default=0.15)

    # PREGRASP->GRASP approach tube.
    parser.add_argument("--approach-layers", type=int, default=6)
    parser.add_argument("--approach-cross-section-points", type=int, default=5)
    parser.add_argument("--approach-corridor-radius-m", type=float, default=0.025)

    # Support pose IK/collision graph.
    parser.add_argument("--corridor-ik-seeds", type=int, default=16)
    parser.add_argument("--corridor-ik-batch-size", type=int, default=512)
    parser.add_argument("--solutions-per-support-pose", type=int, default=2)
    parser.add_argument("--collision-margin-m", type=float, default=0.005)
    parser.add_argument("--moving-link-prefix", action="append", default=["arm_r_"], help="Can be repeated")
    parser.add_argument("--block-unknown", action="store_true")
    parser.add_argument("--check-self-collision", action="store_true")
    parser.add_argument("--edge-max-joint-delta-deg", type=float, default=38.0)
    parser.add_argument("--home-seed-max-joint-delta-deg", type=float, default=50.0)
    parser.add_argument("--edge-step-deg", type=float, default=10.0)
    parser.add_argument("--edge-parent-trials", type=int, default=3)

    # Candidate membership in the successful branch union.  Position radii are
    # derived from bridge inflation + the sampled corridor width, with optional extra.
    parser.add_argument("--trajectory-extra-position-m", type=float, default=0.0)
    parser.add_argument("--trajectory-orientation-slack-deg", type=float, default=8.0)
    parser.add_argument("--trajectory-approach-slack-deg", type=float, default=5.0)

    parser.add_argument("--candidate-draw-topk", type=int, default=500)
    args = parser.parse_args()

    started = time.perf_counter()
    project_root = args.project_root.expanduser().resolve()
    cycle_root = args.cycle_root.expanduser().resolve()
    query_slug = safe_slug(args.query)
    capture = cycle_root / "capture"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else cycle_root / "rfs_prototype" / "v2_candidate_centric"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_npz = (
        args.bridge_npz.expanduser().resolve()
        if args.bridge_npz is not None
        else cycle_root / "rfs_prototype" / "bridge_calibration.npz"
    )
    if not bridge_npz.is_file():
        raise FileNotFoundError(f"Missing bridge calibration: {bridge_npz}")

    rgb_path = capture / "rgb.png"
    depth_path = capture / "depth_m.npy"
    K_path = capture / "intrinsics.npy"
    Twc_path = capture / "T_world_camera.npy"
    mask_path = capture / "grounded_sam" / query_slug / "mask.npy"
    dgn_path = capture / "dgn2" / query_slug / "official_leap_1024_target_ranked.npz"
    robot_state_path = capture / "robot_state.json"
    robot_urdf = project_root / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
    collision_yaml = project_root / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
    for p in [rgb_path, depth_path, K_path, Twc_path, mask_path, dgn_path, robot_state_path, robot_urdf, collision_yaml]:
        if not p.is_file():
            raise FileNotFoundError(p)

    core_root = project_root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(core_root))
    from core.config import IKConfig, MapperConfig, RIGHT_ARM_NAMES  # noqa: E402
    from core.ik import CuroboGpuIK  # noqa: E402
    from core.perception_collision import RGBDFrame, CuroboRGBDMapper, CuroboRobotSphereModel  # noqa: E402

    print("[V2 1/9] loading LEAP candidates + bridge + captured scene ...", flush=True)
    rgb = load_rgb(rgb_path)
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    K = np.asarray(np.load(K_path), dtype=np.float64)
    T_world_camera = np.asarray(np.load(Twc_path), dtype=np.float64)
    mask = np.asarray(np.load(mask_path), dtype=bool)
    robot_state = load_json(robot_state_path)
    measured = {str(k): float(v) for k, v in robot_state["joint_positions_by_name"].items()}
    q_home = np.asarray([measured[n] for n in RIGHT_ARM_NAMES], dtype=np.float64)
    T_world_base = world_from_base(project_root)
    T_base_world = np.linalg.inv(T_world_base)

    with np.load(bridge_npz, allow_pickle=False) as z:
        T_leap_from_wuji_mean = np.asarray(z["T_leap_from_wuji2_wrist_mean"], dtype=np.float64)
        flange_from_wrist = np.asarray(z["flange_from_wuji2_wrist"], dtype=np.float64)
        bridge_position_inflation_m = float(np.asarray(z["recommended_position_inflation_m"]).reshape(()))
        bridge_orientation_inflation_deg = float(np.asarray(z["recommended_orientation_inflation_deg"]).reshape(()))
    wrist_from_flange = np.linalg.inv(flange_from_wrist)

    with np.load(dgn_path, allow_pickle=False) as z:
        dgn_R = np.asarray(z["rotation_world"], dtype=np.float64)
        dgn_t = np.asarray(z["translation_world"], dtype=np.float64)
        dgn_score = np.asarray(z["score"], dtype=np.float64)
        target_sorted = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)

    candidates = build_candidate_poses(
        dgn_R,
        dgn_t,
        target_sorted,
        T_leap_from_wuji_mean,
        wrist_from_flange,
        args.pregrasp_retreat_m,
    )
    n_candidates = len(target_sorted)
    T_world_home_flange = T_world_base @ urdf_fk_to_link(robot_urdf, measured, "arm_r_link_tf")
    home_xyz = T_world_home_flange[:3, 3]
    target_pts = target_world_points(depth, K, T_world_camera, mask)
    target_center = np.median(target_pts, axis=0)
    target_hi = np.percentile(target_pts, 98.0, axis=0)
    print(
        f"    candidates={n_candidates} | target center={np.round(target_center,4).tolist()} | HOME flange={np.round(home_xyz,4).tolist()}",
        flush=True,
    )
    print(
        f"    bridge inflation={1000*bridge_position_inflation_m:.0f} mm / {bridge_orientation_inflation_deg:.1f} deg",
        flush=True,
    )

    print("[V2 2/9] candidate-specific coarse IK for LEAP GRASP and PREGRASP roots ...", flush=True)
    endpoint_cfg = IKConfig(
        device=args.device,
        num_seeds=args.endpoint_ik_seeds,
        batch_size=args.endpoint_ik_batch_size,
        return_seeds=args.endpoint_ik_seeds,
        minimum_inner_limit_margin_rad=math.radians(args.coarse_joint_margin_deg),
    )
    endpoint_solver = CuroboGpuIK(robot_urdf, endpoint_cfg)
    grasp_base = np.asarray([T_base_world @ T for T in candidates["approx_flange_grasp_world"]], dtype=np.float64)
    pre_base = np.asarray([T_base_world @ T for T in candidates["approx_flange_pregrasp_world"]], dtype=np.float64)
    ik_started = time.perf_counter()
    grasp_ik = endpoint_solver.solve(grasp_base, return_seeds=args.endpoint_ik_seeds)
    pre_ik = endpoint_solver.solve(pre_base, return_seeds=args.endpoint_ik_seeds)
    grasp_direct = np.any(grasp_ik.accepted, axis=1)
    pre_direct = np.any(pre_ik.accepted, axis=1)
    print(
        f"    GRASP direct coarse IK={int(np.count_nonzero(grasp_direct))}/{n_candidates} | "
        f"PREGRASP={int(np.count_nonzero(pre_direct))}/{n_candidates} | wall={time.perf_counter()-ik_started:.1f}s",
        flush=True,
    )

    print("[V2 3/9] building LEAP target reach region (inflated union, not point-cloud labels) ...", flush=True)
    target_pos_radius = bridge_position_inflation_m + float(args.target_extra_position_inflation_m)
    target_rot_radius = math.radians(
        bridge_orientation_inflation_deg + float(args.target_extra_orientation_inflation_deg)
    )
    reachable_ref_T = candidates["approx_flange_grasp_world"][grasp_direct]
    target_region_pass, target_nearest_pos, target_nearest_rot, target_nearest_ref_local = pose_region_membership(
        candidates["approx_flange_grasp_world"],
        reachable_ref_T,
        target_pos_radius,
        target_rot_radius,
    )
    # Direct IK is always kept even if numerical region search has a corner case.
    target_region_pass |= grasp_direct
    direct_ref_global = np.flatnonzero(grasp_direct)
    target_nearest_ref_rank = np.full(n_candidates, -1, dtype=np.int64)
    valid_ref = target_nearest_ref_local >= 0
    target_nearest_ref_rank[valid_ref] = direct_ref_global[target_nearest_ref_local[valid_ref]]
    print(
        f"    target reach region PASS={int(np.count_nonzero(target_region_pass))}/{n_candidates} "
        f"(direct={int(np.count_nonzero(grasp_direct))}, inflation-rescued={int(np.count_nonzero(target_region_pass & ~grasp_direct))})",
        flush=True,
    )

    print("[V2 4/9] selecting diverse LEAP PREGRASP/GRASP corridor anchors ...", flush=True)
    eligible = np.flatnonzero(grasp_direct & pre_direct)
    anchor_ranks = select_anchor_ranks(
        eligible,
        candidates["approx_flange_grasp_world"],
        args.anchor_count,
        args.anchor_source_topk,
        args.anchor_min_position_sep_m,
        math.radians(args.anchor_min_orientation_sep_deg),
    )
    if not anchor_ranks:
        raise RuntimeError("No candidate has both coarse GRASP and PREGRASP IK; cannot build trajectory space")
    print(f"    anchors={len(anchor_ranks)} ranks={anchor_ranks[:12]}{' ...' if len(anchor_ranks)>12 else ''}", flush=True)

    print("[V2 5/9] sampling HOME->PREGRASP and PREGRASP->GRASP rough trajectory tubes ...", flush=True)
    supports: list[SupportPose] = []
    anchor_pre = []
    anchor_grasp = []
    for bid, rank in enumerate(anchor_ranks):
        Tp = candidates["approx_flange_pregrasp_world"][rank]
        Tg = candidates["approx_flange_grasp_world"][rank]
        anchor_pre.append(Tp)
        anchor_grasp.append(Tg)
        supports.extend(
            make_home_family_support(
                bid,
                "home_direct",
                T_world_home_flange,
                Tp,
                float(target_hi[2]),
                args.home_direct_layers,
                args.home_cross_section_points,
                args.home_corridor_radius_m,
                args.overhead_clearance_m,
            )
        )
        supports.extend(
            make_home_family_support(
                bid,
                "home_lifted",
                T_world_home_flange,
                Tp,
                float(target_hi[2]),
                args.home_lifted_layers,
                args.home_cross_section_points,
                args.home_corridor_radius_m,
                args.overhead_clearance_m,
            )
        )
        supports.extend(
            make_approach_support(
                bid,
                Tp,
                Tg,
                args.approach_layers,
                args.approach_cross_section_points,
                args.approach_corridor_radius_m,
            )
        )
    anchor_pre = np.asarray(anchor_pre, dtype=np.float64)
    anchor_grasp = np.asarray(anchor_grasp, dtype=np.float64)
    support_world_T = np.stack([s.world_from_flange for s in supports])
    support_base_T = np.asarray([T_base_world @ T for T in support_world_T], dtype=np.float64)
    print(f"    sampled arm-flange support poses={len(supports)} across {len(anchor_ranks)} candidate branches", flush=True)

    print("[V2 6/9] cuRobo IK + non-target RGB-D ESDF screening for trajectory support states ...", flush=True)
    corridor_cfg = IKConfig(
        device=args.device,
        num_seeds=args.corridor_ik_seeds,
        batch_size=args.corridor_ik_batch_size,
        return_seeds=args.corridor_ik_seeds,
        minimum_inner_limit_margin_rad=math.radians(args.coarse_joint_margin_deg),
    )
    corridor_solver = CuroboGpuIK(robot_urdf, corridor_cfg)
    support_ik = corridor_solver.solve(support_base_T, return_seeds=args.corridor_ik_seeds)

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
                f"moving-link prefixes {args.moving_link_prefix} match zero collision spheres; sample={sphere_names[:20]}"
            )

    def named_with_q(q: np.ndarray) -> dict[str, float]:
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
            sr = sphere_model.check_self_collision(named)
            if not bool(sr["self_collision_pass"]):
                collision_cache[key] = (False, -math.inf, 0)
                return collision_cache[key]
        spheres = sphere_model.spheres_from_named_joints(named, T_world_base)
        if moving_mask is not None and len(moving_mask) == len(spheres):
            spheres = spheres[moving_mask]
        c = observed_map.check_spheres(spheres[:, :3], spheres[:, 3], "grasp", args.collision_margin_m)
        # Candidate-centric V2 uses the point cloud only as a NON-TARGET obstacle
        # field.  The SAM target is the destination and is deliberately excluded here;
        # exact post-retarget hand/target contact semantics remain downstream.
        scene_collision = np.asarray(c["scene_collision"], dtype=bool)
        unknown = np.asarray(c["unknown"], dtype=bool)
        blocked = bool(np.any(scene_collision) or (args.block_unknown and np.any(unknown)))
        scene_d = np.asarray(c["scene_distance_m"], dtype=np.float64)
        clearance = float(np.min(scene_d - spheres[:, 3])) if len(scene_d) else math.inf
        result = (not blocked, clearance, int(np.count_nonzero(unknown)))
        collision_cache[key] = result
        return result

    support_solutions: list[list[SupportSolution]] = [[] for _ in supports]
    for si in range(len(supports)):
        seeds = np.flatnonzero(support_ik.accepted[si])
        if len(seeds):
            order = sorted(
                [int(s) for s in seeds],
                key=lambda s: (
                    float(np.linalg.norm(support_ik.q_rad[si, s] - q_home)),
                    -float(support_ik.inner_limit_margin_rad[si, s]),
                ),
            )
            for seed in order:
                q = np.asarray(support_ik.q_rad[si, seed], dtype=np.float64)
                ok, clearance, unknown_count = q_collision_free(q)
                if ok:
                    support_solutions[si].append(
                        SupportSolution(
                            q_rad=q,
                            clearance_m=clearance,
                            unknown_count=unknown_count,
                            ik_margin_rad=float(support_ik.inner_limit_margin_rad[si, seed]),
                        )
                    )
                    if len(support_solutions[si]) >= max(1, int(args.solutions_per_support_pose)):
                        break
        if si % 500 == 0 or si + 1 == len(supports):
            free_pose_count = sum(bool(x) for x in support_solutions[: si + 1])
            print(f"    support collision screen {si+1:5d}/{len(supports)} | free poses={free_pose_count}", flush=True)

    print("[V2 7/9] extracting HOME-connected layered trajectory-space branches ...", flush=True)
    support_by_key: dict[tuple[int, str, int], list[int]] = {}
    max_layer: dict[tuple[int, str], int] = {}
    for si, s in enumerate(supports):
        support_by_key.setdefault((s.branch_id, s.family, s.layer), []).append(si)
        max_layer[(s.branch_id, s.family)] = max(max_layer.get((s.branch_id, s.family), -1), int(s.layer))

    edge_collision_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], bool] = {}
    edge_check_count = 0

    def q_edge_free(q0: np.ndarray, q1: np.ndarray) -> bool:
        nonlocal edge_check_count
        a0 = tuple(np.round(np.asarray(q0, dtype=np.float64), 3).tolist())
        a1 = tuple(np.round(np.asarray(q1, dtype=np.float64), 3).tolist())
        key = (a0, a1) if a0 <= a1 else (a1, a0)
        if key in edge_collision_cache:
            return edge_collision_cache[key]
        edge_check_count += 1
        max_delta = float(np.max(np.abs(q1 - q0)))
        sample_count = max(2, int(math.ceil(max_delta / math.radians(args.edge_step_deg))) + 1)
        ok = True
        for k in range(1, sample_count - 1):
            alpha = k / (sample_count - 1)
            q = (1.0 - alpha) * q0 + alpha * q1
            if not q_collision_free(q)[0]:
                ok = False
                break
        edge_collision_cache[key] = ok
        return ok

    connected_support = np.zeros(len(supports), dtype=bool)
    graph_edges: list[tuple[int, int]] = []
    branch_success = np.zeros(len(anchor_ranks), dtype=bool)
    branch_reason: list[str] = ["UNTESTED"] * len(anchor_ranks)
    branch_connected_counts = np.zeros(len(anchor_ranks), dtype=np.int32)

    # State is (support_index, solution_index).  Only one parent is needed to prove
    # membership in the reachable graph, but many offsets/solutions survive and form
    # the rough trajectory *space*.
    def connect_layer(
        prev_states: list[tuple[int | None, np.ndarray]],
        current_indices: list[int],
        seed_from_home: bool = False,
    ) -> list[tuple[int, np.ndarray]]:
        curr_states: list[tuple[int, np.ndarray]] = []
        for si in current_indices:
            for sol in support_solutions[si]:
                q = sol.q_rad
                candidates_prev = []
                for psi, pq in prev_states:
                    dmax = float(np.max(np.abs(q - pq)))
                    if seed_from_home:
                        if dmax > math.radians(args.home_seed_max_joint_delta_deg):
                            continue
                    elif dmax > math.radians(args.edge_max_joint_delta_deg):
                        continue
                    candidates_prev.append((float(np.linalg.norm(q - pq)), psi, pq))
                candidates_prev.sort(key=lambda x: x[0])
                for _dist, psi, pq in candidates_prev[: max(1, int(args.edge_parent_trials))]:
                    if q_edge_free(pq, q):
                        connected_support[si] = True
                        if psi is not None:
                            graph_edges.append((int(psi), int(si)))
                        curr_states.append((int(si), q))
                        break
        return curr_states

    for bid in range(len(anchor_ranks)):
        home_final_states: list[tuple[int, np.ndarray]] = []
        for family in ("home_direct", "home_lifted"):
            prev: list[tuple[int | None, np.ndarray]] = [(None, q_home)]
            alive = True
            for layer in range(max_layer[(bid, family)] + 1):
                indices = support_by_key.get((bid, family, layer), [])
                prev = connect_layer(prev, indices, seed_from_home=(layer == 0))
                if not prev:
                    alive = False
                    break
            if alive:
                home_final_states.extend(prev)
        if not home_final_states:
            branch_reason[bid] = "NO_HOME_TO_PREGRASP_CORRIDOR"
            branch_connected_counts[bid] = int(
                sum(1 for si, s in enumerate(supports) if s.branch_id == bid and connected_support[si])
            )
            print(
                f"    branch {bid+1:02d}/{len(anchor_ranks)} rank={anchor_ranks[bid]:4d} "
                f"status=FAIL connected_nodes={branch_connected_counts[bid]} reason={branch_reason[bid]}",
                flush=True,
            )
            continue

        prev = home_final_states
        approach_alive = True
        for layer in range(max_layer[(bid, "approach")] + 1):
            indices = support_by_key.get((bid, "approach", layer), [])
            prev = connect_layer(prev, indices, seed_from_home=False)
            if not prev:
                approach_alive = False
                break
        if approach_alive and prev:
            branch_success[bid] = True
            branch_reason[bid] = "PASS_HOME_PREGRASP_GRASP_CORRIDOR"
        else:
            branch_reason[bid] = "NO_PREGRASP_TO_GRASP_APPROACH_CORRIDOR"
        branch_connected_counts[bid] = int(
            sum(1 for si, s in enumerate(supports) if s.branch_id == bid and connected_support[si])
        )
        print(
            f"    branch {bid+1:02d}/{len(anchor_ranks)} rank={anchor_ranks[bid]:4d} "
            f"status={'PASS' if branch_success[bid] else 'FAIL'} connected_nodes={branch_connected_counts[bid]} "
            f"reason={branch_reason[bid]}",
            flush=True,
        )

    successful_ids = np.flatnonzero(branch_success)
    print(
        f"    successful rough trajectory branches={len(successful_ids)}/{len(anchor_ranks)} | edge checks={edge_check_count}",
        flush=True,
    )

    print("[V2 8/9] first candidate filter = target reach region AND rough trajectory space ...", flush=True)
    trajectory_pass = np.zeros(n_candidates, dtype=bool)
    matched_branch = np.full(n_candidates, -1, dtype=np.int32)
    pre_match_dist = np.full(n_candidates, np.inf, dtype=np.float64)
    grasp_match_dist = np.full(n_candidates, np.inf, dtype=np.float64)
    orientation_match_rad = np.full(n_candidates, np.inf, dtype=np.float64)
    approach_match_rad = np.full(n_candidates, np.inf, dtype=np.float64)

    # These radii are not arbitrary endpoint thresholds: they are the calibrated
    # LEAP->Wuji bridge uncertainty plus the sampled width of the route tubes.
    pre_match_radius = (
        bridge_position_inflation_m
        + float(args.home_corridor_radius_m)
        + float(args.trajectory_extra_position_m)
    )
    grasp_match_radius = (
        bridge_position_inflation_m
        + float(args.approach_corridor_radius_m)
        + float(args.trajectory_extra_position_m)
    )
    orientation_match_radius = math.radians(
        bridge_orientation_inflation_deg + float(args.trajectory_orientation_slack_deg)
    )
    approach_match_radius = math.radians(
        bridge_orientation_inflation_deg
        + float(args.trajectory_orientation_slack_deg)
        + float(args.trajectory_approach_slack_deg)
    )

    anchor_approach = anchor_grasp[:, :3, 3] - anchor_pre[:, :3, 3]
    anchor_approach /= np.maximum(np.linalg.norm(anchor_approach, axis=1, keepdims=True), 1.0e-12)

    for rank in range(n_candidates):
        if not target_region_pass[rank] or len(successful_ids) == 0:
            continue
        best = None
        cp = candidates["approx_flange_pregrasp_world"][rank]
        cg = candidates["approx_flange_grasp_world"][rank]
        ca = candidates["approx_approach_axis_world"][rank]
        for bid in successful_ids:
            bid = int(bid)
            dp = float(np.linalg.norm(cp[:3, 3] - anchor_pre[bid, :3, 3]))
            dg = float(np.linalg.norm(cg[:3, 3] - anchor_grasp[bid, :3, 3]))
            dr = rotation_angle_rad(anchor_grasp[bid, :3, :3].T @ cg[:3, :3])
            dot = float(np.clip(np.dot(ca, anchor_approach[bid]), -1.0, 1.0))
            da = float(math.acos(dot))
            score = max(
                dp / max(pre_match_radius, 1.0e-9),
                dg / max(grasp_match_radius, 1.0e-9),
                dr / max(orientation_match_radius, 1.0e-9),
                da / max(approach_match_radius, 1.0e-9),
            )
            row = (score, dp, dg, dr, da, bid)
            if best is None or row < best:
                best = row
        if best is not None:
            score, dp, dg, dr, da, bid = best
            pre_match_dist[rank] = dp
            grasp_match_dist[rank] = dg
            orientation_match_rad[rank] = dr
            approach_match_rad[rank] = da
            matched_branch[rank] = int(bid)
            trajectory_pass[rank] = bool(
                dp <= pre_match_radius
                and dg <= grasp_match_radius
                and dr <= orientation_match_radius
                and da <= approach_match_radius
            )

    overall_pass = target_region_pass & trajectory_pass
    print(
        f"    target region PASS={int(np.count_nonzero(target_region_pass))}/{n_candidates} | "
        f"trajectory-space PASS={int(np.count_nonzero(trajectory_pass))}/{n_candidates} | "
        f"first-filter PASS={int(np.count_nonzero(overall_pass))}/{n_candidates}",
        flush=True,
    )

    filter_rows = []
    for rank in range(n_candidates):
        idx = int(candidates["source_candidate_index"][rank])
        if overall_pass[rank]:
            status = "PASS"
            reason = "TARGET_REACH_AND_HOME_PREGRASP_GRASP_TRAJECTORY_SPACE"
        elif not target_region_pass[rank]:
            status = "REJECT"
            reason = "REJECT_TARGET_REACH_REGION"
        else:
            status = "REJECT"
            reason = "REJECT_NO_HOME_PREGRASP_GRASP_TRAJECTORY_SPACE"
        bid = int(matched_branch[rank])
        filter_rows.append(
            {
                "target_rank": int(rank),
                "candidate_index": idx,
                "official_score": float(dgn_score[idx]),
                "status": status,
                "reason": reason,
                "target_reach": {
                    "direct_coarse_grasp_ik": bool(grasp_direct[rank]),
                    "inflated_region_pass": bool(target_region_pass[rank]),
                    "nearest_reachable_position_m": None if not np.isfinite(target_nearest_pos[rank]) else float(target_nearest_pos[rank]),
                    "nearest_reachable_orientation_deg": None if not np.isfinite(target_nearest_rot[rank]) else float(np.degrees(target_nearest_rot[rank])),
                    "nearest_reachable_rank": int(target_nearest_ref_rank[rank]) if target_nearest_ref_rank[rank] >= 0 else None,
                },
                "pregrasp": {
                    "direct_coarse_ik": bool(pre_direct[rank]),
                    "leap_root_position_world_m": candidates["leap_pregrasp_world"][rank, :3, 3].tolist(),
                    "approx_flange_position_world_m": candidates["approx_flange_pregrasp_world"][rank, :3, 3].tolist(),
                },
                "grasp": {
                    "leap_root_position_world_m": candidates["leap_grasp_world"][rank, :3, 3].tolist(),
                    "approx_flange_position_world_m": candidates["approx_flange_grasp_world"][rank, :3, 3].tolist(),
                },
                "trajectory_space": {
                    "pass": bool(trajectory_pass[rank]),
                    "matched_branch_id": bid if bid >= 0 else None,
                    "matched_anchor_rank": int(anchor_ranks[bid]) if bid >= 0 else None,
                    "pregrasp_position_distance_m": None if not np.isfinite(pre_match_dist[rank]) else float(pre_match_dist[rank]),
                    "grasp_position_distance_m": None if not np.isfinite(grasp_match_dist[rank]) else float(grasp_match_dist[rank]),
                    "orientation_distance_deg": None if not np.isfinite(orientation_match_rad[rank]) else float(np.degrees(orientation_match_rad[rank])),
                    "approach_axis_distance_deg": None if not np.isfinite(approach_match_rad[rank]) else float(np.degrees(approach_match_rad[rank])),
                },
            }
        )

    diag_validation = None
    exact_accepted_ranks: set[int] = set()
    diag_path = args.cover_diagnostic_json
    if diag_path is not None:
        diag_path = diag_path.expanduser().resolve()
    elif (project_root / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/bottle_cover_ik_diag_first8.json").is_file():
        diag_path = project_root / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/bottle_cover_ik_diag_first8.json"
    if diag_path is not None and diag_path.is_file():
        diag = load_json(diag_path)
        by_idx = {int(row["candidate_index"]): row for row in diag.get("records", [])}
        joined = []
        for fr in filter_rows:
            idx = int(fr["candidate_index"])
            if idx in by_idx:
                joined.append((by_idx[idx], fr))
        exact = [(d, f) for d, f in joined if d.get("classification") == "ACCEPTED"]
        no_raw = [(d, f) for d, f in joined if d.get("classification") == "NO_RAW_CUROBO_SUCCESS"]
        margin_only = [
            (d, f)
            for d, f in joined
            if str(d.get("classification", "")).startswith("RAW_SUCCESS_REJECTED_JOINT_MARGIN")
        ]
        index_to_rank = {int(row["candidate_index"]): int(row["target_rank"]) for row in filter_rows}
        exact_accepted_ranks = {index_to_rank[int(d["candidate_index"])] for d, _f in exact}
        diag_validation = {
            "diagnostic_json": str(diag_path),
            "matched_count": len(joined),
            "exact_accepted_count": len(exact),
            "exact_accepted_target_region_retained": int(sum(1 for _d, f in exact if f["target_reach"]["inflated_region_pass"])),
            "exact_accepted_overall_retained": int(sum(1 for _d, f in exact if f["status"] == "PASS")),
            "no_raw_count": len(no_raw),
            "no_raw_rejected_by_target_region": int(sum(1 for _d, f in no_raw if not f["target_reach"]["inflated_region_pass"])),
            "no_raw_additionally_rejected_by_trajectory_space": int(
                sum(
                    1
                    for _d, f in no_raw
                    if f["target_reach"]["inflated_region_pass"] and not f["trajectory_space"]["pass"]
                )
            ),
            "no_raw_overall_rejected": int(sum(1 for _d, f in no_raw if f["status"] == "REJECT")),
            "joint_margin_only_count": len(margin_only),
            "joint_margin_only_overall_retained": int(sum(1 for _d, f in margin_only if f["status"] == "PASS")),
            "exact_accepted_details": [
                {
                    "target_rank": int(f["target_rank"]),
                    "candidate_index": int(f["candidate_index"]),
                    "target_region_pass": bool(f["target_reach"]["inflated_region_pass"]),
                    "trajectory_space_pass": bool(f["trajectory_space"]["pass"]),
                    "overall_status": f["status"],
                    "matched_anchor_rank": f["trajectory_space"]["matched_anchor_rank"],
                }
                for _d, f in exact
            ],
        }
        print(
            "    exact-COVER diagnostic: "
            f"target retained={diag_validation['exact_accepted_target_region_retained']}/{diag_validation['exact_accepted_count']} | "
            f"overall retained={diag_validation['exact_accepted_overall_retained']}/{diag_validation['exact_accepted_count']} | "
            f"NO_RAW overall rejected={diag_validation['no_raw_overall_rejected']}/{diag_validation['no_raw_count']}",
            flush=True,
        )
        if diag_validation["exact_accepted_target_region_retained"] < diag_validation["exact_accepted_count"]:
            print(
                "    [WARN] Target reach region false-rejected a known exact-COVER PASS; do not integrate this filter.",
                flush=True,
            )
        elif diag_validation["exact_accepted_overall_retained"] < diag_validation["exact_accepted_count"]:
            print(
                "    [NOTE] A known exact-COVER PASS was rejected only by the rough trajectory space. "
                "Exact-COVER does not prove path feasibility, so inspect the corridor visualization before changing thresholds.",
                flush=True,
            )

    print("[V2 9/9] saving candidate spaces, filter report, and camera-view overlays ...", flush=True)
    report_path = output_dir / "candidate_centric_rfs_v2_report.json"
    filter_path = output_dir / "candidate_centric_rfs_v2_filter.json"
    map_path = output_dir / "candidate_centric_rfs_v2_map.npz"
    target_overlay = output_dir / "target_reach_region_overlay.png"
    trajectory_overlay = output_dir / "trajectory_space_overlay.png"
    candidate_overlay = output_dir / "candidate_filter_overlay.png"

    support_has_free = np.asarray([bool(x) for x in support_solutions], dtype=bool)
    support_solution_count = np.asarray([len(x) for x in support_solutions], dtype=np.int16)
    support_best_q = np.full((len(supports), len(RIGHT_ARM_NAMES)), np.nan, dtype=np.float64)
    support_clearance = np.full(len(supports), np.nan, dtype=np.float64)
    support_unknown = np.full(len(supports), -1, dtype=np.int32)
    for i, sols in enumerate(support_solutions):
        if sols:
            support_best_q[i] = sols[0].q_rad
            support_clearance[i] = sols[0].clearance_m
            support_unknown[i] = sols[0].unknown_count

    np.savez_compressed(
        map_path,
        source_candidate_index=candidates["source_candidate_index"],
        leap_grasp_world=candidates["leap_grasp_world"],
        leap_pregrasp_world=candidates["leap_pregrasp_world"],
        approx_flange_grasp_world=candidates["approx_flange_grasp_world"],
        approx_flange_pregrasp_world=candidates["approx_flange_pregrasp_world"],
        approx_approach_axis_world=candidates["approx_approach_axis_world"],
        grasp_direct_coarse_ik=grasp_direct,
        pregrasp_direct_coarse_ik=pre_direct,
        target_reach_region_pass=target_region_pass,
        trajectory_space_pass=trajectory_pass,
        first_filter_pass=overall_pass,
        matched_branch_id=matched_branch,
        anchor_rank=np.asarray(anchor_ranks, dtype=np.int32),
        anchor_success=branch_success,
        anchor_pregrasp_world=anchor_pre,
        anchor_grasp_world=anchor_grasp,
        support_world_from_flange=support_world_T,
        support_branch_id=np.asarray([s.branch_id for s in supports], dtype=np.int16),
        support_family=np.asarray([s.family for s in supports]),
        support_stage=np.asarray([s.stage for s in supports]),
        support_layer=np.asarray([s.layer for s in supports], dtype=np.int16),
        support_offset_id=np.asarray([s.offset_id for s in supports], dtype=np.int16),
        support_has_collision_free_ik=support_has_free,
        support_solution_count=support_solution_count,
        support_best_q_rad=support_best_q,
        support_clearance_m=support_clearance,
        support_unknown_count=support_unknown,
        support_home_connected=connected_support,
        graph_edges=np.asarray(graph_edges, dtype=np.int32).reshape(-1, 2),
        home_flange_world=T_world_home_flange,
        target_center_world_m=target_center,
        bridge_position_inflation_m=np.asarray(bridge_position_inflation_m),
        bridge_orientation_inflation_deg=np.asarray(bridge_orientation_inflation_deg),
        target_region_position_radius_m=np.asarray(target_pos_radius),
        target_region_orientation_radius_deg=np.asarray(np.degrees(target_rot_radius)),
        trajectory_pre_match_radius_m=np.asarray(pre_match_radius),
        trajectory_grasp_match_radius_m=np.asarray(grasp_match_radius),
        trajectory_orientation_match_deg=np.asarray(np.degrees(orientation_match_radius)),
        trajectory_approach_match_deg=np.asarray(np.degrees(approach_match_radius)),
    )

    filter_payload = {
        "schema_version": 2,
        "status": "PASS",
        "policy": "candidate-centric pre-retarget coarse filter; preserve DGN2 score order among PASS candidates",
        "candidate_semantics": "reachability is indexed by DGN2 LEAP root 6-D candidates; approximate arm flange is used only for coarse IK/corridor construction",
        "pointcloud_semantics": "RGB-D/ESDF is obstacle geometry only; point-cloud points are never labeled reachable/unreachable",
        "first_filter_rule": "TARGET_REACH_REGION AND HOME_TO_PREGRASP_TO_GRASP_ROUGH_TRAJECTORY_SPACE",
        "candidate_count": n_candidates,
        "pass_count": int(np.count_nonzero(overall_pass)),
        "reject_target_reach_count": int(np.count_nonzero(~target_region_pass)),
        "reject_trajectory_space_count": int(np.count_nonzero(target_region_pass & ~trajectory_pass)),
        "rows": filter_rows,
    }
    filter_path.write_text(json.dumps(filter_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    branch_rows = []
    for bid, rank in enumerate(anchor_ranks):
        branch_rows.append(
            {
                "branch_id": bid,
                "anchor_rank": int(rank),
                "candidate_index": int(candidates["source_candidate_index"][rank]),
                "official_score": float(dgn_score[int(candidates["source_candidate_index"][rank])]),
                "status": "PASS" if branch_success[bid] else "FAIL",
                "reason": branch_reason[bid],
                "connected_support_pose_count": int(branch_connected_counts[bid]),
                "pregrasp_approx_flange_world": anchor_pre[bid].tolist(),
                "grasp_approx_flange_world": anchor_grasp[bid].tolist(),
            }
        )

    report = {
        "schema_version": 2,
        "status": "PASS",
        "architecture": "candidate-centric LEAP target reach region + HOME->PREGRASP->GRASP rough arm trajectory-space union",
        "project_root": str(project_root),
        "cycle_root": str(cycle_root),
        "query": args.query,
        "does_not_start_isaac": True,
        "does_not_modify_production_pipeline": True,
        "semantic_contract": {
            "reachability_subject": "DexGraspNet2 LEAP-hand root 6-D candidate",
            "arm_mapping": "calibrated mean LEAP-root -> final Wuji2-wrist -> right-arm flange bridge used only for coarse pre-retarget arm queries",
            "target_space": "inflated union around LEAP candidates whose approximate flange GRASP has coarse right-arm IK",
            "trajectory_space": "union of HOME-connected arm-state corridor branches sampled around candidate-derived PREGRASP/GRASP pairs",
            "pointcloud_role": "non-target obstacle field only; no point-cloud reachability classification",
            "downstream_mandatory": "LEAP->Wuji2 retarget, exact COVER IK, final motion/collision verification, Isaac/PhysX execution",
        },
        "home_flange_world": T_world_home_flange.tolist(),
        "target_center_world_m": target_center.tolist(),
        "candidate_endpoints": {
            "count": n_candidates,
            "grasp_direct_coarse_ik": int(np.count_nonzero(grasp_direct)),
            "pregrasp_direct_coarse_ik": int(np.count_nonzero(pre_direct)),
            "ik_seeds": args.endpoint_ik_seeds,
            "coarse_joint_margin_deg": args.coarse_joint_margin_deg,
            "position_tolerance_m": endpoint_cfg.position_tolerance_m,
            "orientation_tolerance_deg": math.degrees(endpoint_cfg.orientation_tolerance_rad),
            "official_pregrasp_retreat_m": args.pregrasp_retreat_m,
        },
        "target_reach_region": {
            "bridge_position_inflation_m": bridge_position_inflation_m,
            "bridge_orientation_inflation_deg": bridge_orientation_inflation_deg,
            "position_radius_m": target_pos_radius,
            "orientation_radius_deg": math.degrees(target_rot_radius),
            "direct_reference_count": int(np.count_nonzero(grasp_direct)),
            "pass_count": int(np.count_nonzero(target_region_pass)),
            "inflation_rescued_count": int(np.count_nonzero(target_region_pass & ~grasp_direct)),
        },
        "trajectory_space": {
            "anchor_count": len(anchor_ranks),
            "successful_branch_count": int(np.count_nonzero(branch_success)),
            "home_direct_layers": args.home_direct_layers,
            "home_lifted_layers": args.home_lifted_layers,
            "home_cross_section_points": args.home_cross_section_points,
            "home_corridor_radius_m": args.home_corridor_radius_m,
            "overhead_clearance_m": args.overhead_clearance_m,
            "approach_layers": args.approach_layers,
            "approach_cross_section_points": args.approach_cross_section_points,
            "approach_corridor_radius_m": args.approach_corridor_radius_m,
            "support_pose_count": len(supports),
            "support_pose_with_collision_free_ik_count": int(np.count_nonzero(support_has_free)),
            "home_connected_support_pose_count": int(np.count_nonzero(connected_support)),
            "graph_edge_count": len(graph_edges),
            "edge_collision_check_count": edge_check_count,
            "collision": {
                "moving_link_prefixes": args.moving_link_prefix,
                "margin_m": args.collision_margin_m,
                "block_unknown": bool(args.block_unknown),
                "check_self_collision": bool(args.check_self_collision),
                "target_layer_used_as_obstacle": False,
                "map_id": observed_map.map_id,
            },
            "candidate_membership": {
                "pregrasp_position_radius_m": pre_match_radius,
                "grasp_position_radius_m": grasp_match_radius,
                "orientation_radius_deg": math.degrees(orientation_match_radius),
                "approach_axis_radius_deg": math.degrees(approach_match_radius),
            },
            "branches": branch_rows,
        },
        "first_filter": {
            "pass_count": int(np.count_nonzero(overall_pass)),
            "reject_target_reach_count": int(np.count_nonzero(~target_region_pass)),
            "reject_trajectory_space_count": int(np.count_nonzero(target_region_pass & ~trajectory_pass)),
        },
        "exact_cover_diagnostic_validation": diag_validation,
        "outputs": {
            "map_npz": str(map_path),
            "filter_json": str(filter_path),
            "target_reach_overlay_png": str(target_overlay),
            "trajectory_space_overlay_png": str(trajectory_overlay),
            "candidate_filter_overlay_png": str(candidate_overlay),
        },
        "wall_time_s": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_target_reach_overlay(
        rgb,
        mask,
        candidates["leap_grasp_world"],
        grasp_direct,
        target_region_pass,
        overall_pass,
        K,
        T_world_camera,
        target_pos_radius,
        target_overlay,
        args.candidate_draw_topk,
    )
    support_branch_id = np.asarray([s.branch_id for s in supports], dtype=np.int16)
    save_trajectory_space_overlay(
        rgb,
        mask,
        home_xyz,
        support_world_T[:, :3, 3],
        connected_support,
        np.asarray(graph_edges, dtype=np.int32).reshape(-1, 2),
        branch_success,
        support_branch_id,
        anchor_pre,
        anchor_grasp,
        K,
        T_world_camera,
        trajectory_overlay,
    )
    save_candidate_filter_overlay(
        rgb,
        mask,
        candidates["leap_pregrasp_world"],
        candidates["leap_grasp_world"],
        target_region_pass,
        trajectory_pass,
        overall_pass,
        K,
        T_world_camera,
        candidate_overlay,
        args.candidate_draw_topk,
        exact_accepted_ranks,
    )

    print("=" * 92)
    print("CANDIDATE-CENTRIC RFS V2 DONE")
    print(f"LEAP candidates                    : {n_candidates}")
    print(f"GRASP direct coarse IK             : {int(np.count_nonzero(grasp_direct))}")
    print(f"PREGRASP direct coarse IK          : {int(np.count_nonzero(pre_direct))}")
    print(f"TARGET REACH REGION                : {int(np.count_nonzero(target_region_pass))}")
    print(f"trajectory anchors                 : {len(anchor_ranks)}")
    print(f"successful HOME->PRE->GRASP space  : {int(np.count_nonzero(branch_success))}/{len(anchor_ranks)} branches")
    print(f"FIRST FILTER PASS                  : {int(np.count_nonzero(overall_pass))}/{n_candidates}")
    print(f"reject target reach                : {int(np.count_nonzero(~target_region_pass))}")
    print(f"reject trajectory space            : {int(np.count_nonzero(target_region_pass & ~trajectory_pass))}")
    if diag_validation is not None:
        print(
            "known exact COVER target retained   : "
            f"{diag_validation['exact_accepted_target_region_retained']}/{diag_validation['exact_accepted_count']}"
        )
        print(
            "known exact COVER overall retained  : "
            f"{diag_validation['exact_accepted_overall_retained']}/{diag_validation['exact_accepted_count']}"
        )
        print(
            "NO_RAW overall rejected             : "
            f"{diag_validation['no_raw_overall_rejected']}/{diag_validation['no_raw_count']}"
        )
    print(f"target overlay     : {target_overlay}")
    print(f"trajectory overlay : {trajectory_overlay}")
    print(f"candidate overlay  : {candidate_overlay}")
    print(f"report             : {report_path}")
    print(f"filter             : {filter_path}")
    print(f"map                : {map_path}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
