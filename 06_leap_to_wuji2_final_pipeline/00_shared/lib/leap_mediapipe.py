#!/usr/bin/env python3
"""LEAP Hand q16 forward kinematics and MediaPipe-21 landmark conversion.

The LEAP hand has four actuated joints per finger, but MediaPipe stores four
landmarks (MCP/PIP/DIP/TIP) for each non-thumb finger.  LEAP's two proximal
degrees of freedom are therefore represented geometrically, not as two extra
landmarks: MCP is the origin of j0/j4/j8 after both upstream transforms have
been applied; PIP and DIP are the origins of j2/j6/j10 and j3/j7/j11; TIP is a
point on the distal end of the terminal fingertip mesh.

LEAP has no pinky.  A deterministic virtual pinky is extrapolated from the
middle/ring geometry.  This is an explicit compatibility policy, not a measured
LEAP link, and is recorded in every output manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
FINAL_PIPELINE_ROOT = HERE.parent
LEAP_URDF = (
    FINAL_PIPELINE_ROOT
    / "models/robot_models/urdf/leap_hand.urdf"
)

LEAP_JOINT_ORDER = [
    "j12", "j13", "j14", "j15",
    "j1", "j0", "j2", "j3",
    "j5", "j4", "j6", "j7",
    "j9", "j8", "j10", "j11",
]

MEDIAPIPE_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Distal surface centers measured directly from the LEAP terminal STL meshes in
# their URDF link-local frames (one-millimetre cap at the most distal -Y end).
NONTHUMB_TIP_LOCAL_M = np.array([-0.003589, -0.049085, 0.014430])
THUMB_TIP_LOCAL_M = np.array([-0.003589, -0.061685, -0.014370])

# LEAP lacks a fifth finger.  1.0 extrapolates the MCP lateral spacing; 0.88
# shortens the ring-relative phalange vectors to a plausible pinky length.
VIRTUAL_PINKY_LATERAL_SCALE = 1.0
VIRTUAL_PINKY_LENGTH_SCALE = 0.88

# MediaPipe point 0 is an anatomical wrist landmark, not necessarily the URDF
# root.  LEAP's hand_base_link root lies inside its palm mechanism.  This point
# is the centre of the proximal face on the palm's inner surface, measured from
# the official hand_base_link STL/box bounds.  Keeping it explicit prevents the
# wrist->finger vectors from becoming artificially short.
LEAP_WRIST_LOCAL_M = np.array([-0.10009525, -0.03710693, -0.03472240])


def _origin_matrix(node: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if node is None:
        return result
    xyz = np.fromstring(node.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(node.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def _axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(axis * float(angle)).as_matrix()
    return result


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class LeapFk:
    def __init__(self, urdf_path: Path = LEAP_URDF):
        self.urdf_path = urdf_path.resolve()
        root = ET.parse(self.urdf_path).getroot()
        links = {node.get("name") for node in root.findall("link")}
        joints: Dict[str, Joint] = {}
        children: Dict[str, list[Joint]] = {}
        child_links = set()
        for node in root.findall("joint"):
            axis_node = node.find("axis")
            axis = np.fromstring(
                axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0",
                sep=" ", dtype=np.float64,
            )
            item = Joint(
                name=str(node.get("name")),
                kind=str(node.get("type")),
                parent=str(node.find("parent").get("link")),
                child=str(node.find("child").get("link")),
                origin=_origin_matrix(node.find("origin")),
                axis=axis / np.linalg.norm(axis),
            )
            joints[item.name] = item
            children.setdefault(item.parent, []).append(item)
            child_links.add(item.child)
        roots = sorted(links - child_links)
        if roots != ["hand_base_link"]:
            raise RuntimeError(f"Expected LEAP root hand_base_link, got {roots}")
        self.root = roots[0]
        self.joints = joints
        self.children = children

    def solve(self, qpos: Dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        missing = [name for name in LEAP_JOINT_ORDER if name not in qpos]
        if missing:
            raise KeyError(f"LEAP qpos missing joints: {missing}")
        link_tf = {self.root: np.eye(4, dtype=np.float64)}
        joint_origin_tf: dict[str, np.ndarray] = {}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                before_motion = link_tf[parent] @ joint.origin
                joint_origin_tf[joint.name] = before_motion
                child_tf = before_motion
                if joint.kind in ("revolute", "continuous"):
                    child_tf = child_tf @ _axis_matrix(joint.axis, qpos[joint.name])
                elif joint.kind != "fixed":
                    raise ValueError(f"Unsupported joint type {joint.kind}: {joint.name}")
                link_tf[joint.child] = child_tf
                stack.append(joint.child)
        return link_tf, joint_origin_tf


def _translation(tf: np.ndarray) -> np.ndarray:
    return np.asarray(tf[:3, 3], dtype=np.float64)


def _transform_point(tf: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (tf @ np.r_[np.asarray(point, dtype=np.float64), 1.0])[:3]


def leap_qpos_to_mediapipe21(qpos: Dict[str, float]) -> tuple[np.ndarray, dict]:
    """Convert one LEAP q16 dictionary to raw right-hand MediaPipe landmarks."""
    link_tf, joint_tf = LeapFk().solve(qpos)
    points = np.zeros((21, 3), dtype=np.float64)

    # 0: anatomical wrist point expressed in hand_base_link coordinates.
    points[0] = _transform_point(link_tf["hand_base_link"], LEAP_WRIST_LOCAL_M)

    # Thumb: omit the first physical CMC-axis origin from the 4-point MediaPipe
    # chain; its effect is already present in all downstream FK positions.
    points[1] = _translation(joint_tf["j13"])
    points[2] = _translation(joint_tf["j14"])
    points[3] = _translation(joint_tf["j15"])
    points[4] = _transform_point(link_tf["thumb_fingertip"], THUMB_TIP_LOCAL_M)

    finger_specs = [
        (5, "j0", "j2", "j3", "fingertip"),
        (9, "j4", "j6", "j7", "fingertip_2"),
        (13, "j8", "j10", "j11", "fingertip_3"),
    ]
    for offset, mcp, pip, dip, tip_link in finger_specs:
        points[offset] = _translation(joint_tf[mcp])
        points[offset + 1] = _translation(joint_tf[pip])
        points[offset + 2] = _translation(joint_tf[dip])
        points[offset + 3] = _transform_point(link_tf[tip_link], NONTHUMB_TIP_LOCAL_M)

    # Virtual pinky: extrapolate ring base away from middle, then copy the ring
    # articulation relative to its MCP with a shorter length scale.
    points[17] = points[13] + VIRTUAL_PINKY_LATERAL_SCALE * (points[13] - points[9])
    for dst, src in zip((18, 19, 20), (14, 15, 16)):
        points[dst] = points[17] + VIRTUAL_PINKY_LENGTH_SCALE * (points[src] - points[13])

    if not np.all(np.isfinite(points)):
        raise RuntimeError("Generated MediaPipe landmarks contain non-finite values")
    if np.linalg.matrix_rank(np.stack([points[5] - points[0], points[9] - points[0]])) < 2:
        raise RuntimeError("LEAP wrist/index/middle landmarks are degenerate")

    metadata = {
        "schema_version": 1,
        "source": "LEAP q16 exact URDF forward kinematics",
        "leap_urdf": str(LEAP_URDF),
        "landmark_order": MEDIAPIPE_NAMES,
        "unit": "m",
        "frame": "LEAP hand_base_link; raw points are later normalized by official Retargeter",
        "nonthumb_tip_local_m": NONTHUMB_TIP_LOCAL_M.tolist(),
        "thumb_tip_local_m": THUMB_TIP_LOCAL_M.tolist(),
        "wrist_local_m": LEAP_WRIST_LOCAL_M.tolist(),
        "wrist_policy": "centre of LEAP palm proximal face on the inner surface; not the URDF root origin",
        "virtual_pinky": {
            "measured": False,
            "policy": "ring geometry extrapolated laterally from middle-to-ring spacing",
            "lateral_scale": VIRTUAL_PINKY_LATERAL_SCALE,
            "length_scale": VIRTUAL_PINKY_LENGTH_SCALE,
        },
    }
    return points, metadata


def load_leap_pose(path: Path, index: int = 0) -> tuple[str, Dict[str, float]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if "poses" in payload:
        record = payload["poses"][index]
    else:
        record = payload
    source = record.get("qpos", record.get("joint_values_by_name_rad"))
    if not isinstance(source, dict):
        raise ValueError("Input JSON must contain qpos or joint_values_by_name_rad")
    missing = [name for name in LEAP_JOINT_ORDER if name not in source]
    if missing:
        raise KeyError(f"LEAP pose missing joints: {missing}")
    qpos = {name: float(source[name]) for name in LEAP_JOINT_ORDER}
    return str(record.get("name", f"pose_{index:03d}")), qpos
