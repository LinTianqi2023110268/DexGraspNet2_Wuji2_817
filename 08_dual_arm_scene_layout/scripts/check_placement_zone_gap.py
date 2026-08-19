#!/usr/bin/env python3
"""Offline invariant check for the calibrated SourceZone/PlacementZone X gap."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAYOUT_JSON = ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
CALIBRATED_USDA = ROOT / "08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda"
MASS_FIXED_USDA = ROOT / "08_dual_arm_scene_layout/scenes/manual_layout_calibrated_mass_fixed.usda"
DRAFT_USDA = ROOT / "08_dual_arm_scene_layout/scenes/manual_layout_draft.usda"

EXPECTED_SOURCE_CENTER = [-0.4238227717751965, -0.15291664032601016, 0.46]
EXPECTED_SOURCE_SIZE = [0.5, 0.30000001192092896, 0.0010000000474974513]
EXPECTED_PLACEMENT_CENTER = [0.27617723418526796, -0.1446419350251421, 0.46]
EXPECTED_PLACEMENT_SIZE = [0.800000011920929, 0.30000001192092896, 0.0010000000474974513]
EXPECTED_TABLE_CENTER = [0.06347617107116313, -0.14842441124362116, 0.44]
EXPECTED_TABLE_SIZE = [1.600000023841858, 0.4000000059604645, 0.03999999910593033]
EXPECTED_DUAL_ARM_MOUNT = [0.0, 0.16, 0.8]
EXPECTED_CAMERA_POSITION = [3.725290298461914e-09, 0.08499996900558474, 0.9600000381469727]
EXPECTED_CAMERA_TARGET = [-0.4238227717751965, -0.15291664032601016, 0.46]
EXPECTED_GAP_M = 0.05


def assert_close_vec(name: str, actual, expected, tol: float = 1e-12) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: length {len(actual)} != {len(expected)}")
    for idx, (a, e) in enumerate(zip(actual, expected)):
        if abs(float(a) - float(e)) > tol:
            raise AssertionError(f"{name}[{idx}]: {a!r} != {e!r}")


def placement_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'def Cube "PlacementZone"\s*\{(?P<body>.*?)\n\s*\}', text, re.S)
    if not match:
        raise AssertionError(f"{path}: PlacementZone block not found")
    return match.group("body")


def parse_tuple(block: str, field: str) -> list[float]:
    match = re.search(rf"{re.escape(field)} = \(([^)]*)\)", block)
    if not match:
        raise AssertionError(f"{field} not found")
    return [float(part.strip()) for part in match.group(1).split(",")]


def main() -> int:
    layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
    geom = layout["geometry"]
    transforms = layout["transforms"]

    source_center = transforms["source_zone"]["position_world_m"]
    source_size = geom["source_zone_size_m"]
    placement_center = transforms["placement_zone"]["position_world_m"]
    placement_size = geom["placement_zone_size_m"]

    assert_close_vec("source_center", source_center, EXPECTED_SOURCE_CENTER)
    assert_close_vec("source_size", source_size, EXPECTED_SOURCE_SIZE)
    assert_close_vec("placement_center", placement_center, EXPECTED_PLACEMENT_CENTER)
    assert_close_vec("placement_size", placement_size, EXPECTED_PLACEMENT_SIZE)
    assert_close_vec("table_center", transforms["table"]["position_world_m"], EXPECTED_TABLE_CENTER)
    assert_close_vec("table_size", geom["table_size_m"], EXPECTED_TABLE_SIZE)
    assert_close_vec("dual_arm_mount", transforms["dual_arm_mount"]["position_world_m"], EXPECTED_DUAL_ARM_MOUNT)
    assert_close_vec("camera_position", layout["camera"]["camera_position_world_m"], EXPECTED_CAMERA_POSITION)
    assert_close_vec("camera_target", layout["camera"]["target_position_world_m"], EXPECTED_CAMERA_TARGET)

    source_right_x = source_center[0] + source_size[0] / 2
    placement_left_x = placement_center[0] - placement_size[0] / 2
    gap_m = placement_left_x - source_right_x
    if abs(gap_m - EXPECTED_GAP_M) >= 1e-9:
        raise AssertionError(f"gap_m={gap_m:.18f}, expected {EXPECTED_GAP_M:.18f}")

    for path in (CALIBRATED_USDA, MASS_FIXED_USDA):
        block = placement_block(path)
        translate = parse_tuple(block, "double3 xformOp:translate")
        scale = parse_tuple(block, "float3 xformOp:scale")
        assert_close_vec(f"{path.name}:PlacementZone translate", translate, EXPECTED_PLACEMENT_CENTER)
        assert_close_vec(f"{path.name}:PlacementZone scale", scale, [0.8, 0.3, 0.001])

    draft_block = placement_block(DRAFT_USDA)
    draft_translate = parse_tuple(draft_block, "double3 xformOp:translate")
    draft_scale = parse_tuple(draft_block, "float3 xformOp:scale")
    assert_close_vec("manual_layout_draft.usda:PlacementZone translate", draft_translate, [0.30, 0.0, 0.0005])
    assert_close_vec("manual_layout_draft.usda:PlacementZone scale", draft_scale, [1.0, 0.3, 0.001])

    draft_source_right_x = -0.50 + 0.50 / 2
    draft_placement_left_x = 0.30 - 1.00 / 2
    draft_gap_m = draft_placement_left_x - draft_source_right_x
    if abs(draft_gap_m - EXPECTED_GAP_M) >= 1e-12:
        raise AssertionError(f"draft_gap_m={draft_gap_m:.18f}")

    print(f"source_center_x={source_center[0]:.17f}")
    print(f"source_size_x={source_size[0]:.17f}")
    print(f"source_right_x={source_right_x:.17f}")
    print(f"placement_center_x={placement_center[0]:.17f}")
    print(f"placement_size_x={placement_size[0]:.17f}")
    print(f"placement_left_x={placement_left_x:.17f}")
    print(f"gap_m={gap_m:.18f}")
    print(f"draft_gap_m={draft_gap_m:.18f}")
    print("placement_zone_gap_invariants=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
