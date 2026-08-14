#!/usr/bin/env python3
"""Deterministic, footprint-aware placement allocation for the green zone.

This module is deliberately independent of Isaac Sim.  Offline waypoint
generation calls it to reserve a geometrically valid placement target.  The
Isaac Lab runtime commits that reservation only after a full pick/place cycle
has physically passed, so a failed attempt never consumes a slot.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def placement_zone_bounds(layout: dict) -> tuple[np.ndarray, np.ndarray, float]:
    centre = np.asarray(layout["transforms"]["placement_zone"]["position_world_m"], dtype=np.float64)
    size = np.asarray(layout["geometry"]["placement_zone_size_m"], dtype=np.float64)
    table_centre = np.asarray(layout["transforms"]["table"]["position_world_m"], dtype=np.float64)
    table_size = np.asarray(layout["geometry"]["table_size_m"], dtype=np.float64)
    return centre[:2] - 0.5 * size[:2], centre[:2] + 0.5 * size[:2], float(table_centre[2] + 0.5 * table_size[2])


def oriented_surface_offsets(surface_points_object: np.ndarray, world_from_object: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world-axis offsets from object origin for the current stable orientation."""
    points = np.asarray(surface_points_object, dtype=np.float64)
    rotation = np.asarray(world_from_object, dtype=np.float64)[:3, :3]
    rotated = points @ rotation.T
    return rotated.min(axis=0), rotated.max(axis=0)


def _axis_samples(lower: float, upper: float, step: float) -> np.ndarray:
    if lower > upper:
        return np.empty(0, dtype=np.float64)
    count = max(1, int(np.floor((upper - lower) / step)) + 1)
    values = lower + np.arange(count, dtype=np.float64) * step
    if upper - values[-1] > 0.5 * step:
        values = np.append(values, upper)
    return values


def _overlaps(candidate_min: np.ndarray, candidate_max: np.ndarray, record: dict, clearance: float) -> bool:
    occupied_min = np.asarray(record["footprint_world_xy_min_m"], dtype=np.float64) - clearance
    occupied_max = np.asarray(record["footprint_world_xy_max_m"], dtype=np.float64) + clearance
    return bool(np.all(candidate_max >= occupied_min) and np.all(occupied_max >= candidate_min))


def read_registry(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "placements": []}
    registry = load_json(path)
    if int(registry.get("schema_version", -1)) != 1 or not isinstance(registry.get("placements"), list):
        raise RuntimeError(f"Invalid placement registry: {path}")
    return registry


def allocate_placement(
    *,
    project_root: Path,
    layout: dict,
    policy: dict,
    surface_points_object: np.ndarray,
    world_from_object_initial: np.ndarray,
    requested_slot_index: int | None = None,
) -> dict:
    """Select one free root position whose complete footprint stays in the zone."""
    zone_min, zone_max, table_top = placement_zone_bounds(layout)
    oriented_min, oriented_max = oriented_surface_offsets(surface_points_object, world_from_object_initial)
    footprint_mode = str(policy.get("footprint_xy_mode", "oriented_aabb"))
    if footprint_mode == "orientation_invariant_radius":
        radius = float(np.max(np.linalg.norm(np.asarray(surface_points_object), axis=1)))
        offset_min = oriented_min.copy()
        offset_max = oriented_max.copy()
        offset_min[:2] = -radius
        offset_max[:2] = radius
    elif footprint_mode == "oriented_aabb":
        offset_min, offset_max = oriented_min, oriented_max
    else:
        raise ValueError(f"Unknown footprint_xy_mode: {footprint_mode}")
    edge = float(policy["edge_margin_m"])
    clearance = float(policy["inter_object_clearance_m"])
    step = np.asarray(policy["grid_step_xy_m"], dtype=np.float64)
    if np.any(step <= 0.0):
        raise ValueError("grid_step_xy_m must be positive")

    root_min = zone_min + edge - offset_min[:2]
    root_max = zone_max - edge - offset_max[:2]
    xs = _axis_samples(float(root_min[0]), float(root_max[0]), float(step[0]))
    ys = _axis_samples(float(root_min[1]), float(root_max[1]), float(step[1]))
    if not len(xs) or not len(ys):
        raise RuntimeError("Object footprint cannot fit inside the placement zone")

    preferred_y = float(policy.get("preferred_world_y_m", 0.5 * (zone_min[1] + zone_max[1])))
    ys = np.asarray(sorted(ys, key=lambda value: (abs(value - preferred_y), value)))
    raw_candidates = [np.asarray([x, y], dtype=np.float64) for x in xs for y in ys]

    registry_path = resolve_project_path(project_root, policy["occupancy_registry"])
    registry = read_registry(registry_path)
    free = []
    for raw_index, xy in enumerate(raw_candidates):
        footprint_min = xy + offset_min[:2]
        footprint_max = xy + offset_max[:2]
        if any(_overlaps(footprint_min, footprint_max, record, clearance) for record in registry["placements"]):
            continue
        free.append((raw_index, xy, footprint_min, footprint_max))

    if not free:
        raise RuntimeError(f"No collision-free placement remains in {registry_path}")
    free_index = 0 if requested_slot_index is None else int(requested_slot_index)
    if free_index < 0 or free_index >= len(free):
        raise IndexError(f"Requested free slot {free_index}, available range is 0..{len(free) - 1}")
    raw_index, xy, footprint_min, footprint_max = free[free_index]
    # Root Z is retained as a nominal visualization/reference value only.  The
    # placement IK uses the explicitly configured hand-height rule.
    root_z = table_top - float(oriented_min[2])
    return {
        "schema_version": 1,
        "allocation_mode": "automatic_first_free" if requested_slot_index is None else "requested_free_slot",
        "requested_free_slot_index": requested_slot_index,
        "selected_free_slot_index": free_index,
        "selected_raw_grid_index": raw_index,
        "object_root_place_world_m": [float(xy[0]), float(xy[1]), root_z],
        "footprint_world_xy_min_m": footprint_min.tolist(),
        "footprint_world_xy_max_m": footprint_max.tolist(),
        "surface_offset_world_min_m": offset_min.tolist(),
        "surface_offset_world_max_m": offset_max.tolist(),
        "footprint_xy_mode": footprint_mode,
        "zone_world_xy_min_m": zone_min.tolist(),
        "zone_world_xy_max_m": zone_max.tolist(),
        "table_top_world_z_m": table_top,
        "release_hand_height_above_grasp_m": float(
            policy["release_hand_height_above_grasp_m"]
        ),
        "transfer_clearance_m": float(policy["transfer_clearance_m"]),
        "retreat_clearance_m": float(policy["retreat_clearance_m"]),
        "edge_margin_m": edge,
        "inter_object_clearance_m": clearance,
        "occupancy_registry": str(registry_path),
        "occupied_count_before": len(registry["placements"]),
        "free_candidate_count_before": len(free),
    }


def commit_placement(registry_path: Path, record: dict) -> None:
    """Upsert one physically successful placement using an atomic replace."""
    registry_path = registry_path.resolve()
    registry = read_registry(registry_path)
    placement_id = str(record["placement_id"])
    record = dict(record)
    record["committed_utc"] = datetime.now(timezone.utc).isoformat()
    registry["placements"] = [
        existing for existing in registry["placements"]
        if str(existing.get("placement_id")) != placement_id
    ]
    registry["placements"].append(record)
    registry["updated_utc"] = record["committed_utc"]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    temporary.replace(registry_path)
