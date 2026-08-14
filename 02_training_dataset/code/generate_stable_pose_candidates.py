#!/usr/bin/env python3
"""Generate tabletop candidates from analytic stable poses and 2D footprints.

This adapter generator deliberately does not run PyBullet.  It computes each
object's quasi-static planar support poses once with Trimesh, packs transformed
XY convex footprints with Shapely, and leaves final physical acceptance to
Isaac Sim 5.0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import MultiPoint, box


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ADAPTER_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import (  # noqa: E402
    load_config,
    prepare_centered_object_asset,
    quat_wxyz_from_matrix,
    sha256,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ADAPTER_ROOT
        / "config"
        / "wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument("--count", type=int, help="Override configured candidate count.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after the highest complete candidate instead of overwriting it.",
    )
    return parser.parse_args()


def successful_grasp_count(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", errors="replace") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def prepare_assets(config: dict) -> list[dict]:
    source_root = Path(config["paths"]["source_mesh_root"])
    single_root = Path(config["paths"]["single_object_output_root"]) / "objects"
    output_root = Path(config["paths"]["output_root"]) / "prepared_assets"
    assets = []
    for item in config["objects"]:
        source_obj = source_root / item["code"] / "coacd" / "decomposed.obj"
        single_dir = single_root / item["code"]
        manifest_path = single_dir / "manifest.json"
        grasp_csv = single_dir / "records" / "successful_grasps.csv"
        if not source_obj.is_file() or not manifest_path.is_file() or not grasp_csv.is_file():
            raise FileNotFoundError(f"Incomplete single-object input for {item['code']}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scale = float(manifest["object_scale"])
        if not math.isclose(scale, float(item["scale"]), rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"Scale mismatch for {item['code']}")
        if sha256(source_obj) != manifest["source_mesh_sha256"]:
            raise RuntimeError(f"Source mesh mismatch for {item['code']}")
        asset = prepare_centered_object_asset(
            source_obj,
            output_root / f"object_{int(item['id']):03d}",
            scale,
            item["code"],
        )
        asset["scale_contract"] = {
            "verified": True,
            "dexgraspnet1_output_manifest": str(manifest_path.resolve()),
            "object_scale": scale,
            "source_mesh_sha256": asset["source_obj_sha256"],
        }
        asset["successful_grasp_count"] = successful_grasp_count(grasp_csv)
        assets.append(asset)
    return assets


def load_single_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected one Trimesh at {path}, got {type(loaded).__name__}")
    return loaded


def compute_pose_libraries(assets: list[dict], config: dict) -> list[dict]:
    generation = config["stable_pose_scene_generation"]
    upright_policy = generation.get("upright_pose_bias", {})
    libraries = []
    for object_index, asset in enumerate(assets):
        mesh = load_single_mesh(Path(asset["centered_combined_obj"]))
        # A non-watertight mesh has no trustworthy volumetric centre of mass.
        # Its convex hull is closed and is also the planar support geometry used
        # internally by Trimesh, so use the hull for both quantities.
        analysis_mesh = mesh if mesh.is_watertight else mesh.convex_hull
        started = time.perf_counter()
        transforms, probabilities = trimesh.poses.compute_stable_poses(
            analysis_mesh,
            sigma=float(generation["center_mass_sigma_m"]),
            n_samples=int(generation["center_mass_samples"]),
            threshold=float(generation["stable_pose_probability_threshold"]),
        )
        elapsed = time.perf_counter() - started
        if len(probabilities) == 0:
            raise RuntimeError(f"No stable poses survived for {asset['object_code']}")
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities /= probabilities.sum()
        pose_extents = []
        for transform in transforms:
            transformed = trimesh.transform_points(mesh.vertices, transform)
            pose_extents.append(np.ptp(transformed, axis=0))
        pose_extents = np.asarray(pose_extents, dtype=np.float64)
        pose_heights = pose_extents[:, 2]
        horizontal_spans = np.maximum(pose_extents[:, 0], pose_extents[:, 1])
        height_to_horizontal = pose_heights / np.maximum(horizontal_spans, 1.0e-12)
        relative_height_threshold = float(
            upright_policy.get("relative_height_threshold", 0.80)
        )
        minimum_height_to_horizontal = float(
            upright_policy.get("minimum_height_to_horizontal_span", 0.80)
        )
        minimum_pose_height_range_ratio = float(
            upright_policy.get("minimum_pose_height_range_ratio", 1.25)
        )
        height_range_ratio = float(
            pose_heights.max() / max(pose_heights.min(), 1.0e-12)
        )
        upright_pose_mask = (
            (pose_heights >= relative_height_threshold * pose_heights.max())
            & (height_to_horizontal >= minimum_height_to_horizontal)
            & (height_range_ratio >= minimum_pose_height_range_ratio)
        )
        library_dir = (
            Path(config["paths"]["output_root"])
            / "stable_pose_libraries"
            / f"object_{object_index + 1:03d}"
        )
        library_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            library_dir / "stable_poses.npz",
            transforms=np.asarray(transforms, dtype=np.float64),
            probabilities=probabilities,
        )
        metadata = {
            "schema_version": 1,
            "object_index": object_index,
            "object_code": asset["object_code"],
            "analysis_geometry": "original_watertight_mesh"
            if mesh.is_watertight
            else "convex_hull_for_non_watertight_mesh",
            "source_watertight": bool(mesh.is_watertight),
            "pose_count": int(len(probabilities)),
            "probabilities": probabilities.tolist(),
            "computation_time_s": elapsed,
            "trimesh_version": trimesh.__version__,
            "source": config["candidate_sampling"]["source"],
            "upright_pose_analysis": {
                "relative_height_threshold": relative_height_threshold,
                "minimum_height_to_horizontal_span": minimum_height_to_horizontal,
                "minimum_pose_height_range_ratio": minimum_pose_height_range_ratio,
                "pose_heights_m": pose_heights.tolist(),
                "height_to_horizontal_span": height_to_horizontal.tolist(),
                "upright_pose_indices": np.flatnonzero(upright_pose_mask).tolist(),
                "upright_probability_mass": float(
                    probabilities[upright_pose_mask].sum()
                ),
            },
        }
        write_json_atomic(library_dir / "stable_pose_manifest.json", metadata)
        libraries.append(
            {
                "mesh": mesh,
                "transforms": np.asarray(transforms, dtype=np.float64),
                "probabilities": probabilities,
                "pose_heights_m": pose_heights,
                "height_to_horizontal_span": height_to_horizontal,
                "upright_pose_mask": upright_pose_mask,
                "metadata": metadata,
            }
        )
        print(
            f"[STABLE-POSE] object={object_index + 1:03d} "
            f"poses={len(probabilities)} time={elapsed:.4f}s "
            f"watertight={mesh.is_watertight}",
            flush=True,
        )
    return libraries


def yaw_transform(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    return transform


def sample_oriented_objects(
    rng: np.random.Generator,
    libraries: list[dict],
    config: dict,
    selected_object_indices: list[int],
    forced_upright_count: int = 0,
) -> list[dict]:
    generation = config["stable_pose_scene_generation"]
    yaw_min, yaw_max = [math.radians(float(v)) for v in generation["yaw_deg"]]
    upright_capable = [
        object_index
        for object_index in selected_object_indices
        if np.asarray(libraries[object_index]["upright_pose_mask"]).any()
    ]
    if forced_upright_count > len(upright_capable):
        raise RuntimeError(
            f"Requested {forced_upright_count} upright objects but selected set has "
            f"only {len(upright_capable)} upright-capable objects"
        )
    forced_upright_indices = set(
        int(value)
        for value in (
            rng.choice(
                upright_capable,
                size=forced_upright_count,
                replace=False,
            )
            if forced_upright_count
            else []
        )
    )
    oriented = []
    for object_index in selected_object_indices:
        library = libraries[object_index]
        base_probabilities = np.asarray(
            library["probabilities"], dtype=np.float64
        )
        sampling_probabilities = base_probabilities
        upright_forced = object_index in forced_upright_indices
        if upright_forced:
            sampling_probabilities = base_probabilities.copy()
            sampling_probabilities[~library["upright_pose_mask"]] = 0.0
            sampling_probabilities /= sampling_probabilities.sum()
        pose_index = int(
            rng.choice(len(sampling_probabilities), p=sampling_probabilities)
        )
        yaw = float(rng.uniform(yaw_min, yaw_max))
        transform = yaw_transform(yaw) @ library["transforms"][pose_index]
        transformed = trimesh.transform_points(library["mesh"].vertices, transform)
        footprint = MultiPoint(transformed[:, :2]).convex_hull
        if footprint.is_empty or footprint.area <= 0.0:
            raise RuntimeError(f"Degenerate footprint for object {object_index}")
        oriented.append(
            {
                "object_index": object_index,
                "stable_pose_index": pose_index,
                "stable_pose_probability": float(base_probabilities[pose_index]),
                "upright_sampling_probability": float(
                    sampling_probabilities[pose_index]
                ),
                "upright_pose_forced": upright_forced,
                "upright_pose_classified": bool(
                    library["upright_pose_mask"][pose_index]
                ),
                "pose_height_m": float(library["pose_heights_m"][pose_index]),
                "pose_height_to_horizontal_span": float(
                    library["height_to_horizontal_span"][pose_index]
                ),
                "yaw_rad": yaw,
                "base_transform": transform,
                "footprint": footprint,
                "footprint_area_m2": float(footprint.area),
                "base_z_min_m": float(transformed[:, 2].min()),
                "base_z_max_m": float(transformed[:, 2].max()),
            }
        )
    return oriented


def pack_footprints(
    rng: np.random.Generator,
    oriented: list[dict],
    config: dict,
    layout_mode: str,
) -> list[dict] | None:
    generation = config["stable_pose_scene_generation"]
    table_size = np.asarray(config["table"]["size_m"], dtype=np.float64)
    edge = float(generation["table_edge_margin_m"])
    clearance = float(generation["minimum_object_clearance_m"])
    table_interior = box(
        -0.5 * table_size[0] + edge,
        -0.5 * table_size[1] + edge,
        0.5 * table_size[0] - edge,
        0.5 * table_size[1] - edge,
    )
    candidate_count = int(generation["xy_candidates_per_object"])
    selection_pool_fraction = float(
        generation.get("layout_selection_pool_fraction", 0.15)
    )
    if not 0.0 < selection_pool_fraction <= 1.0:
        raise ValueError("layout_selection_pool_fraction must be in (0, 1]")

    supported_modes = {"uniform", "compact", "spread"}
    if layout_mode not in supported_modes:
        raise ValueError(
            f"Unsupported layout mode {layout_mode!r}; expected one of "
            f"{sorted(supported_modes)}"
        )

    order_policy = generation.get("object_order", "random")
    ordered = list(oriented)
    if order_policy == "random":
        rng.shuffle(ordered)
    elif order_policy == "random_anchor_then_largest":
        # The anchor identity changes from scene to scene, removing the old
        # largest-object bias.  Packing the remaining large footprints early
        # keeps the rejection rate practical on a crowded 0.50 x 0.30 m table.
        anchor_index = int(rng.integers(len(ordered)))
        anchor = ordered.pop(anchor_index)
        ordered.sort(key=lambda row: row["footprint_area_m2"], reverse=True)
        ordered.insert(0, anchor)
    elif order_policy == "largest_first":
        ordered.sort(key=lambda row: row["footprint_area_m2"], reverse=True)
    else:
        raise ValueError(
            f"Unsupported object_order {order_policy!r}; expected random, "
            "random_anchor_then_largest or largest_first"
        )

    # Compact scenes gather around a random point in the central part of the
    # tabletop.  It is sampled once per scene, rather than being tied to the
    # first object's identity or to the table origin.
    compact_center_fraction = float(
        generation.get("compact_center_sampling_fraction", 0.50)
    )
    if not 0.0 <= compact_center_fraction <= 1.0:
        raise ValueError("compact_center_sampling_fraction must be in [0, 1]")
    interior_min = np.asarray(table_interior.bounds[:2], dtype=np.float64)
    interior_max = np.asarray(table_interior.bounds[2:], dtype=np.float64)
    compact_half_span = 0.5 * (interior_max - interior_min) * compact_center_fraction
    compact_center = rng.uniform(-compact_half_span, compact_half_span)

    placed: list[dict] = []
    packing_order = [int(item["object_index"]) for item in ordered]
    for placement_rank, item in enumerate(ordered):
        footprint = item["footprint"]
        min_x, min_y, max_x, max_y = footprint.bounds
        x_low = table_interior.bounds[0] - min_x
        y_low = table_interior.bounds[1] - min_y
        x_high = table_interior.bounds[2] - max_x
        y_high = table_interior.bounds[3] - max_y
        if x_low > x_high or y_low > y_high:
            return None
        samples = rng.uniform(
            [x_low, y_low], [x_high, y_high], size=(candidate_count, 2)
        )
        feasible = []
        for xy in samples:
            polygon = affinity.translate(footprint, xoff=float(xy[0]), yoff=float(xy[1]))
            padded = polygon.buffer(0.5 * clearance, join_style="mitre")
            if not table_interior.covers(padded):
                continue
            if any(padded.intersects(row["padded_footprint"]) for row in placed):
                continue
            nearest_distance = (
                min(polygon.distance(row["footprint_world"]) for row in placed)
                if placed
                else 0.0
            )
            if layout_mode == "spread":
                # A spread scene favours separation, but samples from the best
                # pool instead of deterministically taking the single maximum.
                score = (
                    nearest_distance if placed else float(np.linalg.norm(xy))
                )
            elif layout_mode == "compact":
                # Favour both the random cluster centre and existing objects.
                # Exact footprint clearance is still enforced above.
                center_distance = float(np.linalg.norm(xy - compact_center))
                score = -(
                    center_distance
                    if not placed
                    else 0.55 * center_distance + 0.45 * nearest_distance
                )
            else:
                score = 0.0
            feasible.append((score, xy, polygon, padded))
        if not feasible:
            return None
        if layout_mode == "uniform":
            selected_index = int(rng.integers(len(feasible)))
            _, xy, polygon, padded = feasible[selected_index]
        else:
            feasible.sort(key=lambda row: row[0], reverse=True)
            pool_size = max(
                1, int(math.ceil(len(feasible) * selection_pool_fraction))
            )
            selected_index = int(rng.integers(pool_size))
            _, xy, polygon, padded = feasible[selected_index]
        placed.append(
            {
                **item,
                "layout_mode": layout_mode,
                "placement_rank": placement_rank,
                "packing_order": packing_order,
                "compact_center_xy_m": compact_center,
                "xy_translation_m": np.asarray(xy, dtype=np.float64),
                "footprint_world": polygon,
                "padded_footprint": padded,
            }
        )
    return sorted(placed, key=lambda row: row["object_index"])


def build_candidate(
    rng: np.random.Generator,
    libraries: list[dict],
    assets: list[dict],
    config: dict,
    layout_mode: str,
    candidate_index: int,
) -> tuple[dict | None, str]:
    pool_size = len(libraries)
    objects_per_scene = int(config["scope"]["objects_per_scene"])
    if not 1 <= objects_per_scene <= pool_size:
        raise ValueError(
            f"objects_per_scene={objects_per_scene} must be in [1, {pool_size}]"
        )
    generation = config["stable_pose_scene_generation"]
    upright_policy = generation.get("upright_pose_bias", {})
    upright_active = bool(upright_policy.get("enabled", False)) and candidate_index >= int(
        upright_policy.get("start_candidate_index", 0)
    )
    forced_upright_count = (
        int(upright_policy.get("minimum_upright_objects_per_scene", 0))
        if upright_active
        else 0
    )
    if forced_upright_count > objects_per_scene:
        raise ValueError(
            "minimum_upright_objects_per_scene cannot exceed objects_per_scene"
        )
    if upright_active and forced_upright_count:
        capable_indices = np.asarray(
            [
                index
                for index, library in enumerate(libraries)
                if np.asarray(library["upright_pose_mask"]).any()
            ],
            dtype=np.int64,
        )
        if len(capable_indices) < forced_upright_count:
            raise RuntimeError(
                f"Only {len(capable_indices)} objects have upright stable poses; "
                f"cannot force {forced_upright_count} per scene"
            )
        guaranteed = np.asarray(
            rng.choice(
                capable_indices,
                size=forced_upright_count,
                replace=False,
            ),
            dtype=np.int64,
        )
        remaining_pool = np.setdiff1d(
            np.arange(pool_size, dtype=np.int64), guaranteed, assume_unique=False
        )
        remaining = np.asarray(
            rng.choice(
                remaining_pool,
                size=objects_per_scene - forced_upright_count,
                replace=False,
            ),
            dtype=np.int64,
        )
        selected_object_indices = sorted(
            int(value) for value in np.concatenate((guaranteed, remaining))
        )
    else:
        selected_object_indices = sorted(
            int(value)
            for value in rng.choice(pool_size, size=objects_per_scene, replace=False)
        )
    oriented = sample_oriented_objects(
        rng,
        libraries,
        config,
        selected_object_indices,
        forced_upright_count=forced_upright_count,
    )
    packed = pack_footprints(rng, oriented, config, layout_mode)
    if packed is None:
        return None, "footprint_packing_failed"
    table_top = float(config["table"]["top_z_m"])
    clearance = float(generation["release_clearance_m"])
    transforms = []
    records = []
    for row in packed:
        transform = np.asarray(row["base_transform"], dtype=np.float64).copy()
        transform[0, 3] += row["xy_translation_m"][0]
        transform[1, 3] += row["xy_translation_m"][1]
        # Trimesh stable transforms put the support hull at z=0.  Correct for
        # numeric residue and release one millimetre above the configured top.
        transform[2, 3] += table_top + clearance - row["base_z_min_m"]
        transforms.append(transform)
        polygon_xy = np.asarray(row["footprint_world"].exterior.coords[:-1])
        records.append(
            {
                "object_index": row["object_index"],
                "object_code": assets[row["object_index"]]["object_code"],
                "stable_pose_index": row["stable_pose_index"],
                "stable_pose_probability": row["stable_pose_probability"],
                "upright_sampling_probability": row[
                    "upright_sampling_probability"
                ],
                "upright_pose_forced": row["upright_pose_forced"],
                "upright_pose_classified": row["upright_pose_classified"],
                "pose_height_m": row["pose_height_m"],
                "pose_height_to_horizontal_span": row[
                    "pose_height_to_horizontal_span"
                ],
                "yaw_rad": row["yaw_rad"],
                "placement_rank": row["placement_rank"],
                "xy_translation_m": row["xy_translation_m"].tolist(),
                "release_clearance_m": clearance,
                "footprint_area_m2": row["footprint_area_m2"],
                "footprint_world_xy_m": polygon_xy.tolist(),
                "analytic_T_world_centered_object": transform.tolist(),
            }
        )
    positions = [transform[:3, 3].tolist() for transform in transforms]
    quaternions = [
        quat_wxyz_from_matrix(transform[:3, :3]).tolist() for transform in transforms
    ]
    return {
        "active_object_indices": selected_object_indices,
        "active_object_ids": [
            int(config["objects"][index]["id"])
            for index in selected_object_indices
        ],
        "layout_sampling": {
            "mode": layout_mode,
            "object_order_policy": generation.get("object_order", "random"),
            "packing_order_object_indices": packed[0]["packing_order"],
            "compact_center_xy_m": packed[0]["compact_center_xy_m"].tolist(),
            "upright_pose_bias": {
                "active": upright_active,
                "minimum_upright_objects_per_scene": forced_upright_count,
                "forced_upright_object_indices": [
                    int(row["object_index"])
                    for row in packed
                    if row["upright_pose_forced"]
                ],
                "classified_upright_object_count": int(
                    sum(bool(row["upright_pose_classified"]) for row in packed)
                ),
            },
        },
        "initial_records": records,
        "positions_world_m": positions,
        "quaternions_world_wxyz": quaternions,
    }, "accepted"


def pairwise_xy_distance_signature(candidate: dict) -> np.ndarray:
    """Describe scene geometry independently of global translation/rotation."""
    xy = np.asarray(candidate["positions_world_m"], dtype=np.float64)[:, :2]
    pairwise = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    upper = np.triu_indices(len(xy), k=1)
    # Sorting makes this signature independent of which pool identities happen
    # to occupy the six positions.  Object-set diversity is recorded separately.
    return np.sort(pairwise[upper])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assets = prepare_assets(config)
    libraries = compute_pose_libraries(assets, config)
    generation = config["stable_pose_scene_generation"]
    count = int(generation["candidate_count"] if args.count is None else args.count)
    output = (
        Path(config["paths"]["output_root"])
        / config["paths"]["candidate_directory_name"]
    )
    output.mkdir(parents=True, exist_ok=True)
    accepted = 0
    global_attempt = 0
    accepted_signatures: list[np.ndarray] = []
    if args.resume:
        existing_paths = sorted(output.glob("candidate_[0-9][0-9][0-9][0-9].json"))
        for expected_index, path in enumerate(existing_paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload["candidate_index"]) != expected_index:
                raise RuntimeError(f"Non-contiguous resume candidate: {path}")
            if int(payload.get("object_pool_size", len(config["objects"]))) != len(
                config["objects"]
            ):
                raise RuntimeError(f"Object-pool mismatch in resume candidate: {path}")
            accepted_signatures.append(pairwise_xy_distance_signature(payload))
            global_attempt = max(global_attempt, int(payload["global_attempt"]))
        accepted = len(existing_paths)
        if accepted:
            print(
                f"[RESUME] candidates={accepted} global_attempt={global_attempt}",
                flush=True,
            )
    rejection_counts: dict[str, int] = {}
    layout_modes = generation.get(
        "layout_mode_sequence", ["uniform", "compact", "spread"]
    )
    if not layout_modes:
        raise ValueError("layout_mode_sequence must contain at least one mode")
    minimum_signature_rms = float(
        generation.get("minimum_pairwise_layout_rms_m", 0.0)
    )
    while accepted < count:
        local_success = False
        # Cycling the modes guarantees that early candidates are not all drawn
        # from the same layout family.  Isaac Sim can therefore accept a varied
        # teaching subset even when it stops after the first five stable scenes.
        layout_mode = str(layout_modes[accepted % len(layout_modes)])
        for _ in range(int(generation["maximum_scene_attempts_per_candidate"])):
            global_attempt += 1
            rng = np.random.default_rng(
                int(config["random_seed"]) + global_attempt * 104729
            )
            result, reason = build_candidate(
                rng, libraries, assets, config, layout_mode, accepted
            )
            if result is None:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            signature = pairwise_xy_distance_signature(result)
            if accepted_signatures and minimum_signature_rms > 0.0:
                nearest_rms = min(
                    float(np.sqrt(np.mean(np.square(signature - previous))))
                    for previous in accepted_signatures
                )
                if nearest_rms < minimum_signature_rms:
                    reason = "layout_too_similar"
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
            payload = {
                "schema_version": 1,
                "generator": f"Trimesh {trimesh.__version__} stable poses + Shapely footprint packing",
                "candidate_type": "analytic_stable_pose_packing",
                "paper_pipeline_role": "adapter candidate before Isaac Sim 5.0 physical verification",
                "candidate_index": accepted,
                "global_attempt": global_attempt,
                "table": config["table"],
                "object_pool_size": len(config["objects"]),
                "objects": [
                    config["objects"][index]
                    for index in result["active_object_indices"]
                ],
                "assets": [
                    assets[index] for index in result["active_object_indices"]
                ],
                "generation_config": generation,
                **result,
            }
            write_json_atomic(output / f"candidate_{accepted:04d}.json", payload)
            accepted_signatures.append(signature)
            accepted += 1
            local_success = True
            print(
                f"[PACKED] candidate={accepted - 1:04d} "
                f"mode={layout_mode} global_attempt={global_attempt}",
                flush=True,
            )
            break
        if not local_success:
            raise RuntimeError(
                f"Could not pack candidate {accepted}; rejections={rejection_counts}"
            )
    write_json_atomic(
        output / "candidate_manifest.json",
        {
            "status": "complete",
            "candidate_type": "analytic_stable_pose_packing",
            "candidate_count": count,
            "global_attempt_count": global_attempt,
            "rejection_counts": rejection_counts,
            "config": str(args.config.resolve()),
        },
    )
    print(f"[COMPLETE] {count} analytic candidates at {output}")


if __name__ == "__main__":
    main()
