#!/usr/bin/env python3
"""Stage 02: filter Wuji2 *pregrasps* by scene and table clearance.

This is the isolated Wuji2 counterpart of DexGraspNet2's
``CollisionChecker`` plus the strict filtering expression in
``src/preprocess/dex_graspness.py``.  Signed hand distance is positive inside
the hand and negative outside.  Therefore the official condition
``distance < -0.0025`` means a strict clearance greater than 2.5 mm.

The official evaluator opens the fingers by 25 mm and moves the hand back by
10 cm before checking collision.  This Wuji2 adaptation uses the reviewed
palm-to-thumb/index-gap direction for that retreat.  Contact and enclosure at
the final target grasp are intentional and are not rejected by this
scene-pregrasp test.

The official repository reads precomputed ``collision_label`` arrays but does
not publish the program that created those arrays.  Here, every scene object
(including the target, because the initial pregrasp must not intersect it) is
represented by a deterministic mesh-surface point cloud, and the existing
Wuji2 1 mm articulated-link SDF is queried at those points.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


from .adapter_common import load_config, write_json_atomic
from .project import PROJECT_ROOT, project_path, source_path

ADAPTER_ROOT = PROJECT_ROOT


STAGE_01 = "01_transformed_object_grasps"
STAGE_NAME = "02_scene_table_collision_filtered"
SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
OUTPUT_POLICY_REVISION = "wuji2-nondestructive-paper-mask-v1"
WUJI2_MODEL_PATH = source_path("wuji2_factory") / (
    "04_pipeline/engine/configurable_object/grasp_generation/utils/"
    "wuji2_hand_model.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ADAPTER_ROOT
        / "02_training_dataset"
        / "config"
        / "wuji2_train60_100seminal_256view_v1.json",
    )
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-grasps-per-object",
        type=int,
        default=None,
        help="Optional Stage-01 prefix for a quick diagnostic run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override collision_grasp_batch_size; useful for CPU background work.",
    )
    return parser.parse_args()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_wuji2_module():
    if not WUJI2_MODEL_PATH.is_file():
        raise FileNotFoundError(WUJI2_MODEL_PATH)
    name = "wuji2_hand_model_for_dgn2_adapter"
    spec = importlib.util.spec_from_file_location(name, WUJI2_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {WUJI2_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_manifests(config: dict, scene_index: int) -> tuple[dict, dict, Path, Path]:
    output_root = Path(config["paths"]["output_root"])
    scene_path = (
        output_root / "scenes" / f"scene_{scene_index:04d}" / "scene_manifest.json"
    )
    stage01_path = (
        output_root
        / config["grasp_label_generation"]["stage_directory_name"]
        / STAGE_01
        / f"scene_{scene_index:04d}"
        / "stage_manifest.json"
    )
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not stage01_path.is_file():
        raise FileNotFoundError(stage01_path)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    stage01 = json.loads(stage01_path.read_text(encoding="utf-8"))
    if int(scene["scene_index"]) != scene_index:
        raise RuntimeError("Scene manifest index mismatch")
    if int(stage01["scene_index"]) != scene_index or stage01["stage"] != STAGE_01:
        raise RuntimeError("Stage-01 manifest contract mismatch")
    return scene, stage01, scene_path, stage01_path


def deterministic_surface_points(
    mesh_path: Path, count: int, seed: int
) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Not a triangle mesh: {mesh_path}")
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise RuntimeError(f"Invalid sampled surface for {mesh_path}")
    return points.astype(np.float32)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        points @ np.asarray(transform[:3, :3], dtype=np.float32).T
        + np.asarray(transform[:3, 3], dtype=np.float32)
    ).astype(np.float32)


def build_scene_object_points(
    scene: dict, count: int, seed: int, cache_root: Path | None = None
) -> dict[int, torch.Tensor]:
    points = {}
    for record in scene["objects"]:
        object_id = int(record["segmentation_id"])
        mesh_path = project_path(record["asset"]["centered_combined_obj"])
        object_seed = seed + object_id
        local = None
        cache_path = None
        if cache_root is not None:
            mesh_hash = str(record["asset"].get("source_obj_sha256", "unknown"))
            cache_key = json_sha256(
                {
                    "schema": CACHE_SCHEMA_VERSION,
                    "mesh_sha256": mesh_hash,
                    "count": count,
                    "seed": object_seed,
                }
            )[:20]
            cache_path = cache_root / "surface_points" / f"{cache_key}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as archive:
                    candidate = np.asarray(archive["points_centered_object"])
                    stored_hash = str(archive["mesh_sha256"].item())
                    stored_seed = int(archive["seed"].item())
                if (
                    candidate.shape == (count, 3)
                    and stored_hash == mesh_hash
                    and stored_seed == object_seed
                    and np.isfinite(candidate).all()
                ):
                    local = candidate.astype(np.float32, copy=False)
        if local is None:
            local = deterministic_surface_points(mesh_path, count, object_seed)
            if cache_path is not None:
                atomic_savez(
                    cache_path,
                    points_centered_object=local,
                    mesh_sha256=np.asarray(
                        str(record["asset"].get("source_obj_sha256", "unknown"))
                    ),
                    seed=np.asarray(object_seed, dtype=np.int64),
                    count=np.asarray(count, dtype=np.int64),
                )
        world = transform_points(
            np.asarray(record["T_world_centered_object"], dtype=np.float32), local
        )
        points[object_id] = torch.from_numpy(world)
    return points


def world_link_transforms_from_base_pose(
    model,
    base_pose: torch.Tensor,
    qpos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return T_world_link when the supplied root is T_world_r_base_link.

    Wuji2HandKinematics internally roots FK at r_wrist and also returns the
    fixed ancestor r_base_link.  This explicit bridge prevents the two roots
    from ever being silently treated as the same frame.
    """
    base_fk = model.forward_kinematics_base(qpos)
    wrist_to_base = base_fk["r_base_link"]
    world_to_wrist = base_pose @ torch.linalg.inv(wrist_to_base)
    transforms = {
        link: world_to_wrist @ wrist_to_link
        for link, wrist_to_link in base_fk.items()
    }
    if not torch.allclose(
        transforms["r_base_link"], base_pose, atol=2.0e-6, rtol=0.0
    ):
        raise RuntimeError("r_base_link to r_wrist bridge failed")
    return transforms


def load_hand_link_vertices(module, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for link_name, geometry in module.MESH_GEOMETRY.items():
        mesh_path = Path(geometry["source_mesh"])
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.unique(np.asarray(mesh.vertices, dtype=np.float32), axis=0)
        if not len(vertices):
            raise RuntimeError(f"No collision vertices for {link_name}: {mesh_path}")
        result[str(link_name)] = torch.as_tensor(vertices, device=device)
    return result


def minimum_table_clearance(
    transforms: dict[str, torch.Tensor],
    local_vertices: dict[str, torch.Tensor],
    table_top_z: float,
) -> torch.Tensor:
    per_link = []
    for link_name, vertices in local_vertices.items():
        transform = transforms[link_name]
        world = torch.einsum(
            "bij,vj->bvi", transform[:, :3, :3], vertices
        ) + transform[:, None, :3, 3]
        per_link.append(world[:, :, 2].amin(dim=1) - table_top_z)
    return torch.stack(per_link, dim=1).amin(dim=1)


def open_fingertips_like_official_width_mapper(
    model,
    module,
    qpos: torch.Tensor,
    label_cfg: dict,
) -> torch.Tensor:
    """Reproduce WidthMapper's 20-step fingertip-target IK for Wuji2."""
    return shift_fingertips_like_official_width_mapper(
        model=model,
        module=module,
        qpos=qpos,
        label_cfg=label_cfg,
        delta_width_m=-float(label_cfg["pregrasp_fingertip_opening_m"]),
        keep_z=False,
    )


def shift_fingertips_like_official_width_mapper(
    model,
    module,
    qpos: torch.Tensor,
    label_cfg: dict,
    delta_width_m,
    keep_z: bool = False,
    direction_mode: str = "surface_normal",
) -> torch.Tensor:
    """Wuji2 counterpart of the official WidthMapper.squeeze_fingers.

    Negative ``delta_width_m`` opens the fingertips; positive values squeeze
    them.  The value may be one scalar shared by all tips or one value per tip
    in ``pregrasp_fingertip_normals`` order. ``surface_normal`` preserves the
    original label/pregrasp behaviour.
    ``opposition_center`` is the Wuji2 squeeze adaptation: the thumb moves
    toward the centroid of the opposing fingertips and those fingers move
    toward the thumb side.
    """
    tip_normals = label_cfg["pregrasp_fingertip_normals"]
    tip_links = list(tip_normals)
    normals_local = torch.tensor(
        [tip_normals[name] for name in tip_links],
        dtype=qpos.dtype,
        device=qpos.device,
    )
    with torch.no_grad():
        initial_fk = model.forward_kinematics_base(qpos)
        tip_positions = torch.stack(
            [initial_fk[name][:, :3, 3] for name in tip_links], dim=1
        )
        tip_rotations = torch.stack(
            [initial_fk[name][:, :3, :3] for name in tip_links], dim=1
        )
        if direction_mode == "surface_normal":
            normals = torch.einsum("bfij,fj->bfi", tip_rotations, normals_local)
        elif direction_mode == "opposition_center":
            thumb = tip_positions[:, :1]
            opposing_center = tip_positions[:, 1:].mean(dim=1, keepdim=True)
            grasp_center = 0.5 * (thumb + opposing_center)
            normals = grasp_center - tip_positions
        else:
            raise ValueError(f"Unknown fingertip direction mode: {direction_mode}")
        if keep_z:
            normals[..., 2] = 0.0
        norms = torch.linalg.norm(normals, dim=-1, keepdim=True)
        if torch.any(norms <= 1.0e-8):
            raise RuntimeError("Cannot normalize a zero fingertip direction")
        normals = normals / norms
        delta = torch.as_tensor(
            delta_width_m, dtype=qpos.dtype, device=qpos.device
        ).reshape(-1)
        if delta.numel() == 1:
            delta = delta.repeat(len(tip_links))
        if delta.numel() != len(tip_links):
            raise ValueError(
                "delta_width_m must be scalar or contain one value per "
                f"fingertip ({len(tip_links)}), got {delta.numel()}"
            )
        targets = tip_positions + delta.reshape(1, -1, 1) * normals
    optimized = qpos.detach().clone().requires_grad_(True)
    # Limits must come from the exact kinematic companion used to build the
    # model.  In the official-USD pipeline this is the official Hand2 Beta1
    # companion URDF from the same release/commit, not the legacy dataset URDF.
    joint_limit_source = getattr(model, "urdf_path", module.ORIGINAL_HAND_URDF)
    joint_table = module.parse_wuji2_joint_table(joint_limit_source)
    lower = torch.tensor(
        [item.lower_limit for item in joint_table],
        dtype=qpos.dtype,
        device=qpos.device,
    )
    upper = torch.tensor(
        [item.upper_limit for item in joint_table],
        dtype=qpos.dtype,
        device=qpos.device,
    )
    for _ in range(int(label_cfg["pregrasp_width_mapper_steps"])):
        fk = model.forward_kinematics_base(optimized)
        positions = torch.stack([fk[name][:, :3, 3] for name in tip_links], dim=1)
        loss = ((positions - targets) ** 2).sum()
        gradient = torch.autograd.grad(loss, optimized)[0]
        with torch.no_grad():
            optimized -= float(label_cfg["pregrasp_width_mapper_learning_rate"]) * gradient
            optimized.clamp_(lower, upper)
    return optimized.detach()


def pregrasp_base_pose(
    model,
    final_base_pose: torch.Tensor,
    retreat_m: float,
) -> torch.Tensor:
    """Move T_world_r_base_link backward along semantic palm +Z."""
    batch = final_base_pose.shape[0]
    zero_qpos = torch.zeros((1, 20), dtype=final_base_pose.dtype, device=final_base_pose.device)
    wrist_fk = model.forward_kinematics_base(zero_qpos)
    base_to_wrist = torch.linalg.inv(wrist_fk["r_base_link"])[0]
    base_to_palm = base_to_wrist @ model.base_to_palm
    retreat_base = base_to_palm[:3, :3] @ torch.tensor(
        [0.0, 0.0, float(retreat_m)],
        dtype=final_base_pose.dtype,
        device=final_base_pose.device,
    )
    local = torch.eye(4, dtype=final_base_pose.dtype, device=final_base_pose.device)
    local[:3, 3] = retreat_base
    return final_base_pose @ local.view(1, 4, 4).expand(batch, -1, -1)


def tiger_mouth_approach_direction_base(
    model, qpos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Wuji2 palm-to-thumb/index-gap direction in r_base_link.

    This is the previously reviewed Wuji2 approach definition.  It is
    configuration-dependent: the thumb/index midpoint is recomputed for every
    GRASP qpos instead of treating the approach as one fixed robot axis.
    """

    fk = model.forward_kinematics_base(qpos)
    base_from_wrist = torch.linalg.inv(fk["r_base_link"])

    def point_in_base(link_name: str) -> torch.Tensor:
        point_wrist = fk[link_name][:, :3, 3]
        return torch.einsum(
            "bij,bj->bi", base_from_wrist[:, :3, :3], point_wrist
        ) + base_from_wrist[:, :3, 3]

    thumb_tip = point_in_base("r_thumb_tip")
    index_tip = point_in_base("r_index_finger_tip")
    gap_midpoint = 0.5 * (thumb_tip + index_tip)

    zero = torch.zeros((1, 20), dtype=qpos.dtype, device=qpos.device)
    zero_fk = model.forward_kinematics_base(zero)
    base_from_wrist_zero = torch.linalg.inv(zero_fk["r_base_link"])[0]
    base_from_palm = base_from_wrist_zero @ model.base_to_palm
    palm_center = base_from_palm[:3, 3].view(1, 3)
    direction = gap_midpoint - palm_center
    direction = direction / torch.linalg.norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    return direction, gap_midpoint


def tiger_mouth_pregrasp_pose(
    model,
    final_base_pose: torch.Tensor,
    grasp_qpos: torch.Tensor,
    retreat_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retreat the root opposite the palm-to-thumb/index-gap approach."""

    direction_base, gap_midpoint = tiger_mouth_approach_direction_base(
        model, grasp_qpos
    )
    direction_world = torch.einsum(
        "bij,bj->bi", final_base_pose[:, :3, :3], direction_base
    )
    direction_world = direction_world / torch.linalg.norm(
        direction_world, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    pregrasp = final_base_pose.clone()
    pregrasp[:, :3, 3] -= float(retreat_m) * direction_world
    return pregrasp, direction_base, gap_midpoint


def pregrasp_cache_contract(
    qpos: np.ndarray,
    object_code: str,
    label_cfg: dict,
) -> dict:
    """Describe every input that changes Wuji2 PREGRASP joint kinematics."""

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "policy_revision": OUTPUT_POLICY_REVISION,
        "object_code": object_code,
        "qpos_sha256": array_sha256(qpos),
        "qpos_shape": list(qpos.shape),
        "opening_m": float(label_cfg["pregrasp_fingertip_opening_m"]),
        "ik_steps": int(label_cfg["pregrasp_width_mapper_steps"]),
        "ik_learning_rate": float(
            label_cfg["pregrasp_width_mapper_learning_rate"]
        ),
        "fingertip_normals": label_cfg["pregrasp_fingertip_normals"],
        "approach_mode": "semantic-palm-to-thumb-index-gap",
    }


def load_or_build_pregrasp_cache(
    model,
    module,
    qpos: np.ndarray,
    object_code: str,
    label_cfg: dict,
    cache_root: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, Path, bool]:
    """Cache q_pre and the q-dependent tiger-mouth direction per object grasp.

    The same single-object qpos library is reused in many cluttered scenes.  Its
    20-step opening IK and thumb/index geometry therefore need to be evaluated
    only once; scene-specific work is just a rigid object/world transform plus
    signed-distance queries.
    """

    qpos = np.asarray(qpos, dtype=np.float32)
    contract = pregrasp_cache_contract(qpos, object_code, label_cfg)
    digest = json_sha256(contract)
    safe_code = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in object_code
    )
    cache_path = cache_root / "pregrasp" / f"{safe_code}_{digest[:20]}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            stored_contract = json.loads(str(archive["contract_json"].item()))
            opened = np.asarray(archive["pregrasp_qpos"], dtype=np.float32)
            direction = np.asarray(
                archive["tiger_mouth_direction_in_r_base_link"], dtype=np.float32
            )
        if (
            stored_contract == contract
            and opened.shape == qpos.shape
            and direction.shape == (len(qpos), 3)
            and np.isfinite(opened).all()
            and np.isfinite(direction).all()
        ):
            return opened, direction, cache_path, True

    opened_parts = []
    direction_parts = []
    for start in range(0, len(qpos), batch_size):
        stop = min(start + batch_size, len(qpos))
        batch = torch.as_tensor(qpos[start:stop], device=device)
        opened = open_fingertips_like_official_width_mapper(
            model, module, batch, label_cfg
        )
        direction, _ = tiger_mouth_approach_direction_base(model, batch)
        opened_parts.append(opened.detach().cpu().numpy().astype(np.float32))
        direction_parts.append(direction.detach().cpu().numpy().astype(np.float32))
    opened_all = np.concatenate(opened_parts, axis=0)
    direction_all = np.concatenate(direction_parts, axis=0)
    atomic_savez(
        cache_path,
        contract_json=np.asarray(
            json.dumps(contract, ensure_ascii=False, sort_keys=True)
        ),
        pregrasp_qpos=opened_all,
        tiger_mouth_direction_in_r_base_link=direction_all,
    )
    return opened_all, direction_all, cache_path, False


def collision_metrics(
    model,
    module,
    arrays: dict[str, np.ndarray],
    target_id: int,
    scene_points_cpu: dict[int, torch.Tensor],
    local_hand_vertices: dict[str, torch.Tensor],
    table_top_z: float,
    batch_size: int,
    device: torch.device,
    label_cfg: dict,
    approach_mode: str = "tiger_mouth",
    precomputed_opened_qpos: np.ndarray | None = None,
    precomputed_direction_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(arrays["qpos"].shape[0])
    object_ids = sorted(scene_points_cpu)
    per_object_clearance = np.full((total, len(object_ids)), np.nan, np.float32)
    table_clearance = np.empty(total, np.float32)
    minimum_scene_clearance = np.empty(total, np.float32)
    pregrasp_poses = np.empty((total, 4, 4), np.float32)
    pregrasp_qposes = np.empty_like(arrays["qpos"], dtype=np.float32)
    if precomputed_opened_qpos is not None and np.asarray(
        precomputed_opened_qpos
    ).shape != arrays["qpos"].shape:
        raise ValueError("Cached PREGRASP qpos shape differs from grasp qpos")
    if precomputed_direction_base is not None and np.asarray(
        precomputed_direction_base
    ).shape != (total, 3):
        raise ValueError("Cached tiger-mouth direction shape must be [N,3]")
    start = 0
    active_batch_size = int(batch_size)
    while start < total:
        stop = min(start + active_batch_size, total)
        base_pose = torch.as_tensor(
            arrays["T_world_r_base_link"][start:stop], device=device
        )
        final_qpos = torch.as_tensor(arrays["qpos"][start:stop], device=device)
        if precomputed_opened_qpos is None:
            opened_qpos = open_fingertips_like_official_width_mapper(
                model, module, final_qpos, label_cfg
            )
        else:
            opened_qpos = torch.as_tensor(
                precomputed_opened_qpos[start:stop], device=device
            )
        if approach_mode == "tiger_mouth":
            if precomputed_direction_base is None:
                opened_pose, _, _ = tiger_mouth_pregrasp_pose(
                    model,
                    base_pose,
                    final_qpos,
                    float(label_cfg["pregrasp_retreat_m"]),
                )
            else:
                direction_base = torch.as_tensor(
                    precomputed_direction_base[start:stop], device=device
                )
                direction_world = torch.einsum(
                    "bij,bj->bi", base_pose[:, :3, :3], direction_base
                )
                direction_world = direction_world / torch.linalg.norm(
                    direction_world, dim=-1, keepdim=True
                ).clamp_min(1.0e-8)
                opened_pose = base_pose.clone()
                opened_pose[:, :3, 3] -= (
                    float(label_cfg["pregrasp_retreat_m"]) * direction_world
                )
        elif approach_mode == "semantic_palm":
            opened_pose = pregrasp_base_pose(
                model, base_pose, float(label_cfg["pregrasp_retreat_m"])
            )
        else:
            raise ValueError(f"Unknown approach_mode: {approach_mode}")
        with torch.inference_mode():
            transforms = world_link_transforms_from_base_pose(
                model, opened_pose, opened_qpos
            )
            table_clearance[start:stop] = (
                minimum_table_clearance(transforms, local_hand_vertices, table_top_z)
                .cpu()
                .numpy()
            )
            batch_minimum = torch.full(
                (stop - start,), float("inf"), device=device, dtype=torch.float32
            )
            for object_id in object_ids:
                signed = model._batch_signed_distance_to_hand_transforms(
                    scene_points_cpu[object_id].to(device),
                    transforms,
                    stop - start,
                )
                clearance = -signed.amax(dim=1)
                per_object_clearance[start:stop, object_ids.index(object_id)] = (
                    clearance.cpu().numpy()
                )
                batch_minimum = torch.minimum(batch_minimum, clearance)
            minimum_scene_clearance[start:stop] = batch_minimum.cpu().numpy()
            pregrasp_poses[start:stop] = opened_pose.cpu().numpy()
            pregrasp_qposes[start:stop] = opened_qpos.cpu().numpy()
        start = stop
    return (
        minimum_scene_clearance,
        table_clearance,
        per_object_clearance,
        pregrasp_poses,
        pregrasp_qposes,
    )


def main() -> None:
    args = parse_args()
    if args.scene < 0:
        raise ValueError("--scene must be non-negative")
    if args.max_grasps_per_object is not None and args.max_grasps_per_object <= 0:
        raise ValueError("--max-grasps-per-object must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    config = load_config(args.config)
    label_cfg = config["grasp_label_generation"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    scene, stage01, scene_path, stage01_path = load_manifests(config, args.scene)
    module = load_wuji2_module()
    if list(stage01["label_contract"]["joint_order"]) != list(
        module.RIGHT_HAND_JOINT_ORDER
    ):
        raise RuntimeError("Stage-01 joint order differs from Wuji2 hand model")
    model = module.Wuji2HandKinematics(
        module.ORIGINAL_HAND_URDF, device=device, dtype=torch.float32
    )
    local_hand_vertices = load_hand_link_vertices(module, device)
    label_root = (
        Path(config["paths"]["output_root"])
        / label_cfg["stage_directory_name"]
    )
    cache_root = label_root / "_cache"
    surface_count = int(label_cfg["collision_surface_points_per_object"])
    scene_points_cpu = build_scene_object_points(
        scene,
        surface_count,
        int(label_cfg["collision_random_seed"]),
        cache_root=cache_root,
    )
    clearance_threshold = float(label_cfg["scene_and_table_clearance_m"])
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else label_cfg["collision_grasp_batch_size"]
    )
    table_top_z = float(config["table"]["top_z_m"])
    output_root = (
        label_root
        / STAGE_NAME
        / f"scene_{args.scene:04d}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    total_input = 0
    total_kept = 0
    total_cache_hits = 0
    for record in stage01["object_records"]:
        target_id = int(record["segmentation_id"])
        source_path = project_path(record["output_npz"])
        with np.load(source_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        full_count = int(arrays["qpos"].shape[0])
        if args.max_grasps_per_object is not None:
            arrays = {
                key: value[: args.max_grasps_per_object]
                for key, value in arrays.items()
            }
        tested_count = int(arrays["qpos"].shape[0])
        (
            cached_pregrasp_qpos,
            cached_tiger_direction,
            pregrasp_cache_path,
            pregrasp_cache_hit,
        ) = load_or_build_pregrasp_cache(
            model=model,
            module=module,
            qpos=arrays["qpos"],
            object_code=str(record["object_code"]),
            label_cfg=label_cfg,
            cache_root=cache_root,
            device=device,
            batch_size=batch_size,
        )
        total_cache_hits += int(pregrasp_cache_hit)
        (
            scene_clearance,
            table_clearance,
            per_object,
            pregrasp_poses,
            pregrasp_qposes,
        ) = collision_metrics(
            model=model,
            module=module,
            arrays=arrays,
            target_id=target_id,
            scene_points_cpu=scene_points_cpu,
            local_hand_vertices=local_hand_vertices,
            table_top_z=table_top_z,
            batch_size=batch_size,
            device=device,
            label_cfg=label_cfg,
            approach_mode="tiger_mouth",
            precomputed_opened_qpos=cached_pregrasp_qpos,
            precomputed_direction_base=cached_tiger_direction,
        )
        scene_keep = scene_clearance > clearance_threshold
        table_keep = table_clearance > clearance_threshold
        paper_keep = scene_keep & table_keep
        reject_reason_bits = np.zeros(tested_count, dtype=np.uint8)
        reject_reason_bits[~scene_keep] |= np.uint8(1)
        reject_reason_bits[~table_keep] |= np.uint8(2)

        # Non-destructive policy: every Stage-01 eligible grasp stays on disk.
        # Masks and metrics select subsets later without regenerating labels.
        preserved = dict(arrays)
        preserved.update(
            {
                "source_index_stage01": np.arange(tested_count, dtype=np.int64),
                "single_object_eligible_mask": np.ones(tested_count, dtype=bool),
                "paper_scene_clearance_keep_mask": scene_keep,
                "paper_table_clearance_keep_mask": table_keep,
                "paper_keep_mask": paper_keep,
                "paper_keep_indices": np.flatnonzero(paper_keep).astype(np.int64),
                "paper_reject_reason_bits": reject_reason_bits,
                "minimum_scene_clearance_m": scene_clearance,
                "minimum_table_clearance_m": table_clearance,
                "scene_clearance_by_segmentation_m": per_object,
                "pregrasp_T_world_r_base_link": pregrasp_poses,
                "pregrasp_qpos": pregrasp_qposes,
                "tiger_mouth_direction_in_r_base_link": cached_tiger_direction,
                "scene_segmentation_ids": np.asarray(
                    sorted(scene_points_cpu), dtype=np.int64
                ),
            }
        )
        output_path = output_root / source_path.name
        atomic_savez(output_path, **preserved)
        diagnostic_path = output_root / source_path.name.replace(
            ".npz", "_all_collision_diagnostics.npz"
        )
        atomic_savez(
            diagnostic_path,
            paper_keep_mask=paper_keep,
            paper_scene_clearance_keep_mask=scene_keep,
            paper_table_clearance_keep_mask=table_keep,
            paper_reject_reason_bits=reject_reason_bits,
            minimum_scene_clearance_m=scene_clearance,
            minimum_table_clearance_m=table_clearance,
            scene_clearance_by_segmentation_m=per_object,
            pregrasp_T_world_r_base_link=pregrasp_poses,
            pregrasp_qpos=pregrasp_qposes,
            scene_segmentation_ids=np.asarray(sorted(scene_points_cpu), dtype=np.int64),
        )
        total_input += tested_count
        total_kept += int(paper_keep.sum())
        records.append(
            {
                "segmentation_id": target_id,
                "object_code": record["object_code"],
                "stage01_full_count": full_count,
                "tested_count": tested_count,
                "preserved_count": tested_count,
                "kept_count": int(paper_keep.sum()),
                "paper_keep_count": int(paper_keep.sum()),
                "rejected_scene_count": int((scene_clearance <= clearance_threshold).sum()),
                "rejected_table_count": int((table_clearance <= clearance_threshold).sum()),
                "pregrasp_cache_hit": bool(pregrasp_cache_hit),
                "pregrasp_cache_npz": str(pregrasp_cache_path.resolve()),
                "output_npz": str(output_path.resolve()),
                "all_diagnostics_npz": str(diagnostic_path.resolve()),
            }
        )
        print(
            f"[OBJECT {target_id:03d}] preserved={tested_count} "
            f"paper_keep={int(paper_keep.sum())} "
            f"scene_reject={int((scene_clearance <= clearance_threshold).sum())} "
            f"table_reject={int((table_clearance <= clearance_threshold).sum())} "
            f"pregrasp_cache={'hit' if pregrasp_cache_hit else 'built'}",
            flush=True,
        )
    manifest = {
        "schema_version": 2,
        "stage": STAGE_NAME,
        "status": "collision_metrics_complete_all_eligible_grasps_preserved",
        "training_ready": False,
        "output_policy_revision": OUTPUT_POLICY_REVISION,
        "storage_policy": (
            "non-destructive: retain every Stage-01 eligible grasp and store "
            "paper_keep_mask plus per-grasp diagnostics"
        ),
        "scene_index": int(args.scene),
        "scene_manifest": str(scene_path.resolve()),
        "input_stage_manifest": str(stage01_path.resolve()),
        "official_benchmark": {
            "source_files": [
                "DexGraspNet2/src/utils/collision_checker.py",
                "DexGraspNet2/src/preprocess/dex_graspness.py",
                "DexGraspNet2/src/utils/data_evaluator/simulation_evaluator.py",
            ],
            "signed_distance": "positive inside hand, negative outside",
            "strict_filter": "scene_distance < -0.0025 and table_distance < -0.0025",
            "pregrasp": "open fingertips by 0.025 m and retreat 0.10 m before collision checking",
        },
        "wuji2_implementation": {
            "hand_root_input": "T_world_r_base_link",
            "training_joint_posture": label_cfg["training_joint_field"],
            "collision_joint_posture": "20-step Wuji2 fingertip-target IK opened by 0.025 m",
            "collision_root_pose": (
                "target T_world_r_base_link retreated 0.10 m opposite the "
                "current-GRASP semantic-palm-to-thumb/index-gap direction"
            ),
            "hand_collision_model": str(module.HAND_SDF_MANIFEST_PATH.resolve()),
            "hand_sdf_voxel_size_m": float(model.hand_sdf_voxel_size_m),
            "other_object_surface_points_each": surface_count,
            "surface_sampling_seed": int(label_cfg["collision_random_seed"]),
            "strict_clearance_m": clearance_threshold,
            "paper_mask_field": "paper_keep_mask",
            "reject_reason_bits": {
                "1": "PREGRASP scene clearance is not strictly above threshold",
                "2": "PREGRASP table clearance is not strictly above threshold",
            },
            "table_clearance_method": "exact minimum z over transformed hand collision-mesh vertices",
            "scene_clearance_method": "negative max Wuji2 hand SDF over deterministic surface samples from every scene object, including the target",
        },
        "remaining_required_stages": [
            "Wuji2 fingertip/palm reference-point computation",
            "surface and single-view graspness assignment",
        ],
        "diagnostic_prefix_limit": args.max_grasps_per_object,
        "collision_batch_size": batch_size,
        "pregrasp_cache_hits": total_cache_hits,
        "object_records": records,
        "total_tested": total_input,
        "total_preserved": total_input,
        "total_kept": total_kept,
        "total_paper_keep": total_kept,
    }
    manifest_path = output_root / "stage_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        f"[COMPLETE] scene={args.scene:04d} preserved={total_input} "
        f"paper_keep={total_kept} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
