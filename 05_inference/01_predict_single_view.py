#!/usr/bin/env python3
"""Run the official DexGraspNet2 inference path with a 20-joint Wuji2 model.

The seed proposal, FPS sampling, diffusion generation, and score definition
come directly from ``DexGraspNet2/src/network/graspness_sample.py``.  This
adapter only supplies the Wuji2 checkpoint contract, joint order, camera/world
conversion, and an inspectable output containing every proposal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wuji2_dgn2.project import source_path  # noqa: E402

OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network/official_core"
DEFAULT_DATA_ROOT = source_path("active_training_scene_dataset")
DEFAULT_CKPT = source_path("wuji2_checkpoint")
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs/scene_0036_view_0000_all.npz"

os.chdir(OFFICIAL_ROOT)
sys.path.insert(0, str(OFFICIAL_ROOT))

from src.network.model import get_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.dataset import get_sparse_tensor  # noqa: E402
from src.utils.util import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--scene", type=int, default=36)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grasp-num", type=int, default=1024)
    parser.add_argument("--graspness-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--categorical",
        action="store_true",
        help="Distribute proposals across ground-truth segmentation IDs; official default is off.",
    )
    return parser.parse_args()


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def resolve_adapter_path(path: Path) -> Path:
    """Resolve CLI-relative paths against the isolated adapter, not os.getcwd()."""

    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def classification_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    return {
        "accuracy": float((tp + tn) / max(len(target), 1)),
        "object_precision": float(tp / max(tp + fp, 1)),
        "object_recall": float(tp / max(tp + fn, 1)),
        "object_iou": float(tp / max(tp + fp + fn, 1)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    if args.grasp_num <= 0:
        raise ValueError("--grasp-num must be positive")
    data_root = resolve_adapter_path(args.data_root)
    ckpt_path = resolve_adapter_path(args.ckpt)
    output_path = resolve_adapter_path(args.output)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    allowed_output_roots = (
        (SCRIPT_DIR / "outputs").resolve(),
        (PROJECT_ROOT / "07_wuji2_network_3p3r_sim/01_cases").resolve(),
    )
    if not any(
        output_path == root or root in output_path.parents
        for root in allowed_output_roots
    ):
        raise RuntimeError(
            "Predictions must stay under one of the declared output roots: "
            f"{allowed_output_roots}"
        )

    network_path = data_root / "scenes" / f"scene_{args.scene:04d}" / "network_input.npz"
    if not network_path.is_file():
        raise FileNotFoundError(network_path)
    with np.load(network_path) as archive:
        payload = {key: archive[key] for key in archive.files}
    required = {"pc", "seg", "edge", "extrinsics"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"network_input.npz is missing: {missing}")
    if args.view < 0 or args.view >= len(payload["pc"]):
        raise IndexError(f"view {args.view} is outside [0, {len(payload['pc'])})")

    point_camera = np.asarray(payload["pc"][args.view], dtype=np.float32)
    segmentation = np.asarray(payload["seg"][args.view], dtype=np.int64)
    edge = np.asarray(payload["edge"][args.view], dtype=np.int64)
    world_from_camera = np.asarray(payload["extrinsics"][args.view], dtype=np.float32)
    if point_camera.shape != (40000, 3):
        raise ValueError(f"Expected (40000, 3), got {point_camera.shape}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if int(checkpoint.get("joint_num", -1)) != 20:
        raise RuntimeError("Checkpoint is not a 20-joint Wuji2 checkpoint")
    joint_order = tuple(checkpoint.get("joint_order", ()))
    if len(joint_order) != 20 or len(set(joint_order)) != 20:
        raise RuntimeError("Checkpoint joint_order must contain 20 unique names")
    if int(checkpoint.get("iteration", -1)) <= 0:
        raise RuntimeError(f"Checkpoint has no completed training iteration: {checkpoint.get('iteration')}")

    config = load_config(OFFICIAL_ROOT / "configs/network/train_dex_ours.yaml")
    config.model.joint_num = 20
    config.model.voxel_size = float(config.data.voxel_size)
    model = get_model(config.model)
    # Strict loading is deliberate: no missing or unexpected Wuji2 parameters.
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model.to(device).eval()
    set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    point_tensor = torch.from_numpy(point_camera)[None]
    data = get_sparse_tensor(point_tensor, float(config.data.voxel_size))
    data["seg"] = torch.from_numpy(segmentation)[None]
    data = {key: value.to(device) for key, value in data.items()}
    edge_tensor = torch.from_numpy(edge)[None].to(device)

    start = time.time()
    with torch.no_grad():
        feature = model.get_feature(data)
        objectness_logits, point_graspness = model.pred_score(feature)
        objectness_probability = objectness_logits.softmax(dim=-1)[0, :, 1].cpu().numpy()
        objectness_prediction = objectness_logits.argmax(dim=-1)[0].cpu().numpy()
        point_graspness = point_graspness[0].cpu().numpy()
        del feature, objectness_logits
        result = model.sample(
            data,
            args.grasp_num,
            graspness_scale=args.graspness_scale,
            allow_fail=True,
            cate=args.categorical,
            edge=edge_tensor,
            with_score_parts=True,
            with_point=True,
        )
    rotation_camera, translation_camera, qpos, score, object_index, graspness, log_prob, seed_camera = [
        value.detach().cpu().numpy() for value in result
    ]
    rotation_camera = rotation_camera[0]
    translation_camera = translation_camera[0]
    qpos = qpos[0]
    score = score[0]
    object_index = object_index[0]
    graspness = graspness[0]
    log_prob = log_prob[0]
    seed_camera = seed_camera.reshape(args.grasp_num, 3)

    rotation_world = world_from_camera[:3, :3][None] @ rotation_camera
    translation_world = transform_points(world_from_camera, translation_camera)
    seed_world = transform_points(world_from_camera, seed_camera)
    point_world = transform_points(world_from_camera, point_camera)
    seed_distance, seed_point_index = cKDTree(point_camera).query(seed_camera, k=1)
    seed_point_index = seed_point_index.astype(np.int64)
    seed_segmentation = segmentation[seed_point_index]
    ranking = np.argsort(-score).astype(np.int64)

    arrays_to_check = {
        "rotation_world": rotation_world,
        "translation_world": translation_world,
        "qpos": qpos,
        "score": score,
        "graspness": graspness,
        "log_prob": log_prob,
    }
    nonfinite = {name: int((~np.isfinite(value)).sum()) for name, value in arrays_to_check.items()}
    if any(nonfinite.values()):
        raise RuntimeError(f"Non-finite prediction values: {nonfinite}")
    identity = np.eye(3, dtype=np.float32)
    orthogonality_error = np.linalg.norm(
        np.swapaxes(rotation_world, -1, -2) @ rotation_world - identity,
        axis=(-2, -1),
    )
    determinant = np.linalg.det(rotation_world)
    elapsed = time.time() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        rotation_camera=rotation_camera.astype(np.float32),
        translation_camera=translation_camera.astype(np.float32),
        rotation_world=rotation_world.astype(np.float32),
        translation_world=translation_world.astype(np.float32),
        qpos=qpos.astype(np.float32),
        score=score.astype(np.float32),
        graspness=graspness.astype(np.float32),
        log_prob=log_prob.astype(np.float32),
        object_index=object_index.astype(np.int64),
        seed_point_camera=seed_camera.astype(np.float32),
        seed_point_world=seed_world.astype(np.float32),
        seed_point_index=seed_point_index,
        seed_segmentation=seed_segmentation.astype(np.int64),
        seed_match_distance_m=np.asarray(seed_distance, dtype=np.float32),
        ranking=ranking,
        joint_order=np.asarray(joint_order),
        point_cloud_camera=point_camera,
        point_cloud_world=point_world.astype(np.float32),
        ground_truth_segmentation=segmentation,
        edge=edge,
        predicted_objectness=objectness_prediction.astype(np.int64),
        predicted_object_probability=objectness_probability.astype(np.float32),
        predicted_graspness_log=point_graspness.astype(np.float32),
        T_world_camera=world_from_camera,
        scene_index=np.asarray(args.scene, dtype=np.int64),
        view_index=np.asarray(args.view, dtype=np.int64),
        graspness_scale=np.asarray(args.graspness_scale, dtype=np.float32),
    )

    summary = {
        "status": "wuji2_single_view_inference_complete",
        "official_blueprint": "DexGraspNet2 GraspnessSample.sample",
        "checkpoint": str(ckpt_path),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "strict_checkpoint_load": True,
        "scene": args.scene,
        "view": args.view,
        "point_count": len(point_camera),
        "proposal_count": args.grasp_num,
        "joint_count": len(joint_order),
        "joint_order": list(joint_order),
        "categorical": bool(args.categorical),
        "score_formula": f"log_prob + {args.graspness_scale} * graspness",
        "diffusion_prediction_type": str(config.model.diffusion.scheduler.prediction_type),
        "diffusion_inference_timesteps": int(config.model.diffusion.num_inference_timesteps),
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "objectness_metrics_on_labeled_view": classification_metrics(
            objectness_prediction, segmentation > 0
        ),
        "seed_exact_match_max_distance_m": float(np.max(seed_distance)),
        "seed_segmentation_histogram": {
            str(int(value)): int((seed_segmentation == value).sum())
            for value in np.unique(seed_segmentation)
        },
        "rotation_orthogonality_error_max": float(np.max(orthogonality_error)),
        "rotation_determinant_min_max": [
            float(np.min(determinant)),
            float(np.max(determinant)),
        ],
        "qpos_min_max_rad": [float(np.min(qpos)), float(np.max(qpos))],
        "score_min_max": [float(np.min(score)), float(np.max(score))],
        "best_candidate": {
            "index": int(ranking[0]),
            "seed_segmentation": int(seed_segmentation[ranking[0]]),
            "score": float(score[ranking[0]]),
            "graspness": float(graspness[ranking[0]]),
            "log_prob": float(log_prob[ranking[0]]),
            "translation_world_m": translation_world[ranking[0]].tolist(),
        },
        "output_npz": str(output_path),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
