#!/usr/bin/env python3
"""Run DexGraspNet 2.0 on an arbitrary ``network_input.npz``.

Unlike the official batch script, this entry point does not assume exactly 256
views or a fixed GraspNet/Acronym directory layout.  Its ``rotation`` and
``translation`` outputs remain in the world/table frame expected by the
official physics evaluator.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.network.model import get_model
from src.utils.config import ckpt_to_config
from src.utils.dataset import get_sparse_tensor
from src.utils.robot_model import RobotModel
from src.utils.util import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grasp-num", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strategy",
        choices=("ours", "top10", "graspness", "logprob", "random"),
        default="ours",
    )
    parser.add_argument(
        "--categorical",
        type=int,
        choices=(0, 1),
        default=0,
        help="0 reproduces official prediction; 1 distributes proposals across segmentation IDs",
    )
    parser.add_argument(
        "--per-object",
        action="store_true",
        help=(
            "Select one final grasp for every non-zero segmentation ID. "
            "Requires --categorical 1; the default keeps the official one-grasp-per-view behavior."
        ),
    )
    return parser.parse_args()


def select_indices(
    strategy: str,
    score: torch.Tensor,
    graspness: torch.Tensor,
    log_prob: torch.Tensor,
) -> torch.Tensor:
    if strategy == "ours":
        return score.argmax(dim=1)
    if strategy == "graspness":
        return graspness.argmax(dim=1)
    if strategy == "logprob":
        return log_prob.argmax(dim=1)
    if strategy == "random":
        return torch.randint(score.shape[1], (score.shape[0],))
    top = torch.topk(score, min(10, score.shape[1]), dim=1).indices
    choice = torch.randint(top.shape[1], (top.shape[0], 1))
    return top.gather(1, choice).squeeze(1)


def select_one(
    strategy: str,
    score: torch.Tensor,
    graspness: torch.Tensor,
    log_prob: torch.Tensor,
) -> torch.Tensor:
    """Select one candidate from one already-filtered one-dimensional group."""

    if strategy == "ours":
        return score.argmax()
    if strategy == "graspness":
        return graspness.argmax()
    if strategy == "logprob":
        return log_prob.argmax()
    if strategy == "random":
        return torch.randint(score.numel(), ())
    top = torch.topk(score, min(10, score.numel())).indices
    return top[torch.randint(top.numel(), ())]


def main() -> None:
    args = parse_args()
    if args.per_object and not args.categorical:
        raise ValueError("--per-object requires --categorical 1")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.ckpt.is_file():
        raise FileNotFoundError(
            "Checkpoint is missing: {}. Download the official DexGraspNet2 checkpoint first.".format(
                args.ckpt
            )
        )
    set_seed(args.seed)
    device = torch.device(args.device)

    payload = dict(np.load(args.input))
    required = {"pc", "seg", "edge", "extrinsics"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError("network input is missing keys: {}".format(missing))
    pc_all = torch.as_tensor(payload["pc"], dtype=torch.float32)
    seg_all = torch.as_tensor(payload["seg"], dtype=torch.long)
    edge_all = torch.as_tensor(payload["edge"], dtype=torch.long)
    extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32)
    views = len(pc_all)
    if seg_all.shape[:2] != pc_all.shape[:2] or edge_all.shape[:2] != pc_all.shape[:2]:
        raise ValueError("pc, seg and edge view/point dimensions disagree")
    if extrinsics.shape != (views, 4, 4):
        raise ValueError("extrinsics must have shape (views, 4, 4)")

    config = ckpt_to_config(str(args.ckpt))
    model = get_model(config.model)
    model.config.voxel_size = config.data.voxel_size
    checkpoint = torch.load(str(args.ckpt), map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    robot_model = RobotModel(
        "robot_models/urdf/leap_hand_simplified.urdf",
        "robot_models/meta/leap_hand/meta.yaml",
    )

    # Keep the selected seed point as well as the final grasp.  The official
    # predictor only needs the latter, but retaining both makes the translation
    # parameterization t = voxel_center(seed) + delta / scale inspectable.
    collected = [[] for _ in range(8)]
    with torch.no_grad():
        for start in range(0, views, args.stride):
            end = min(start + args.stride, views)
            data = get_sparse_tensor(pc_all[start:end], config.data.voxel_size)
            data["seg"] = seg_all[start:end]
            data = {key: value.to(device) for key, value in data.items()}
            result = model.sample(
                data,
                args.grasp_num,
                graspness_scale=5,
                allow_fail=True,
                cate=bool(args.categorical),
                edge=edge_all[start:end].to(device),
                with_score_parts=True,
                with_point=True,
            )
            # GraspnessSample returns seed points flattened as (batch * K, 3),
            # whereas the other proposal tensors retain (batch, K, ...).
            result[-1] = result[-1].reshape(end - start, args.grasp_num, 3)
            for bucket, value in zip(collected, result):
                bucket.append(value.cpu())

    rotation, translation, qpos, score, object_indices, graspness, log_prob, seed_points = [
        torch.cat(parts, dim=0) for parts in collected
    ]
    if args.per_object:
        selected_rows = []
        selected_candidates = []
        for row in range(views):
            object_ids = sorted(
                int(value) for value in torch.unique(object_indices[row]).tolist() if value != -1
            )
            for object_id in object_ids:
                candidates = torch.nonzero(
                    object_indices[row] == object_id, as_tuple=False
                ).flatten()
                local = select_one(
                    args.strategy,
                    score[row, candidates],
                    graspness[row, candidates],
                    log_prob[row, candidates],
                )
                selected_rows.append(row)
                selected_candidates.append(int(candidates[local]))
        if not selected_rows:
            raise RuntimeError("No non-zero segmentation categories produced a grasp")
        rows = torch.as_tensor(selected_rows, dtype=torch.long)
        best = torch.as_tensor(selected_candidates, dtype=torch.long)
    else:
        best = select_indices(args.strategy, score, graspness, log_prob)
        rows = torch.arange(views)

    rotation_camera = rotation[rows, best].numpy()
    translation_camera = translation[rows, best].numpy()
    seed_point_camera = seed_points[rows, best].numpy()
    qpos = qpos[rows, best].numpy()
    selected_extrinsics = extrinsics[rows.numpy()]
    voxel_size = float(config.data.voxel_size)
    trans_scale = float(config.model.trans_scale)
    voxel_center_camera = (
        np.floor(seed_point_camera / voxel_size) * voxel_size + voxel_size / 2
    )
    delta_translation_scaled = (
        translation_camera - voxel_center_camera
    ) * trans_scale
    rotation_world = selected_extrinsics[:, :3, :3] @ rotation_camera
    translation_world = (
        selected_extrinsics[:, :3, :3] @ translation_camera[:, :, None]
        + selected_extrinsics[:, :3, 3:]
    )[:, :, 0]
    seed_point_world = (
        selected_extrinsics[:, :3, :3] @ seed_point_camera[:, :, None]
        + selected_extrinsics[:, :3, 3:]
    )[:, :, 0]
    voxel_center_world = (
        selected_extrinsics[:, :3, :3] @ voxel_center_camera[:, :, None]
        + selected_extrinsics[:, :3, 3:]
    )[:, :, 0]

    output = {
        "rotation": rotation_world.astype(np.float32),
        "translation": translation_world.astype(np.float32),
        "rotation_camera": rotation_camera.astype(np.float32),
        "translation_camera": translation_camera.astype(np.float32),
        "seed_point": seed_point_world.astype(np.float32),
        "seed_point_camera": seed_point_camera.astype(np.float32),
        "voxel_center": voxel_center_world.astype(np.float32),
        "voxel_center_camera": voxel_center_camera.astype(np.float32),
        "delta_translation_scaled": delta_translation_scaled.astype(np.float32),
        "voxel_size": np.asarray(voxel_size, dtype=np.float32),
        "trans_scale": np.asarray(trans_scale, dtype=np.float32),
        "object_index": object_indices[rows, best].numpy().astype(np.int64),
        "score": score[rows, best].numpy().astype(np.float32),
        "graspness": graspness[rows, best].numpy().astype(np.float32),
        "log_prob": log_prob[rows, best].numpy().astype(np.float32),
        "view_index": rows.numpy().astype(np.int64),
    }
    output.update(
        {name: qpos[:, index].astype(np.float32) for index, name in enumerate(robot_model.joint_names)}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print("wrote {}".format(args.output.resolve()))
    print("views={}, predictions={}, joints={}".format(views, len(translation_world), len(qpos[0])))
    print("selection scope={}".format("one per object" if args.per_object else "one per view"))
    print("translation frame=world/table; rotation maps LEAP hand coordinates to world/table")


if __name__ == "__main__":
    main()
