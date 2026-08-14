#!/usr/bin/env python3
"""Run the untouched official LEAP checkpoint and rank only target proposals.

The network samples 1024 proposals from the complete scene with ``cate=False``
and the official score ``log_prob + 5 * graspness``.  Grounded-SAM does not
alter generation; its sampled membership is consulted afterwards to retain
only candidates whose seed point belongs to the requested target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout"
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network/official_core"
CHECKPOINT = OFFICIAL_ROOT / "experiments/dex_ours/ckpt/ckpt_50000.pth"
PROPOSAL_COUNT = 1024
GRASPNESS_SCALE = 5.0
RANDOM_SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="ashtray")
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Official sampler random seed; different seeds generate independent diffusion proposals.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Independent 1024-proposal sampler calls after one checkpoint load.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Directory containing network_input.npz; defaults to captures/latest/dgn2/<target>.",
    )
    args = parser.parse_args()
    root = (
        args.input_root.resolve()
        if args.input_root is not None
        else LAYOUT_ROOT / "captures/latest/dgn2" / args.target
    )
    input_path = root / "network_input.npz"
    output_path = root / "official_leap_1024_target_ranked.npz"
    for path in (input_path, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Run with the graspnet2.0 host environment.")

    with np.load(input_path, allow_pickle=False) as archive:
        pc_np = np.asarray(archive["pc"], dtype=np.float32)
        seg_np = np.asarray(archive["seg"], dtype=np.int64)
        edge_np = np.asarray(archive["edge"], dtype=np.int64)
        extrinsics = np.asarray(archive["extrinsics"], dtype=np.float32)
        target_id = int(np.asarray(archive["target_segmentation_id"]).item())
    if pc_np.shape != (1, 40000, 3):
        raise ValueError(f"expected pc=(1,40000,3), got {pc_np.shape}")

    original_cwd = Path.cwd()
    os.chdir(OFFICIAL_ROOT)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from src.network.model import get_model
        from src.utils.config import ckpt_to_config
        from src.utils.dataset import get_sparse_tensor
        from src.utils.robot_model import RobotModel
        from src.utils.util import set_seed

        set_seed(args.seed)
        device = torch.device("cuda:0")
        config = ckpt_to_config(str(CHECKPOINT))
        model = get_model(config.model)
        model.config.voxel_size = config.data.voxel_size
        checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device).eval()
        pc = torch.as_tensor(pc_np, dtype=torch.float32)
        seg = torch.as_tensor(seg_np, dtype=torch.long)
        edge = torch.as_tensor(edge_np, dtype=torch.long)
        if args.rounds < 1:
            raise ValueError("--rounds must be positive")
        chunks: list[list[np.ndarray]] = [[] for _ in range(8)]
        with torch.no_grad():
            sparse = get_sparse_tensor(pc, config.data.voxel_size)
            sparse["seg"] = seg
            sparse = {key: value.to(device) for key, value in sparse.items()}
            for round_index in range(args.rounds):
                result = model.sample(
                    sparse,
                    PROPOSAL_COUNT,
                    graspness_scale=GRASPNESS_SCALE,
                    allow_fail=True,
                    cate=False,
                    edge=edge.to(device),
                    with_score_parts=True,
                    with_point=True,
                )
                for value_index, (bucket, value) in enumerate(zip(chunks, result)):
                    array = value.detach().cpu().numpy()
                    if value_index == 7:
                        array = array.reshape(1, PROPOSAL_COUNT, 3)
                    bucket.append(array)
                print(f"[SAMPLER] round {round_index + 1}/{args.rounds}", flush=True)
        rotation, translation, qpos, score, object_index, graspness, log_prob, seed = [
            np.concatenate(bucket, axis=1) for bucket in chunks
        ]
        proposal_count = PROPOSAL_COUNT * args.rounds
        seed = seed.reshape(1, proposal_count, 3)
        robot = RobotModel(
            "robot_models/urdf/leap_hand_simplified.urdf",
            "robot_models/meta/leap_hand/meta.yaml",
        )
    finally:
        os.chdir(original_cwd)

    tree = cKDTree(pc_np[0].astype(np.float64))
    distance, point_index = tree.query(seed[0].astype(np.float64), k=1)
    if float(np.max(distance)) > 1.0e-6:
        raise RuntimeError(f"seed/input mismatch: maximum {distance.max()} m")
    seed_segmentation = seg_np[0, point_index]
    target_mask = seed_segmentation == target_id
    target_candidates = np.flatnonzero(target_mask)
    if len(target_candidates) == 0:
        raise RuntimeError("Official sampler produced no seed on the segmented target")

    r_wc = extrinsics[0, :3, :3]
    t_wc = extrinsics[0, :3, 3]
    rotation_world = np.einsum("ij,njk->nik", r_wc, rotation[0])
    translation_world = translation[0] @ r_wc.T + t_wc
    seed_world = seed[0] @ r_wc.T + t_wc
    # Rank only after target membership is known.  Scores themselves are the
    # untouched official values.
    target_order = target_candidates[
        np.argsort(-score[0, target_candidates], kind="stable")
    ]
    all_order = np.argsort(-score[0], kind="stable")
    best = int(target_order[0])

    np.savez_compressed(
        output_path,
        rotation_world=rotation_world.astype(np.float32),
        translation_world=translation_world.astype(np.float32),
        rotation_camera=rotation[0].astype(np.float32),
        translation_camera=translation[0].astype(np.float32),
        leap_qpos_rad=qpos[0].astype(np.float32),
        leap_joint_order=np.asarray(robot.joint_names),
        score=score[0].astype(np.float32),
        graspness=graspness[0].astype(np.float32),
        log_prob=log_prob[0].astype(np.float32),
        seed_point_world=seed_world.astype(np.float32),
        seed_point_camera=seed[0].astype(np.float32),
        seed_point_input_index=point_index.astype(np.int64),
        seed_segmentation_id=seed_segmentation.astype(np.int64),
        target_segmentation_id=np.asarray(target_id, dtype=np.int64),
        target_candidate_index=target_candidates.astype(np.int64),
        target_score_descending_candidate_index=target_order.astype(np.int64),
        all_score_descending_candidate_index=all_order.astype(np.int64),
        official_sampler_object_index=object_index[0].astype(np.int64),
        extrinsic_T_world_camera=extrinsics[0].astype(np.float32),
    )
    report = {
        "schema_version": 1,
        "status": "official_leap_target_candidates_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_query": args.target,
        "target_segmentation_id": target_id,
        "proposal_count": proposal_count,
        "sampler_random_seed": args.seed,
        "sampler_rounds": args.rounds,
        "target_proposal_count": int(len(target_candidates)),
        "selection_rule": "target seed membership, then descending official score",
        "official_score_formula": "log_prob + 5 * graspness",
        "selected_candidate_index": best,
        "selected_score": float(score[0, best]),
        "selected_graspness": float(graspness[0, best]),
        "selected_log_prob": float(log_prob[0, best]),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "device": torch.cuda.get_device_name(0),
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] official proposals: {proposal_count}")
    print(f"[PASS] target proposals: {len(target_candidates)}")
    print(f"[BEST TARGET] candidate={best}; score={score[0,best]:.6f}")
    print(f"[OK] {output_path}")


if __name__ == "__main__":
    main()
