#!/usr/bin/env python3
"""Run the untouched official DexGraspNet2 LEAP checkpoint on one case view.

Input
-----
``01_cases/<case>/01_input/view_XXXX_network_input.npz``

Output
------
``01_cases/<case>/01_input/official_leap_1024.npz`` and an audit JSON.

Select a case with ``DGN2_CASE_ID=<case_id>``.  The script deliberately keeps
all 1024 proposals; it does not silently replace the official raw-score rank 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from case_paths import PROJECT_ROOT, active_case_root


CASE_ROOT = active_case_root()
CASE = json.loads((CASE_ROOT / "case.json").read_text(encoding="utf-8"))
VIEW_INPUT = CASE_ROOT / "01_input" / f"{CASE['view_id']}_network_input.npz"
OUTPUT = CASE_ROOT / "01_input" / "official_leap_1024.npz"
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network" / "official_core"
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
    for path in (VIEW_INPUT, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is hidden from this process. Run on the host RTX 4070; "
            "do not change the checkpoint or fall back to CPU."
        )

    original_cwd = Path.cwd()
    os.chdir(OFFICIAL_ROOT)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from src.network.model import get_model
        from src.utils.config import ckpt_to_config
        from src.utils.dataset import get_sparse_tensor
        from src.utils.robot_model import RobotModel
        from src.utils.util import set_seed

        set_seed(RANDOM_SEED)
        device = torch.device("cuda:0")
        with np.load(VIEW_INPUT, allow_pickle=False) as archive:
            pc_np = np.asarray(archive["pc"], dtype=np.float32)
            seg_np = np.asarray(archive["seg"], dtype=np.int64)
            edge_np = np.asarray(archive["edge"], dtype=np.int64)
            extrinsics = np.asarray(archive["extrinsics"], dtype=np.float32)
        if pc_np.shape != (1, 40000, 3):
            raise ValueError(f"expected (1,40000,3), got {pc_np.shape}")

        config = ckpt_to_config(str(CHECKPOINT))
        model = get_model(config.model)
        model.config.voxel_size = config.data.voxel_size
        checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device).eval()

        pc = torch.as_tensor(pc_np, dtype=torch.float32)
        seg = torch.as_tensor(seg_np, dtype=torch.long)
        edge = torch.as_tensor(edge_np, dtype=torch.long)
        with torch.no_grad():
            sparse = get_sparse_tensor(pc, config.data.voxel_size)
            sparse["seg"] = seg
            sparse = {key: value.to(device) for key, value in sparse.items()}
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
        rotation, translation, qpos, score, object_index, graspness, log_prob, seed = [
            value.detach().cpu().numpy() for value in result
        ]
        seed = seed.reshape(1, PROPOSAL_COUNT, 3)

        # Official cate=False returns no category.  Recover it from the exact
        # seed point's segmentation label without altering any prediction.
        tree = cKDTree(pc_np[0].astype(np.float64))
        distance, point_index = tree.query(seed[0].astype(np.float64), k=1)
        if float(np.max(distance)) > 1.0e-6:
            raise RuntimeError(f"seed/input mismatch: {distance.max()} m")
        seed_segmentation = seg_np[0, point_index]

        r_wc = extrinsics[0, :3, :3]
        t_wc = extrinsics[0, :3, 3]
        rotation_world = np.einsum("ij,njk->nik", r_wc, rotation[0])
        translation_world = translation[0] @ r_wc.T + t_wc
        seed_world = seed[0] @ r_wc.T + t_wc
        robot = RobotModel(
            "robot_models/urdf/leap_hand_simplified.urdf",
            "robot_models/meta/leap_hand/meta.yaml",
        )
    finally:
        os.chdir(original_cwd)

    order = np.argsort(-score[0], kind="stable")
    np.savez_compressed(
        OUTPUT,
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
        target_segmentation_id=seed_segmentation.astype(np.int64),
        official_sampler_object_index=object_index[0].astype(np.int64),
        score_descending_candidate_index=order.astype(np.int64),
        extrinsic_T_world_camera=extrinsics[0].astype(np.float32),
        random_seed=np.asarray(RANDOM_SEED, dtype=np.int64),
    )
    top = int(order[0])
    reconstruction_error = float(
        np.max(np.abs(log_prob[0] + GRASPNESS_SCALE * graspness[0] - score[0]))
    )
    report = {
        "schema_version": 1,
        "status": "official_inference_complete",
        "case_id": CASE_ROOT.name,
        "scene_id": CASE["scene_id"],
        "view_id": CASE["view_id"],
        "official_source": str(OFFICIAL_ROOT / "src/eval/predict_dexterous.py"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "input": str(VIEW_INPUT),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "proposal_count": PROPOSAL_COUNT,
        "random_seed": RANDOM_SEED,
        "graspness_scale": GRASPNESS_SCALE,
        "score_formula": "log_prob + 5 * graspness",
        "max_score_reconstruction_error": reconstruction_error,
        "selected_candidate_index": top,
        "selected_score": float(score[0, top]),
        "selected_graspness": float(graspness[0, top]),
        "selected_log_prob": float(log_prob[0, top]),
        "selected_target_segmentation_id": int(seed_segmentation[top]),
        "output": str(OUTPUT),
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] GPU={report['device']}; proposals={PROPOSAL_COUNT}")
    print(
        f"[PASS] rank0={top}; target_seg={seed_segmentation[top]}; "
        f"score={score[0, top]:.6f}"
    )
    print(f"[OK] {OUTPUT}")


if __name__ == "__main__":
    main()
