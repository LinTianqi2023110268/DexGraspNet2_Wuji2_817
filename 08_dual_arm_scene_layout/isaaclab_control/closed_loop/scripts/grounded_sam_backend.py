#!/usr/bin/env python3
"""Closed-loop adapter for the local GroundingDINO + SAM project.

Semantic detection is intentionally limited to the captured RGB image and the
user's text query.  Depth, intrinsics, and camera pose are passed only because
the existing local backend validates/saves the corresponding target point cloud;
they are not semantic inputs to GroundingDINO.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


LOCAL_GROUNDED_SAM_ROOT = Path("/home/lin/Projects/分类抓取开源项/03_检测加分割_GroundedSAM")
LOCAL_BACKEND = LOCAL_GROUNDED_SAM_ROOT / "scripts/grounded_sam_to_pointcloud.py"
DINO_RESULT_JSON = "dino_result.json"


DINO_SUBPROCESS_CODE = r"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

backend_path = Path(sys.argv[1])
rgb_path = Path(sys.argv[2])
query = sys.argv[3]
output_path = Path(sys.argv[4])
device_request = sys.argv[5]

spec = importlib.util.spec_from_file_location("local_grounded_sam_backend", backend_path)
backend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backend)

device = backend.select_device(device_request)
prompt = backend.normalize_query(query)
print(f"[DINO] query: {query!r} -> prompt: {prompt!r}", flush=True)
print(f"[DINO] loading GroundingDINO on {device}", flush=True)
rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
dino, load_report = backend.load_dino(backend.DEFAULT_DINO_CONFIG, backend.DEFAULT_DINO_WEIGHT, device)
boxes, scores, phrases = backend.detect(
    dino,
    rgb,
    prompt,
    box_threshold=0.25,
    text_threshold=0.20,
    device=device,
)
# backend.detect already returns CPU numpy arrays; force a detached CPU-only
# serialization boundary before this CUDA process exits.
boxes = np.asarray(boxes, dtype=np.float32)
scores = np.asarray(scores, dtype=np.float32)
selected = backend.choose_detection(boxes, scores)
output = {
    "schema_version": 1,
    "query_original": query,
    "query_groundingdino": prompt,
    "device": device,
    "thresholds": {"box": 0.25, "text": 0.20},
    "boxes_xyxy_pixels": boxes.tolist(),
    "scores": scores.tolist(),
    "phrases": [str(value) for value in phrases],
    "selected_detection": int(selected),
    "model_load": load_report,
}
output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[DINO] selected box {selected}/{len(boxes)} score={float(scores[selected]):.4f}", flush=True)
"""


SAM_SUBPROCESS_CODE = r"""
import importlib.util
import contextlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

backend_path = Path(sys.argv[1])
rgb_path = Path(sys.argv[2])
depth_path = Path(sys.argv[3])
intrinsics_path = Path(sys.argv[4])
world_from_camera_path = Path(sys.argv[5])
dino_result_path = Path(sys.argv[6])
output_dir = Path(sys.argv[7])
device_request = sys.argv[8]

spec = importlib.util.spec_from_file_location("local_grounded_sam_backend", backend_path)
backend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backend)

device = backend.select_device(device_request)
dino = json.loads(dino_result_path.read_text(encoding="utf-8"))
rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
depth = np.load(depth_path).astype(np.float32)
intrinsic = np.load(intrinsics_path).astype(np.float64)
backend.validate_rgbd(rgb, depth, intrinsic)

boxes = np.asarray(dino["boxes_xyxy_pixels"], dtype=np.float32)
scores = np.asarray(dino["scores"], dtype=np.float32)
phrases = [str(value) for value in dino["phrases"]]
selected = int(dino["selected_detection"])
box = boxes[selected].astype(np.float32)

def _vram_snapshot():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5.0,
        )
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout or "").strip()
        return f"unavailable ({reason})"
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return "; ".join(f"gpu{index}: used/free MiB={line}" for index, line in enumerate(lines)) or "unavailable"


def _log_vram(label):
    print(f"[VRAM] {label}: {_vram_snapshot()}", flush=True)


precision = "autocast_fp16" if device == "cuda" else "fp32"
print(f"[SAM] precision={precision}", flush=True)
print(f"[SAM] loading SAM ViT-B on {device}", flush=True)
_log_vram("before SAM load")
model = backend.sam_model_registry["vit_b"](checkpoint=str(backend.DEFAULT_SAM_WEIGHT)).to(device)
model.eval()
predictor = backend.SamPredictor(model)
_log_vram("after SAM load")

def _autocast_context():
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()

with torch.inference_mode():
    _log_vram("before set_image")
    with _autocast_context():
        predictor.set_image(rgb)
    _log_vram("after set_image")
    with _autocast_context():
        masks, quality, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=False,
            return_logits=False,
        )
    _log_vram("after predict")
mask = masks[0].astype(bool)
sam_quality = float(quality[0])
points_camera, rows, cols, valid_mask = backend.backproject(mask, depth, intrinsic, max_depth=3.0)
colors = rgb[rows, cols]
if len(points_camera) == 0:
    raise RuntimeError("SAM mask contains no valid depth pixels")

np.save(output_dir / "mask.npy", mask)
Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / "mask.png")
np.save(output_dir / "target_points_camera.npy", points_camera)
backend.write_ply(output_dir / "target_points_camera.ply", points_camera, colors)

world_from_camera = np.load(world_from_camera_path).astype(np.float64)
points_world = backend.transform_points(points_camera, world_from_camera)
np.save(output_dir / "target_points_world.npy", points_world)
backend.write_ply(output_dir / "target_points_world.ply", points_world, colors)

overlay = backend.draw_overlay(rgb, boxes, scores, phrases, selected, mask)
Image.fromarray(overlay).save(output_dir / "overlay.png")

detections = [
    {
        "index": index,
        "phrase": phrase,
        "score": float(score),
        "box_xyxy_pixels": [float(value) for value in box_xyxy],
        "selected": index == selected,
    }
    for index, (box_xyxy, score, phrase) in enumerate(zip(boxes, scores, phrases))
]
report = {
    "schema_version": 1,
    "input": {
        "rgb": str(rgb_path.resolve()),
        "depth_m": str(depth_path.resolve()),
        "intrinsics": str(intrinsics_path.resolve()),
        "T_world_camera": str(world_from_camera_path.resolve()),
        "resolution_hw": list(depth.shape),
    },
    "query_original": dino["query_original"],
    "query_groundingdino": dino["query_groundingdino"],
    "device": device,
    "thresholds": dino["thresholds"],
    "detections": detections,
    "selected_detection": selected,
    "sam_predicted_iou": float(sam_quality),
    "mask_pixels": int(mask.sum()),
    "valid_depth_mask_pixels": int(valid_mask.sum()),
    "target_point_count": int(len(points_camera)),
    "coordinate_contract": {
        "camera": "OpenCV: +x image-right, +y image-down, +z camera-forward",
        "world": "points_world = R_world_camera @ points_camera + t_world_camera",
    },
    "model_load": dino["model_load"],
}
(output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[SAM] mask={int(mask.sum())} px, valid target cloud={len(points_camera)} points", flush=True)
"""


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _vram_snapshot() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5.0,
        )
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout or "").strip()
        return f"unavailable ({reason})"
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not values:
        return "unavailable (empty nvidia-smi output)"
    return "; ".join(f"gpu{index}: used/free MiB={line}" for index, line in enumerate(values))


def _log_vram(label: str) -> None:
    print(f"[VRAM] {label}: {_vram_snapshot()}", flush=True)


def _normalize_result(output_root: Path, query: str) -> None:
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected_index = int(result["selected_detection"])
    selected = result["detections"][selected_index]
    result.update(
        {
            "query": query,
            "grounding_score": float(selected["score"]),
            "box_xyxy": [float(value) for value in selected["box_xyxy_pixels"]],
            "backend": {
                "adapter": "closed_loop.scripts.grounded_sam_backend",
                "source_project": str(LOCAL_GROUNDED_SAM_ROOT),
                "detector": "GroundingDINO Swin-T OGC",
                "segmenter": "Segment Anything ViT-B",
                "semantic_inputs": ["rgb", "text"],
            },
        }
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_backend(image_path: Path, text_query: str, output_root: Path) -> None:
    image_path = _require(image_path.resolve(), "RGB image")
    capture_root = image_path.parent
    depth = _require(capture_root / "depth_m.npy", "aligned depth")
    intrinsics = _require(capture_root / "intrinsics.npy", "camera intrinsics")
    world_from_camera = _require(capture_root / "T_world_camera.npy", "camera pose")
    _require(LOCAL_BACKEND, "local Grounded-SAM backend")

    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(output_root / "matplotlib_cache"))
    env.setdefault("PYTHONUNBUFFERED", "1")

    dino_result = output_root / DINO_RESULT_JSON
    dino_command = [
        sys.executable,
        "-c",
        DINO_SUBPROCESS_CODE,
        str(LOCAL_BACKEND),
        str(image_path),
        text_query,
        str(dino_result),
        "auto",
    ]
    _log_vram("before DINO")
    completed = subprocess.run(dino_command, cwd=LOCAL_GROUNDED_SAM_ROOT, env=env, text=True)
    _log_vram("DINO finished")
    if completed.returncode != 0:
        raise RuntimeError(f"GroundingDINO subprocess failed with exit code {completed.returncode}")
    _require(dino_result, DINO_RESULT_JSON)
    _log_vram("after DINO process exit")

    sam_command = [
        sys.executable,
        "-c",
        SAM_SUBPROCESS_CODE,
        str(LOCAL_BACKEND),
        str(image_path),
        str(depth),
        str(intrinsics),
        str(world_from_camera),
        str(dino_result),
        str(output_root),
        "auto",
    ]
    _log_vram("before SAM")
    completed = subprocess.run(sam_command, cwd=LOCAL_GROUNDED_SAM_ROOT, env=env, text=True)
    _log_vram("SAM finished")
    if completed.returncode != 0:
        raise RuntimeError(f"SAM subprocess failed with exit code {completed.returncode}")
    _log_vram("after SAM process exit")

    for name in ("mask.npy", "overlay.png", "result.json"):
        _require(output_root / name, name)
    _normalize_result(output_root, text_query)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    query = args.text.strip()
    if not query:
        raise ValueError("--text must not be empty")
    run_backend(args.image, query, args.output.resolve())


if __name__ == "__main__":
    main()
