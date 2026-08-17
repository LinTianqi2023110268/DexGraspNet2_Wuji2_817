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


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


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
    command = [
        sys.executable,
        str(LOCAL_BACKEND),
        "--rgb",
        str(image_path),
        "--depth",
        str(depth),
        "--intrinsics",
        str(intrinsics),
        "--world-from-camera",
        str(world_from_camera),
        "--query",
        text_query,
        "--output",
        str(output_root),
        "--device",
        "auto",
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(output_root / "matplotlib_cache"))
    completed = subprocess.run(command, cwd=LOCAL_GROUNDED_SAM_ROOT, env=env, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Grounded-SAM backend failed with exit code {completed.returncode}")
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
