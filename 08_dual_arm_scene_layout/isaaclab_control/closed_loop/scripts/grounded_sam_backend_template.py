#!/usr/bin/env python3
"""GroundingDINO + SAM backend contract.

Codex must replace the body of ``run_backend`` with the locally installed,
already-approved GroundingDINO/SAM implementation and model checkpoints.

HARD CONTRACT:
- Inputs to GroundingDINO are ONLY ``image_path`` and ``text_query`` (+ model params).
- Do not inspect scene_manifest.json, object_code, segmentation IDs, USD/URDF, or 3-D geometry
  to decide what the requested object is.
- SAM receives the RGB image and GroundingDINO box(es).
- Choose the highest-confidence requested instance unless an explicit instance selector is added.
- Write:
    output_root/mask.npy      bool HxW
    output_root/overlay.png
    output_root/result.json
- result.json must contain at least:
    query, grounding_score, box_xyxy, sam_predicted_iou, backend
"""
from __future__ import annotations
import argparse
from pathlib import Path

def run_backend(image_path: Path, text_query: str, output_root: Path) -> None:
    raise RuntimeError(
        "Grounded-SAM local backend is intentionally not guessed in the patch. "
        "Have Codex inspect the installed GroundingDINO/SAM environment and wire this adapter "
        "without changing the RGB+text-only semantic contract."
    )

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    run_backend(a.image.resolve(), a.text.strip(), a.output.resolve())

if __name__ == "__main__":
    main()
