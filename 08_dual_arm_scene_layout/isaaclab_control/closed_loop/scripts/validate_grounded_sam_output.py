#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--query", required=True)
    args = p.parse_args()
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    mask_path = args.output_root / "mask.npy"
    overlay_path = args.output_root / "overlay.png"
    result_path = args.output_root / "result.json"
    for path in (mask_path, overlay_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    mask = np.asarray(np.load(mask_path), dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"mask/rgb mismatch: {mask.shape} vs {rgb.shape[:2]}")
    count = int(mask.sum())
    if count < 50:
        raise RuntimeError(f"Grounded-SAM mask too small: {count} pixels")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recorded = str(result.get("query", "")).strip()
    if recorded and recorded.lower() != args.query.strip().lower():
        raise RuntimeError(f"query mismatch: requested={args.query!r}, result={recorded!r}")
    print(json.dumps({
        "status": "PASS",
        "query": args.query,
        "mask_pixels": count,
        "mask_fraction": float(count / mask.size),
        "overlay": str(overlay_path.resolve()),
        "result": str(result_path.resolve()),
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
