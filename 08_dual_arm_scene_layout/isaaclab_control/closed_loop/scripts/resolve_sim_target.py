#!/usr/bin/env python3
"""Simulation-only binding of a vision-selected mask to one rigid object.

This script runs AFTER GroundingDINO+SAM.  It must never be used to help
GroundingDINO identify the requested object.  Its only purpose is to bind the
already-selected visual mask to a simulator rigid body so contact/lift metrics
and object placement can be measured.

The first implementation uses the median 3-D mask point and the nearest settled
object origin.  Codex may upgrade this to mask-point/surface overlap while
preserving the semantic ordering.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def backproject_selected(depth, K, mask):
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(u) < 50:
        raise RuntimeError("Too few valid target depth pixels")
    z = depth[v, u].astype(np.float64)
    x = (u - K[0,2]) * z / K[0,0]
    y = (v - K[1,2]) * z / K[1,1]
    return np.stack([x,y,z,np.ones_like(z)], axis=1)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--settled-manifest", type=Path, required=True)
    p.add_argument("--max-origin-distance-m", type=float, default=0.25)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.capture_root.resolve()
    depth = np.asarray(np.load(root/"depth_m.npy"), dtype=np.float64)
    K = np.asarray(np.load(root/"intrinsics.npy"), dtype=np.float64)
    Twc = np.asarray(np.load(root/"T_world_camera.npy"), dtype=np.float64)
    mask = np.asarray(np.load(a.mask), dtype=bool)
    pts_c = backproject_selected(depth, K, mask)
    pts_w = (Twc @ pts_c.T).T[:, :3]
    center = np.median(pts_w, axis=0)

    manifest = json.loads(a.settled_manifest.read_text(encoding="utf-8"))
    rows = []
    for record in manifest["objects"]:
        T = np.asarray(record["settled_pose_layout_world"], dtype=np.float64)
        d = float(np.linalg.norm(T[:3,3] - center))
        rows.append((d, record))
    rows.sort(key=lambda x: x[0])
    if not rows or rows[0][0] > a.max_origin_distance_m:
        raise RuntimeError(
            f"Vision mask could not be bound to a simulator object; nearest origin distance="
            f"{rows[0][0] if rows else None}"
        )
    best_d, best = rows[0]
    second = rows[1][0] if len(rows) > 1 else None
    out = {
        "schema_version": 1,
        "status": "PASS",
        "semantic_source": "GroundingDINO+SAM mask only",
        "sim_binding_method": "nearest settled rigid origin to median masked RGB-D point",
        "mask_centroid_world_m": center.tolist(),
        "segmentation_id": int(best["segmentation_id"]),
        "object_pool_index": int(best["object_pool_index"]),
        "object_code": str(best["object_code"]),
        "nearest_origin_distance_m": best_d,
        "second_nearest_origin_distance_m": second,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
