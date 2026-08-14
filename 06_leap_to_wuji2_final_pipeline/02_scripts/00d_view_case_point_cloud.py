#!/usr/bin/env python3
"""Export/show one frozen single-view point cloud with per-object colors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from case_paths import active_case_root


COLORS = np.asarray(
    [
        [230, 126, 34, 255], [52, 152, 219, 255], [46, 204, 113, 255],
        [155, 89, 182, 255], [241, 196, 15, 255], [231, 76, 60, 255],
        [26, 188, 156, 255], [127, 140, 141, 255],
    ],
    dtype=np.uint8,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    case_root = active_case_root()
    case = json.loads((case_root / "case.json").read_text(encoding="utf-8"))
    source = case_root / "01_input" / f"{case['view_id']}_network_input.npz"
    output = case_root / "01_input" / "single_view_point_cloud.glb"
    with np.load(source, allow_pickle=False) as archive:
        pc_camera = np.asarray(archive["pc"][0], dtype=np.float64)
        seg = np.asarray(archive["seg"][0], dtype=np.int64)
        transform = np.asarray(archive["extrinsics"][0], dtype=np.float64)
    pc_world = pc_camera @ transform[:3, :3].T + transform[:3, 3]
    rgba = np.full((len(pc_world), 4), [145, 145, 145, 255], dtype=np.uint8)
    object_ids = [int(value) for value in np.unique(seg) if int(value) != 0]
    for index, seg_id in enumerate(object_ids):
        rgba[seg == seg_id] = COLORS[index % len(COLORS)]
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.points.PointCloud(pc_world, colors=rgba),
        geom_name="single_view_40000_points",
    )
    output.write_bytes(scene.export(file_type="glb"))
    print(f"[PASS] points={len(pc_world)}; visible objects={object_ids}")
    print(f"[OK] {output}")
    if args.show:
        scene.show()


if __name__ == "__main__":
    main()
