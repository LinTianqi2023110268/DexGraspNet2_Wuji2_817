#!/usr/bin/env python3
"""Generate the cuRobo collision-sphere model for the project robot.

This is a one-time conversion step.  It reads the existing combined URDF and
mesh assets, fits cuRobo collision spheres, computes the self-collision ignore
matrix, and writes a new YAML under ``core/generated``.  It never modifies the
vendor Wuji2/dual-arm repositories.

The combined URDF uses ROS-style package names that do not exactly match this
repository's folder layout.  A small stable symlink-only asset adapter is
created under ``core/generated/asset_aliases`` so cuRobo's official
RobotBuilder can resolve the package paths without editing the URDF.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def _package_names(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    names: set[str] = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        m = re.match(r"package://([^/]+)/", filename)
        if m:
            names.add(m.group(1))
    return names


def _ensure_symlink(link: Path, target: Path) -> None:
    if not target.exists():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    desired = os.path.relpath(target, start=link.parent)
    if link.is_symlink():
        if os.readlink(link) == desired:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"asset alias path exists and is not a symlink: {link}")
    link.symlink_to(desired, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home()/"Projects/DexGraspNet2_Wuji2")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sphere-density", type=float, default=1.0)
    parser.add_argument("--collision-samples", type=int, default=1000)
    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing generated YAML")
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    description_root = root / "01_environment/vendor/wuji-description"
    urdf = description_root / "dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
    generated_root = root / "08_dual_arm_scene_layout/isaaclab_control/core/generated"
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else generated_root / "dual_arm_right_wuji2_curobo.yml"
    )
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force only after review")

    package_targets = {
        "dual_arm": description_root / "dual_arm",
        # URDF package://wuji_hand2_description/meshes/... corresponds to
        # the body package stored under hand2/hand2_beta1/body in this repo.
        "wuji_hand2_description": description_root / "hand2/hand2_beta1/body",
    }
    packages = _package_names(urdf)
    unknown = sorted(packages - set(package_targets))
    if unknown:
        raise RuntimeError(
            f"URDF contains unmapped package:// prefixes {unknown}; add an explicit adapter mapping"
        )
    alias_root = generated_root / "asset_aliases"
    for package in sorted(packages):
        _ensure_symlink(alias_root / package, package_targets[package])
    print(f"[ASSETS] package aliases: {sorted(packages)} -> {alias_root}")

    import torch
    from curobo.robot_builder import RobotBuilder
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; run this inside curobo_v2")

    builder = RobotBuilder(
        urdf_path=str(urdf),
        asset_path=str(alias_root),
        tool_frames=["arm_r_link_tf", "r_wrist"],
        device_cfg=DeviceCfg(device="cuda:0", dtype=torch.float32),
    )
    print("[1/3] fitting collision spheres ...", flush=True)
    builder.fit_collision_spheres(
        sphere_density=float(args.sphere_density),
        use_collision_mesh=True,
        compute_metrics=bool(args.compute_metrics),
    )
    print("[2/3] computing self-collision ignore matrix ...", flush=True)
    builder.compute_collision_matrix(
        prune_collisions=True,
        num_samples=int(args.collision_samples),
    )
    print("[3/3] writing generated cuRobo robot YAML ...", flush=True)
    config = builder.build()
    output.parent.mkdir(parents=True, exist_ok=True)
    builder.save(config, str(output))
    print(f"[PASS] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
