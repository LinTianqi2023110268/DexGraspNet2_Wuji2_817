#!/usr/bin/env python3
"""Read-only audit of a Wuji2 USD articulation.

Run this script with the USD Python libraries shipped by Isaac Sim.  It does
not launch Kit, PhysX, a GUI, or a GPU process; it only opens the composed USD
stage and emits stable JSON that can be compared between assets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


INTERESTING_PREFIXES = (
    "physics:",
    "physx",
    "drive:",
    "material:",
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Sdf.AssetPath):
        return value.path
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return str(value)


def _authored_attributes(prim: Usd.Prim) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith(INTERESTING_PREFIXES):
            continue
        if not attr.HasAuthoredValueOpinion():
            continue
        result[name] = _json_value(attr.Get())
    return dict(sorted(result.items()))


def _relationship_targets(prim: Usd.Prim) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for rel in prim.GetRelationships():
        name = rel.GetName()
        if not (name.startswith(INTERESTING_PREFIXES) or name in {"body0", "body1"}):
            continue
        targets = [str(path) for path in rel.GetTargets()]
        if targets:
            result[name] = targets
    return dict(sorted(result.items()))


def audit(path: Path) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")

    type_counts: Counter[str] = Counter()
    articulation_roots: list[str] = []
    rigid_bodies: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    collision_prims: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []

    for prim in stage.Traverse():
        type_counts[prim.GetTypeName() or "(typeless)"] += 1
        path_text = str(prim.GetPath())
        attrs = _authored_attributes(prim)
        rels = _relationship_targets(prim)

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path_text)

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(
                {
                    "path": path_text,
                    "type": prim.GetTypeName(),
                    "attributes": attrs,
                }
            )

        if prim.IsA(UsdPhysics.Joint):
            joints.append(
                {
                    "path": path_text,
                    "name": prim.GetName(),
                    "type": prim.GetTypeName(),
                    "attributes": attrs,
                    "relationships": rels,
                    "applied_schemas": list(prim.GetAppliedSchemas()),
                }
            )

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(
                {
                    "path": path_text,
                    "type": prim.GetTypeName(),
                    "attributes": attrs,
                    "relationships": rels,
                }
            )

        if prim.IsA(UsdShade.Material) or prim.HasAPI(UsdPhysics.MaterialAPI):
            materials.append(
                {
                    "path": path_text,
                    "attributes": attrs,
                    "relationships": rels,
                }
            )

    root_layer = stage.GetRootLayer()
    default_prim = stage.GetDefaultPrim()
    return {
        "asset": str(path.resolve()),
        "root_layer": root_layer.realPath,
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "frames_per_second": stage.GetFramesPerSecond(),
        "time_codes_per_second": stage.GetTimeCodesPerSecond(),
        "type_counts": dict(sorted(type_counts.items())),
        "articulation_roots": articulation_roots,
        "rigid_bodies": rigid_bodies,
        "joints": joints,
        "collision_prims": collision_prims,
        "materials": materials,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(args.usd)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered)


if __name__ == "__main__":
    main()
