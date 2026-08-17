#!/usr/bin/env python3
"""Build PREGRASP..RETREAT Cartesian targets with multi-object placement allocation.

This is intentionally a GEOMETRY builder only.  It performs no CPU/Pinocchio IK.
It writes the legacy-compatible filename ``full_arm_waypoint_ik.npz`` because
the current Route-C V2 runtime only consumes waypoint names + Cartesian flange
targets from that file; all q7 values are solved later by cuRobo.

The placement plan uses the existing footprint-aware ``placement_allocator`` and
therefore automatically avoids slots committed by previous successful cycles.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_SCRIPTS = PROJECT_ROOT/"08_dual_arm_scene_layout/isaaclab_control/runtime/scripts"
sys.path.insert(0, str(RUNTIME_SCRIPTS))
from placement_allocator import allocate_placement, load_json

LAYOUT_JSON = PROJECT_ROOT/"08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
PLACEMENT_POLICY = PROJECT_ROOT/"08_dual_arm_scene_layout/isaaclab_control/runtime/config/placement_policy.json"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-root", type=Path, required=True)
    p.add_argument("--placement-policy", type=Path, default=PLACEMENT_POLICY)
    p.add_argument("--placement-slot-index", type=int)
    a = p.parse_args()
    case_root = a.case_root.resolve()
    arm_targets = case_root/"07_arm_execution/arm_flange_targets.npz"
    if not arm_targets.is_file():
        raise FileNotFoundError(arm_targets)
    with np.load(arm_targets, allow_pickle=False) as z:
        names = np.asarray(z["waypoint_names"])
        flange = np.asarray(z["world_from_right_flange"], dtype=np.float64)
        wrist = np.asarray(z["world_from_wuji2_wrist"], dtype=np.float64)
        world_from_source = np.asarray(z["world_from_source_zone"], dtype=np.float64)

    if [str(x) for x in names.tolist()] != ["pregrasp","cover","grasp","squeeze","lift"]:
        raise RuntimeError(f"unexpected pick stages: {names.tolist()}")
    layout = load_json(LAYOUT_JSON)
    policy = load_json(a.placement_policy)
    case = load_json(case_root/"case.json")
    target_id = int(case["target_segmentation_id"])
    manifests = sorted((case_root/"01_input").glob("scene_*_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one scene manifest, got {manifests}")
    scene = load_json(manifests[0])
    target = next(r for r in scene["objects"] if int(r["segmentation_id"]) == target_id)
    world_from_object_initial = world_from_source @ np.asarray(target["pose_world_object"], dtype=np.float64)
    grasp_i = 2
    flange_from_object = np.linalg.inv(flange[grasp_i]) @ world_from_object_initial
    surface_path = case_root/"01_input"/f"object_{target_id:03d}_surface_points.npy"
    if not surface_path.is_file():
        raise FileNotFoundError(surface_path)
    placement = allocate_placement(
        project_root=PROJECT_ROOT,
        layout=layout,
        policy=policy,
        surface_points_object=np.load(surface_path),
        world_from_object_initial=world_from_object_initial,
        requested_slot_index=a.placement_slot_index,
    )

    object_place = world_from_object_initial.copy()
    object_place[:3,3] = np.asarray(placement["object_root_place_world_m"], dtype=np.float64)
    object_transfer = object_place.copy()
    object_transfer[2,3] += float(placement["transfer_clearance_m"])

    grasp_flange = flange[grasp_i].copy()
    grasp_rot = grasp_flange[:3,:3].copy()
    flange_from_wrist = np.linalg.inv(grasp_flange) @ wrist[grasp_i]
    object_offset = grasp_rot @ flange_from_object[:3,3]
    wrist_offset = grasp_rot @ flange_from_wrist[:3,3]
    release_wrist_z = float(wrist[grasp_i][2,3] + placement["release_hand_height_above_grasp_m"])

    transfer = object_transfer @ np.linalg.inv(flange_from_object)
    place = grasp_flange.copy()
    place[0,3] = object_place[0,3] - object_offset[0]
    place[1,3] = object_place[1,3] - object_offset[1]
    place[2,3] = release_wrist_z - wrist_offset[2]
    release = place.copy()
    retreat = release.copy()
    retreat[2,3] += float(placement["retreat_clearance_m"])

    full_names = np.concatenate([names, np.asarray(["transfer","place","release","retreat"])])
    full_flange = np.concatenate([flange, transfer[None], place[None], release[None], retreat[None]], axis=0)
    output_npz = case_root/"07_arm_execution/full_arm_waypoint_ik.npz"
    np.savez_compressed(
        output_npz,
        waypoint_names=full_names,
        world_from_right_flange=full_flange.astype(np.float64),
        producer=np.asarray("closed_loop_cartesian_route_no_cpu_ik"),
    )
    placement["hand_height_frame"] = "Wuji2 r_wrist origin"
    placement["grasp_hand_world_z_m"] = float(wrist[grasp_i][2,3])
    placement["release_hand_world_z_m"] = release_wrist_z
    placement["induced_object_root_release_world_m"] = (place @ flange_from_object)[:3,3].tolist()
    report = {
        "schema_version": 2,
        "status": "CARTESIAN_ROUTE_READY_FOR_CUROBO",
        "ik_solver": None,
        "cpu_ik_used": False,
        "placement_policy": str(a.placement_policy.resolve()),
        "placement_plan": placement,
        "waypoint_names": [str(x) for x in full_names.tolist()],
        "output_npz": str(output_npz),
    }
    report_path = case_root/"07_arm_execution/full_arm_waypoint_ik_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"PASS","output":str(output_npz),"placement":placement}, ensure_ascii=False))

if __name__ == "__main__":
    main()
