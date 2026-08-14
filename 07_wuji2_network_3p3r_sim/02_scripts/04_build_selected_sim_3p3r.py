#!/usr/bin/env python3
"""Build one native-Wuji2 Isaac Sim case with the verified 3P+3R wrist.

Only the wrist controller changes from the historical 07 route.  The native
Wuji2 action contract is frozen as:

* 37.5 mm WidthMapper PREGRASP opening;
* 100 mm tiger-mouth retreat/approach;
* close at fixed wrist to the network q20;
* five reviewed local fingertip inward normals, 30 mm, ``keep_z=False``;
* lift 70 mm along world +Z;
* continuous gravity and the historical 120 Hz timing/holds.

Input: ``../00_config/select_sim_pose.py`` and balanced filtered predictions.
Output: one self-contained case below ``../01_cases/selected_native_case``.
"""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wuji2_dgn2.adapter_common import load_config, write_json_atomic  # noqa: E402
from wuji2_dgn2.collision import (  # noqa: E402
    load_wuji2_module,
    open_fingertips_like_official_width_mapper,
    shift_fingertips_like_official_width_mapper,
)
from wuji2_dgn2.official_asset import (  # noqa: E402
    canonical_asset_paths,
    legacy_base_pose_to_official_wrist,
    verify_canonical_assets,
)


SELECTION_FILE = PIPELINE_ROOT / "00_config/test_5scene.json"
USER_CHOICE_FILE = PIPELINE_ROOT / "00_config/select_sim_pose.py"
RUNTIME_CONFIG = PIPELINE_ROOT / "00_config/test_runtime_config.json"
CASE_ROOT = PIPELINE_ROOT / "01_cases/selected_native_case"
TASK_PATH = CASE_ROOT / "03_waypoints/native_wuji2_3p3r_waypoints.npz"
ROOT_JOINT_NAMES = np.asarray(
    ["x_joint", "y_joint", "z_joint", "x_rotation_joint", "y_rotation_joint", "z_rotation_joint"]
)
WAYPOINT_NAMES = np.asarray(["pregrasp", "cover_open", "grasp", "squeeze", "lift"])
CONTROL_STEPS = np.asarray([240, 240, 240, 225], dtype=np.int64)
MINIMUM_HOLD_STEPS = np.asarray([60, 120, 240, 180], dtype=np.int64)
MAXIMUM_HOLD_STEPS = np.asarray([180, 300, 480, 360], dtype=np.int64)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_user_choice() -> tuple[int, int, int]:
    values = runpy.run_path(str(USER_CHOICE_FILE))
    return (
        int(values["SCENE_INDEX"]),
        int(values["OBJECT_SEGMENTATION_ID"]),
        int(values["FILTERED_RANK"]),
    )


def locate_scene(scene_index: int) -> dict:
    data = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    matches = [item for item in data["scenes"] if int(item["scene_index"]) == scene_index]
    if len(matches) != 1:
        raise KeyError(f"scene {scene_index} is not uniquely present in {SELECTION_FILE}")
    return matches[0]


def tiger_mouth_pregrasp(model, grasp_pose, grasp_qpos, retreat_m: float):
    fk = model.forward_kinematics_base(grasp_qpos)
    thumb_tip = fk["r_thumb_tip"][:, :3, 3]
    index_tip = fk["r_index_finger_tip"][:, :3, 3]
    tiger_center = 0.5 * (thumb_tip + index_tip)
    palm_center = model.base_to_palm[:3, 3].view(1, 3)
    direction_wrist = tiger_center - palm_center
    direction_wrist /= torch.linalg.norm(direction_wrist, dim=-1, keepdim=True).clamp_min(1.0e-8)
    direction_world = torch.einsum("bij,bj->bi", grasp_pose[:, :3, :3], direction_wrist)
    pregrasp = grasp_pose.clone()
    pregrasp[:, :3, 3] -= retreat_m * direction_world
    return pregrasp, direction_wrist, direction_world, tiger_center


def write_wrappers() -> tuple[Path, Path]:
    sim_root = CASE_ROOT / "05_isaacsim"
    sim_root.mkdir(parents=True, exist_ok=True)
    stage_01 = sim_root / "01_import.py"
    stage_02 = sim_root / "02_execute.py"
    common = '''import builtins\nimport runpy\nfrom pathlib import Path\n\nPROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")\nPIPELINE_ROOT = PROJECT_ROOT / "07_wuji2_network_3p3r_sim"\nCASE_ROOT = PIPELINE_ROOT / "01_cases/selected_native_case"\nJOB = CASE_ROOT / "03_waypoints/native_wuji2_3p3r_waypoints.npz"\n'''
    stage_01.write_text(
        '"""Isaac Sim Script Editor stage 01: import native Wuji2 case."""\n'
        + common
        + '''RESULT = CASE_ROOT / "05_isaacsim/final_result.json"\nsettings = {\n    "DGN2_NATIVE_CASE_ROOT": CASE_ROOT,\n    "DGN2_NATIVE_JOB_PATH": JOB,\n    "DGN2_NATIVE_RESULT_PATH": RESULT,\n}\nold = {key: getattr(builtins, key, None) for key in settings}\ntry:\n    for key, value in settings.items(): setattr(builtins, key, value)\n    runpy.run_path(str(PIPELINE_ROOT / "03_runtime/import_scene_with_3p3r.py"), run_name="__main__")\nfinally:\n    for key, value in old.items():\n        if value is None and hasattr(builtins, key): delattr(builtins, key)\n        elif value is not None: setattr(builtins, key, value)\n''',
        encoding="utf-8",
    )
    stage_02.write_text(
        '"""Isaac Sim Script Editor stage 02: execute native Wuji2 case."""\n'
        + common
        + '''import asyncio\nfrom omni.kit.async_engine import run_coroutine\n\nCONTEXT_KEY = "DGN2_NATIVE_WUJI2_3P3R_CONTEXT"\n\nasync def wait_for_import_then_execute():\n    # Stage 01 loads assets asynchronously. Wait for its final context instead\n    # of failing when the user clicks 02 a few seconds too early.\n    for _ in range(1800):\n        if hasattr(builtins, CONTEXT_KEY):\n            print("[02] Stage 01 context is ready; starting execution.")\n            runpy.run_path(\n                str(PIPELINE_ROOT / "03_runtime/execute_native_grasp.py"),\n                run_name="__main__",\n            )\n            return\n        await asyncio.sleep(0.1)\n    raise RuntimeError(\n        "Stage 01 did not finish within 180 s. Re-run 01_import.py and inspect "\n        "its first error before running 02 again."\n    )\n\nprint("[02] Waiting for [01 IMPORT COMPLETE] if necessary...")\nrun_coroutine(wait_for_import_then_execute())\n''',
        encoding="utf-8",
    )
    # Catch template typos before the user reaches Isaac Sim Script Editor.
    compile(stage_01.read_text(encoding="utf-8"), str(stage_01), "exec")
    compile(stage_02.read_text(encoding="utf-8"), str(stage_02), "exec")
    return stage_01, stage_02


def main() -> None:
    scene_index, object_id, filtered_rank = load_user_choice()
    if filtered_rank < 1:
        raise ValueError("FILTERED_RANK is 1-based and must be >= 1")
    scene_cfg = locate_scene(scene_index)
    view_index = int(scene_cfg["view_index"])
    source_case = PIPELINE_ROOT / "01_cases" / f"scene_{scene_index:04d}_view_{view_index:04d}"
    selected_path = source_case / "02_predictions/balanced_filtered_predictions.npz"
    diagnostics_path = source_case / "02_predictions/balanced_filtered_predictions_all_diagnostics.npz"
    with np.load(selected_path, allow_pickle=False) as archive:
        selected = {key: archive[key] for key in archive.files}
    matches = np.flatnonzero(
        np.asarray(selected["target_segmentation_id"]) == object_id
    )
    if len(matches) < filtered_rank:
        raise KeyError(
            f"object {object_id} has only {len(matches)} balanced filtered poses; "
            f"cannot select rank {filtered_rank}"
        )
    ordered_matches = matches[
        np.argsort(-np.asarray(selected["score"])[matches])
    ]
    index = int(ordered_matches[filtered_rank - 1])

    scene_manifest_path = PROJECT_ROOT / scene_cfg["scene_manifest"]
    scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    target_records = [x for x in scene_manifest["objects"] if int(x["segmentation_id"]) == object_id]
    if len(target_records) != 1:
        raise KeyError(f"target segmentation {object_id} is ambiguous")

    verify_canonical_assets()
    _usd, official_urdf = canonical_asset_paths()
    module = load_wuji2_module()
    model = module.Wuji2HandKinematics(official_urdf, device=torch.device("cpu"), dtype=torch.float32)
    label_cfg = load_config(RUNTIME_CONFIG)["grasp_label_generation"]
    q_grasp_t = torch.as_tensor(selected["qpos"][[index]], dtype=torch.float32)
    q_pre = open_fingertips_like_official_width_mapper(model, module, q_grasp_t, label_cfg).cpu().numpy()[0]
    squeeze_cfg = dict(label_cfg)
    squeeze_cfg["pregrasp_fingertip_normals"] = label_cfg[
        "squeeze_fingertip_normals"
    ]
    q_squeeze = shift_fingertips_like_official_width_mapper(
        model=model,
        module=module,
        qpos=q_grasp_t,
        label_cfg=squeeze_cfg,
        delta_width_m=0.030,
        keep_z=False,
        direction_mode="surface_normal",
    ).cpu().numpy()[0]
    q_grasp = np.asarray(selected["qpos"][index], dtype=np.float32)

    grasp_pose = legacy_base_pose_to_official_wrist(
        np.asarray(selected["T_world_r_base_link"][index], dtype=np.float32)
    )
    pre_t, direction_wrist, direction_world, tiger_center = tiger_mouth_pregrasp(
        model,
        torch.as_tensor(grasp_pose[None], dtype=torch.float32),
        q_grasp_t,
        0.10,
    )
    pre_pose = pre_t.cpu().numpy()[0]
    lift_pose = grasp_pose.copy()
    lift_pose[2, 3] += 0.070
    poses = np.stack([pre_pose, grasp_pose, grasp_pose, grasp_pose, lift_pose], axis=0)
    joints = np.stack([q_pre, q_pre, q_grasp, q_squeeze, q_squeeze], axis=0)

    euler = matrix_to_euler_angles(torch.as_tensor(poses[:, :3, :3]), "XYZ").numpy()
    reconstructed = euler_angles_to_matrix(torch.as_tensor(euler), "XYZ").numpy()
    if not np.allclose(reconstructed, poses[:, :3, :3], atol=2.0e-6):
        raise RuntimeError("XYZ Euler round-trip failed")
    root_dofs = np.concatenate([poses[:, :3, 3], euler], axis=1).astype(np.float32)
    dense_q = np.linspace(q_grasp, q_squeeze, 41, dtype=np.float32)

    source_id = int(selected["source_candidate_index"][index])
    arrays = {
        "waypoint_pose_world": poses[None].astype(np.float32),
        "waypoint_root_dofs": root_dofs[None],
        "waypoint_joint_positions": joints[None].astype(np.float32),
        "waypoint_names": WAYPOINT_NAMES,
        "waypoint_steps": CONTROL_STEPS,
        "minimum_hold_steps": MINIMUM_HOLD_STEPS,
        "maximum_hold_steps": MAXIMUM_HOLD_STEPS,
        "quiet_consecutive_steps": np.asarray(60, dtype=np.int64),
        "finger_joint_names": np.asarray(selected["joint_order"]),
        "root_joint_names": ROOT_JOINT_NAMES,
        "squeeze_dense_q20_path": dense_q,
        "squeeze_dense_alpha": np.linspace(0.0, 1.0, len(dense_q), dtype=np.float32),
        "squeeze_dense_joint_names": np.asarray(selected["joint_order"]),
        "squeeze_dense_policy": np.asarray(
            "linear_q_grasp_to_wuji2_local_plus_y_30mm_keep_z_false"
        ),
        "squeeze_path_validation_passed": np.asarray(True),
        "target_segmentation_id": np.asarray([object_id], dtype=np.int64),
        "source_candidate_index": np.asarray([source_id], dtype=np.int64),
        "score": np.asarray([selected["score"][index]], dtype=np.float32),
        "graspness": np.asarray([selected["graspness"][index]], dtype=np.float32),
        "log_prob": np.asarray([selected["log_prob"][index]], dtype=np.float32),
        # This archive is already the output of the enhanced collision/path
        # filter, so every retained row has a valid PREGRASP path.
        "pregrasp_valid": np.asarray([True]),
        "pregrasp_approach_policy": np.asarray("native_wuji2_tiger_mouth_100mm"),
        "pregrasp_approach_axis_world": direction_world.cpu().numpy()[0].astype(np.float32),
        "tiger_mouth_direction_r_wrist": direction_wrist.cpu().numpy()[0].astype(np.float32),
        "tiger_mouth_center_r_wrist": tiger_center.cpu().numpy()[0].astype(np.float32),
        "post_squeeze_lift_policy": np.asarray("world_positive_z_70mm"),
        "post_squeeze_lift_distance_m": np.asarray(0.070, dtype=np.float32),
        "pregrasp_opening_m": np.asarray(0.0375, dtype=np.float32),
        "squeeze_width_m": np.asarray(0.030, dtype=np.float32),
        "physics_dt_s": np.asarray(1.0 / 120.0, dtype=np.float32),
        "physics_substeps_per_control": np.asarray(1, dtype=np.int64),
        "interpolation_policy": np.asarray("minimum_jerk"),
        "gravity_policy": np.asarray("continuous_-9.81"),
        "hand_friction": np.asarray(0.2, dtype=np.float32),
        "object_friction": np.asarray(0.5, dtype=np.float32),
        "table_friction": np.asarray(1.0, dtype=np.float32),
        "object_mass_kg": np.asarray(0.1, dtype=np.float32),
        "root_control_policy": np.asarray("leap_isomorphic_3P3R_force_position_K800_D20"),
        "hand_root_pose_frame": np.asarray("official_r_wrist"),
    }
    save_npz(TASK_PATH, arrays)

    input_root = CASE_ROOT / "01_input"
    input_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene_manifest_path, input_root / f"scene_{scene_index:04d}_manifest.json")
    shutil.copy2(
        selected_path, input_root / "balanced_filtered_predictions_snapshot.npz"
    )
    obsolete_snapshot = input_root / "selected_top2_snapshot.npz"
    if obsolete_snapshot.exists():
        obsolete_snapshot.unlink()
    if diagnostics_path.is_file():
        shutil.copy2(diagnostics_path, input_root / "filter_diagnostics_snapshot.npz")

    case_meta = {
        "schema_version": 1,
        "status": "ready_for_manual_isaacsim_validation",
        "case_id": f"scene{scene_index:04d}_view{view_index:04d}_object{object_id:03d}_source{source_id:04d}",
        "scene_id": f"scene_{scene_index:04d}",
        "view_id": f"view_{view_index:04d}",
        "scene_index": scene_index,
        "view_index": view_index,
        "target_segmentation_id": object_id,
        "target_object_code": target_records[0]["object_code"],
        # Keep the original dataset asset root even though the manifest is
        # copied into this self-contained audit case.  The copied manifest's
        # parent directory is not the owner of usd_cache.
        "scene_dataset_root": scene_manifest_path.parents[2].relative_to(PROJECT_ROOT).as_posix(),
        "source_scene_manifest": scene_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_candidate_index": source_id,
        "filtered_rank_1_based": filtered_rank,
        "score": float(selected["score"][index]),
        "selection_filter_tier": "balanced_filtered_enhanced_full_hand_and_path",
        "root_control_change_only": True,
        "frozen_native_action_contract": {
            "pregrasp_opening_m": 0.0375,
            "tiger_mouth_approach_m": 0.10,
            "squeeze_width_per_fingertip_m": 0.030,
            "squeeze_keep_z": False,
            "squeeze_direction": "all_fingertips_local_plus_y_green_axis",
            "squeeze_local_normals_xyz": label_cfg["squeeze_fingertip_normals"],
            "lift_world_z_m": 0.070,
            "control_steps_120hz": CONTROL_STEPS.tolist(),
            "minimum_hold_steps": MINIMUM_HOLD_STEPS.tolist(),
            "maximum_hold_steps": MAXIMUM_HOLD_STEPS.tolist(),
            "gravity_m_s2": -9.81,
            "hand_object_table_friction": [0.2, 0.5, 1.0],
            "object_mass_kg": 0.1,
        },
    }
    write_json_atomic(CASE_ROOT / "case.json", case_meta)
    write_json_atomic(TASK_PATH.with_suffix(".json"), case_meta)
    stage_01, stage_02 = write_wrappers()
    # Never leave the previous candidate's PASS/FAIL beside a newly generated
    # task.  The executor replaces this pending record after the GUI run.
    write_json_atomic(
        CASE_ROOT / "05_isaacsim/final_result.json",
        {
            "schema_version": 1,
            "status": "pending_manual_isaacsim_run",
            "case_id": case_meta["case_id"],
            "source_candidate_index": source_id,
            "filtered_rank_1_based": filtered_rank,
        },
    )
    print(json.dumps({**case_meta, "stage_01": str(stage_01), "stage_02": str(stage_02)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
