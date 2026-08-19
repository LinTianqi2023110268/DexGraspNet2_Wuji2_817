#!/usr/bin/env python3
"""Build the two-stage Isaac Sim job for the fixed-root SQUEEZE retry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from case_paths import PROJECT_ROOT, SHARED_ROOT, active_case_root  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from src.wuji2_dgn2.collision import (  # noqa: E402
    load_wuji2_module,
    shift_fingertips_like_official_width_mapper,
)
sys.path.insert(0, str(SHARED_ROOT / "lib"))
from pinky_ring_coupling import (  # noqa: E402
    apply_pinky_ring_coupling,
    load_pinky_policy,
)

CASE_ROOT = active_case_root()
SOURCE = CASE_ROOT / "04_squeeze/squeeze_official.npz"
VALIDATION = CASE_ROOT / "04_squeeze/squeeze_official_report.json"
LEAP_JOB = CASE_ROOT / "01_input/leap_official_waypoints.npz"
BASE_JOB = (
    SHARED_ROOT / "isaacsim/runtime_contract_template.npz"
)
SIM_ROOT = CASE_ROOT / "06_isaacsim"
OUTPUT = SIM_ROOT / "final_waypoints.npz"
NATIVE_DIRECTION_CONFIG = (
    SHARED_ROOT / "config/wuji2_native_width_mapper.json"
)
WUJI_URDF = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/hand2/hand2_beta1/body/urdf/right.urdf"
)
PINKY_POLICY = SHARED_ROOT / "config/pinky_ring_coupling.json"
LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"

# 只影响PREGRASP和COVER的大拇指，不改变GRASP、SQUEEZE或LIFT。
# 正值表示拇指指尖沿已确认内收法向的反方向额外张开多少米。
PREGRASP_THUMB_EXTRA_OPEN_M = 0.03


def write_case_wrappers(case_id: str) -> None:
    """Write the single verified LEAP-style-root Script Editor entry pair."""

    def write_pair(output_dir: Path, branch: str, result_name: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        relative_job = Path("..") / "final_waypoints.npz" if output_dir != SIM_ROOT else Path("final_waypoints.npz")
        common_header = f'''import builtins
import runpy
from pathlib import Path

PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
PIPELINE_ROOT = PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline"
CASE_ROOT = Path("{CASE_ROOT.resolve().as_posix()}")
ROOT = CASE_ROOT / "06_isaacsim/{output_dir.relative_to(SIM_ROOT).as_posix()}"
JOB = (ROOT / "{relative_job.as_posix()}").resolve()
RESULT = ROOT / "{result_name}"
'''
        common_body = '''
keys = ("DGN2_AB_BRANCH", "DGN2_AB_JOB_PATH", "DGN2_AB_RESULT_PATH", "DGN2_CASE_ROOT")
old = {key: getattr(builtins, key, None) for key in keys}
try:
    setattr(builtins, "DGN2_AB_BRANCH", "__BRANCH__")
    setattr(builtins, "DGN2_AB_JOB_PATH", JOB)
    setattr(builtins, "DGN2_AB_RESULT_PATH", RESULT)
    setattr(builtins, "DGN2_CASE_ROOT", CASE_ROOT)
    runpy.run_path(str(COMMON), run_name="__main__")
finally:
    for key in keys:
        if old[key] is None:
            if hasattr(builtins, key):
                delattr(builtins, key)
        else:
            setattr(builtins, key, old[key])
'''.replace("__BRANCH__", branch)
        import_text = (
            '"""Stage 01: import this case and pause at Wuji2 PREGRASP."""\n\n'
            + common_header
            + 'COMMON = PIPELINE_ROOT / "00_shared/isaacsim/common_import.py"\n'
            + 'for path in (COMMON, JOB, JOB.with_suffix(".json")):\n'
            + '    if not path.is_file():\n        raise FileNotFoundError(path)\n'
            + common_body
        )
        execute_text = (
            '"""Stage 02: execute COVER, GRASP, dense SQUEEZE and LIFT."""\n\n'
            + common_header
            + 'COMMON = PIPELINE_ROOT / "00_shared/isaacsim/common_execute.py"\n'
            + 'for path in (COMMON, JOB):\n'
            + '    if not path.is_file():\n        raise FileNotFoundError(path)\n'
            + common_body
        )
        (output_dir / "01_import.py").write_text(import_text, encoding="utf-8")
        (output_dir / "02_execute.py").write_text(execute_text, encoding="utf-8")

    write_pair(SIM_ROOT, "wuji2_leap_root_drive", "final_result.json")
    (SIM_ROOT / "README.md").write_text(
        "# Isaac Sim verified execution entry\n\n"
        "This directory uses the only currently verified Wuji2 execution "
        "method: official Wuji2 USD, LEAP-equivalent 3-prismatic + "
        "3-revolute force-position root drive (K=800, D=20), and native "
        "thumb PREGRASP IK along the direction opposite local +Y.\n\n"
        "In Isaac Sim 5.0 Script Editor, run `01_import.py`, wait for "
        "`[01 IMPORT COMPLETE]`, then run `02_execute.py`.\n",
        encoding="utf-8",
    )


def scalar(data: dict[str, np.ndarray], key: str):
    return np.asarray(data[key]).item()


def source_zone_transform_from_layout() -> np.ndarray:
    """Return T_W_S without using SourceZone display scale.

    The SourceZone prim is a visual marker with authored scale.  That scale is
    not a rigid transform and must never be applied to retarget wrist/root
    poses.  The current calibrated layout records SourceZone as an unrotated
    rigid frame at ``position_world_m``.
    """
    layout = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))
    source = layout["transforms"]["source_zone"]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(source["position_world_m"], dtype=np.float64)
    return transform


def main() -> None:
    for path in (
        SOURCE,
        VALIDATION,
        LEAP_JOB,
        BASE_JOB,
        NATIVE_DIRECTION_CONFIG,
        WUJI_URDF,
        PINKY_POLICY,
        LAYOUT_JSON,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    native_direction = json.loads(
        NATIVE_DIRECTION_CONFIG.read_text(encoding="utf-8")
    )
    if not bool(native_direction.get("directions_approved", False)):
        raise RuntimeError("Wuji2 fingertip directions have not been approved")
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))
    immutable = report.get("immutable_contract") or {}
    if not all(
        immutable.get(key) is True
        for key in (
            "grasp_q20_unchanged",
            "root_6d_unchanged",
            "pinky_ring_coupled",
        )
    ):
        raise RuntimeError("SQUEEZE retry did not preserve the accepted contracts")
    with np.load(SOURCE, allow_pickle=False) as archive:
        retry = {key: archive[key] for key in archive.files}
    with np.load(BASE_JOB, allow_pickle=False) as archive:
        base = {key: archive[key] for key in archive.files}
    with np.load(LEAP_JOB, allow_pickle=False) as archive:
        leap = {key: archive[key] for key in archive.files}

    names = [str(value) for value in base["finger_joint_names"].tolist()]
    retry_names = [str(value) for value in retry["wuji2_joint_names"].tolist()]
    if names != retry_names:
        raise RuntimeError("q20 joint order differs from the audited Isaac Sim job")
    stages = [str(value) for value in base["waypoint_names"].tolist()]
    if stages != ["pregrasp", "cover", "grasp", "squeeze", "lift"]:
        raise RuntimeError(f"unexpected stage contract: {stages}")

    dense = np.asarray(retry["wuji2_q20_path"], dtype=np.float64)
    alpha = np.asarray(retry["path_alpha"], dtype=np.float64)
    limits = np.asarray(base["companion_joint_limits_rad"], dtype=np.float64)
    if dense.shape[1] != 20 or alpha.shape != (len(dense),):
        raise RuntimeError(f"invalid dense SQUEEZE path {dense.shape}, {alpha.shape}")
    if np.any(dense < limits[:, 0][None] - 1.0e-7) or np.any(
        dense > limits[:, 1][None] + 1.0e-7
    ):
        raise RuntimeError("dense SQUEEZE path violates official Wuji2 limits")
    max_step = float(np.max(np.abs(np.diff(dense, axis=0))))
    if max_step > 0.05:
        raise RuntimeError(f"dense SQUEEZE path is too discontinuous: {max_step:.6f} rad")

    pre_i, cover_i, grasp_i, squeeze_i, lift_i = range(5)
    stage_q = np.asarray(base["waypoint_joint_positions"], dtype=np.float64).copy()
    pre_cover_coupled, pre_cover_pinky_audit = apply_pinky_ring_coupling(
        stage_q[0, [pre_i, cover_i]],
        names,
        limits,
        load_pinky_policy(PINKY_POLICY),
    )
    stage_q[0, pre_i] = pre_cover_coupled[0]
    stage_q[0, cover_i] = pre_cover_coupled[1]
    stage_q[0, grasp_i] = dense[0]
    stage_q[0, squeeze_i] = dense[-1]
    stage_q[0, lift_i] = dense[-1]
    thumb_indices = [
        names.index(name)
        for name in (
            "r_thumb_cmc_flex",
            "r_thumb_cmc_abd",
            "r_thumb_mcp",
            "r_thumb_ip",
        )
    ]
    module = load_wuji2_module()
    if names != list(module.RIGHT_HAND_JOINT_ORDER):
        raise RuntimeError("Wuji2 q20 order differs from the native IK contract")
    model = module.Wuji2HandKinematics(
        WUJI_URDF, device=torch.device("cpu"), dtype=torch.float32
    )
    label_cfg = {
        "pregrasp_fingertip_normals": native_direction[
            "local_inward_normals_xyz"
        ],
        "pregrasp_width_mapper_steps": int(native_direction["ik_steps"]),
        "pregrasp_width_mapper_learning_rate": float(
            native_direction["ik_learning_rate"]
        ),
    }
    q_pre_before = torch.as_tensor(
        stage_q[0, pre_i][None], dtype=torch.float32
    )
    tip_order = list(native_direction["tip_links_in_solver_order"])
    if tip_order[0] != "r_thumb_tip" or len(tip_order) != 5:
        raise RuntimeError(f"unexpected fingertip IK order: {tip_order}")
    with torch.no_grad():
        fk_before = model.forward_kinematics_base(q_pre_before)
        thumb_before = fk_before["r_thumb_tip"][0, :3, 3]
        thumb_rotation = fk_before["r_thumb_tip"][0, :3, :3]
        inward_local = torch.as_tensor(
            native_direction["local_inward_normals_xyz"]["r_thumb_tip"],
            dtype=torch.float32,
        )
        inward_r_wrist = thumb_rotation @ inward_local
        inward_r_wrist /= torch.linalg.norm(inward_r_wrist)
        outward_r_wrist = -inward_r_wrist
    q_pre_after = shift_fingertips_like_official_width_mapper(
        model=model,
        module=module,
        qpos=q_pre_before,
        label_cfg=label_cfg,
        delta_width_m=[-PREGRASP_THUMB_EXTRA_OPEN_M, 0.0, 0.0, 0.0, 0.0],
        keep_z=bool(native_direction["keep_z"]),
        direction_mode="surface_normal",
    )
    with torch.no_grad():
        fk_after = model.forward_kinematics_base(q_pre_after)
        thumb_after = fk_after["r_thumb_tip"][0, :3, 3]
        thumb_displacement = thumb_after - thumb_before
        thumb_open_projected_m = float(
            torch.dot(thumb_displacement, outward_r_wrist)
        )
        thumb_open_actual_m = float(torch.linalg.norm(thumb_displacement))
        thumb_open_target_error_m = float(
            torch.linalg.norm(
                thumb_after
                - (thumb_before + PREGRASP_THUMB_EXTRA_OPEN_M * outward_r_wrist)
            )
        )
    if thumb_open_projected_m <= 0.0:
        raise RuntimeError("thumb PREGRASP IK moved opposite the approved open direction")
    q_pre_after_np = q_pre_after.detach().cpu().numpy()[0].astype(np.float64)
    nonthumb_indices = [i for i in range(len(names)) if i not in thumb_indices]
    if not np.allclose(
        q_pre_after_np[nonthumb_indices],
        stage_q[0, pre_i, nonthumb_indices],
        atol=1.0e-6,
    ):
        raise RuntimeError("thumb-only PREGRASP IK changed a non-thumb joint")
    thumb_joint_delta = (
        q_pre_after_np[thumb_indices] - stage_q[0, pre_i, thumb_indices]
    )
    stage_q[0, pre_i] = q_pre_after_np
    stage_q[0, cover_i] = q_pre_after_np
    if np.any(stage_q < limits[:, 0][None, None, :] - 1.0e-7) or np.any(
        stage_q > limits[:, 1][None, None, :] + 1.0e-7
    ):
        raise RuntimeError(
            "PREGRASP_THUMB_EXTRA_OPEN_M drives a waypoint outside "
            "the official Wuji2 joint limits"
        )

    grasp_pose = np.asarray(retry["fixed_wuji2_root_pose_world"], dtype=np.float64)
    wrist_from_palm = np.asarray(
        base["wuji2_semantic_palm_frame_in_r_wrist"], dtype=np.float64
    )
    world_from_source = source_zone_transform_from_layout()
    source_from_world = np.linalg.inv(world_from_source)
    approach_axis_source = grasp_pose[:3, :3] @ wrist_from_palm[:3, 2]
    approach_axis_source /= np.linalg.norm(approach_axis_source)
    approach_axis_world = world_from_source[:3, :3] @ approach_axis_source
    approach_axis_world /= np.linalg.norm(approach_axis_world)
    retreat_m = float(scalar(base, "official_dgn2_pregrasp_retreat_m"))
    lift_m = float(scalar(base, "post_squeeze_lift_distance_m"))
    gravity_direction = np.asarray([0.0, 0.0, -1.0])
    approach_dot_gravity = float(np.dot(approach_axis_world, gravity_direction))
    is_top = approach_dot_gravity > float(np.cos(np.pi / 3.0))

    stage_pose = np.repeat(grasp_pose[None, None], 5, axis=1)
    stage_pose[0, pre_i, :3, 3] -= retreat_m * approach_axis_source
    lift_axis = (
        -approach_axis_source
        if is_top
        else source_from_world[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    lift_axis /= np.linalg.norm(lift_axis)
    stage_pose[0, lift_i, :3, 3] += lift_m * lift_axis
    root_euler = matrix_to_euler_angles(
        torch.as_tensor(stage_pose[0, :, :3, :3]), "XYZ"
    ).numpy()
    reconstructed_rotation = euler_angles_to_matrix(
        torch.as_tensor(root_euler), "XYZ"
    ).numpy()
    if not np.allclose(
        reconstructed_rotation, stage_pose[0, :, :3, :3], atol=2.0e-6
    ):
        raise RuntimeError("Wuji2 root XYZ Euler conversion failed round-trip audit")
    root_dofs = np.concatenate(
        [stage_pose[0, :, :3, 3], root_euler], axis=1
    ).astype(np.float32)
    lift_policy = (
        "top_retreat_semantic_palm_negative_z_200mm"
        if is_top
        else "side_lift_world_positive_z_200mm"
    )

    output = {
        "waypoint_joint_positions": stage_q.astype(np.float32),
        "waypoint_pose_world": stage_pose.astype(np.float32),
        "waypoint_pose_frame": np.asarray("SourceZone"),
        "coordinate_convention": np.asarray("T_A_B maps coordinates from frame B into frame A"),
        "waypoint_root_dofs": root_dofs[None],
        "waypoint_names": np.asarray(stages),
        "waypoint_steps": np.asarray(base["waypoint_steps"], dtype=np.int64),
        "finger_joint_names": np.asarray(names),
        "root_joint_names": np.asarray(leap["root_joint_names"]),
        "companion_joint_limits_rad": np.asarray(
            base["companion_joint_limits_rad"], dtype=np.float32
        ),
        "source_leap_waypoint_pose_world": np.asarray(
            leap["waypoint_pose_world"], dtype=np.float32
        ),
        "source_leap_waypoint_pose_frame": np.asarray("SourceZone"),
        "source_leap_waypoint_joint_positions": np.asarray(
            leap["waypoint_joint_positions"], dtype=np.float32
        ),
        "pregrasp_valid": np.asarray(leap["pregrasp_valid"], dtype=bool),
        "scene_penetration": np.asarray(leap["scene_penetration"], dtype=np.float32),
        "table_penetration": np.asarray(leap["table_penetration"], dtype=np.float32),
        "source_candidate_index": np.asarray(
            leap["source_candidate_index"], dtype=np.int64
        ),
        "score": np.asarray(leap["score"], dtype=np.float32),
        "graspness": np.asarray(leap["graspness"], dtype=np.float32),
        "log_prob": np.asarray(leap["log_prob"], dtype=np.float32),
        "seed_point_world": np.asarray(leap["seed_point_world"], dtype=np.float32),
        "target_segmentation_id": np.asarray(
            leap["target_segmentation_id"], dtype=np.int64
        ),
        "hand_root_pose_frame": np.asarray("official_r_wrist_four_tip_kabsch"),
        "hand_root_transform_frame": np.asarray("SourceZone"),
        "four_real_tip_grasp_q20": dense[0].astype(np.float32),
        "four_real_tip_grasp_pose_world": grasp_pose.astype(np.float32),
        "four_real_tip_target_world_m": np.asarray(
            retry["leap_four_tip_world_m"][0], dtype=np.float32
        ),
        "four_real_tip_actual_world_m": np.asarray(
            retry["wuji2_four_tip_world_m"][0], dtype=np.float32
        ),
        "retarget_validation_passed": np.asarray(True),
        "native_pregrasp_opening_m": np.asarray(
            base["native_pregrasp_opening_m"], dtype=np.float32
        ),
        "official_dgn2_pregrasp_retreat_m": np.asarray(
            base["official_dgn2_pregrasp_retreat_m"], dtype=np.float32
        ),
        "official_dgn2_pregrasp_local_transform": np.asarray(
            base["official_dgn2_pregrasp_local_transform"], dtype=np.float32
        ),
        "wuji2_semantic_palm_frame_in_r_wrist": np.asarray(
            base["wuji2_semantic_palm_frame_in_r_wrist"], dtype=np.float32
        ),
        "post_squeeze_lift_distance_m": np.asarray(
            base["post_squeeze_lift_distance_m"], dtype=np.float32
        ),
        "simulation_gravity_policy": np.asarray(
            "leap_style_zero_until_squeeze_then_continuous"
        ),
        "simulation_runtime_usd": np.asarray(base["simulation_runtime_usd"]),
        "leap_style_root_joint_names": np.asarray(leap["root_joint_names"]),
        "leap_style_root_effective_stiffness": np.asarray(800.0, np.float32),
        "leap_style_root_effective_damping": np.asarray(20.0, np.float32),
        "leap_style_root_virtual_link_mass_kg": np.asarray(0.05, np.float32),
        "leap_style_root_virtual_link_inertia_kg_m2": np.asarray(1.0e-4, np.float32),
        "pregrasp_thumb_extra_open_m_requested": np.asarray(
            PREGRASP_THUMB_EXTRA_OPEN_M, np.float32
        ),
        "pregrasp_thumb_inward_direction_local_xyz": np.asarray(
            native_direction["local_inward_normals_xyz"]["r_thumb_tip"],
            dtype=np.float32,
        ),
        "pregrasp_thumb_outward_direction_local_xyz": np.asarray(
            -np.asarray(
                native_direction["local_inward_normals_xyz"]["r_thumb_tip"],
                dtype=np.float32,
            ),
            dtype=np.float32,
        ),
        "pregrasp_thumb_outward_direction_r_wrist_xyz": (
            outward_r_wrist.detach().cpu().numpy().astype(np.float32)
        ),
        "pregrasp_thumb_extra_open_m_actual": np.asarray(
            thumb_open_actual_m, np.float32
        ),
        "pregrasp_thumb_extra_open_m_projected": np.asarray(
            thumb_open_projected_m, np.float32
        ),
        "pregrasp_thumb_extra_open_target_error_m": np.asarray(
            thumb_open_target_error_m, np.float32
        ),
        "pregrasp_thumb_ik_joint_delta_rad": thumb_joint_delta.astype(np.float32),
        "pregrasp_thumb_ik_steps": np.asarray(
            native_direction["ik_steps"], np.int64
        ),
        "pinky_ring_coupling_policy": np.asarray(
            json.dumps(
                load_pinky_policy(PINKY_POLICY),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "pregrasp_cover_pinky_coupling_audit": np.asarray(
            json.dumps(
                pre_cover_pinky_audit,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }
    output["wuji2_semantic_palm_approach_axis_source"] = approach_axis_source.astype(np.float32)
    output["wuji2_semantic_palm_approach_axis_world"] = approach_axis_world.astype(np.float32)
    output["post_squeeze_lift_policy"] = np.asarray(lift_policy)
    output["approach_axis_dot_gravity"] = np.asarray(approach_dot_gravity, np.float32)
    output["is_top_grasp"] = np.asarray(is_top)
    output["squeeze_dense_q20_path"] = dense.astype(np.float32)
    output["squeeze_dense_alpha"] = alpha.astype(np.float32)
    output["squeeze_dense_joint_names"] = np.asarray(names)
    output["squeeze_dense_policy"] = np.asarray(
        json.dumps(report["effective_overrides"], sort_keys=True, separators=(",", ":"))
    )
    output["squeeze_path_validation_passed"] = np.asarray(True)
    output["retarget_source_npz"] = np.asarray(str(SOURCE))
    output["pregrasp_approach_policy"] = np.asarray(
        "wuji2_semantic_palm_positive_z_100mm"
    )
    output["mediapipe_rotation_scope"] = np.asarray(
        "Retargeter-local keypoint preprocessing only; never multiply into T_SourceZone_LEAP, T_SourceZone_r_wrist, T_world_r_wrist, or T_world_flange"
    )
    output["retarget_offset_scope"] = np.asarray(
        "wrist_offset_cm and thumb_offset_cm are in retarget-local frame after MediaPipe normalization and mediapipe_rotation; not World, SourceZone, or LEAP-root frame"
    )

    SIM_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **output)
    metadata = {
        "schema_version": 1,
        "status": "READY",
        "source": str(SOURCE),
        "base_scene_and_runtime_contract": str(BASE_JOB),
        "output": str(OUTPUT),
        "stages": stages,
        "coordinate_convention": "T_A_B maps coordinates from frame B into frame A",
        "waypoint_pose_world_actual_frame": "SourceZone",
        "waypoint_pose_frame": "SourceZone",
        "source_leap_waypoint_pose_frame": "SourceZone",
        "control_steps": [int(value) for value in base["waypoint_steps"].tolist()],
        "squeeze_dense_samples": int(len(dense)),
        "maximum_joint_step_rad": max_step,
        "inside_official_wuji2_limits": True,
        "grasp_q20_unchanged": True,
        "root_6d_unchanged": True,
        "pinky_policy": load_pinky_policy(PINKY_POLICY),
        "pregrasp_cover_pinky_coupling_audit": pre_cover_pinky_audit,
        "pregrasp": (
            "start from the audited open pose; move only r_thumb_tip by native "
            "IK along the direction opposite its approved local +Y inward "
            "normal; pinky flexion copies ring and its MCP abduction adds the "
            "configured outward bias"
        ),
        "pregrasp_thumb_extra_open_m_requested": PREGRASP_THUMB_EXTRA_OPEN_M,
        "pregrasp_thumb_inward_direction_local_xyz": native_direction[
            "local_inward_normals_xyz"
        ]["r_thumb_tip"],
        "pregrasp_thumb_outward_direction_local_xyz": (
            -np.asarray(
                native_direction["local_inward_normals_xyz"]["r_thumb_tip"],
                dtype=np.float64,
            )
        ).tolist(),
        "pregrasp_thumb_outward_direction_r_wrist_xyz": (
            outward_r_wrist.detach().cpu().numpy().tolist()
        ),
        "pregrasp_thumb_extra_open_m_actual": thumb_open_actual_m,
        "pregrasp_thumb_extra_open_m_projected": thumb_open_projected_m,
        "pregrasp_thumb_extra_open_target_error_m": thumb_open_target_error_m,
        "pregrasp_thumb_ik_joint_delta_rad": thumb_joint_delta.tolist(),
        "pregrasp_thumb_ik_steps": int(native_direction["ik_steps"]),
        "approach": "official 100 mm along calibrated Wuji2 semantic-palm +Z, computed in SourceZone then mapped to layout world for world consumers",
        "wuji2_semantic_palm_approach_axis_source": approach_axis_source.tolist(),
        "wuji2_semantic_palm_approach_axis_world": approach_axis_world.tolist(),
        "mediapipe_rotation_scope": (
            "Retargeter-local keypoint preprocessing only; never multiply into "
            "T_SourceZone_LEAP, T_SourceZone_r_wrist, T_world_r_wrist, or T_world_flange"
        ),
        "retarget_offset_scope": (
            "wrist_offset_cm and thumb_offset_cm are in retarget-local frame after "
            "MediaPipe normalization and mediapipe_rotation; not World, SourceZone, "
            "or LEAP-root frame"
        ),
        "gravity": "0 before/through SQUEEZE, then -9.81 m/s^2",
        "lift_policy": lift_policy,
        "runtime_usd": str(scalar(base, "simulation_runtime_usd")),
        "success_criterion": "target object center rises by at least 30 mm",
        "root_control": (
            "LEAP-equivalent 3-prismatic + 3-revolute force-position "
            "drives; effective K=800, D=20"
        ),
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_case_wrappers(CASE_ROOT.name)
    print(f"[PASS] dense path={dense.shape}; max step={max_step:.6f} rad")
    print(f"[PASS] official Wuji2 limits; top grasp={is_top}")
    print(f"[OK] job={OUTPUT}")


if __name__ == "__main__":
    main()
