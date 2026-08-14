"""Stage 02 executor for the verified LEAP-root-driven Wuji2 workflow."""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path

import numpy as np
import omni.timeline

from isaacsim.core.utils.types import ArticulationAction


CONTEXT_KEY = "DGN2_OFFICIAL_LEAP_WUJI2_AB_CONTEXT"
CALLBACK_NAME = "dgn2_official_leap_wuji2_ab_execution"
LIFT_THRESHOLD_M = 0.03
PHYSICS_SUBSTEPS_PER_CONTROL = 2

if not hasattr(builtins, CONTEXT_KEY):
    raise RuntimeError("Run the matching 01_import.py in this Isaac Sim session first")
ctx = getattr(builtins, CONTEXT_KEY)
BRANCH = str(getattr(builtins, "DGN2_AB_BRANCH", ""))
if BRANCH != "wuji2_leap_root_drive":
    raise RuntimeError(
        "Only the verified wuji2_leap_root_drive method is accepted, got "
        f"{BRANCH!r}"
    )
if BRANCH != ctx["branch"]:
    raise RuntimeError(f"loaded branch is {ctx['branch']}, but this executor is {BRANCH}")

world = ctx["world"]
hand = ctx["hand"]
objects = ctx["objects"]
targets = np.asarray(ctx["targets"], dtype=np.float32)
root_poses = np.asarray(ctx["root_poses"], dtype=np.float32)
waypoint_names = list(ctx["waypoint_names"])
control_steps = list(ctx["waypoint_steps"])
target_seg = int(ctx["target_segmentation_id"])
timeline = omni.timeline.get_timeline_interface()
if targets.shape[0] != 5 or len(control_steps) != 4:
    raise RuntimeError(f"expected five stages/four durations, got {targets.shape}")
if world.physics_callback_exists(CALLBACK_NAME):
    world.remove_physics_callback(CALLBACK_NAME)

initial_positions = {
    seg_id: np.asarray(wrapper.get_world_pose()[0], dtype=np.float64)
    for seg_id, wrapper in objects.items()
}
state = {
    "segment": 1,
    "physics_step": 0,
    "done": False,
    "gravity_restored": False,
    # Read-only diagnostics: snapshot every object's centre after each stage.
    # This does not change the controller, contacts, gravity, or success rule.
    "stage_object_positions": {
        "initial": {
            str(seg_id): position.tolist()
            for seg_id, position in initial_positions.items()
        }
    },
}
dense_squeeze_targets = (
    None
    if ctx.get("squeeze_dense_targets") is None
    else np.asarray(ctx["squeeze_dense_targets"], dtype=np.float32)
)


def rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        q = np.empty(4, dtype=np.float64)
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q[:] = [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale]
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q[:] = [(matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale]
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q[:] = [(matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
        result = q
    return result / np.linalg.norm(result)


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        value = a + alpha * (b - a)
        return value / np.linalg.norm(value)
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    return (
        math.sin((1.0 - alpha) * theta) * a
        + math.sin(alpha * theta) * b
    ) / math.sin(theta)


def finish() -> None:
    final_positions = {
        seg_id: np.asarray(wrapper.get_world_pose()[0], dtype=np.float64)
        for seg_id, wrapper in objects.items()
    }
    displacements = {
        seg_id: final_positions[seg_id] - initial_positions[seg_id]
        for seg_id in objects
    }
    lifted = {
        seg_id: bool(delta[2] > LIFT_THRESHOLD_M)
        for seg_id, delta in displacements.items()
    }
    target_success = bool(lifted[target_seg] and ctx["pregrasp_valid"])
    any_success = bool(any(lifted.values()) and ctx["pregrasp_valid"])
    report = {
        "schema_version": 1,
        "status": "manual_gui_ab_branch_complete",
        "branch": BRANCH,
        "scene": ctx["scene_id"],
        "view": ctx["view_id"],
        "root_control_policy": ctx["root_control_policy"],
        "root_gain_audit": ctx.get("root_gain_audit"),
        "source_candidate_index": ctx["source_candidate_index"],
        "score": ctx["score"],
        "target_segmentation_id": target_seg,
        "target_object_code": ctx["target_code"],
        "pregrasp_approach_policy": ctx["pregrasp_approach_policy"],
        "pregrasp_approach_axis_world": ctx["pregrasp_approach_axis_world"],
        "post_squeeze_lift_policy": ctx["post_squeeze_lift_policy"],
        "squeeze_dense_policy": ctx.get("squeeze_dense_policy"),
        "squeeze_dense_sample_count": (
            0 if dense_squeeze_targets is None else len(dense_squeeze_targets)
        ),
        "pregrasp_valid": ctx["pregrasp_valid"],
        "target_specific_success": target_success,
        "official_any_object_success": any_success,
        "target_lift_delta_m": float(displacements[target_seg][2]),
        "target_lateral_displacement_m": float(
            np.linalg.norm(displacements[target_seg][:2])
        ),
        "displacement_xyz_m_by_segmentation_id": {
            str(key): value.tolist() for key, value in displacements.items()
        },
        "lifted_by_segmentation_id": {
            str(key): value for key, value in lifted.items()
        },
        "stage_object_positions_m": state["stage_object_positions"],
    }
    result_path = Path(ctx["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if world.physics_callback_exists(CALLBACK_NAME):
        world.remove_physics_callback(CALLBACK_NAME)
    timeline.pause()
    state["done"] = True
    print("\n[02 EXECUTION COMPLETE]")
    print(f"branch={BRANCH}")
    print(f"target-specific result={'PASS' if target_success else 'FAIL'}")
    print(f"official any-object result={'PASS' if any_success else 'FAIL'}")
    print(f"target lift={1000.0 * displacements[target_seg][2]:+.2f} mm")
    print(f"target lateral={1000.0 * np.linalg.norm(displacements[target_seg][:2]):.2f} mm")
    print(f"wrote {result_path}")


def on_physics_step(_step_size: float) -> None:
    if state["done"]:
        return
    segment = int(state["segment"])
    physical_step = int(state["physics_step"])
    total = control_steps[segment - 1] * PHYSICS_SUBSTEPS_PER_CONTROL
    control_step = physical_step // PHYSICS_SUBSTEPS_PER_CONTROL
    alpha = float(control_step + 1) / float(control_steps[segment - 1])
    if segment == 3 and dense_squeeze_targets is not None:
        dense_position = alpha * float(len(dense_squeeze_targets) - 1)
        dense_low = min(
            int(math.floor(dense_position)), len(dense_squeeze_targets) - 1
        )
        dense_high = min(dense_low + 1, len(dense_squeeze_targets) - 1)
        dense_alpha = dense_position - float(dense_low)
        joint_target = (
            (1.0 - dense_alpha) * dense_squeeze_targets[dense_low]
            + dense_alpha * dense_squeeze_targets[dense_high]
        )
    else:
        joint_target = targets[segment - 1] + (
            targets[segment] - targets[segment - 1]
        ) * alpha
    hand.apply_action(ArticulationAction(joint_positions=joint_target))
    state["physics_step"] += 1
    if state["physics_step"] < total:
        return
    print(
        f"reached {waypoint_names[segment]}: "
        f"{control_steps[segment - 1]} control / {total} physics steps"
    )
    state["stage_object_positions"][str(waypoint_names[segment])] = {
        str(seg_id): np.asarray(wrapper.get_world_pose()[0], dtype=np.float64).tolist()
        for seg_id, wrapper in objects.items()
    }
    if segment == 3 and not state["gravity_restored"]:
        world.get_physics_context().set_gravity(-9.81)
        state["gravity_restored"] = True
        print("gravity restored to -9.81 m/s^2 after SQUEEZE")
    if segment == 4:
        finish()
        return
    state["segment"] += 1
    state["physics_step"] = 0


world.add_physics_callback(CALLBACK_NAME, on_physics_step)
timeline.play()
print("\n[02 EXECUTION STARTED]")
print(f"branch={BRANCH}; schedule=COVER -> GRASP -> SQUEEZE -> LIFT")
print(f"official control steps={control_steps}; physics substeps/control=2")
if dense_squeeze_targets is not None:
    print(
        f"SQUEEZE uses {len(dense_squeeze_targets)} dense q20 samples; "
        f"policy={ctx['squeeze_dense_policy']}"
    )
