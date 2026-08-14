"""Execute native Wuji2 waypoints through the driven 3P+3R wrist.

The motion/hold schedule is the historical 07 native policy.  Unlike the old
executor, this file never calls ``set_world_pose`` on the hand.
"""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path

import numpy as np
import omni.timeline
from isaacsim.core.utils.types import ArticulationAction


CONTEXT_KEY = "DGN2_NATIVE_WUJI2_3P3R_CONTEXT"
CALLBACK_NAME = "dgn2_native_wuji2_3p3r_execution"
LIFT_THRESHOLD_M = 0.03
INITIAL_MIN_HOLD = 180
INITIAL_MAX_HOLD = 360
JOINT_SPEED_THRESHOLD_RAD_S = 0.02
OBJECT_LINEAR_SPEED_THRESHOLD_M_S = 0.003
OBJECT_ANGULAR_SPEED_THRESHOLD_RAD_S = 0.05

if not hasattr(builtins, CONTEXT_KEY):
    raise RuntimeError("Run the matching 01_import.py in this Isaac Sim session first")
ctx = getattr(builtins, CONTEXT_KEY)
if ctx["branch"] != "wuji2_native_3p3r":
    raise RuntimeError(f"wrong loaded branch: {ctx['branch']}")

world = ctx["world"]
hand = ctx["hand"]
objects = ctx["objects"]
targets = np.asarray(ctx["targets"], dtype=np.float32)
names = list(ctx["waypoint_names"])
steps = list(ctx["waypoint_steps"])
min_holds = list(ctx["minimum_hold_steps"])
max_holds = list(ctx["maximum_hold_steps"])
quiet_required = int(ctx["quiet_consecutive_steps"])
substeps = int(ctx["physics_substeps_per_control"])
interpolation = str(ctx["interpolation_policy"])
target_seg = int(ctx["target_segmentation_id"])
dense_squeeze = np.asarray(ctx["squeeze_dense_targets"], dtype=np.float32)
timeline = omni.timeline.get_timeline_interface()

if names != ["pregrasp", "cover_open", "grasp", "squeeze", "lift"]:
    raise RuntimeError(f"wrong waypoint order: {names}")
if targets.shape[0] != 5 or len(steps) != 4:
    raise RuntimeError(f"expected five targets and four durations, got {targets.shape}")
if not (len(min_holds) == len(max_holds) == 4):
    raise RuntimeError("hold schedule must contain four endpoints")
if interpolation != "minimum_jerk":
    raise RuntimeError(f"native policy requires minimum_jerk, got {interpolation}")
if substeps != 1:
    raise RuntimeError(f"native policy requires one control per 120 Hz step, got {substeps}")
if world.physics_callback_exists(CALLBACK_NAME):
    world.remove_physics_callback(CALLBACK_NAME)

initial_positions = {
    seg: np.asarray(wrapper.get_world_pose()[0], dtype=np.float64)
    for seg, wrapper in objects.items()
}
state = {
    "mode": "initial_hold",
    "segment": 1,
    "step": 0,
    "hold_step": 0,
    "quiet_count": 0,
    "done": False,
    "holds": [],
    "stage_object_positions": {
        "initial": {str(k): v.tolist() for k, v in initial_positions.items()}
    },
}


def current_speed() -> dict:
    joint_velocity = np.asarray(hand.get_joint_velocities(), dtype=np.float32)
    max_joint = float(np.max(np.abs(joint_velocity)))
    max_linear = 0.0
    max_angular = 0.0
    for wrapper in objects.values():
        linear = np.asarray(wrapper.get_linear_velocity(), dtype=np.float32)
        angular = np.asarray(wrapper.get_angular_velocity(), dtype=np.float32)
        max_linear = max(max_linear, float(np.linalg.norm(linear)))
        max_angular = max(max_angular, float(np.linalg.norm(angular)))
    return {
        "quiet": bool(
            max_joint <= JOINT_SPEED_THRESHOLD_RAD_S
            and max_linear <= OBJECT_LINEAR_SPEED_THRESHOLD_M_S
            and max_angular <= OBJECT_ANGULAR_SPEED_THRESHOLD_RAD_S
        ),
        "max_joint_rad_s": max_joint,
        "max_object_linear_m_s": max_linear,
        "max_object_angular_rad_s": max_angular,
    }


def apply_target(value: np.ndarray) -> None:
    # Position targets are converted by PhysX/USD drives to bounded motor
    # efforts. No hand root teleport is issued here.
    hand.apply_action(ArticulationAction(joint_positions=np.asarray(value, dtype=np.float32)))


def record_stage(name: str) -> None:
    state["stage_object_positions"][name] = {
        str(seg): np.asarray(wrapper.get_world_pose()[0], dtype=np.float64).tolist()
        for seg, wrapper in objects.items()
    }


def finish() -> None:
    final_positions = {
        seg: np.asarray(wrapper.get_world_pose()[0], dtype=np.float64)
        for seg, wrapper in objects.items()
    }
    displacement = {seg: final_positions[seg] - initial_positions[seg] for seg in objects}
    lifted = {seg: bool(delta[2] > LIFT_THRESHOLD_M) for seg, delta in displacement.items()}
    success = bool(lifted[target_seg] and ctx["pregrasp_valid"])
    report = {
        "schema_version": 1,
        "status": "native_wuji2_3p3r_execution_complete",
        "target_specific_success": success,
        "scene": ctx["scene_id"],
        "view": ctx["view_id"],
        "target_segmentation_id": target_seg,
        "target_object_code": ctx["target_code"],
        "source_candidate_index": ctx["source_candidate_index"],
        "score": ctx["score"],
        "root_control_policy": ctx["root_control_policy"],
        "root_gain_audit": ctx["root_gain_audit"],
        "native_action_policy": {
            "approach": ctx["pregrasp_approach_policy"],
            "squeeze": ctx["squeeze_dense_policy"],
            "lift": ctx["post_squeeze_lift_policy"],
            "control_steps": steps,
            "minimum_hold_steps": min_holds,
            "maximum_hold_steps": max_holds,
            "interpolation": interpolation,
            "gravity": "continuous -9.81 m/s^2",
        },
        "target_lift_delta_m": float(displacement[target_seg][2]),
        "target_lateral_displacement_m": float(np.linalg.norm(displacement[target_seg][:2])),
        "displacement_xyz_m_by_segmentation_id": {
            str(k): v.tolist() for k, v in displacement.items()
        },
        "lifted_by_segmentation_id": {str(k): v for k, v in lifted.items()},
        "stage_object_positions_m": state["stage_object_positions"],
        "hold_diagnostics": state["holds"],
    }
    path = Path(ctx["result_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if world.physics_callback_exists(CALLBACK_NAME):
        world.remove_physics_callback(CALLBACK_NAME)
    timeline.pause()
    state["done"] = True
    print("\n[02 EXECUTION COMPLETE]")
    print(f"result={'PASS' if success else 'FAIL'}")
    print(f"target lift={1000.0 * displacement[target_seg][2]:+.2f} mm")
    print(f"wrote {path}")


def hold_step(endpoint_index: int | None) -> None:
    target = targets[0] if endpoint_index is None else targets[endpoint_index]
    apply_target(target)
    speed = current_speed()
    state["hold_step"] += 1
    state["quiet_count"] = state["quiet_count"] + 1 if speed["quiet"] else 0
    if endpoint_index is None:
        minimum, maximum, label = INITIAL_MIN_HOLD, INITIAL_MAX_HOLD, "initial"
    else:
        hold_index = endpoint_index - 1
        minimum, maximum, label = min_holds[hold_index], max_holds[hold_index], names[endpoint_index]
    settled = state["hold_step"] >= minimum and state["quiet_count"] >= quiet_required
    timed_out = state["hold_step"] >= maximum
    if not (settled or timed_out):
        return
    state["holds"].append({
        "stage": label,
        "hold_steps": int(state["hold_step"]),
        "settled": bool(settled),
        "maximum_hold_timeout": bool(timed_out and not settled),
        **{k: v for k, v in speed.items() if k != "quiet"},
    })
    print(f"held {label} for {state['hold_step']} steps; {'quiet' if settled else 'timeout'}")
    state["hold_step"] = 0
    state["quiet_count"] = 0
    if endpoint_index is None:
        state["mode"] = "motion"
        state["segment"] = 1
        state["step"] = 0
    elif endpoint_index == 4:
        finish()
    else:
        state["mode"] = "motion"
        state["segment"] = endpoint_index + 1
        state["step"] = 0


def on_physics_step(_step_size: float) -> None:
    if state["done"]:
        return
    if state["mode"] == "initial_hold":
        hold_step(None)
        return
    if state["mode"] == "hold":
        hold_step(int(state["segment"]))
        return

    segment = int(state["segment"])
    count = int(steps[segment - 1])
    linear_alpha = float(state["step"] + 1) / float(count)
    alpha = 10.0 * linear_alpha**3 - 15.0 * linear_alpha**4 + 6.0 * linear_alpha**5
    if segment == 3:
        dense_position = alpha * float(len(dense_squeeze) - 1)
        low = min(int(math.floor(dense_position)), len(dense_squeeze) - 1)
        high = min(low + 1, len(dense_squeeze) - 1)
        fraction = dense_position - low
        target = (1.0 - fraction) * dense_squeeze[low] + fraction * dense_squeeze[high]
    else:
        target = targets[segment - 1] + alpha * (targets[segment] - targets[segment - 1])
    apply_target(target)
    state["step"] += 1
    if state["step"] < count:
        return
    record_stage(names[segment])
    print(f"reached {names[segment]} after {count} physics/control steps")
    state["mode"] = "hold"
    state["hold_step"] = 0
    state["quiet_count"] = 0


world.add_physics_callback(CALLBACK_NAME, on_physics_step)
timeline.play()
print("\n[02 EXECUTION STARTED]")
print("schedule=settle -> COVER_OPEN -> GRASP -> SQUEEZE -> LIFT")
print(f"control steps={steps}; holds={min_holds}/{max_holds}; interpolation=minimum_jerk")
print("gravity=-9.81 throughout; hand root=set by 3P+3R joint targets, never teleported")
