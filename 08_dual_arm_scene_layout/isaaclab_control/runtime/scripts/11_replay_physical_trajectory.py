#!/usr/bin/env python3
"""Replay the verified physical trajectory in real time, without physics."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPLAY = (
    PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/outputs/"
    "full_pick_place_25s_dog_candidate3800/physical_replay_30fps.npz"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-file", type=Path)
    parser.add_argument("--done-file", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_arguments()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def find_one_rigid_prim(prefix: str) -> Usd.Prim:
    matches = [
        prim for prim in get_current_stage().Traverse()
        if str(prim.GetPath()).startswith(prefix + "/")
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"Expected one rigid body under {prefix}, got {paths}")
    return matches[0]


def configure_view(metadata: dict) -> None:
    view = metadata.get("viewer_camera", {})
    stage = get_current_stage()
    for path in view.get("hide_prims", []):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    target = np.asarray(view.get("target_world_m", [0.0, -0.145, 0.50]))
    yaw = math.radians(float(view.get("yaw_about_world_z_deg", -90.0)))
    distance = float(view.get("horizontal_distance_m", 1.45))
    eye = target + np.asarray([
        distance * math.cos(yaw),
        distance * math.sin(yaw),
        float(view.get("height_above_target_m", 0.75)),
    ])
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")


def run() -> None:
    replay_path = ARGS.replay.resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"Replay does not exist: {replay_path}")
    with np.load(replay_path, allow_pickle=False) as archive:
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        states = np.asarray(archive["state"]).astype(str)
        joint_q = np.asarray(archive["joint_position_rad"], dtype=np.float32)
        object_poses = np.asarray(archive["object_pose_world_wxyz"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if not (len(time_s) == len(states) == joint_q.shape[0] == object_poses.shape[0]):
        raise RuntimeError("Replay arrays have inconsistent frame counts")

    if not stage_utils.open_stage(metadata["stage"]):
        raise RuntimeError(f"Cannot open replay stage: {metadata['stage']}")
    stage = get_current_stage()
    duplicate = stage.GetPrimAtPath("/World/Layout/TableAssembly/TestScene0000")
    if duplicate.IsValid():
        stage.RemovePrim(duplicate.GetPath())
    UsdGeom.Xform.Define(stage, "/World/TaskObjects")

    objects: list[RigidObject] = []
    for record in metadata["objects"]:
        root = record["reference_root_path"]
        add_reference_to_stage(record["simulation_usd"], root)
        rigid = find_one_rigid_prim(root)
        objects.append(RigidObject(RigidObjectCfg(prim_path=str(rigid.GetPath()), spawn=None)))

    simulation = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=ARGS.device))
    robot = Articulation(ArticulationCfg(
        prim_path=metadata["robot_prim"],
        spawn=None,
        actuators={"replay_only": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )},
    ))
    simulation.reset()
    configure_view(metadata)
    if list(robot.joint_names) != list(metadata["joint_names"]):
        raise RuntimeError("Replay robot joint order no longer matches the recording")
    zero_velocity = torch.zeros((1, robot.num_joints), device=robot.device)

    def show_frame(index: int) -> None:
        q = torch.as_tensor(joint_q[index], device=robot.device).reshape(1, -1)
        robot.write_joint_state_to_sim(q, zero_velocity)
        for object_index, obj in enumerate(objects):
            pose = torch.as_tensor(
                object_poses[index, object_index], device=obj.device
            ).reshape(1, 7)
            obj.write_root_pose_to_sim(pose)
        simulation.render()

    duration = float(time_s[-1])
    show_frame(0)
    if ARGS.ready_file:
        ARGS.ready_file.resolve().parent.mkdir(parents=True, exist_ok=True)
        ARGS.ready_file.resolve().touch()
    if ARGS.start_file:
        start_file = ARGS.start_file.resolve()
        print(f"[REPLAY WAITING] create {start_file} to begin")
        while SIMULATION_APP.is_running() and not start_file.exists():
            time.sleep(0.02)

    print(f"[REPLAY READY] frames={len(time_s)}; recorded={metadata['record_fps']:.1f} FPS")
    print(f"[REPLAY START] wall-clock duration={duration:.2f} s; physics disabled")
    start = time.perf_counter()
    last_index, last_state, rendered = -1, "", 0
    while SIMULATION_APP.is_running():
        elapsed = min(time.perf_counter() - start, duration)
        index = int(np.searchsorted(time_s, elapsed, side="right") - 1)
        index = max(0, min(index, len(time_s) - 1))
        if index != last_index:
            show_frame(index)
            rendered += 1
            last_index = index
            if states[index] != last_state:
                last_state = states[index]
                print(f"[{elapsed:6.2f}s] STATE: {last_state}")
            print(
                f"\rREPLAY {elapsed:6.2f}/{duration:.2f}s | frame {index + 1:4d}/{len(time_s)}",
                end="", flush=True,
            )
        else:
            time.sleep(0.001)
        if elapsed >= duration:
            break
    wall = time.perf_counter() - start
    print(f"\n[REPLAY COMPLETE] wall={wall:.2f}s; rendered={rendered}; skipped={len(time_s)-rendered}")
    if ARGS.done_file:
        ARGS.done_file.resolve().parent.mkdir(parents=True, exist_ok=True)
        ARGS.done_file.resolve().touch()


def main() -> int:
    try:
        run()
        return 0
    except Exception as error:
        print(f"[REPLAY ERROR] {type(error).__name__}: {error}")
        return 1
    finally:
        SIMULATION_APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
