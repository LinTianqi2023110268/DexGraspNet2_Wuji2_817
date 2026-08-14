#!/usr/bin/env python3
"""Execute a prepared DexGraspNet2 LEAP grasp job in Isaac Sim 5.0.

This is stage two only.  It never loads the neural network or its checkpoint.
Prepare the job first with ``prepare_isaacsim5_job.py`` in ``graspnet2.0``,
then run this file from the Isaac Sim 5.0 / Isaac Lab 2.2 environment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.chdir(str(REPO_ROOT))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

from isaaclab.app import AppLauncher


ISAAC_SIM_ROOT = Path(os.environ.get("ISAAC_PATH", "/home/lin/isaacsim")).expanduser().resolve()
MINIMAL_EXPERIENCE = ISAAC_SIM_ROOT / "apps" / "isaacsim.exp.base.python.kit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--env-spacing", type=float, default=0.8)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace GUI physics to wall-clock time without changing official step counts",
    )
    parser.add_argument(
        "--gui-start-delay",
        type=float,
        default=0.0,
        help="Render the initial pregrasp for this many seconds before execution",
    )
    parser.add_argument("--hold", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    if not MINIMAL_EXPERIENCE.is_file():
        raise FileNotFoundError("Isaac Sim 5.0 experience not found: {}".format(MINIMAL_EXPERIENCE))
    parser.set_defaults(experience=str(MINIMAL_EXPERIENCE))
    return parser


ARGS = build_parser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import numpy as np
import torch
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.prims import RigidPrim
import isaaclab.sim as sim_utils
from omni.physx.scripts import physicsUtils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


class ReplaySimulationContext(SimulationContext):
    """Avoid Isaac Lab render presets that assume a Lab-owned experience."""

    def _apply_render_settings_from_cfg(self) -> None:
        return


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
             (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                               (matrix[0, 1] + matrix[1, 0]) / s,
                               (matrix[0, 2] + matrix[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray([(matrix[0, 2] - matrix[2, 0]) / s,
                               (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                               (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray([(matrix[1, 0] - matrix[0, 1]) / s,
                               (matrix[0, 2] + matrix[2, 0]) / s,
                               (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)


def set_gravity(stage, magnitude: float) -> None:
    scene = UsdPhysics.Scene.Get(stage, Sdf.Path("/physicsScene"))
    if not scene:
        raise RuntimeError("Missing /physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(float(magnitude))


def bind_physics_material_at_asset_root(stage, root_path: str, material_path: str) -> int:
    """Bind above referenced/instanced colliders and audit inherited binding.

    Isaac Lab's nested helper tries to author directly on every collision
    instance proxy, which USD correctly rejects.  A stronger-than-descendants
    physics binding on the non-instanced asset root is inherited by all those
    colliders and is the supported USD composition pattern.
    """

    root = stage.GetPrimAtPath(root_path)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not root.IsValid() or not material:
        raise RuntimeError(
            "Invalid physics binding root/material: {} -> {}".format(
                root_path, material_path
            )
        )
    binding_api = (
        UsdShade.MaterialBindingAPI(root)
        if root.HasAPI(UsdShade.MaterialBindingAPI)
        else UsdShade.MaterialBindingAPI.Apply(root)
    )
    binding_api.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )

    colliders = [
        prim
        for prim in Usd.PrimRange(
            root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not colliders:
        raise RuntimeError("No collision prims found below {}".format(root_path))
    failures = []
    for collider in colliders:
        result = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
            materialPurpose="physics"
        )
        bound_material = result[0] if isinstance(result, tuple) else result
        if not bound_material or bound_material.GetPath() != material.GetPath():
            failures.append(str(collider.GetPath()))
    if failures:
        raise RuntimeError(
            "Physics material inheritance audit failed below {}: {}".format(
                root_path, failures[:5]
            )
        )
    return len(colliders)


def build_hand_cfg(cache_dir: Path) -> sim_utils.UrdfFileCfg:
    return sim_utils.UrdfFileCfg(
        asset_path=str((REPO_ROOT / "robot_models/urdf/leap_hand_simplified_free.urdf").resolve()),
        usd_dir=str(cache_dir / "leap_hand_free"),
        usd_file_name="leap_hand_simplified_free.usd",
        force_usd_conversion=False,
        make_instanceable=False,
        fix_base=True,
        merge_fixed_joints=True,
        self_collision=False,
        replace_cylinders_with_capsules=True,
        collider_type="convex_hull",
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=800.0, damping=20.0
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    )


def build_object_cfg(urdf: Path, cache_dir: Path, code: str) -> sim_utils.UrdfFileCfg:
    return sim_utils.UrdfFileCfg(
        asset_path=str(urdf.resolve()),
        # Keep official GraspNet meshes in a separate cache namespace.  Older
        # teaching demos used locally generated CoACD URDFs with the same
        # object codes and must never be silently reused here.
        usd_dir=str(cache_dir / "official_meshdata_objects" / code),
        usd_file_name="{}_official_meshdata.usd".format(code),
        force_usd_conversion=False,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        # Isaac Gym loads the official nontextured_simplified mesh with VHACD
        # enabled.  Isaac Sim's closest importer-side equivalent is convex
        # decomposition, not one enclosing convex hull.
        collider_type="convex_decomposition",
        joint_drive=None,
        link_density=500.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            max_depenetration_velocity=1000.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        activate_contact_sensors=False,
    )


def find_rigid_body_pattern(stage, code: str) -> str:
    prefix = "/World/envs/env_000/Object_{}".format(code)
    paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(paths) != 1:
        raise RuntimeError("Expected one merged rigid body for {}, found {}".format(code, paths))
    return paths[0].replace("/env_000/", "/env_.*/")


def create_scene(sim: ReplaySimulationContext, manifest: dict, count: int):
    sim_utils.DomeLightCfg(intensity=900.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=900.0)
    )
    # Isaac Lab 2.2's GroundPlaneCfg references
    # ``${ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd``.
    # A standalone Isaac Sim 5.0 install has no Nucleus root, so create the
    # PhysX plane directly and avoid any remote/default-environment asset.
    physicsUtils.add_ground_plane(
        sim.stage,
        "/World/Ground",
        "Z",
        100.0,
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.35, 0.35, 0.35),
    )
    ground_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )
    ground_material.func("/World/PhysicsMaterials/Ground", ground_material)
    sim_utils.bind_physics_material("/World/Ground", "/World/PhysicsMaterials/Ground")
    prim_utils.create_prim("/World/envs", "Xform")
    cache_dir = REPO_ROOT / "data" / "isaacsim5_cache"
    hand_cfg = build_hand_cfg(cache_dir)
    object_cfgs = {
        obj["code"]: build_object_cfg(Path(obj["simulation_urdf"]), cache_dir, obj["code"])
        for obj in manifest["objects"]
    }

    columns = min(4, count)
    origins = []
    for index in range(count):
        row, column = divmod(index, columns)
        origin = np.asarray(
            [column * float(ARGS.env_spacing), row * float(ARGS.env_spacing), 0.0],
            dtype=np.float64,
        )
        origins.append(origin)
        env_path = "/World/envs/env_{:03d}".format(index)
        prim_utils.create_prim(env_path, "Xform")
        hand_cfg.func(
            env_path + "/Hand", hand_cfg, translation=tuple(origin), orientation=(1.0, 0.0, 0.0, 0.0)
        )
        for obj in manifest["objects"]:
            pose = np.asarray(obj["pose_world_object"], dtype=np.float64)
            object_cfgs[obj["code"]].func(
                env_path + "/Object_{}".format(obj["code"]),
                object_cfgs[obj["code"]],
                translation=tuple(origin + pose[:3, 3]),
                orientation=matrix_to_quaternion_wxyz(pose[:3, :3]),
            )

    # Official IsaacGym SimulationEvaluator creates every scene object with
    # collision filter=2.  Reproduce that with one self-filtering collision
    # group per environment.  LEAP and Ground stay outside this group.
    object_object_filtered_pairs = 0
    for index in range(count):
        group_path = "/World/CollisionGroups/env_{:03d}_SceneObjects".format(index)
        group = UsdPhysics.CollisionGroup.Define(sim.stage, group_path)
        group.CreateFilteredGroupsRel().AddTarget(group.GetPath())
        for obj in manifest["objects"]:
            prefix = "/World/envs/env_{:03d}/Object_{}".format(index, obj["code"])
            matches = [
                prim
                for prim in sim.stage.Traverse()
                if str(prim.GetPath()).startswith(prefix)
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Expected one object rigid body below {}, found {}".format(
                        prefix, [str(prim.GetPath()) for prim in matches]
                    )
                )
            physicsUtils.add_collision_to_collision_group(sim.stage, prefix, group_path)
        object_object_filtered_pairs += len(
            list(itertools.combinations(manifest["objects"], 2))
        )

    hand_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.2, dynamic_friction=0.2, restitution=0.0
    )
    hand_material.func("/World/PhysicsMaterials/Leap", hand_material)
    object_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )
    object_material.func("/World/PhysicsMaterials/Object", object_material)
    material_audit = {"hand_colliders": 0, "object_colliders": 0}
    for index in range(count):
        material_audit["hand_colliders"] += bind_physics_material_at_asset_root(
            sim.stage,
            "/World/envs/env_{:03d}/Hand".format(index),
            "/World/PhysicsMaterials/Leap",
        )
        for obj in manifest["objects"]:
            object_root = "/World/envs/env_{:03d}/Object_{}".format(
                index, obj["code"]
            )
            material_audit["object_colliders"] += bind_physics_material_at_asset_root(
                sim.stage,
                object_root,
                "/World/PhysicsMaterials/Object",
            )
    print(
        "[DexGraspNet2/IsaacSim5] material audit: hand friction=0.2 on "
        "{} colliders; object friction=1.0 on {} colliders".format(
            material_audit["hand_colliders"],
            material_audit["object_colliders"],
        ),
        flush=True,
    )
    print(
        "[DexGraspNet2/IsaacSim5] official object-object collision filter: "
        "{} pairs".format(object_object_filtered_pairs),
        flush=True,
    )

    hand = Articulation(
        ArticulationCfg(
            prim_path="/World/envs/env_.*/Hand",
            spawn=None,
            actuators={
                "leap_all_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=800.0, damping=20.0
                )
            },
        )
    )
    object_views = {
        obj["code"]: RigidPrim(
            prim_paths_expr=find_rigid_body_pattern(sim.stage, obj["code"]),
            name="object_{}_view".format(obj["code"]),
            reset_xform_properties=False,
        )
        for obj in manifest["objects"]
    }
    material_audit["object_object_filtered_pairs"] = object_object_filtered_pairs
    return hand, object_views, np.asarray(origins), material_audit


def build_joint_targets(hand, job: dict, start: int, end: int) -> torch.Tensor:
    root_names = [str(value) for value in job["root_joint_names"].tolist()]
    finger_names = [str(value) for value in job["finger_joint_names"].tolist()]
    root = np.asarray(job["waypoint_root_dofs"][start:end], dtype=np.float32)
    fingers = np.asarray(job["waypoint_joint_positions"][start:end], dtype=np.float32)
    targets = torch.zeros(
        (end - start, root.shape[1], len(hand.joint_names)), dtype=torch.float32, device=hand.device
    )
    imported = list(hand.joint_names)
    missing = [name for name in root_names + finger_names if name not in imported]
    if missing:
        raise RuntimeError("Isaac Sim imported LEAP joints are missing: {}".format(missing))
    for source, names in ((root, root_names), (fingers, finger_names)):
        for source_index, name in enumerate(names):
            targets[:, :, imported.index(name)] = torch.as_tensor(
                source[:, :, source_index], device=hand.device
            )
    return targets


def main() -> None:
    print("[DexGraspNet2/IsaacSim5] loading prepared job", flush=True)
    job_path = ARGS.job.resolve()
    manifest_path = (
        ARGS.job_manifest.resolve() if ARGS.job_manifest else job_path.with_suffix(".json")
    )
    job = dict(np.load(job_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_scene_manifest_path = project_path(manifest["scene_manifest"])
    source_scene_manifest = json.loads(
        source_scene_manifest_path.read_text(encoding="utf-8")
    )
    target_object_code = (
        source_scene_manifest.get("top1_selection", {}).get("seed_object_code")
    )
    total = len(job["source_grasp_indices"])
    start = int(ARGS.start)
    end = total if ARGS.end is None else int(ARGS.end)
    if start < 0 or end <= start or end > total:
        raise ValueError("Invalid grasp slice [{}, {}) for {} predictions".format(start, end, total))
    count = end - start
    print(
        "[DexGraspNet2/IsaacSim5] grasp slice [{}, {}), count={}".format(start, end, count),
        flush=True,
    )

    # The paper evaluator uses a 60 Hz control loop and two PhysX substeps.
    # Isaac Sim exposes the physics step directly, so integrate at 120 Hz and
    # hold every interpolated 60 Hz target for exactly two substeps.
    control_hz = 60.0
    physics_substeps_per_control = 2
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / (control_hz * physics_substeps_per_control),
        device=ARGS.device,
        gravity=(0.0, 0.0, 0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.2, dynamic_friction=0.2, restitution=0.0
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.2,
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8 * 1024 * 1024,
        ),
    )
    print("[DexGraspNet2/IsaacSim5] creating simulation context", flush=True)
    sim = ReplaySimulationContext(sim_cfg)
    print("[DexGraspNet2/IsaacSim5] importing LEAP Hand and object assets", flush=True)
    hand, object_views, origins, material_audit = create_scene(sim, manifest, count)
    print("[DexGraspNet2/IsaacSim5] resetting PhysX scene", flush=True)
    sim.reset()
    if not ARGS.headless:
        sim.set_camera_view(
            eye=(-0.75, -0.85, 0.55),
            target=(-0.13, -0.14, 0.06),
        )
    for view in object_views.values():
        view.initialize()
    if hand.num_instances != count:
        raise RuntimeError("Expected {} LEAP instances, got {}".format(count, hand.num_instances))

    print("[DexGraspNet2/IsaacSim5] mapping five waypoint targets", flush=True)
    targets = build_joint_targets(hand, job, start, end)
    velocity = torch.zeros_like(targets[:, 0])
    hand.write_joint_state_to_sim(targets[:, 0], velocity)
    hand.set_joint_position_target(targets[:, 0])
    hand.write_data_to_sim()
    sim.forward()
    if ARGS.gui_start_delay < 0.0:
        raise ValueError("--gui-start-delay must be non-negative")
    if ARGS.gui_start_delay and not ARGS.headless:
        wait_until = time.perf_counter() + ARGS.gui_start_delay
        while SIMULATION_APP.is_running() and time.perf_counter() < wait_until:
            frame_start = time.perf_counter()
            sim.render()
            remaining = (1.0 / 60.0) - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)

    # Use the actual imported rigid-link roots as the lift baseline.  URDF
    # importers may place the rigid body below an asset Xform with a fixed
    # offset, so comparing its final link pose to the manifest's asset pose can
    # create false 3 cm lifts for untouched objects.
    initial_heights = {}
    for code, view in object_views.items():
        positions, _ = view.get_world_poses(usd=False)
        positions = (
            positions.detach().cpu().numpy()
            if hasattr(positions, "detach")
            else np.asarray(positions)
        )
        initial_heights[code] = positions[:, 2].copy()

    steps = list(manifest["waypoint_steps"])
    for waypoint in range(1, 5):
        print(
            "[DexGraspNet2/IsaacSim5] executing waypoint {}/4 ({} steps)".format(
                waypoint, int(steps[waypoint - 1])
            ),
            flush=True,
        )
        begin = targets[:, waypoint - 1]
        finish = targets[:, waypoint]
        for step in range(int(steps[waypoint - 1])):
            alpha = float(step + 1) / float(steps[waypoint - 1])
            target = begin + (finish - begin) * alpha
            for _ in range(physics_substeps_per_control):
                tick_start = time.perf_counter()
                hand.set_joint_position_target(target)
                hand.write_data_to_sim()
                sim.step()
                hand.update(sim_cfg.dt)
                if ARGS.realtime and not ARGS.headless:
                    remaining = sim_cfg.dt - (time.perf_counter() - tick_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
        if waypoint == 3:
            set_gravity(sim.stage, 9.81)
            sim.forward()

    lifted = np.zeros(count, dtype=bool)
    lifted_by_object = {}
    lift_delta_by_object = {}
    final_heights = {}
    for code, view in object_views.items():
        positions, _ = view.get_world_poses(usd=False)
        positions = positions.detach().cpu().numpy() if hasattr(positions, "detach") else np.asarray(positions)
        final_heights[code] = positions[:, 2]
        lift_delta = positions[:, 2] - initial_heights[code]
        lift_delta_by_object[code] = lift_delta
        object_lifted = lift_delta > 0.03
        lifted_by_object[code] = object_lifted
        lifted |= object_lifted
    pregrasp_valid = np.asarray(job["pregrasp_valid"][start:end], dtype=bool)
    successes = lifted & pregrasp_valid
    target_lifted = (
        lifted_by_object[target_object_code]
        if target_object_code in lifted_by_object
        else np.zeros(count, dtype=bool)
    )
    target_successes = target_lifted & pregrasp_valid

    output_path = ARGS.output
    if output_path is None:
        output_path = job_path.with_name("sim_success_{:04d}_{:04d}.npy".format(start, end))
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, successes)
    report = {
        "schema_version": 1,
        "simulator": "Isaac Sim 5.0 + Isaac Lab 2.2",
        "robot": "paper LEAP Hand",
        "source_job": str(job_path),
        "grasp_slice": [start, end],
        "success_count": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "successes": successes.tolist(),
        "official_success_rule": "any scene object rises more than 0.03 m",
        "target_object_code_from_seed_segmentation": target_object_code,
        "target_successes": target_successes.tolist(),
        "lifted_before_collision_filter": lifted.tolist(),
        "lifted_by_object": {
            key: value.tolist() for key, value in lifted_by_object.items()
        },
        "pregrasp_valid": pregrasp_valid.tolist(),
        "final_object_heights": {key: value.tolist() for key, value in final_heights.items()},
        "initial_object_heights_from_sim": {
            key: value.tolist() for key, value in initial_heights.items()
        },
        "lift_delta_by_object": {
            key: value.tolist() for key, value in lift_delta_by_object.items()
        },
        "physics_material_audit": material_audit,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote {}".format(output_path))
    print("wrote {}".format(report_path))
    print("successes={}/{}".format(int(successes.sum()), count))
    print(
        "target {} successes={}/{}".format(
            target_object_code, int(target_successes.sum()), count
        )
    )

    if ARGS.hold and not ARGS.headless:
        while SIMULATION_APP.is_running():
            sim.render()


exit_code = 0
try:
    main()
except BaseException:
    exit_code = 1
    traceback.print_exc()
finally:
    shutdown_finished = threading.Event()

    def force_exit_after_timeout() -> None:
        if not shutdown_finished.wait(15.0):
            print(
                "[DexGraspNet2/IsaacSim5] shutdown exceeded 15 s; forcing process exit",
                flush=True,
            )
            os._exit(exit_code)

    if ARGS.headless:
        threading.Thread(target=force_exit_after_timeout, daemon=True).start()
    SIMULATION_APP.close(wait_for_replicator=False)
    shutdown_finished.set()

if exit_code:
    raise SystemExit(exit_code)
