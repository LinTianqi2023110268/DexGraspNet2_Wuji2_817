#!/usr/bin/env python3
"""Run one monitored RGB-D-to-pick-and-place cycle in one Isaac Lab session.

The perception/network/retargeting products are frozen, audited inputs.  This
program owns one AppLauncher and one SimulationContext, creates the six dynamic
scene objects, and executes the selected ashtray candidate continuously:

HOME -> PREGRASP -> COVER -> GRASP -> SQUEEZE -> LIFT -> TRANSFER -> PLACE
     -> RELEASE -> RETREAT -> HOME.

No arm or hand root is teleported after physics starts.  The right arm uses the
validated ft04 implicit Force Drive, and Wuji2 keeps its official USD drives.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ISAACLAB_CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
sys.path.insert(0, str(ISAACLAB_CONTROL_ROOT))

from core.bridge import CuroboWorkerClient
from core.config import DEFAULT_INITIAL_RIGHT_ARM_DEG
from core.runtime_math import pose_from_position_quaternion_wxyz, rebase_pick_waypoints


DEFAULT_CONFIG = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/runtime/config/full_pick_place.json"
DEFAULT_CASE = (
    PROJECT_ROOT
    / "06_leap_to_wuji2_final_pipeline/01_cases/"
    "live_dynamic_scene0000_dog_candidate3800"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--dry-run", action="store_true", help="Stop after loading and auditing inputs.")
    parser.add_argument(
        "--stop-after-lift", action="store_true",
        help="Run the monitored pipeline only through LIFT_HOLD and validate real object lift.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_arguments()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.physx.scripts import physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# Import the already validated robot setup instead of duplicating it.
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

from placement_allocator import commit_placement, oriented_surface_offsets  # noqa: E402


def load_short_motion_helpers():
    """Load pure helper definitions without constructing another AppLauncher."""
    path = Path(__file__).with_name("01_run_short_motion.py")
    text = path.read_text(encoding="utf-8")
    marker = "ARGS = parse_arguments()"
    if marker not in text:
        raise RuntimeError("short-motion helper contract changed")
    # Reusing that module directly would start a second Kit application.  Keep
    # the full-flow runner independent and duplicate only the tiny audited
    # force-gain algorithm below instead.
    return None


TRACE_FIELDS = [
    "time_s", "state", "progress", "flange_error_mm", "flange_error_deg",
    "wuji2_wrist_error_mm", "wuji2_wrist_error_deg",
    "target_object_x_m", "target_object_y_m", "target_object_z_m",
    "object_lift_mm", "object_follow_error_mm", "max_arm_qdot_rad_s",
    "max_arm_joint_goal_error_deg", "max_wuji2_joint_target_error_deg",
    "target_contact_finger_count", "target_contact_force_max_n",
    "thumb_target_force_n", "index_target_force_n", "middle_target_force_n",
    "ring_target_force_n", "pinky_target_force_n", "palm_target_force_n",
    "thumb_any_force_n", "index_any_force_n", "middle_any_force_n",
    "ring_any_force_n", "pinky_any_force_n", "palm_any_force_n",
]

CONTACT_GROUPS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def contact_group_for_body(body_name: str) -> str | None:
    """Assign every physical Wuji2 body to one readable contact group."""
    if body_name.startswith("r_thumb"):
        return "thumb"
    if body_name.startswith("r_index_finger"):
        return "index"
    if body_name.startswith("r_middle_finger"):
        return "middle"
    if body_name.startswith("r_ring_finger"):
        return "ring"
    if body_name.startswith("r_pinky"):
        return "pinky"
    if body_name in {"r_wrist", "r_base_link"}:
        return "palm"
    return None


def load_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_demo_view(config: dict) -> None:
    """Hide engineering helpers and place the user viewport in a stable table view."""
    view = config.get("viewer_camera", {})
    if not bool(view.get("enabled", True)) or bool(getattr(ARGS, "headless", False)):
        return
    stage = get_current_stage()
    for path in view.get(
        "hide_prims",
        ["/World/Sensors/TopD435iVirtual/Frustum", "/World/Markers"],
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    target = np.asarray(view.get("target_world_m", [0.0, -0.145, 0.50]), dtype=np.float64)
    yaw = math.radians(float(view.get("yaw_about_world_z_deg", -90.0)))
    distance = float(view.get("horizontal_distance_m", 1.45))
    eye = target + np.asarray(
        [distance * math.cos(yaw), distance * math.sin(yaw), float(view.get("height_above_target_m", 0.75))],
        dtype=np.float64,
    )
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    print(
        f"[VIEWPORT] yaw={math.degrees(yaw):.1f} deg; "
        f"eye={np.round(eye, 3).tolist()} -> target={np.round(target, 3).tolist()}"
    )


def run_offline_with_gui_heartbeat(label: str, function):
    """Keep Kit responsive while pure NumPy/Pinocchio work runs off the UI thread."""
    if bool(getattr(ARGS, "headless", False)):
        return function()
    print(f"[{label}] working; Isaac Sim GUI heartbeat remains active")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function)
        while not future.done():
            SIMULATION_APP.update()
            time.sleep(0.01)
        result = future.result()
    print(f"[{label}] complete in {time.perf_counter() - started:.2f} wall s")
    return result


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale]
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = [(matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale]
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = [(matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def set_reference_transform(stage: Usd.Stage, root_path: str, pose: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(root_path)
    transform = Gf.Matrix4d(1.0)
    quaternion = matrix_to_quaternion_wxyz(pose[:3, :3])
    transform.SetRotate(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:]))))
    transform.SetTranslate(Gf.Vec3d(*map(float, pose[:3, 3])))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)


def find_one_rigid_prim(stage: Usd.Stage, prefix: str) -> Usd.Prim:
    matches = [
        prim for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rigid body below {prefix}, got {[str(x.GetPath()) for x in matches]}")
    return matches[0]


def spawn_dynamic_objects(
    case_root: Path, source_world: np.ndarray,
) -> tuple[list[RigidObject], int, str]:
    """Reference six audited object USDs before the first simulation reset."""
    manifest_paths = sorted((case_root / "01_input").glob("scene_*_manifest.json"))
    if len(manifest_paths) != 1:
        raise RuntimeError(f"Expected one scene manifest, got {manifest_paths}")
    scene = load_json(manifest_paths[0])
    target_segmentation_id = int(load_json(case_root / "case.json")["target_segmentation_id"])
    stage = get_current_stage()
    static_root = "/World/Layout/TableAssembly/TestScene0000"
    if stage.GetPrimAtPath(static_root).IsValid():
        stage.RemovePrim(static_root)
    UsdGeom.Xform.Define(stage, "/World/TaskObjects")

    material_path = "/World/PhysicsMaterials/TaskObjects"
    sim_utils.spawn_rigid_body_material(
        material_path,
        sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    wrappers: list[RigidObject] = []
    root_paths: list[str] = []
    target_index = -1
    target_rigid_path = ""
    for index, record in enumerate(scene["objects"]):
        seg_id = int(record["segmentation_id"])
        root_path = f"/World/TaskObjects/Object_{seg_id:03d}"
        add_reference_to_stage(str(Path(record["simulation_usd"])), root_path)
        world_pose = source_world @ np.asarray(record["pose_world_object"], dtype=np.float64)
        set_reference_transform(stage, root_path, world_pose)
        rigid_prim = find_one_rigid_prim(stage, root_path)
        PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim).CreateDisableGravityAttr().Set(False)
        UsdPhysics.MassAPI.Apply(rigid_prim).CreateMassAttr().Set(0.1)
        sim_utils.bind_physics_material(root_path, material_path)
        wrappers.append(RigidObject(RigidObjectCfg(prim_path=str(rigid_prim.GetPath()), spawn=None)))
        root_paths.append(root_path)
        if seg_id == target_segmentation_id:
            target_index = index
            target_rigid_path = str(rigid_prim.GetPath())

    group_path = "/World/CollisionGroups/TaskObjects"
    group = UsdPhysics.CollisionGroup.Define(stage, group_path)
    group.CreateFilteredGroupsRel().AddTarget(group.GetPath())
    for root_path in root_paths:
        physicsUtils.add_collision_to_collision_group(stage, root_path, group_path)
    print(f"[SCENE] dynamic objects={len(wrappers)}, filtered object pairs={len(list(itertools.combinations(root_paths, 2)))}")
    if target_index < 0:
        raise RuntimeError(f"Target segmentation id {target_segmentation_id} is absent")
    return wrappers, target_index, target_rigid_path


def create_target_contact_sensors(
    stage: Usd.Stage, robot_root: str, target_rigid_path: str,
) -> dict[str, tuple[str, ContactSensor]]:
    """Read whole-hand contacts, grouped by five fingers and palm.

    ``force_matrix_w`` is filtered to the selected target object.  The sensor's
    ``net_forces_w`` is retained separately so that target contact can be
    distinguished from accidental table/scene contact without changing physics.
    """
    sensors: dict[str, tuple[str, ContactSensor]] = {}
    grouped_paths = {group: [] for group in CONTACT_GROUPS}
    for prim in stage.Traverse():
        body_path = str(prim.GetPath())
        if not body_path.startswith(robot_root + "/") or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        group = contact_group_for_body(prim.GetName())
        if group is None:
            continue
        grouped_paths[group].append(body_path)
        sim_utils.activate_contact_sensors(body_path, threshold=0.0, stage=stage)
        sensors[f"{group}:{prim.GetName()}"] = (group, ContactSensor(ContactSensorCfg(
            prim_path=body_path,
            update_period=0.0,
            history_length=0,
            track_pose=False,
            track_air_time=False,
            filter_prim_paths_expr=[target_rigid_path],
        )))
    missing = [group for group, paths in grouped_paths.items() if not paths]
    if missing:
        raise RuntimeError(f"Missing Wuji2 contact body groups: {missing}")
    print(f"[CONTACT AUDIT] whole Wuji2 hand -> {target_rigid_path}")
    for group, paths in grouped_paths.items():
        print(f"  {group}: {[Path(path).name for path in paths]}")
    return sensors


def set_force_drive_type(config: dict) -> None:
    stage = get_current_stage()
    requested = set(config["right_arm_joints"])
    found = set()
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(config["robot_prim"] + "/"):
            continue
        if prim.GetName() not in requested:
            continue
        drive = UsdPhysics.DriveAPI(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        found.add(prim.GetName())
    if found != requested:
        raise RuntimeError(f"Force Drive mapping failed: missing={sorted(requested - found)}")


def create_robot(config: dict) -> Articulation:
    from isaaclab.actuators import ImplicitActuatorCfg
    actuators = {
        "native_left_and_wuji2": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_.*", "r_.*"], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )
    }
    for group in config["right_arm_force_natural_frequency_groups"]:
        actuators[group["name"]] = ImplicitActuatorCfg(
            joint_names_expr=group["joint_names_expr"], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )
    return Articulation(ArticulationCfg(prim_path=config["robot_prim"], spawn=None, actuators=actuators))


def apply_ft04_gains(robot: Articulation, arm_ids: list[int], config: dict) -> list[dict]:
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[0].to(robot.device)
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    audit = []
    for group in config["right_arm_force_natural_frequency_groups"]:
        ids, names = robot.find_joints(group["joint_names_expr"], preserve_order=True)
        if any(int(index) not in arm_ids for index in ids):
            raise RuntimeError(f"ft04 group matched non-arm DOF: {names}")
        frequency = float(group["natural_frequency_rad_s"])
        zeta = float(group.get("damping_ratio", 1.0))
        equivalent_mass = torch.clamp(torch.diag(mass_matrix)[ids], min=1.0e-6)
        stiffness[0, ids] = equivalent_mass * frequency * frequency
        damping[0, ids] = 2.0 * zeta * frequency * equivalent_mass
        audit.append({"names": names, "nf": frequency})
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.reset()
    return audit


def quintic(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def quat_error_deg(actual: torch.Tensor, target: torch.Tensor) -> float:
    dot = torch.abs(torch.sum(actual * target)).clamp(0.0, 1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot)))


def pose_errors(body_pose: torch.Tensor, target_matrix: np.ndarray | None) -> tuple[float, float]:
    """Return world-pose translation/orientation error for one rigid body."""
    if target_matrix is None:
        return 0.0, 0.0
    target_position = torch.as_tensor(
        target_matrix[:3, 3], device=body_pose.device, dtype=body_pose.dtype
    )
    target_quaternion = torch.as_tensor(
        matrix_to_quaternion_wxyz(target_matrix[:3, :3]),
        device=body_pose.device, dtype=body_pose.dtype,
    )
    position_error_mm = 1000.0 * float(torch.linalg.vector_norm(body_pose[:3] - target_position))
    return position_error_mm, quat_error_deg(body_pose[3:7], target_quaternion)


def run() -> Path:
    config = load_json(ARGS.config)
    case_root = ARGS.case_root.resolve()
    case_metadata = load_json(case_root / "case.json")
    stage_path = project_path(config["stage"]).resolve()
    if not stage_utils.open_stage(str(stage_path)):
        raise RuntimeError(f"Cannot open stage: {stage_path}")
    configure_demo_view(config)

    print("\n[PIPELINE INPUT AUDIT]")
    capture_root = project_path(
        config.get("capture_root", case_metadata.get("live_capture_root", ""))
    ).resolve()
    target_key = str(config.get("target_key", "")).strip()
    if not target_key:
        raise RuntimeError("Config must declare target_key for auditable live-perception inputs")
    frozen_inputs = sorted((case_root / "01_input").glob("view_*_network_input.npz"))
    if not frozen_inputs:
        frozen_inputs = sorted((case_root / "01_input").glob("live_*_network_input.npz"))
    if len(frozen_inputs) != 1:
        raise RuntimeError(f"Expected one frozen network input, got {frozen_inputs}")
    case_prediction = case_root / "01_input/official_leap_1024.npz"
    grounded_sam_result = capture_root / "grounded_sam" / target_key / "result.json"
    grounded_sam_mask = capture_root / "grounded_sam" / target_key / "mask.npy"
    target_dgn2_root = capture_root / "dgn2" / target_key
    live_perception = grounded_sam_result.is_file()
    required_inputs = {
        "frozen RGB-D/40K input": frozen_inputs[0],
        "DGN2 prediction": case_prediction if case_prediction.is_file() else target_dgn2_root / "official_leap_1024_target_ranked.npz",
        "selected DGN2 candidate metadata": case_root / "case.json",
        "retargeted waypoints": case_root / "06_isaacsim/final_waypoints.npz",
        "Cartesian arm target plan (legacy q ignored)": case_root / "07_arm_execution/full_arm_waypoint_ik.npz",
        "depth_m": capture_root / "depth_m.npy",
        "intrinsics": capture_root / "intrinsics.npy",
        "T_world_camera": capture_root / "T_world_camera.npy",
        "GroundedSAM target mask": grounded_sam_mask,
    }
    if "live_" in frozen_inputs[0].name and live_perception:
        required_inputs["GroundedSAM"] = grounded_sam_result
    for label, path in required_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"  [PASS] {label}: {path}")
    layout = load_json(PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json")
    with np.load(case_root / "07_arm_execution/arm_flange_targets.npz", allow_pickle=False) as archive:
        source_world = np.asarray(archive["world_from_source_zone"], dtype=np.float64)
    objects, target_object_index, target_rigid_path = spawn_dynamic_objects(case_root, source_world)
    set_force_drive_type(config)

    simulation = SimulationContext(sim_utils.SimulationCfg(
        dt=float(config["physics_dt_s"]), render_interval=int(config["render_interval"]), device=ARGS.device,
    ))
    robot = create_robot(config)
    contact_sensors = create_target_contact_sensors(
        get_current_stage(), config["robot_prim"], target_rigid_path
    )
    simulation.reset()
    dt = float(config["physics_dt_s"])
    robot.update(dt)
    for obj in objects:
        obj.update(dt)
    for _, sensor in contact_sensors.values():
        sensor.reset()

    if robot.num_joints != int(config["expected_total_actuated_joints"]) or not robot.is_fixed_base:
        raise RuntimeError(f"Robot audit failed: joints={robot.num_joints}, fixed={robot.is_fixed_base}")
    arm_ids, arm_names = robot.find_joints(config["right_arm_joints"], preserve_order=True)
    flange_ids, flange_names = robot.find_bodies([config["flange_body"]], preserve_order=True)
    wrist_ids, wrist_names = robot.find_bodies([config["wuji2_wrist_body"]], preserve_order=True)
    if (
        arm_names != config["right_arm_joints"]
        or flange_names != [config["flange_body"]]
        or wrist_names != [config["wuji2_wrist_body"]]
    ):
        raise RuntimeError("Right-arm/flange/Wuji2-wrist mapping changed")
    flange_id = int(flange_ids[0])
    wrist_id = int(wrist_ids[0])
    gain_audit = apply_ft04_gains(robot, [int(x) for x in arm_ids], config)

    with np.load(case_root / "07_arm_execution/full_arm_waypoint_ik.npz", allow_pickle=False) as archive:
        stage_names = [str(x) for x in archive["waypoint_names"]]
        flange_targets = np.asarray(archive["world_from_right_flange"], dtype=np.float32)
    initial_arm_q = np.deg2rad(np.asarray(DEFAULT_INITIAL_RIGHT_ARM_DEG, dtype=np.float32))
    with np.load(case_root / "07_arm_execution/arm_flange_targets.npz", allow_pickle=False) as archive:
        pick_wrist_targets = np.asarray(archive["world_from_wuji2_wrist"], dtype=np.float32)
        source_world = np.asarray(archive["world_from_source_zone"], dtype=np.float64)
        flange_from_wrist = np.asarray(archive["flange_from_wuji2_wrist"], dtype=np.float64)
    if pick_wrist_targets.shape[0] != 5:
        raise RuntimeError(
            f"Expected five pick-stage Wuji2 wrist targets, got {pick_wrist_targets.shape}"
        )
    if flange_targets.shape[0] != len(stage_names):
        raise RuntimeError(
            f"Flange target/name count mismatch: {flange_targets.shape[0]} vs {len(stage_names)}"
        )
    # The pick file intentionally contains only PREGRASP..LIFT.  The complete
    # arm plan also contains TRANSFER/PLACE/RELEASE/RETREAT.  Recover those
    # Wuji2 wrist monitor targets from the fixed assembly transform instead of
    # indexing past the five-row pick array or inventing a new pose.
    wrist_targets = np.asarray(
        flange_targets, dtype=np.float64
    ) @ flange_from_wrist[None]
    wrist_targets[:5] = pick_wrist_targets
    wrist_targets = wrist_targets.astype(np.float32)
    with np.load(case_root / "06_isaacsim/final_waypoints.npz", allow_pickle=False) as archive:
        hand_names = [str(x) for x in archive["finger_joint_names"]]
        hand_stage_names = [str(x) for x in archive["waypoint_names"]]
        hand_q5 = np.asarray(archive["waypoint_joint_positions"][0], dtype=np.float32)
        squeeze_dense = np.asarray(archive["squeeze_dense_q20_path"], dtype=np.float32)
    hand_index = {name: index for index, name in enumerate(hand_stage_names)}
    hand_ids, matched_hand_names = robot.find_joints(hand_names, preserve_order=True)
    if matched_hand_names != hand_names or len(hand_ids) != 20:
        raise RuntimeError("Wuji2 20-DOF mapping changed")

    joint_state = robot.data.joint_pos.clone()
    joint_state[:, arm_ids] = torch.as_tensor(initial_arm_q, device=robot.device).reshape(1, 7)
    joint_velocity = torch.zeros_like(joint_state)
    robot.write_joint_state_to_sim(joint_state, joint_velocity)
    robot.reset()
    command = joint_state.clone()
    robot.set_joint_position_target(command)

    def step() -> None:
        robot.set_joint_position_target(command)
        robot.write_data_to_sim()
        for dynamic_object in objects:
            dynamic_object.write_data_to_sim()
        simulation.step()
        robot.update(dt)
        for dynamic_object in objects:
            dynamic_object.update(dt)
        for _, sensor in contact_sensors.values():
            sensor.update(dt, force_recompute=True)

    def contact_forces() -> tuple[dict[str, float], dict[str, float]]:
        target = {group: 0.0 for group in CONTACT_GROUPS}
        any_contact = {group: 0.0 for group in CONTACT_GROUPS}
        for group, sensor in contact_sensors.values():
            target_matrix = sensor.data.force_matrix_w
            target[group] = max(
                target[group], float(torch.linalg.vector_norm(target_matrix, dim=-1).max())
            )
            net_force = sensor.data.net_forces_w
            any_contact[group] = max(
                any_contact[group], float(torch.linalg.vector_norm(net_force, dim=-1).max())
            )
        return target, any_contact

    print(
        f"\n[AUDIT PASS] fixed-base 35-DOF robot; right arm={arm_names}; Wuji2=20 DOF; "
        f"flange={flange_names[0]}; wrist={wrist_names[0]}"
    )
    print(f"[CONTROL] ft04 implicit Force Drive={gain_audit}; official Wuji2 drives unchanged")
    print("[STATE] INITIAL_SETTLE")
    for _ in range(round(float(config["initial_hold_s"]) / dt)):
        step()

    # Real operation captures RGB-D only after gravity has settled the scene.
    # Cached predictions retain a valid object-relative grasp, so carry that
    # grasp with the measured rigid-body delta and re-solve the seven arm joints.
    manifest_path = next((case_root / "01_input").glob("scene_*_manifest.json"))
    scene_manifest = load_json(manifest_path)
    target_segmentation_id = int(load_json(case_root / "case.json")["target_segmentation_id"])
    target_record = next(
        record for record in scene_manifest["objects"]
        if int(record["segmentation_id"]) == target_segmentation_id
    )
    object_before_settle = source_world @ np.asarray(
        target_record["pose_world_object"], dtype=np.float64
    )
    target_object = objects[target_object_index]
    object_after_settle = pose_from_position_quaternion_wxyz(
        target_object.data.root_pos_w[0].cpu().numpy(),
        target_object.data.root_quat_w[0].cpu().numpy(),
    )
    wrist_targets[:5], object_delta = rebase_pick_waypoints(
        wrist_targets[:5], object_before_settle, object_after_settle
    )
    flange_targets[:5] = wrist_targets[:5] @ np.linalg.inv(flange_from_wrist)[None]
    q_current = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
    measured_joint_state = {
        str(name): float(value)
        for name, value in zip(
            robot.joint_names,
            robot.data.joint_pos[0].detach().cpu().numpy(),
        )
    }
    hand_state_for_stage = {
        "pregrasp": "pregrasp", "cover": "cover", "grasp": "grasp",
        "squeeze": "squeeze", "lift": "squeeze", "transfer": "squeeze",
        "place": "squeeze", "release": "pregrasp", "retreat": "pregrasp",
    }
    phase_for_stage = {
        "pregrasp": "pregrasp", "cover": "cover", "grasp": "grasp",
        "squeeze": "squeeze", "lift": "lift", "transfer": "lift",
        "place": "lift", "release": "lift", "retreat": "lift",
    }
    collision_joint_states = []
    collision_phases = []
    for stage_name in stage_names:
        if stage_name not in hand_state_for_stage or stage_name not in phase_for_stage:
            raise RuntimeError(f"Route-C V2 has no phase/hand collision policy for {stage_name}")
        named = dict(measured_joint_state)
        hand_stage = hand_state_for_stage[stage_name]
        for name, value in zip(hand_names, hand_q5[hand_index[hand_stage]]):
            named[name] = float(value)
        collision_joint_states.append(named)
        collision_phases.append(phase_for_stage[stage_name])

    world_from_base = np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T
    base_from_world = np.linalg.inv(world_from_base)
    flange_targets_base = np.stack(
        [base_from_world @ np.asarray(target, dtype=np.float64) for target in flange_targets]
    )

    def solve_route_c_v2():
        with CuroboWorkerClient(PROJECT_ROOT) as client:
            map_report = client.build_map(
                required_inputs["depth_m"],
                required_inputs["intrinsics"],
                required_inputs["T_world_camera"],
                required_inputs["GroundedSAM target mask"],
            )
            solve_report = client.solve_ik(
                flange_targets_base,
                q_current,
                select_chain=True,
                collision_context={
                    "phases": collision_phases,
                    "joint_positions_by_target": collision_joint_states,
                    "T_world_base": world_from_base,
                    "margin_m": 0.0,
                },
            )
        return map_report, solve_report

    map_report, route_c_v2_plan = run_offline_with_gui_heartbeat(
        "ROUTE-C V2 GPU IK + OBSERVED ESDF", solve_route_c_v2
    )
    if route_c_v2_plan["selected"] is None:
        raise RuntimeError(
            "Route-C V2 found no continuous collision-feasible IK chain: "
            f"ik={route_c_v2_plan['ik_accepted_per_target']}, "
            f"collision={route_c_v2_plan['accepted_per_target']}"
        )
    arm_q = np.asarray(
        [item["q_rad"] for item in route_c_v2_plan["selected"]], dtype=np.float32
    )
    if len(arm_q) != len(stage_names):
        raise RuntimeError(f"Route-C V2 arm target count mismatch: {len(arm_q)} != {len(stage_names)}")
    runtime_ik_report = []
    for stage_name, selected, collision in zip(
        stage_names, route_c_v2_plan["selected"], route_c_v2_plan["selected_collision"]
    ):
        runtime_ik_report.append({
            "stage": stage_name,
            "position_error_mm": 1000.0 * float(selected["position_error_m"]),
            "orientation_error_deg": math.degrees(float(selected["orientation_error_rad"])),
            "minimum_limit_margin_deg": math.degrees(float(selected["inner_limit_margin_rad"])),
            "observed_scene_collision_pass": bool(collision["observed_scene_collision_pass"]),
            "unknown_space_exposure": bool(collision["unknown_space_exposure"]),
            "unknown_sphere_count": int(collision["unknown_sphere_count"]),
        })
    ik_pass = bool(route_c_v2_plan["ik_pass"] and route_c_v2_plan["selected"] is not None)
    observed_scene_collision_pass = bool(route_c_v2_plan["observed_scene_collision_pass"])
    unknown_space_exposure = bool(any(route_c_v2_plan["unknown_space_exposure"]))
    unknown_space_safe_enough = not unknown_space_exposure
    path_pass = None

    planning_output = project_path(config["output_directory"]).resolve()
    planning_output.mkdir(parents=True, exist_ok=True)
    route_c_v2_planning_path = planning_output / "route_c_v2_planning.json"
    route_c_v2_planning_path.write_text(json.dumps({
        "schema_version": 1,
        "production_planner": "core Route-C V2 cuRobo GPU IK + observed Mapper/TSDF/ESDF",
        "q_current_rad": q_current.tolist(),
        "stage_names": stage_names,
        "map": map_report,
        "solve": route_c_v2_plan,
        "statuses": {
            "ik_pass": ik_pass,
            "observed_scene_collision_pass": observed_scene_collision_pass,
            "unknown_space_exposure": unknown_space_exposure,
            "unknown_space_safe_enough": unknown_space_safe_enough,
            "path_pass": path_pass,
            "physics_grasp_pass": None,
        },
        "limitations": [
            "single-view unknown/unobserved/occluded space is not certified free",
            "target contact allowance is phase-wide for GRASP/SQUEEZE/LIFT, not yet finger-link-specific",
            "continuous-path observed collision is not yet evaluated; path_pass is null",
        ],
    }, indent=2) + "\n", encoding="utf-8")

    print("[STATE] SETTLED_SCENE_RELOCALIZED")
    print(f"  object delta xyz mm={np.round(1000.0 * object_delta[:3, 3], 3).tolist()}")
    print(f"  q_current deg={np.degrees(q_current).round(3).tolist()}")
    for name, item in zip(stage_names, runtime_ik_report):
        print(
            f"  {name}: {item['position_error_mm']:.3f} mm / "
            f"{item['orientation_error_deg']:.3f} deg; "
            f"limit margin={item['minimum_limit_margin_deg']:.2f} deg; "
            f"observed_collision={item['observed_scene_collision_pass']}; "
            f"unknown={item['unknown_space_exposure']}"
        )

    if ARGS.dry_run:
        print(f"[DRY RUN COMPLETE] {route_c_v2_planning_path}")
        return route_c_v2_planning_path

    actual_arm_start = robot.data.joint_pos[:, arm_ids].clone()
    static_bias = command[:, arm_ids].clone() - actual_arm_start
    print("[STATIC BIAS DEG]", torch.rad2deg(static_bias).cpu().numpy().round(4).tolist()[0])
    current_desired_arm = actual_arm_start.clone()
    initial_object_position = target_object.data.root_pos_w[0].clone()

    trace: list[dict] = []
    contact_force_peaks = {group: 0.0 for group in CONTACT_GROUPS}
    any_contact_force_peaks = {group: 0.0 for group in CONTACT_GROUPS}
    simulation_time = 0.0
    action_wall_start = time.perf_counter()
    action_limit_s = config.get("action_duration_limit_s")
    print(
        "\n[ACTION TIMER START] simulation=0.00 s"
        + ("" if action_limit_s is None else f" | limit={float(action_limit_s):.2f} s")
    )
    telemetry_stride = max(1, round(1.0 / (float(config["telemetry_hz"]) * dt)))
    replay_fps = float(config.get("replay_record_fps", 30.0))
    replay_period_s = 1.0 / replay_fps
    next_replay_capture_s = 0.0
    replay_time_s: list[float] = []
    replay_state: list[str] = []
    replay_joint_position_rad: list[np.ndarray] = []
    replay_object_pose_world_wxyz: list[np.ndarray] = []
    previous_flange_target = None
    previous_object_target = None

    def capture_replay_frame(state: str, force: bool = False) -> None:
        """Sample actual simulation state without influencing the controller.

        The physical solver still runs at 120 Hz.  This side-channel stores
        only 30 display frames per simulated second for deterministic replay.
        """
        nonlocal next_replay_capture_s
        if not force and simulation_time + 0.5 * dt < next_replay_capture_s:
            return
        replay_time_s.append(float(simulation_time))
        replay_state.append(str(state))
        replay_joint_position_rad.append(
            robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).copy()
        )
        replay_object_pose_world_wxyz.append(np.stack([
            torch.cat((obj.data.root_pos_w[0], obj.data.root_quat_w[0]))
            .detach().cpu().numpy().astype(np.float32)
            for obj in objects
        ]))
        next_replay_capture_s += replay_period_s

    def flange_errors(target_matrix: np.ndarray | None) -> tuple[float, float]:
        return pose_errors(robot.data.body_pose_w[0, flange_id], target_matrix)

    def wrist_errors(target_matrix: np.ndarray | None) -> tuple[float, float]:
        return pose_errors(robot.data.body_pose_w[0, wrist_id], target_matrix)

    def monitor(
        state: str, progress: float, target_matrix: np.ndarray | None,
        step_index: int, wrist_target_matrix: np.ndarray | None = None,
    ) -> tuple[float, float]:
        nonlocal previous_flange_target, previous_object_target
        object_position = target_object.data.root_pos_w[0]
        target_forces, any_forces = contact_forces()
        for group, force in target_forces.items():
            contact_force_peaks[group] = max(contact_force_peaks[group], force)
        for group, force in any_forces.items():
            any_contact_force_peaks[group] = max(any_contact_force_peaks[group], force)
        contact_count = sum(target_forces[group] > 1.0e-3 for group in CONTACT_GROUPS[:-1])
        contact_force_max = max(target_forces.values(), default=0.0)
        position_error_mm, orientation_error = flange_errors(target_matrix)
        wrist_error_mm, wrist_orientation_error = wrist_errors(wrist_target_matrix)
        object_lift = 1000.0 * float(object_position[2] - initial_object_position[2])
        follow_error = 0.0
        if previous_flange_target is not None and previous_object_target is not None:
            follow_error = 1000.0 * float(torch.linalg.vector_norm(object_position - previous_object_target))
        row = {
            "time_s": simulation_time, "state": state, "progress": progress,
            "flange_error_mm": position_error_mm, "flange_error_deg": orientation_error,
            "wuji2_wrist_error_mm": wrist_error_mm,
            "wuji2_wrist_error_deg": wrist_orientation_error,
            "target_object_x_m": float(object_position[0]), "target_object_y_m": float(object_position[1]),
            "target_object_z_m": float(object_position[2]), "object_lift_mm": object_lift,
            "object_follow_error_mm": follow_error,
            "max_arm_qdot_rad_s": float(torch.max(torch.abs(robot.data.joint_vel[:, arm_ids]))),
            "max_arm_joint_goal_error_deg": float(
                torch.max(torch.abs(torch.rad2deg(current_desired_arm - robot.data.joint_pos[:, arm_ids])))
            ),
            "max_wuji2_joint_target_error_deg": float(
                torch.max(torch.abs(torch.rad2deg(command[:, hand_ids] - robot.data.joint_pos[:, hand_ids])))
            ),
            "target_contact_finger_count": contact_count,
            "target_contact_force_max_n": contact_force_max,
            **{f"{group}_target_force_n": force for group, force in target_forces.items()},
            **{f"{group}_any_force_n": force for group, force in any_forces.items()},
        }
        trace.append(row)
        capture_replay_frame(state)
        if step_index % telemetry_stride == 0:
            print(
                f"\r{state:<18} {100.0 * progress:6.1f}% | flange={position_error_mm:6.2f} mm/"
                f"{orientation_error:5.2f} deg | wrist={wrist_error_mm:6.2f} mm/"
                f"{wrist_orientation_error:5.2f} deg | object_lift={object_lift:7.2f} mm"
                f" | contact={contact_count}/5 max={contact_force_max:7.3f} N",
                end="", flush=True,
            )
        return position_error_mm, orientation_error

    def execute_segment(
        state: str, arm_goal: np.ndarray | None, hand_goal: np.ndarray | None,
        duration_s: float, target_matrix: np.ndarray | None,
    ) -> None:
        nonlocal simulation_time, previous_flange_target, previous_object_target, current_desired_arm
        print(f"\n\n>>> STATE: {state} | duration={duration_s:.2f}s")
        start_arm = command[:, arm_ids].clone()
        start_hand = command[:, hand_ids].clone()
        if arm_goal is None:
            goal_arm = start_arm
            goal_desired_arm = current_desired_arm.clone()
        else:
            goal_desired_arm = torch.as_tensor(
                arm_goal, device=robot.device, dtype=command.dtype
            ).reshape(1, 7)
            # Carry the latest equilibrium bias into the next smooth segment.
            # It will be re-estimated at the endpoint from actual joint feedback.
            current_bias = command[:, arm_ids] - current_desired_arm
            goal_arm = goal_desired_arm + current_bias
        goal_hand = start_hand if hand_goal is None else torch.as_tensor(
            hand_goal, device=robot.device, dtype=command.dtype
        ).reshape(1, 20)
        count = max(2, round(duration_s / dt))
        object_at_start = target_object.data.root_pos_w[0].clone()
        flange_at_start = robot.data.body_pose_w[0, flange_id, :3].clone()
        for index in range(count + 1):
            progress = index / count
            alpha = quintic(progress)
            command[:, arm_ids] = start_arm + alpha * (goal_arm - start_arm)
            command[:, hand_ids] = start_hand + alpha * (goal_hand - start_hand)
            step()
            simulation_time += dt
            current_flange = robot.data.body_pose_w[0, flange_id, :3]
            previous_flange_target = current_flange
            previous_object_target = object_at_start + (current_flange - flange_at_start)
            wrist_target = None if target_matrix is None else wrist_targets[index_of[state.lower().replace("to_", "")]] if state.lower().replace("to_", "") in index_of else None
            monitor(state, progress, target_matrix, index, wrist_target)
        current_desired_arm = goal_desired_arm
        print()

    def refine_arm_endpoint(
        state: str,
        desired_arm: np.ndarray,
        target_matrix: np.ndarray,
        position_limit_mm: float,
        orientation_limit_deg: float,
    ) -> None:
        """Remove pose-dependent steady-state offset without changing drive gains.

        The offline IK solution remains the desired *actual* joint state.  A
        bounded integral correction only adjusts the position-drive command so
        that gravity/contact load does not leave the arm short of that state.
        """
        nonlocal simulation_time, current_desired_arm
        desired = torch.as_tensor(desired_arm, device=robot.device, dtype=command.dtype).reshape(1, 7)
        current_desired_arm = desired
        bias = command[:, arm_ids].clone() - desired
        gain = float(config["endpoint_refinement"]["integral_gain_per_s"])
        max_bias = math.radians(float(config["endpoint_refinement"]["max_command_bias_deg"]))
        max_steps = round(float(config["endpoint_refinement"]["max_duration_s"]) / dt)
        stable_steps_required = round(float(config["endpoint_refinement"]["stable_duration_s"]) / dt)
        stable_steps = 0
        lower = robot.data.soft_joint_pos_limits[:, arm_ids, 0]
        upper = robot.data.soft_joint_pos_limits[:, arm_ids, 1]
        print(
            f"\n>>> STATE: {state} | closed-loop endpoint refinement; "
            f"gate={position_limit_mm:.1f} mm/{orientation_limit_deg:.1f} deg"
        )
        for index in range(max_steps):
            actual = robot.data.joint_pos[:, arm_ids]
            joint_error = desired - actual
            bias = torch.clamp(bias + gain * joint_error * dt, -max_bias, max_bias)
            command[:, arm_ids] = torch.maximum(torch.minimum(desired + bias, upper), lower)
            step()
            simulation_time += dt
            stage_key = state.lower().replace("_refine", "")
            position_error_mm, orientation_error_deg = monitor(
                state, (index + 1) / max_steps, target_matrix, index,
                wrist_targets[index_of[stage_key]],
            )
            if position_error_mm <= position_limit_mm and orientation_error_deg <= orientation_limit_deg:
                stable_steps += 1
                if stable_steps >= stable_steps_required:
                    print(
                        f"\n[ENDPOINT PASS] {state}: {position_error_mm:.2f} mm/"
                        f"{orientation_error_deg:.2f} deg; command bias(deg)="
                        f"{torch.rad2deg(bias).cpu().numpy().round(3).tolist()[0]}"
                    )
                    return
            else:
                stable_steps = 0
        position_error_mm, orientation_error_deg = flange_errors(target_matrix)
        raise RuntimeError(
            f"{state} endpoint failed {position_error_mm:.2f} mm/{orientation_error_deg:.2f} deg "
            f"against {position_limit_mm:.1f} mm/{orientation_limit_deg:.1f} deg gate"
        )

    def hold(state: str, duration_s: float, target_matrix: np.ndarray | None) -> None:
        nonlocal simulation_time
        print(f"\n>>> STATE: {state} | hold={duration_s:.2f}s")
        count = max(1, round(duration_s / dt))
        for index in range(count):
            step()
            simulation_time += dt
            stage_key = state.lower().replace("_hold", "").replace("_settle", "")
            wrist_target = wrist_targets[index_of[stage_key]] if stage_key in index_of else None
            monitor(state, (index + 1) / count, target_matrix, index, wrist_target)
        print()

    index_of = {name: index for index, name in enumerate(stage_names)}
    hand_index = {name: index for index, name in enumerate(hand_stage_names)}
    durations = config["durations_s"]

    capture_replay_frame("ACTION_START", force=True)

    execute_segment("FORM_PREGRASP", None, hand_q5[hand_index["pregrasp"]], durations["form_pregrasp"], None)
    execute_segment("TO_PREGRASP", arm_q[index_of["pregrasp"]], None, durations["to_pregrasp"], flange_targets[index_of["pregrasp"]])
    refine_arm_endpoint(
        "PREGRASP_REFINE", arm_q[index_of["pregrasp"]], flange_targets[index_of["pregrasp"]],
        float(config["stage_tolerances"]["pregrasp_position_mm"]),
        float(config["stage_tolerances"]["pregrasp_orientation_deg"]),
    )
    execute_segment("COVER", arm_q[index_of["cover"]], hand_q5[hand_index["cover"]], durations["cover"], flange_targets[index_of["cover"]])
    refine_arm_endpoint(
        "COVER_REFINE", arm_q[index_of["cover"]], flange_targets[index_of["cover"]],
        float(config["stage_tolerances"]["contact_position_mm"]),
        float(config["stage_tolerances"]["contact_orientation_deg"]),
    )
    execute_segment("GRASP", arm_q[index_of["grasp"]], hand_q5[hand_index["grasp"]], durations["grasp"], flange_targets[index_of["grasp"]])
    refine_arm_endpoint(
        "GRASP_REFINE", arm_q[index_of["grasp"]], flange_targets[index_of["grasp"]],
        float(config["stage_tolerances"]["contact_position_mm"]),
        float(config["stage_tolerances"]["contact_orientation_deg"]),
    )

    print("\n>>> STATE: SQUEEZE | dense retargeted 41-point hand path")
    squeeze_duration = float(durations["squeeze"])
    steps_each = max(1, round(squeeze_duration / (len(squeeze_dense) - 1) / dt))
    for path_index in range(1, len(squeeze_dense)):
        start_hand = command[:, hand_ids].clone()
        goal_hand = torch.as_tensor(squeeze_dense[path_index], device=robot.device, dtype=command.dtype).reshape(1, 20)
        for local in range(steps_each):
            alpha = quintic((local + 1) / steps_each)
            command[:, hand_ids] = start_hand + alpha * (goal_hand - start_hand)
            squeeze_arm = torch.as_tensor(
                arm_q[index_of["squeeze"]], device=robot.device, dtype=command.dtype
            ).reshape(1, 7)
            command[:, arm_ids] = squeeze_arm + (command[:, arm_ids] - current_desired_arm)
            current_desired_arm = squeeze_arm
            step(); simulation_time += dt
            monitor(
                "SQUEEZE",
                (path_index - 1 + (local + 1) / steps_each) / (len(squeeze_dense) - 1),
                flange_targets[index_of["squeeze"]],
                (path_index - 1) * steps_each + local,
                wrist_targets[index_of["squeeze"]],
            )
    print()
    refine_arm_endpoint(
        "SQUEEZE_REFINE", arm_q[index_of["squeeze"]], flange_targets[index_of["squeeze"]],
        float(config["stage_tolerances"]["squeeze_contact_position_mm"]),
        float(config["stage_tolerances"]["squeeze_contact_orientation_deg"]),
    )
    hold("SQUEEZE_HOLD", durations["squeeze_hold"], flange_targets[index_of["squeeze"]])
    execute_segment("LIFT", arm_q[index_of["lift"]], hand_q5[hand_index["squeeze"]], durations["lift"], flange_targets[index_of["lift"]])
    refine_arm_endpoint(
        "LIFT_REFINE", arm_q[index_of["lift"]], flange_targets[index_of["lift"]],
        float(config["stage_tolerances"]["contact_position_mm"]),
        float(config["stage_tolerances"]["contact_orientation_deg"]),
    )
    hold("LIFT_HOLD", durations["lift_hold"], flange_targets[index_of["lift"]])
    if not ARGS.stop_after_lift:
        execute_segment("TRANSFER", arm_q[index_of["transfer"]], None, durations["transfer"], flange_targets[index_of["transfer"]])
        execute_segment("PLACE", arm_q[index_of["place"]], None, durations["place"], flange_targets[index_of["place"]])
        hold("PLACE_SETTLE", durations["place_hold"], flange_targets[index_of["place"]])
        execute_segment("RELEASE", arm_q[index_of["release"]], hand_q5[hand_index["pregrasp"]], durations["release"], flange_targets[index_of["release"]])
        hold("RELEASE_SETTLE", durations["release_hold"], flange_targets[index_of["release"]])
        execute_segment("RETREAT", arm_q[index_of["retreat"]], None, durations["retreat"], flange_targets[index_of["retreat"]])
        execute_segment("RETURN_HOME", initial_arm_q, None, durations["return_home"], None)
        hold("FINAL_HOLD", durations["final_hold"], None)

    capture_replay_frame("ACTION_END", force=True)

    final_object_position = target_object.data.root_pos_w[0].detach().cpu().numpy()
    action_wall_duration_s = time.perf_counter() - action_wall_start
    print(
        f"\n[ACTION TIMER STOP] simulation={simulation_time:.2f} s | "
        f"wall={action_wall_duration_s:.2f} s"
        + ("" if action_limit_s is None else f" | limit={float(action_limit_s):.2f} s")
    )
    final_object_pose = pose_from_position_quaternion_wxyz(
        final_object_position,
        target_object.data.root_quat_w[0].detach().cpu().numpy(),
    )
    final_surface_points = np.load(
        case_root / "01_input" / f"object_{target_segmentation_id:03d}_surface_points.npy"
    )
    final_offset_min, final_offset_max = oriented_surface_offsets(
        final_surface_points, final_object_pose
    )
    final_footprint_min = final_object_position[:2] + final_offset_min[:2]
    final_footprint_max = final_object_position[:2] + final_offset_max[:2]
    placement = layout["transforms"]["placement_zone"]
    centre = np.asarray(placement["position_world_m"], dtype=np.float64)
    size = np.asarray(layout["geometry"]["placement_zone_size_m"], dtype=np.float64)
    zone_min = centre[:2] - 0.5 * size[:2]
    zone_max = centre[:2] + 0.5 * size[:2]
    in_green = bool(
        np.all(final_footprint_min >= zone_min)
        and np.all(final_footprint_max <= zone_max)
    )
    max_lift_mm = max(row["object_lift_mm"] for row in trace)
    action_duration_s = float(simulation_time)
    duration_limit_s = config.get("action_duration_limit_s")
    duration_pass = (
        True if duration_limit_s is None
        else action_duration_s <= float(duration_limit_s) + dt
    )
    passed = (
        max_lift_mm >= float(config["object_lift_pass_mm"])
        and (ARGS.stop_after_lift or in_green)
        and duration_pass
    )

    # The configuration owns the evidence directory.  Earlier versions rebuilt
    # the directory name from ``case_root`` here; consequently a slow baseline
    # and a report-speed run of the same candidate overwrote each other's
    # report/trace.  Keeping the configured directory makes the two experiments
    # auditable without changing any simulation or controller behavior.
    output = project_path(config["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader(); writer.writerows(trace)
    scene_manifest_path = sorted((case_root / "01_input").glob("scene_*_manifest.json"))[0]
    scene_manifest = load_json(scene_manifest_path)
    replay_path = output / "physical_replay_30fps.npz"
    replay_metadata = {
        "schema_version": 1,
        "stage": str(stage_path),
        "robot_prim": config["robot_prim"],
        "joint_names": list(robot.joint_names),
        "object_rigid_prim_paths": [str(obj.cfg.prim_path) for obj in objects],
        "objects": [
            {
                "segmentation_id": int(record["segmentation_id"]),
                "reference_root_path": f"/World/TaskObjects/Object_{int(record['segmentation_id']):03d}",
                "simulation_usd": str(Path(record["simulation_usd"]).resolve()),
            }
            for record in scene_manifest["objects"]
        ],
        "viewer_camera": config.get("viewer_camera", {}),
        "record_fps": replay_fps,
        "action_duration_s": float(simulation_time),
        "source_report": str(output / "report.json"),
    }
    np.savez_compressed(
        replay_path,
        time_s=np.asarray(replay_time_s, dtype=np.float32),
        state=np.asarray(replay_state),
        joint_position_rad=np.stack(replay_joint_position_rad),
        object_pose_world_wxyz=np.stack(replay_object_pose_world_wxyz),
        metadata_json=np.asarray(json.dumps(replay_metadata)),
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "mode": "through_lift_only" if ARGS.stop_after_lift else "full_pick_place",
        "target": (
            f"{load_json(case_root / 'case.json')['target_object_code']} segmentation "
            f"{int(load_json(case_root / 'case.json')['target_segmentation_id'])}"
        ),
        "candidate_index": int(load_json(case_root / "case.json")["source_candidate_index"]),
        "control": "Isaac Lab 2.2 + core Route-C V2 persistent cuRobo worker; ft04 right-arm Force Drive; official Wuji2 drives",
        "cached_pipeline_inputs": {key: str(value) for key, value in required_inputs.items()},
        "route_c_v2_planning_report": str(route_c_v2_planning_path),
        "ik_pass": ik_pass,
        "observed_scene_collision_pass": observed_scene_collision_pass,
        "unknown_space_exposure": unknown_space_exposure,
        "unknown_space_safe_enough": unknown_space_safe_enough,
        "path_pass": path_pass,
        "physics_grasp_pass": passed,
        "max_object_lift_mm": max_lift_mm,
        "action_duration_s": action_duration_s,
        "action_wall_duration_s": action_wall_duration_s,
        "action_duration_limit_s": duration_limit_s,
        "action_duration_pass": duration_pass,
        "final_object_position_world_m": final_object_position.tolist(),
        "final_object_footprint_world_xy_min_m": final_footprint_min.tolist(),
        "final_object_footprint_world_xy_max_m": final_footprint_max.tolist(),
        "final_object_inside_green_zone_xy": in_green,
        "target_contact_force_peak_n_by_finger": contact_force_peaks,
        "any_contact_force_peak_n_by_hand_group": any_contact_force_peaks,
        "trace_csv": str(trace_path),
        "physical_replay_30fps": str(replay_path),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if passed and not ARGS.stop_after_lift:
        ik_report = load_json(case_root / "07_arm_execution/full_arm_waypoint_ik_report.json")
        placement_plan = ik_report.get("placement_plan")
        if placement_plan is None:
            raise RuntimeError("Full-cycle PASS has no auditable placement plan")
        policy = load_json(Path(ik_report["placement_policy"]))
        if bool(policy.get("commit_on_physical_success", True)):
            registry_path = Path(placement_plan["occupancy_registry"])
            commit_placement(registry_path, {
                "placement_id": case_root.name,
                "case_root": str(case_root),
                "target_segmentation_id": target_segmentation_id,
                "target_object_code": load_json(case_root / "case.json")["target_object_code"],
                "object_root_world_m": final_object_pose[:3, 3].tolist(),
                "footprint_world_xy_min_m": final_footprint_min.tolist(),
                "footprint_world_xy_max_m": final_footprint_max.tolist(),
                "planned_free_slot_index": placement_plan["selected_free_slot_index"],
                "physical_report": str(report_path),
            })
            report["placement_registry_commit"] = str(registry_path)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"[PLACEMENT COMMITTED] {registry_path}")
    print(f"\n[FULL PIPELINE {report['status']}] {report_path}")
    print(
        f"max object lift={max_lift_mm:.2f} mm; final in green zone={in_green}; "
        f"action duration={action_duration_s:.2f} s"
        + ("" if duration_limit_s is None else f" / limit={float(duration_limit_s):.2f} s")
    )
    return report_path


def main() -> int:
    try:
        report_path = run()
        if ARGS.dry_run:
            return 0
        return 0 if load_json(report_path)["status"] == "PASS" else 2
    except Exception as error:
        print(f"\n[FULL PIPELINE ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        SIMULATION_APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
