#!/usr/bin/env python3
"""Settle the real dynamic test scene, then capture one aligned RGB-D frame.

This is the first stage of the live monitored pipeline.  The six objects use
the same editable USD assets and physics properties as grasp execution.  The
robot holds the validated ft04 initial posture while gravity remains enabled.
Only after both the objects and robot have settled is one RGB-D frame saved.

The output ``settled_scene_manifest.json`` is the authoritative geometry for
all later collision checking, retargeting, IK and physical execution.  This
prevents a cached pre-settle grasp from being applied to a post-settle object.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAYOUT_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout"
DEFAULT_STAGE = LAYOUT_ROOT / "scenes/manual_layout_calibrated_mass_fixed.usda"
DEFAULT_SCENE = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/"
    "wuji2_test60_10upright_10view_v1/scenes/scene_0000/scene_manifest.json"
)
DEFAULT_OUTPUT = LAYOUT_ROOT / "captures/live_dynamic_scene0000"
ROBOT_PRIM = "/World/Layout/DualArmMount/DualArm"
CAMERA_PRIM = "/World/Sensors/TopD435iVirtual/Camera"
TASK_ROOT = "/World/Layout/TableAssembly/TestScene0000"
SOURCE_ZONE_PRIM = "/World/Layout/TableAssembly/SourceZone"
RIGHT_ARM_NAMES = [f"arm_r_joint_{index}" for index in range(1, 8)]
INITIAL_ARM_Q_RAD = np.deg2rad([50.0, -70.0, 0.0, 40.0, 35.0, 0.0, 25.0])
FT04_NF = [50.0, 40.0, 45.0, 40.0, 55.0, 50.0, 40.0]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--scene-manifest", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settle-time-s", type=float, default=4.0)
    parser.add_argument("--settled-speed-m-s", type=float, default=0.01)
    parser.add_argument("--settled-angular-speed-rad-s", type=float, default=0.10)
    parser.add_argument(
        "--settle-only", action="store_true",
        help="Save settled object poses without initializing RTX Camera in this process.",
    )
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
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def gf_matrix_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.asarray([[float(matrix[row][column]) for column in range(4)] for row in range(4)])


def rigid_world_transform(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    row_matrix = gf_matrix_to_numpy(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    raw = row_matrix[:3, :3]
    scales = np.linalg.norm(raw, axis=1)
    normalized = raw / scales[:, None]
    u, _, vt = np.linalg.svd(normalized)
    rotation_row = u @ vt
    if np.linalg.det(rotation_row) < 0.0:
        u[:, -1] *= -1.0
        rotation_row = u @ vt
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_row.T
    result[:3, 3] = row_matrix[3, :3]
    return result


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = [
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            values = [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                      (matrix[0, 1] + matrix[1, 0]) / scale,
                      (matrix[0, 2] + matrix[2, 0]) / scale]
        elif index == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            values = [(matrix[0, 2] - matrix[2, 0]) / scale,
                      (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                      (matrix[1, 2] + matrix[2, 1]) / scale]
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            values = [(matrix[1, 0] - matrix[0, 1]) / scale,
                      (matrix[0, 2] + matrix[2, 0]) / scale,
                      (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values)


def set_reference_transform(stage: Usd.Stage, root_path: str, pose: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(root_path)
    quaternion = matrix_to_quaternion_wxyz(pose[:3, :3])
    transform = Gf.Matrix4d(1.0)
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
        raise RuntimeError(f"Expected one rigid body under {prefix}, got {len(matches)}")
    return matches[0]


def set_force_drive_type(stage: Usd.Stage) -> None:
    requested = set(RIGHT_ARM_NAMES)
    found = set()
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(ROBOT_PRIM + "/") or prim.GetName() not in requested:
            continue
        drive = UsdPhysics.DriveAPI(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        found.add(prim.GetName())
    if found != requested:
        raise RuntimeError(f"Right-arm Force Drive mapping failed: {sorted(requested - found)}")


def create_robot() -> Articulation:
    actuators = {
        "native_left_and_wuji2": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_.*", "r_.*"], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )
    }
    for index, name in enumerate(RIGHT_ARM_NAMES):
        actuators[f"ft04_j{index + 1}"] = ImplicitActuatorCfg(
            joint_names_expr=[name], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )
    return Articulation(ArticulationCfg(prim_path=ROBOT_PRIM, spawn=None, actuators=actuators))


def apply_ft04(robot: Articulation, arm_ids: list[int]) -> None:
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[0].to(robot.device)
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    for joint_id, frequency in zip(arm_ids, FT04_NF):
        equivalent_mass = torch.clamp(mass_matrix[joint_id, joint_id], min=1.0e-6)
        stiffness[0, joint_id] = equivalent_mass * frequency * frequency
        damping[0, joint_id] = 2.0 * frequency * equivalent_mass
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.reset()


def camera_calibration(stage: Usd.Stage, width: int, height: int) -> tuple[np.ndarray, np.ndarray, dict]:
    camera = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA_PRIM))
    focal_mm = float(camera.GetFocalLengthAttr().Get())
    horizontal_mm = float(camera.GetHorizontalApertureAttr().Get())
    vertical_mm = float(camera.GetVerticalApertureAttr().Get())
    clipping = camera.GetClippingRangeAttr().Get()
    intrinsic = np.asarray(
        [[focal_mm * width / horizontal_mm, 0.0, width / 2.0],
         [0.0, focal_mm * height / vertical_mm, height / 2.0],
         [0.0, 0.0, 1.0]], dtype=np.float64,
    )
    row = gf_matrix_to_numpy(
        UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = np.column_stack((row[0, :3], -row[1, :3], -row[2, :3]))
    world_from_camera[:3, 3] = row[3, :3]
    return intrinsic, world_from_camera, {
        "focal_length_mm": focal_mm,
        "horizontal_aperture_mm": horizontal_mm,
        "vertical_aperture_mm": vertical_mm,
        "near_far_m": [float(clipping[0]), float(clipping[1])],
    }


def depth_preview(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("Camera returned no valid depth")
    values = depth[valid]
    near, far = np.percentile(values, [1.0, 99.0])
    if far <= near:
        far = near + 1.0e-6
    image = np.zeros(depth.shape, dtype=np.float32)
    image[valid] = np.clip((far - depth[valid]) / (far - near), 0.0, 1.0)
    return np.round(image * 255).astype(np.uint8), {
        "valid_pixel_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "minimum_depth_m": float(values.min()),
        "maximum_depth_m": float(values.max()),
    }


def spawn_objects(stage: Usd.Stage, scene: dict, world_from_source: np.ndarray):
    old = stage.GetPrimAtPath(TASK_ROOT)
    if old.IsValid():
        stage.RemovePrim(TASK_ROOT)
    root = UsdGeom.Xform.Define(stage, TASK_ROOT).GetPrim()
    root.CreateAttribute("dgn2:dynamicScene", Sdf.ValueTypeNames.Bool).Set(True)
    material_path = "/World/PhysicsMaterials/TaskObjects"
    sim_utils.spawn_rigid_body_material(
        material_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    objects = []
    records = []
    for record in scene["objects"]:
        seg_id = int(record["segmentation_id"])
        code = str(record["object_code"])
        root_path = f"{TASK_ROOT}/Object_{seg_id:02d}_{code.split('-', 2)[1]}"
        simulation_usd = (
            PROJECT_ROOT
            / "02_training_dataset/data/scene_datasets/wuji2_test60_10upright_10view_v1/"
            f"usd_cache/object_{int(record['object_pool_index']):03d}/flat/"
            f"object_{int(record['object_pool_index']):03d}_editable.usd"
        )
        if not simulation_usd.is_file():
            raise FileNotFoundError(simulation_usd)
        add_reference_to_stage(str(simulation_usd), root_path)
        world_pose = world_from_source @ np.asarray(record["T_world_centered_object"], dtype=np.float64)
        set_reference_transform(stage, root_path, world_pose)
        prim = stage.GetPrimAtPath(root_path)
        prim.CreateAttribute("dgn2:segmentationId", Sdf.ValueTypeNames.Int).Set(seg_id)
        prim.CreateAttribute("dgn2:objectPoolIndex", Sdf.ValueTypeNames.Int).Set(int(record["object_pool_index"]))
        prim.CreateAttribute("dgn2:objectCode", Sdf.ValueTypeNames.String).Set(code)
        prim.CreateAttribute("dgn2:classLabel", Sdf.ValueTypeNames.String).Set(code.split("-", 2)[1])
        rigid = find_one_rigid_prim(stage, root_path)
        PhysxSchema.PhysxRigidBodyAPI.Apply(rigid).CreateDisableGravityAttr().Set(False)
        UsdPhysics.MassAPI.Apply(rigid).CreateMassAttr().Set(0.1)
        sim_utils.bind_physics_material(root_path, material_path)
        objects.append(RigidObject(RigidObjectCfg(prim_path=str(rigid.GetPath()), spawn=None)))
        records.append({
            "segmentation_id": seg_id,
            "object_pool_index": int(record["object_pool_index"]),
            "object_code": code,
            "asset": record["asset"],
            "simulation_usd": str(simulation_usd.resolve()),
            "root_path": root_path,
            "rigid_path": str(rigid.GetPath()),
            "initial_pose_source_object": record["T_world_centered_object"],
        })
    return objects, records


def main() -> Path:
    output = ARGS.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = load_json(ARGS.scene_manifest)
    print("[INIT 1/8] opening calibrated stage", flush=True)
    if not stage_utils.open_stage(str(ARGS.stage.resolve())):
        raise RuntimeError(f"Cannot open stage: {ARGS.stage}")
    stage = get_current_stage()
    print("[INIT 2/8] reading SourceZone transform", flush=True)
    world_from_source = rigid_world_transform(stage, SOURCE_ZONE_PRIM)
    source_from_world = np.linalg.inv(world_from_source)
    print("[INIT 3/8] spawning six dynamic object USDs", flush=True)
    objects, object_records = spawn_objects(stage, scene, world_from_source)
    print("[INIT 4/8] applying ft04 Force Drive type", flush=True)
    set_force_drive_type(stage)

    width, height = 1280, 720
    print("[INIT 5/8] constructing SimulationContext", flush=True)
    simulation = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=ARGS.device)
    )
    print("[INIT 6/8] registering robot and rigid-object tensor views", flush=True)
    robot = create_robot()
    camera = None
    if not ARGS.settle_only:
        camera = Camera(CameraCfg(
            prim_path=CAMERA_PRIM, update_period=0.0, width=width, height=height,
            data_types=["rgb", "distance_to_image_plane"],
            update_latest_camera_pose=False, spawn=None,
        ))
    print("[INIT 7/8] SimulationContext.reset", flush=True)
    simulation.reset()
    print("[INIT 8/8] tensor views ready", flush=True)
    dt = simulation.get_physics_dt()
    print("[POSTRESET 1/7] robot.update", flush=True)
    robot.update(dt)
    print("[POSTRESET 2/7] object.update", flush=True)
    for obj in objects:
        obj.update(dt)

    print("[POSTRESET 3/7] joint-name audit", flush=True)
    arm_ids, arm_names = robot.find_joints(RIGHT_ARM_NAMES, preserve_order=True)
    if arm_names != RIGHT_ARM_NAMES or robot.num_joints != 35 or not robot.is_fixed_base:
        raise RuntimeError("35-DOF fixed-base robot audit failed")
    print("[POSTRESET 4/7] reading mass matrix and applying ft04", flush=True)
    apply_ft04(robot, [int(value) for value in arm_ids])
    print("[POSTRESET 5/7] auditing authored initial joint state", flush=True)
    state = robot.data.joint_pos.clone()
    authored_arm = state[:, arm_ids]
    expected_arm = torch.as_tensor(INITIAL_ARM_Q_RAD, device=robot.device).reshape(1, 7)
    authored_error_deg = float(torch.max(torch.abs(torch.rad2deg(authored_arm - expected_arm))))
    if authored_error_deg > 0.1:
        raise RuntimeError(
            f"Authored right-arm initial state differs from calibrated posture by {authored_error_deg:.3f} deg"
        )
    print("[POSTRESET 6/7] holding authored initial state without tensor teleport", flush=True)
    command = state.clone()
    robot.set_joint_position_target(command)
    print("[POSTRESET 7/7] ready for physics loop", flush=True)

    print("\n>>> STATE: SCENE_SETTLE")
    step_count = max(1, round(ARGS.settle_time_s / dt))
    for index in range(step_count):
        robot.set_joint_position_target(command)
        robot.write_data_to_sim()
        for obj in objects:
            obj.write_data_to_sim()
        simulation.step(render=(camera is not None and index >= step_count - 12))
        robot.update(dt)
        for obj in objects:
            obj.update(dt)
        if index % max(1, round(0.5 / dt)) == 0 or index == step_count - 1:
            linear = max(float(torch.linalg.vector_norm(obj.data.root_lin_vel_w[0])) for obj in objects)
            angular = max(float(torch.linalg.vector_norm(obj.data.root_ang_vel_w[0])) for obj in objects)
            arm_error = float(torch.max(torch.abs(torch.rad2deg(
                robot.data.joint_pos[:, arm_ids] - state[:, arm_ids]
            ))))
            print(
                f"\rSCENE_SETTLE {100*(index+1)/step_count:6.1f}% | "
                f"object v={linear:.4f} m/s w={angular:.4f} rad/s | arm={arm_error:.3f} deg",
                end="", flush=True,
            )
    print()
    max_linear = max(float(torch.linalg.vector_norm(obj.data.root_lin_vel_w[0])) for obj in objects)
    max_angular = max(float(torch.linalg.vector_norm(obj.data.root_ang_vel_w[0])) for obj in objects)
    if max_linear > ARGS.settled_speed_m_s or max_angular > ARGS.settled_angular_speed_rad_s:
        raise RuntimeError(
            f"Scene is not settled: v={max_linear:.5f} m/s, w={max_angular:.5f} rad/s"
        )

    settled_objects = []
    for wrapper, record in zip(objects, object_records):
        pose = wrapper.data.root_pose_w[0].detach().cpu().numpy()
        world_from_object = np.eye(4, dtype=np.float64)
        world_from_object[:3, :3] = quaternion_wxyz_to_matrix(pose[3:7])
        world_from_object[:3, 3] = pose[:3]
        source_from_object = source_from_world @ world_from_object
        settled_objects.append({
            **record,
            "pose_world_object": source_from_object.tolist(),
            "T_world_centered_object": source_from_object.tolist(),
            "settled_pose_layout_world": world_from_object.tolist(),
            "settled_linear_velocity_world_m_s": wrapper.data.root_lin_vel_w[0].detach().cpu().tolist(),
            "settled_angular_velocity_world_rad_s": wrapper.data.root_ang_vel_w[0].detach().cpu().tolist(),
        })

    settled_manifest = {
        "schema_version": 2,
        "status": "dynamic_scene_settled",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scene_manifest": str(ARGS.scene_manifest.resolve()),
        "coordinate_contract": {
            "object_pose": "pose_world_object means T_SourceZone_centeredObject",
            "layout_bridge": "T_layoutWorld_object = T_layoutWorld_SourceZone @ pose_world_object",
        },
        "table": scene["table"],
        "world_from_source_zone": world_from_source.tolist(),
        "objects": settled_objects,
        "settled_object_speed_limits": {
            "linear_m_s": ARGS.settled_speed_m_s,
            "angular_rad_s": ARGS.settled_angular_speed_rad_s,
            "observed_max_linear_m_s": max_linear,
            "observed_max_angular_rad_s": max_angular,
        },
    }
    settled_path = output / "settled_scene_manifest.json"
    settled_path.write_text(json.dumps(settled_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[PASS] dynamic scene settled")
    print(f"[POSES]  {settled_path}")
    if ARGS.settle_only:
        report_path = output / "settle_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "stage": "SCENE_SETTLE",
                    "settled_scene_manifest": str(settled_path),
                    "observed_max_linear_m_s": max_linear,
                    "observed_max_angular_rad_s": max_angular,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        return report_path

    assert camera is not None
    print(">>> STATE: RGBD_CAPTURE")
    intrinsic, world_from_camera, camera_model = camera_calibration(stage, width, height)
    for _ in range(12):
        robot.set_joint_position_target(command)
        robot.write_data_to_sim()
        for obj in objects:
            obj.write_data_to_sim()
        simulation.step(render=True)
        robot.update(dt)
        for obj in objects:
            obj.update(dt)
        camera.update(dt=dt, force_recompute=True)
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
    depth = np.squeeze(
        camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy().astype(np.float32)
    )
    preview, depth_stats = depth_preview(depth)
    Image.fromarray(rgb, mode="RGB").save(output / "rgb.png")
    Image.fromarray(preview, mode="L").save(output / "depth_preview.png")
    np.save(output / "depth_m.npy", depth)
    np.save(output / "intrinsics.npy", intrinsic)
    np.save(output / "T_world_camera.npy", world_from_camera)

    capture = {
        "schema_version": 2,
        "status": "single_dynamic_rgbd_capture_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture_backend": "Isaac Lab 2.2 Camera after dynamic gravity settle",
        "camera_prim": CAMERA_PRIM,
        "resolution_wh": [width, height],
        "rgb": {"file": "rgb.png", "shape": list(rgb.shape)},
        "depth": {"file": "depth_m.npy", "shape": list(depth.shape), **depth_stats},
        "intrinsics": {"file": "intrinsics.npy", "K": intrinsic.tolist()},
        "extrinsics": {"file": "T_world_camera.npy", "matrix": world_from_camera.tolist()},
        "camera_model": camera_model,
        "settled_scene_manifest": str(settled_path),
        "settled_object_speed_limits": {
            "linear_m_s": ARGS.settled_speed_m_s,
            "angular_rad_s": ARGS.settled_angular_speed_rad_s,
            "observed_max_linear_m_s": max_linear,
            "observed_max_angular_rad_s": max_angular,
        },
        "grounded_sam_compatible": True,
    }
    capture_path = output / "capture_manifest.json"
    capture_path.write_text(json.dumps(capture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[PASS] RGB-D: {rgb.shape}, {depth.shape}, valid={depth_stats['valid_fraction']:.4f}")
    print(f"[OUTPUT] {capture_path}")
    print(f"[POSES]  {settled_path}")
    return capture_path


try:
    RESULT = main()
finally:
    SIMULATION_APP.close()
