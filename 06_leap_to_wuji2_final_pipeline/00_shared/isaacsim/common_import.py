"""Stage 01 importer for the verified LEAP-root-driven Wuji2 workflow.

Run a case-specific ``01_import.py`` entry instead of running this file
directly.  The old direct-pose root branch is intentionally not accepted.
"""

from __future__ import annotations

import builtins
import itertools
import json
import math
from pathlib import Path

import numpy as np
import omni.timeline
from omni.kit.async_engine import run_coroutine
from omni.physx.scripts import physicsUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.articulations import move_articulation_root
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    create_new_stage_async,
    get_current_stage,
)
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view


PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
PIPELINE_ROOT = PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline"
if not hasattr(builtins, "DGN2_CASE_ROOT"):
    raise RuntimeError("Run a case-specific 01_import.py, not common_import.py")
EXP_ROOT = Path(getattr(builtins, "DGN2_CASE_ROOT")).resolve()
if EXP_ROOT.parent != (PIPELINE_ROOT / "01_cases").resolve():
    raise RuntimeError(f"Case is outside 01_cases: {EXP_ROOT}")
if not (EXP_ROOT / "case.json").is_file():
    raise FileNotFoundError(EXP_ROOT / "case.json")
BRANCH = str(getattr(builtins, "DGN2_AB_BRANCH", ""))
ALLOWED_BRANCHES = {"wuji2_leap_root_drive"}
if BRANCH not in ALLOWED_BRANCHES:
    raise RuntimeError(
        f"This cleaned importer accepts only {sorted(ALLOWED_BRANCHES)}, got {BRANCH!r}"
    )
for required_key in ("DGN2_AB_JOB_PATH", "DGN2_AB_RESULT_PATH"):
    if not hasattr(builtins, required_key):
        raise RuntimeError(f"case wrapper did not provide {required_key}")
JOB_PATH = Path(getattr(builtins, "DGN2_AB_JOB_PATH"))
RESULT_PATH = Path(getattr(builtins, "DGN2_AB_RESULT_PATH"))
CASE_META = json.loads((EXP_ROOT / "case.json").read_text(encoding="utf-8"))
SCENE_MANIFEST_PATH = (
    EXP_ROOT / "01_input" / f"{CASE_META['scene_id']}_manifest.json"
)
NATIVE_DIRECTION_CONFIG = EXP_ROOT / "00_config/wuji2_native_width_mapper.json"
WUJI_USD = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/hand2/hand2_beta1/body/usd/right/wujihand2.usd"
)
HAND_USD = WUJI_USD

CONTEXT_KEY = "DGN2_OFFICIAL_LEAP_WUJI2_AB_CONTEXT"
CALLBACK_NAME = "dgn2_official_leap_wuji2_ab_execution"
PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0
LEAP_ROOT_JOINT_NAMES = (
    "x_joint", "y_joint", "z_joint",
    "x_rotation_joint", "y_rotation_joint", "z_rotation_joint",
)
LEAP_ROOT_LINK_MASS_KG = 0.05
LEAP_ROOT_LINK_INERTIA_KG_M2 = 1.0e-4
LEAP_ROOT_STIFFNESS = 800.0
LEAP_ROOT_DAMPING = 20.0
LEAP_ROOT_LIMIT = 6.28


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def set_reference_transform(stage, root_path: str, pose: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(root_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing referenced prim {root_path}")
    transform = Gf.Matrix4d(1.0)
    quat = matrix_to_quaternion_wxyz(pose[:3, :3])
    transform.SetRotate(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))
    transform.SetTranslate(Gf.Vec3d(*map(float, pose[:3, 3])))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)


def find_one_api_prim(stage, prefix: str, api) -> Usd.Prim:
    matches = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(api)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {api.__name__} below {prefix}, got "
            f"{[str(item.GetPath()) for item in matches]}"
        )
    return matches[0]


def bind_material(stage, root_path: str, material: PhysicsMaterial) -> int:
    root = stage.GetPrimAtPath(root_path)
    binding = (
        UsdShade.MaterialBindingAPI(root)
        if root.HasAPI(UsdShade.MaterialBindingAPI)
        else UsdShade.MaterialBindingAPI.Apply(root)
    )
    binding.Bind(
        material.material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return sum(
        1
        for prim in Usd.PrimRange(
            root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )


def _define_virtual_root_body(stage, path: str) -> Usd.Prim:
    """Create one collider-free root-chain body exactly like LEAP's URDF links."""
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr().Set(LEAP_ROOT_LINK_MASS_KG)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(
            LEAP_ROOT_LINK_INERTIA_KG_M2,
            LEAP_ROOT_LINK_INERTIA_KG_M2,
            LEAP_ROOT_LINK_INERTIA_KG_M2,
        )
    )
    return prim


def _configure_leap_root_drive(joint, drive_name: str) -> None:
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_name)
    drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    # USD angular drive attributes are degree-based, whereas Isaac/PhysX
    # controller gains are reported per radian. Author the converted value so
    # the effective gain is already 800/20 at first physics initialization.
    angular_scale = np.pi / 180.0 if drive_name == "angular" else 1.0
    drive.CreateStiffnessAttr().Set(LEAP_ROOT_STIFFNESS * angular_scale)
    drive.CreateDampingAttr().Set(LEAP_ROOT_DAMPING * angular_scale)
    # LEAP's six virtual URDF joints have no effort field.  The converted USD
    # therefore does not impose a finite maxForce on these six root drives.


def create_leap_style_wuji2_root(stage, wrist_path: str) -> Usd.Prim:
    """Attach LEAP's exact 3P+3R root topology to official Wuji2 ``r_wrist``.

    The referenced Wuji2 USD is never edited.  All six virtual links/joints are
    authored only in the current stage layer.
    """
    wrist = stage.GetPrimAtPath(wrist_path)
    if not wrist.IsValid() or not wrist.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"official Wuji2 wrist rigid body missing: {wrist_path}")
    old_fixed = stage.GetPrimAtPath("/World/Hand/root_joint")
    if not old_fixed.IsValid() or not old_fixed.IsA(UsdPhysics.FixedJoint):
        raise RuntimeError("official Wuji2 fixed root_joint was not found")
    UsdPhysics.Joint(old_fixed).CreateJointEnabledAttr().Set(False)

    # The official composed asset already has one articulation-root opinion.
    # Remove it only from this session layer before installing the LEAP-style
    # fixed-base root joint.
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith("/World/Hand") and prim.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    adapter = "/World/Hand/LeapStyleRoot"
    UsdGeom.Xform.Define(stage, adapter)
    bodies = [
        _define_virtual_root_body(stage, f"{adapter}/hand_root_{index}")
        for index in range(6)
    ]
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{adapter}/root_joint")
    fixed.CreateBody1Rel().SetTargets([bodies[0].GetPath()])
    articulation_root = UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim()).GetPrim()
    physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)
    physx_articulation.CreateSolverPositionIterationCountAttr().Set(8)
    physx_articulation.CreateSolverVelocityIterationCountAttr().Set(0)

    linear_axes = (UsdPhysics.Tokens.x, UsdPhysics.Tokens.y, UsdPhysics.Tokens.z)
    for index, (name, axis) in enumerate(zip(LEAP_ROOT_JOINT_NAMES[:3], linear_axes)):
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{adapter}/{name}")
        joint.CreateBody0Rel().SetTargets([bodies[index].GetPath()])
        joint.CreateBody1Rel().SetTargets([bodies[index + 1].GetPath()])
        joint.CreateAxisAttr().Set(axis)
        joint.CreateLowerLimitAttr().Set(-LEAP_ROOT_LIMIT)
        joint.CreateUpperLimitAttr().Set(LEAP_ROOT_LIMIT)
        _configure_leap_root_drive(joint, "linear")

    angular_axes = (UsdPhysics.Tokens.x, UsdPhysics.Tokens.y, UsdPhysics.Tokens.z)
    angular_children = (bodies[4].GetPath(), bodies[5].GetPath(), Sdf.Path(wrist_path))
    for offset, (name, axis, child) in enumerate(
        zip(LEAP_ROOT_JOINT_NAMES[3:], angular_axes, angular_children)
    ):
        parent_index = 3 + offset
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{adapter}/{name}")
        joint.CreateBody0Rel().SetTargets([bodies[parent_index].GetPath()])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateAxisAttr().Set(axis)
        # USD angular joint limits are authored in degrees; ArticulationAction
        # still exposes the resulting joint position in radians.
        limit_deg = float(np.degrees(LEAP_ROOT_LIMIT))
        joint.CreateLowerLimitAttr().Set(-limit_deg)
        joint.CreateUpperLimitAttr().Set(limit_deg)
        _configure_leap_root_drive(joint, "angular")
    return articulation_root


def map_joint_targets(hand, source_names: list[str], source_values: np.ndarray) -> np.ndarray:
    imported = list(hand.dof_names)
    missing = [name for name in source_names if name not in imported]
    if missing:
        raise RuntimeError(f"{BRANCH} USD misses joints {missing}")
    targets = np.zeros((5, len(imported)), dtype=np.float32)
    for source_index, name in enumerate(source_names):
        targets[:, imported.index(name)] = source_values[:, source_index]
    return targets


async def import_branch() -> None:
    if BRANCH in {
        "wuji2_native_actions",
        "wuji2_four_real_tip",
        "wuji2_four_real_tip_hammer37",
        "wuji2_thumb_leap_squeeze",
    }:
        direction_config = json.loads(
            NATIVE_DIRECTION_CONFIG.read_text(encoding="utf-8")
        )
        if not bool(direction_config.get("directions_approved", False)):
            raise RuntimeError(
                "Wuji2 native SQUEEZE is disabled: fingertip directions are "
                "pending per-finger visual calibration."
            )
    for path in (JOB_PATH, JOB_PATH.with_suffix(".json"), SCENE_MANIFEST_PATH, HAND_USD):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(JOB_PATH, allow_pickle=False) as archive:
        job = {key: archive[key] for key in archive.files}
    if BRANCH in {
        "wuji2_four_real_tip",
        "wuji2_four_real_tip_hammer37",
        "wuji2_thumb_leap_squeeze",
        "wuji2_all_tip_leap_squeeze",
        "wuji2_unified_weighted",
        "wuji2_leap_root_drive",
        "wuji2_hammer37_unified_weighted",
    }:
        required_contract = {
            "pregrasp_approach_policy",
            "wuji2_semantic_palm_frame_in_r_wrist",
            "wuji2_semantic_palm_approach_axis_world",
            "official_dgn2_pregrasp_retreat_m",
            "post_squeeze_lift_policy",
            "post_squeeze_lift_distance_m",
            "is_top_grasp",
        }
        missing_contract = sorted(required_contract.difference(job))
        if missing_contract:
            raise RuntimeError(
                f"Wuji2 semantic-palm approach contract is incomplete: "
                f"{missing_contract}"
            )
        policy = str(np.asarray(job["pregrasp_approach_policy"]).item())
        expected_policy = "wuji2_semantic_palm_positive_z_100mm"
        if policy != expected_policy:
            raise RuntimeError(
                f"wrong Wuji2 approach policy: {policy!r}; "
                f"expected {expected_policy!r}"
            )
        stage_names = [str(value) for value in job["waypoint_names"].tolist()]
        pre_index = stage_names.index("pregrasp")
        cover_index = stage_names.index("cover")
        grasp_index = stage_names.index("grasp")
        contract_poses = np.asarray(
            job["waypoint_pose_world"][0], dtype=np.float64
        )
        approach_delta = (
            contract_poses[cover_index, :3, 3]
            - contract_poses[pre_index, :3, 3]
        )
        approach_distance = float(np.linalg.norm(approach_delta))
        expected_distance = float(job["official_dgn2_pregrasp_retreat_m"])
        if not np.isclose(approach_distance, expected_distance, atol=5.0e-7):
            raise RuntimeError(
                "Wuji2 semantic-palm approach distance changed: "
                f"{approach_distance} vs {expected_distance} m"
            )
        measured_axis = approach_delta / approach_distance
        recorded_axis = np.asarray(
            job["wuji2_semantic_palm_approach_axis_world"], dtype=np.float64
        )
        recorded_axis /= np.linalg.norm(recorded_axis)
        wrist_from_palm = np.asarray(
            job["wuji2_semantic_palm_frame_in_r_wrist"], dtype=np.float64
        )
        palm_blue_axis = (
            contract_poses[grasp_index, :3, :3] @ wrist_from_palm[:3, 2]
        )
        palm_blue_axis /= np.linalg.norm(palm_blue_axis)
        if float(np.dot(measured_axis, recorded_axis)) < 1.0 - 1.0e-6:
            raise RuntimeError("stored and measured Wuji2 approach axes disagree")
        if float(np.dot(measured_axis, palm_blue_axis)) < 1.0 - 1.0e-6:
            raise RuntimeError(
                "PREGRASP->COVER is not along semantic-palm blue +Z"
            )
        squeeze_index = stage_names.index("squeeze")
        lift_index = stage_names.index("lift")
        lift_delta = (
            contract_poses[lift_index, :3, 3]
            - contract_poses[squeeze_index, :3, 3]
        )
        lift_distance = float(np.linalg.norm(lift_delta))
        expected_lift_distance = float(job["post_squeeze_lift_distance_m"])
        if not np.isclose(lift_distance, expected_lift_distance, atol=5.0e-7):
            raise RuntimeError(
                f"post-SQUEEZE lift distance changed: {lift_distance} vs "
                f"{expected_lift_distance} m"
            )
        is_top_grasp = bool(np.asarray(job["is_top_grasp"]).item())
        lift_policy = str(np.asarray(job["post_squeeze_lift_policy"]).item())
        expected_lift_axis = (
            -palm_blue_axis
            if is_top_grasp
            else np.asarray([0.0, 0.0, 1.0])
        )
        measured_lift_axis = lift_delta / lift_distance
        if float(np.dot(measured_lift_axis, expected_lift_axis)) < 1.0 - 1.0e-6:
            raise RuntimeError(
                "post-SQUEEZE motion violates the frozen top/side rule"
            )
    scene = json.loads(SCENE_MANIFEST_PATH.read_text(encoding="utf-8"))
    target_seg = int(job["target_segmentation_id"][0])
    target_records = [
        item for item in scene["objects"]
        if int(item["segmentation_id"]) == target_seg
    ]
    if len(target_records) != 1:
        raise RuntimeError(f"target segmentation {target_seg} is ambiguous")
    target_code = str(target_records[0]["code"])

    previous = getattr(builtins, CONTEXT_KEY, None)
    if previous is not None:
        old_world = previous.get("world")
        if old_world is not None and old_world.physics_callback_exists(CALLBACK_NAME):
            old_world.remove_physics_callback(CALLBACK_NAME)
        if old_world is not None:
            await old_world.stop_async()
    World.clear_instance()
    await create_new_stage_async()
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
    )
    await world.initialize_simulation_context_async()
    world.get_physics_context().set_gravity(-9.81)
    stage = get_current_stage()
    physicsUtils.add_ground_plane(
        stage,
        "/World/Ground",
        "Z",
        100.0,
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.35, 0.35, 0.35),
    )
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(900.0)

    ground_material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/Ground",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    hand_material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/Hand",
        static_friction=0.2,
        dynamic_friction=0.2,
        restitution=0.0,
    )
    object_material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/Object",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    bind_material(stage, "/World/Ground", ground_material)

    add_reference_to_stage(str(HAND_USD), "/World/Hand")
    leap_style_rigid_body_count = 0
    if BRANCH == "wuji2_leap_root_drive":
        articulation_root = create_leap_style_wuji2_root(stage, "/World/Hand/r_wrist")
        # LEAP's evaluator disables gravity on the complete hand asset while
        # restoring world gravity for scene objects after SQUEEZE. Apply the
        # same runtime property without modifying the Wuji2 source USD.
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/Hand") and prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rigid.CreateDisableGravityAttr().Set(True)
                rigid.CreateMaxDepenetrationVelocityAttr().Set(1000.0)
                leap_style_rigid_body_count += 1
    elif BRANCH.startswith("leap"):
        articulation_root = find_one_api_prim(
            stage, "/World/Hand", UsdPhysics.ArticulationRootAPI
        )
    else:
        fixed_root = stage.GetPrimAtPath("/World/Hand/root_joint")
        wrist = stage.GetPrimAtPath("/World/Hand/r_wrist")
        if not fixed_root.IsValid() or not fixed_root.IsA(UsdPhysics.FixedJoint):
            raise RuntimeError("official Wuji2 fixed root_joint was not found")
        if not wrist.IsValid() or not wrist.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("official Wuji2 r_wrist rigid body was not found")
        UsdPhysics.Joint(fixed_root).CreateJointEnabledAttr().Set(False)
        move_articulation_root(fixed_root, wrist)
        articulation_root = wrist
    hand = world.scene.add(
        SingleArticulation(
            prim_path=str(articulation_root.GetPath()),
            name=f"official_{BRANCH}_hand",
            reset_xform_properties=False,
        )
    )
    hand_collider_count = bind_material(stage, "/World/Hand", hand_material)

    objects = {}
    initial_object_poses = {}
    rigid_prims = {}
    object_root_paths = {}
    object_collider_count = 0
    for record in scene["objects"]:
        seg_id = int(record["segmentation_id"])
        root_path = f"/World/Objects/Object_{seg_id:03d}"
        add_reference_to_stage(str(Path(record["simulation_usd"])), root_path)
        pose = np.asarray(record["pose_world_object"], dtype=np.float64)
        set_reference_transform(stage, root_path, pose)
        rigid_prim = find_one_api_prim(stage, root_path, UsdPhysics.RigidBodyAPI)
        PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim).CreateDisableGravityAttr().Set(False)
        UsdPhysics.MassAPI.Apply(rigid_prim).CreateMassAttr().Set(0.1)
        wrapper = world.scene.add(
            SingleRigidPrim(
                prim_path=str(rigid_prim.GetPath()),
                name=f"object_{seg_id:03d}",
                reset_xform_properties=False,
            )
        )
        object_collider_count += bind_material(stage, root_path, object_material)
        objects[seg_id] = wrapper
        rigid_prims[seg_id] = rigid_prim
        object_root_paths[seg_id] = root_path
        initial_object_poses[seg_id] = pose

    # Match the original evaluator's object actor filter=2: scene objects do
    # not collide with one another, while hand-object and ground-object remain.
    group_path = "/World/CollisionGroups/OfficialSceneObjects"
    group = UsdPhysics.CollisionGroup.Define(stage, group_path)
    group.CreateFilteredGroupsRel().AddTarget(group.GetPath())
    for root_path in object_root_paths.values():
        physicsUtils.add_collision_to_collision_group(stage, root_path, group_path)
    filtered_pairs = len(list(itertools.combinations(rigid_prims.values(), 2)))

    await world.reset_async()
    root_gain_audit = None
    if BRANCH == "wuji2_leap_root_drive":
        controller = hand.get_articulation_controller()
        effective_kps, effective_kds = controller.get_gains()
        effective_kps = np.asarray(effective_kps, dtype=np.float32).copy()
        effective_kds = np.asarray(effective_kds, dtype=np.float32).copy()
        imported_dofs = list(hand.dof_names)
        root_indices = []
        for name in LEAP_ROOT_JOINT_NAMES:
            if name not in imported_dofs:
                raise RuntimeError(f"LEAP-style root DOF was not imported: {name}")
            root_indices.append(imported_dofs.index(name))
        effective_kps[root_indices] = LEAP_ROOT_STIFFNESS
        effective_kds[root_indices] = LEAP_ROOT_DAMPING
        # Set effective PhysX gains after articulation initialization. This is
        # the same numeric contract as IsaacLab's ImplicitActuatorCfg and
        # avoids USD degree/radian authoring ambiguity for revolute drives.
        controller.set_gains(kps=effective_kps, kds=effective_kds)
        read_kps, read_kds = controller.get_gains()
        read_kps = np.asarray(read_kps, dtype=np.float32)
        read_kds = np.asarray(read_kds, dtype=np.float32)
        if not np.allclose(read_kps[root_indices], LEAP_ROOT_STIFFNESS):
            raise RuntimeError("effective LEAP root stiffness readback failed")
        if not np.allclose(read_kds[root_indices], LEAP_ROOT_DAMPING):
            raise RuntimeError("effective LEAP root damping readback failed")
        root_gain_audit = {
            "joint_names": list(LEAP_ROOT_JOINT_NAMES),
            "joint_indices": root_indices,
            "effective_stiffness": read_kps[root_indices].tolist(),
            "effective_damping": read_kds[root_indices].tolist(),
            "hand_rigid_bodies_with_gravity_disabled": leap_style_rigid_body_count,
        }
    finger_names = [str(value) for value in job["finger_joint_names"].tolist()]
    finger_values = np.asarray(job["waypoint_joint_positions"][0], dtype=np.float32)
    if BRANCH.startswith("leap") or BRANCH == "wuji2_leap_root_drive":
        root_names = [str(value) for value in job["root_joint_names"].tolist()]
        if tuple(root_names) != LEAP_ROOT_JOINT_NAMES:
            raise RuntimeError(
                f"root joint order changed: {root_names}; expected {LEAP_ROOT_JOINT_NAMES}"
            )
        root_values = np.asarray(job["waypoint_root_dofs"][0], dtype=np.float32)
        source_names = root_names + finger_names
        source_values = np.concatenate([root_values, finger_values], axis=1)
        targets = map_joint_targets(hand, source_names, source_values)
        root_poses = np.asarray(job["waypoint_pose_world"][0], dtype=np.float32)
    else:
        targets = map_joint_targets(hand, finger_names, finger_values)
        root_poses = np.asarray(job["waypoint_pose_world"][0], dtype=np.float32)
    limits = np.asarray(
        hand._articulation_view.get_dof_limits()
    ).reshape(-1, hand.num_dof, 2)[0]
    below = targets < limits[:, 0][None, :] - 1.0e-6
    above = targets > limits[:, 1][None, :] + 1.0e-6
    if np.any(below | above):
        violations = []
        for stage_index, joint_index in np.argwhere(below | above):
            violations.append(
                {
                    "stage": str(job["waypoint_names"][stage_index]),
                    "joint": str(hand.dof_names[joint_index]),
                    "q_rad": float(targets[stage_index, joint_index]),
                    "lower_rad": float(limits[joint_index, 0]),
                    "upper_rad": float(limits[joint_index, 1]),
                }
            )
        raise RuntimeError(f"retargeted targets violate composed USD limits: {violations}")

    def reset_actors() -> None:
        for seg_id, wrapper in objects.items():
            pose = initial_object_poses[seg_id]
            wrapper.set_world_pose(
                position=pose[:3, 3],
                orientation=matrix_to_quaternion_wxyz(pose[:3, :3]),
            )
            wrapper.set_linear_velocity(np.zeros(3, dtype=np.float32))
            wrapper.set_angular_velocity(np.zeros(3, dtype=np.float32))
        hand.set_joint_positions(targets[0])
        hand.set_joint_velocities(np.zeros(hand.num_dof, dtype=np.float32))
        hand.apply_action(ArticulationAction(joint_positions=targets[0]))

    # Reproduce the official two-step warm-up and exact second actor reset.
    reset_actors()
    world.step(render=False)
    world.step(render=False)
    reset_actors()
    # The reviewed four-real-tip branch keeps real gravity throughout the
    # complete approach, closure, squeeze and lift sequence. Older A/B
    # branches retain their original official-evaluator gravity schedule.
    if BRANCH in {
        "wuji2_four_real_tip",
        "wuji2_four_real_tip_hammer37",
        "wuji2_thumb_leap_squeeze",
        "wuji2_all_tip_leap_squeeze",
        "wuji2_unified_weighted",
        "wuji2_hammer37_unified_weighted",
    }:
        world.get_physics_context().set_gravity(-9.81)
    else:
        world.get_physics_context().set_gravity(0.0)
    await world.pause_async()
    # Do not call world.render() recursively from this async Script Editor
    # coroutine. The GUI renders on its next Kit frame; an explicit nested
    # update produces asyncio re-entry warnings in headless validation.

    # Capture the thin wrapper's override at module-load time, just like
    # JOB_PATH above.  The wrapper restores builtins immediately after
    # scheduling this coroutine; resolving the value here used to lose the
    # override and silently write the report into the shared legacy folder.
    result_path = RESULT_PATH
    context = {
        "branch": BRANCH,
        "scene_id": str(CASE_META["scene_id"]),
        "view_id": str(CASE_META["view_id"]),
        "world": world,
        "hand": hand,
        "objects": objects,
        "targets": targets,
        "root_poses": root_poses,
        "waypoint_names": [str(value) for value in job["waypoint_names"].tolist()],
        "waypoint_steps": [int(value) for value in job["waypoint_steps"].tolist()],
        "target_segmentation_id": target_seg,
        "target_code": target_code,
        "pregrasp_valid": bool(job["pregrasp_valid"][0]),
        "result_path": result_path,
        "source_candidate_index": int(job["source_candidate_index"][0]),
        "score": float(job["score"][0]),
        "filtered_object_pairs": filtered_pairs,
        "pregrasp_approach_policy": (
            str(np.asarray(job["pregrasp_approach_policy"]).item())
            if "pregrasp_approach_policy" in job
            else "source_branch_default"
        ),
        "pregrasp_approach_axis_world": (
            np.asarray(
                job["wuji2_semantic_palm_approach_axis_world"],
                dtype=np.float64,
            ).tolist()
            if "wuji2_semantic_palm_approach_axis_world" in job
            else None
        ),
        "post_squeeze_lift_policy": (
            str(np.asarray(job["post_squeeze_lift_policy"]).item())
            if "post_squeeze_lift_policy" in job
            else "source_branch_default"
        ),
        "squeeze_dense_targets": None,
        "squeeze_dense_policy": None,
        "root_control_policy": (
            "leap_3_prismatic_3_revolute_force_position_K800_D20"
        ),
        "root_gain_audit": root_gain_audit,
    }
    if BRANCH in {
        "wuji2_thumb_leap_squeeze",
        "wuji2_all_tip_leap_squeeze",
        "wuji2_unified_weighted",
        "wuji2_leap_root_drive",
        "wuji2_hammer37_unified_weighted",
    }:
        required_dense = {
            "squeeze_dense_q20_path",
            "squeeze_dense_alpha",
            "squeeze_dense_joint_names",
            "squeeze_dense_policy",
            "squeeze_path_validation_passed",
        }
        missing_dense = sorted(required_dense.difference(job))
        if missing_dense:
            raise RuntimeError(f"dense SQUEEZE contract is incomplete: {missing_dense}")
        if not bool(np.asarray(job["squeeze_path_validation_passed"]).item()):
            raise RuntimeError("dense SQUEEZE path was not validated")
        dense_names = [
            str(value) for value in job["squeeze_dense_joint_names"].tolist()
        ]
        dense_source = np.asarray(job["squeeze_dense_q20_path"], dtype=np.float32)
        if (
            dense_source.ndim != 2
            or dense_source.shape[0] < 2
            or dense_source.shape[1] != len(dense_names)
        ):
            raise RuntimeError(
                f"expected dense SQUEEZE path (N,{len(dense_names)}), N>=2; "
                f"got {dense_source.shape}"
            )
        # Root DOFs must stay at GRASP throughout SQUEEZE.  Initializing with
        # zeros would pull the LEAP-style virtual wrist back to world origin.
        dense_targets = np.repeat(
            targets[grasp_index][None, :], len(dense_source), axis=0
        ).astype(np.float32)
        imported_names = list(hand.dof_names)
        missing_usd = [name for name in dense_names if name not in imported_names]
        if missing_usd:
            raise RuntimeError(f"official Wuji2 USD misses dense-path joints {missing_usd}")
        for source_index, name in enumerate(dense_names):
            dense_targets[:, imported_names.index(name)] = dense_source[:, source_index]
        dense_below = dense_targets < limits[:, 0][None, :] - 1.0e-6
        dense_above = dense_targets > limits[:, 1][None, :] + 1.0e-6
        if np.any(dense_below | dense_above):
            violations = []
            for sample_index, joint_index in np.argwhere(
                dense_below | dense_above
            ):
                violations.append({
                    "sample": int(sample_index),
                    "joint": str(hand.dof_names[joint_index]),
                    "q_rad": float(dense_targets[sample_index, joint_index]),
                    "usd_lower_rad": float(limits[joint_index, 0]),
                    "usd_upper_rad": float(limits[joint_index, 1]),
                })
            raise RuntimeError(
                "dense SQUEEZE path violates official composed USD limits: "
                f"{violations}"
            )
        if not np.allclose(dense_targets[0], targets[grasp_index], atol=2.0e-7):
            raise RuntimeError("dense SQUEEZE path does not start at mapped GRASP target")
        if not np.allclose(dense_targets[-1], targets[squeeze_index], atol=2.0e-7):
            raise RuntimeError("dense SQUEEZE path does not end at mapped SQUEEZE target")
        context["squeeze_dense_targets"] = dense_targets
        context["squeeze_dense_policy"] = str(
            np.asarray(job["squeeze_dense_policy"]).item()
        )
    setattr(builtins, CONTEXT_KEY, context)
    set_camera_view(
        eye=np.asarray([0.62, -0.62, 0.48]),
        target=np.asarray([0.0, 0.0, 0.07]),
        camera_prim_path="/OmniverseKit_Persp",
    )
    print("\n[01 IMPORT COMPLETE]")
    print(f"branch={BRANCH}; candidate={context['source_candidate_index']}")
    print(f"target seg={target_seg}; code={target_code}")
    print(f"score={context['score']:.6f}; pregrasp_valid={context['pregrasp_valid']}")
    print(f"hand USD={HAND_USD}")
    print(f"hand colliders friction=0.2: {hand_collider_count}")
    print(f"object colliders friction=1.0: {object_collider_count}")
    print(f"object-object filtered pairs: {filtered_pairs}")
    print(f"approach policy={context['pregrasp_approach_policy']}")
    print(f"root control={context['root_control_policy']}")
    if context["root_gain_audit"] is not None:
        print(f"root gain readback={context['root_gain_audit']}")
    if context["pregrasp_approach_axis_world"] is not None:
        print(
            "approach axis world="
            f"{context['pregrasp_approach_axis_world']}"
        )
    print(f"post-SQUEEZE policy={context['post_squeeze_lift_policy']}")
    if context["squeeze_dense_targets"] is not None:
        print(
            f"dense SQUEEZE={len(context['squeeze_dense_targets'])} samples; "
            f"policy={context['squeeze_dense_policy']}"
        )
    print("stage paused at PREGRASP; run the matching 02_execute.py")


run_coroutine(import_branch())
