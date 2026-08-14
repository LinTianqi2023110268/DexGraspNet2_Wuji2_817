"""Isaac Sim 5.0 Script Editor：创建可手动调整的桌面与双臂基础场景。

当前阶段故意不创建相机、不启动物理、不运行第二个 Isaac Sim 进程。

使用方法：
1. 打开 Isaac Sim 5.0，新建空 Stage；
2. 在 Window -> Script Editor 中打开本文件并点击 Run；
3. 在 Stage 树中选择 ``/World/Layout/TableAssembly`` 调整整张桌子；
4. 选择 ``/World/Layout/DualArmMount`` 调整整台双臂机械臂；
5. 调整满意后运行 ``02_export_calibrated_layout.py``。

Script Editor 会从 /tmp 执行临时副本，因此本文件不能依赖 ``__file__``。
"""

from __future__ import annotations

import builtins
from pathlib import Path

import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics


PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
ASSEMBLED_USD_PATH = (
    PROJECT_ROOT
    / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/usd/dual_arm_right_wuji2.usd"
)
DRAFT_USD = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/scenes/manual_layout_draft.usda"
)

# 大桌面：左侧0.50×0.30 m为抓取区；右侧1.00×0.30 m为放置区。
TABLE_SIZE_M = (1.60, 0.40, 0.04)
SOURCE_ZONE_SIZE_M = (0.50, 0.30, 0.001)
PLACEMENT_ZONE_SIZE_M = (1.00, 0.30, 0.001)
TABLE_CENTER_LOCAL_M = (0.0, 0.0, -0.02)  # 上表面恰好为世界z=0
SOURCE_ZONE_CENTER_LOCAL_M = (-0.50, 0.0, 0.0005)
PLACEMENT_ZONE_CENTER_LOCAL_M = (0.25, 0.0, 0.0005)

# 仅为无穿透、容易看见的初始值。最终值由你在Property面板手动调整。
DUAL_ARM_MOUNT_TRANSLATE_M = (0.0, 0.42, 0.80)
DUAL_ARM_MOUNT_ROTATE_XYZ_DEG = (0.0, 0.0, 0.0)

TABLE_ASSEMBLY_PATH = "/World/Layout/TableAssembly"
DUAL_ARM_MOUNT_PATH = "/World/Layout/DualArmMount"
DUAL_ARM_PATH = f"{DUAL_ARM_MOUNT_PATH}/DualArm"
MARKERS_PATH = "/World/Markers"

UPDATE_SUBSCRIPTION_KEY = "DGN2_LAYOUT_UPDATE_SUBSCRIPTION"
MEASUREMENT_WINDOW_KEY = "DGN2_LAYOUT_MEASUREMENT_WINDOW"
MEASURE_ONCE_KEY = "DGN2_LAYOUT_MEASURE_ONCE"


def set_xform_xyz(
    prim: Usd.Prim,
    translate_m=(0.0, 0.0, 0.0),
    rotate_xyz_deg=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
) -> None:
    """将局部变换写成Property面板中容易手调的Translate/Rotate/Scale。"""

    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*map(float, translate_m)))
    api.SetRotate(
        Gf.Vec3f(*map(float, rotate_xyz_deg)),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    api.SetScale(Gf.Vec3f(*map(float, scale)))


def create_colored_cube(
    stage: Usd.Stage,
    path: str,
    size_m,
    center_m,
    color,
    opacity: float,
    collision: bool,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    cube.CreateDisplayOpacityAttr([float(opacity)])
    set_xform_xyz(cube.GetPrim(), translate_m=center_m, scale=size_m)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def create_marker(stage: Usd.Stage, path: str, color) -> None:
    marker = UsdGeom.Sphere.Define(stage, path)
    marker.CreateRadiusAttr(0.012)
    marker.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    set_xform_xyz(marker.GetPrim())


def create_line(stage: Usd.Stage, path: str, color) -> None:
    line = UsdGeom.BasisCurves.Define(stage, path)
    line.CreateTypeAttr("linear")
    line.CreateCurveVertexCountsAttr([2])
    line.CreatePointsAttr([Gf.Vec3f(0), Gf.Vec3f(0)])
    line.CreateWidthsAttr([0.006, 0.006])
    line.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])


def world_position(stage: Usd.Stage, path: str) -> Gf.Vec3d:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return Gf.Vec3d(matrix.ExtractTranslation())


def set_marker_position(stage: Usd.Stage, path: str, position: Gf.Vec3d) -> None:
    # Markers父级保持单位变换，因此世界坐标可直接写入局部平移。
    set_xform_xyz(stage.GetPrimAtPath(path), translate_m=tuple(position))


def set_line(
    stage: Usd.Stage, path: str, start: Gf.Vec3d, end: Gf.Vec3d
) -> None:
    curve = UsdGeom.BasisCurves(stage.GetPrimAtPath(path))
    curve.GetPointsAttr().Set([Gf.Vec3f(start), Gf.Vec3f(end)])


def distance(a: Gf.Vec3d, b: Gf.Vec3d) -> float:
    return float((a - b).GetLength())


def reference_dual_arm_with_wuji2(stage: Usd.Stage) -> None:
    """引用已验证的Isaac原生整机：左夹爪保留、右侧为官方Wuji2 USD。"""

    robot = stage.DefinePrim(DUAL_ARM_PATH, "Xform")
    robot.GetReferences().AddReference(str(ASSEMBLED_USD_PATH))
    if not stage.GetPrimAtPath(DUAL_ARM_PATH).IsValid():
        raise RuntimeError(f"Referenced robot not found at {DUAL_ARM_PATH}")


def build_stage() -> Usd.Stage:
    if not ASSEMBLED_USD_PATH.is_file():
        raise FileNotFoundError(ASSEMBLED_USD_PATH)

    omni.timeline.get_timeline_interface().stop()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No open stage. Create a new empty stage first.")
    if stage.GetPrimAtPath("/World/Layout").IsValid():
        raise RuntimeError(
            "This stage already contains /World/Layout. Open a new empty stage; "
            "the script refuses to overwrite an existing layout."
        )

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Layout")

    table_assembly = UsdGeom.Xform.Define(stage, TABLE_ASSEMBLY_PATH)
    set_xform_xyz(table_assembly.GetPrim())
    create_colored_cube(
        stage,
        f"{TABLE_ASSEMBLY_PATH}/Table",
        TABLE_SIZE_M,
        TABLE_CENTER_LOCAL_M,
        color=(0.42, 0.42, 0.45),
        opacity=1.0,
        collision=True,
    )
    create_colored_cube(
        stage,
        f"{TABLE_ASSEMBLY_PATH}/SourceZone",
        SOURCE_ZONE_SIZE_M,
        SOURCE_ZONE_CENTER_LOCAL_M,
        color=(0.10, 0.35, 0.95),
        opacity=0.38,
        collision=False,
    )
    create_colored_cube(
        stage,
        f"{TABLE_ASSEMBLY_PATH}/PlacementZone",
        PLACEMENT_ZONE_SIZE_M,
        PLACEMENT_ZONE_CENTER_LOCAL_M,
        color=(0.10, 0.80, 0.30),
        opacity=0.38,
        collision=False,
    )

    mount = UsdGeom.Xform.Define(stage, DUAL_ARM_MOUNT_PATH)
    set_xform_xyz(
        mount.GetPrim(),
        translate_m=DUAL_ARM_MOUNT_TRANSLATE_M,
        rotate_xyz_deg=DUAL_ARM_MOUNT_ROTATE_XYZ_DEG,
    )
    reference_dual_arm_with_wuji2(stage)

    # 物理场景存在，但时间线始终停止；当前只做几何摆放。
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr(9.81)

    environment = UsdGeom.Xform.Define(stage, "/World/Environment")
    light = UsdLux.DistantLight.Define(
        stage, f"{environment.GetPath()}/DistantLight"
    )
    light.CreateIntensityAttr(1000.0)
    UsdGeom.XformCommonAPI(light.GetPrim()).SetRotate(
        Gf.Vec3f(-35.0, 25.0, 0.0),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    UsdGeom.Xform.Define(stage, MARKERS_PATH)
    create_marker(stage, f"{MARKERS_PATH}/TableCenter", (1.0, 1.0, 1.0))
    create_marker(stage, f"{MARKERS_PATH}/SourceZoneCenter", (0.1, 0.35, 0.95))
    create_marker(stage, f"{MARKERS_PATH}/PlacementZoneCenter", (0.1, 0.8, 0.3))
    create_marker(stage, f"{MARKERS_PATH}/RobotRoot", (0.95, 0.15, 0.10))
    labels = UsdGeom.Xform.Define(stage, f"{MARKERS_PATH}/DistanceLabels")
    labels.GetPrim().CreateAttribute(
        "dgn2:units", Sdf.ValueTypeNames.String, custom=True
    ).Set("meters")
    create_line(
        stage,
        f"{MARKERS_PATH}/DistanceLabels/RobotToTable",
        (0.95, 0.15, 0.10),
    )
    create_line(
        stage,
        f"{MARKERS_PATH}/DistanceLabels/RobotToSource",
        (0.1, 0.35, 0.95),
    )
    create_line(
        stage,
        f"{MARKERS_PATH}/DistanceLabels/SourceToPlacement",
        (0.95, 0.75, 0.05),
    )
    return stage


def install_measurements() -> None:
    """手动拖动时刷新标记、连线和小型数值窗口，不运行物理。"""

    if hasattr(builtins, UPDATE_SUBSCRIPTION_KEY):
        setattr(builtins, UPDATE_SUBSCRIPTION_KEY, None)
    if hasattr(builtins, MEASUREMENT_WINDOW_KEY):
        try:
            getattr(builtins, MEASUREMENT_WINDOW_KEY).visible = False
        except Exception:
            pass

    window = ui.Window("DGN2 Manual Scene Layout", width=500, height=215)
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label("Move TableAssembly and DualArmMount in the Property panel.")
            robot_table_label = ui.Label("")
            robot_source_label = ui.Label("")
            source_place_label = ui.Label("")
            ui.Label("Blue = source zone; green = placement zone.")
            ui.Label("Timeline is STOPPED. Camera is deliberately deferred.")
    setattr(builtins, MEASUREMENT_WINDOW_KEY, window)

    def measure_once() -> dict[str, float]:
        current_stage = omni.usd.get_context().get_stage()
        if current_stage is None:
            return {}
        table = world_position(current_stage, TABLE_ASSEMBLY_PATH)
        source = world_position(
            current_stage, f"{TABLE_ASSEMBLY_PATH}/SourceZone"
        )
        placement = world_position(
            current_stage, f"{TABLE_ASSEMBLY_PATH}/PlacementZone"
        )
        robot = world_position(current_stage, DUAL_ARM_MOUNT_PATH)

        for name, position in {
            "TableCenter": table,
            "SourceZoneCenter": source,
            "PlacementZoneCenter": placement,
            "RobotRoot": robot,
        }.items():
            set_marker_position(
                current_stage, f"{MARKERS_PATH}/{name}", position
            )

        set_line(
            current_stage,
            f"{MARKERS_PATH}/DistanceLabels/RobotToTable",
            robot,
            table,
        )
        set_line(
            current_stage,
            f"{MARKERS_PATH}/DistanceLabels/RobotToSource",
            robot,
            source,
        )
        set_line(
            current_stage,
            f"{MARKERS_PATH}/DistanceLabels/SourceToPlacement",
            source,
            placement,
        )

        values = {
            "robot_to_table_m": distance(robot, table),
            "robot_to_source_m": distance(robot, source),
            "source_to_placement_m": distance(source, placement),
        }
        labels_prim = current_stage.GetPrimAtPath(
            f"{MARKERS_PATH}/DistanceLabels"
        )
        for key, value in values.items():
            labels_prim.CreateAttribute(
                f"dgn2:{key}", Sdf.ValueTypeNames.Double, custom=True
            ).Set(value)

        robot_table_label.text = (
            f"Robot root -> table center: {values['robot_to_table_m']:.4f} m"
        )
        robot_source_label.text = (
            f"Robot root -> source center: {values['robot_to_source_m']:.4f} m"
        )
        source_place_label.text = (
            f"Source -> placement center: {values['source_to_placement_m']:.4f} m"
        )
        return values

    counter = {"value": 0}

    def on_update(_event) -> None:
        counter["value"] += 1
        if counter["value"] % 6 == 0:
            measure_once()

    setattr(builtins, MEASURE_ONCE_KEY, measure_once)
    subscription = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(on_update, name="DGN2 scene layout measurements")
    )
    setattr(builtins, UPDATE_SUBSCRIPTION_KEY, subscription)
    values = measure_once()
    print("Layout measurements:", {k: round(v, 6) for k, v in values.items()})


stage = build_stage()
install_measurements()

DRAFT_USD.parent.mkdir(parents=True, exist_ok=True)
if not omni.usd.get_context().save_as_stage(str(DRAFT_USD)):
    raise RuntimeError(f"Could not save draft stage: {DRAFT_USD}")

# 保存会重新载入Stage，重新绑定数值窗口。
install_measurements()

print("\n[LAYOUT DRAFT CREATED]")
print("Saved:", DRAFT_USD)
print("Move table:", TABLE_ASSEMBLY_PATH)
print("Move robot:", DUAL_ARM_MOUNT_PATH)
print("Timeline remains STOPPED; camera work is deferred.")
print("When satisfied, run scripts/02_export_calibrated_layout.py")
