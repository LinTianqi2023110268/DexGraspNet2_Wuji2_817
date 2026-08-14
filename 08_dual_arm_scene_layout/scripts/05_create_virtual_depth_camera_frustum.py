"""Isaac Sim 5.0 Script Editor：创建覆盖蓝色抓取区的虚拟深度相机和视锥。

设计目的：当前不知道真实D435i内参，因此先建立一个坐标和接口严格、视野必定
覆盖SourceZone的虚拟针孔相机，把RGB-D -> GroundedSAM -> 点云 -> DGN2串通。

使用方法：
1. 打开/保持 ``manual_layout_calibrated.usda``；
2. 时间线保持STOPPED；
3. 在Script Editor运行本文件；
4. 检查绿色视锥是否覆盖整个蓝区。

本脚本不会启动物理，不需要Isaac Lab，不会修改机械臂或Wuji2上游USD。
"""

from __future__ import annotations

import builtins
import carb.settings
import json
import math
from pathlib import Path

import numpy as np
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
LAYOUT_JSON = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/outputs/manual_layout_calibrated.json"
)
CAMERA_CONFIG_JSON = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/outputs/virtual_depth_camera_preview.json"
)

# 顶部两个D435i外壳中选位置更高的d435i_2作为虚拟光心锚点。
CAMERA_ANCHOR_PATH = (
    "/World/Layout/DualArmMount/DualArm/arm_base_link_d435i_2"
)
SOURCE_ZONE_PATH = "/World/Layout/TableAssembly/SourceZone"
SENSORS_PATH = "/World/Sensors"
RIG_PATH = f"{SENSORS_PATH}/TopD435iVirtual"
CAMERA_PATH = f"{RIG_PATH}/Camera"
FRUSTUM_PATH = f"{RIG_PATH}/Frustum"
WINDOW_KEY = "DGN2_VIRTUAL_DEPTH_CAMERA_WINDOW"

WIDTH = 1280
HEIGHT = 720
ASPECT = WIDTH / HEIGHT
FOV_MARGIN = 1.15
NEAR_M = 0.05
FAR_M = 3.0
HORIZONTAL_APERTURE_MM = 20.955


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-9:
        raise ValueError("Cannot normalize a zero-length direction")
    return vector / norm


def world_matrix(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing required prim: {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return np.asarray(matrix, dtype=np.float64)


def transform_point_row(point_xyz, matrix_row_major: np.ndarray) -> np.ndarray:
    homogeneous = np.asarray([*point_xyz, 1.0], dtype=np.float64)
    return (homogeneous @ matrix_row_major)[:3]


def source_zone_corners_world(stage: Usd.Stage):
    prim = stage.GetPrimAtPath(SOURCE_ZONE_PATH)
    cube = UsdGeom.Cube(prim)
    if not prim.IsValid() or not cube:
        raise RuntimeError(f"Expected Cube source zone: {SOURCE_ZONE_PATH}")
    matrix = world_matrix(stage, SOURCE_ZONE_PATH)
    half = 0.5 * float(cube.GetSizeAttr().Get() or 1.0)
    center = transform_point_row((0.0, 0.0, 0.0), matrix)
    corners = np.asarray(
        [
            transform_point_row((sx * half, sy * half, 0.0), matrix)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ],
        dtype=np.float64,
    )
    return center, corners


def camera_basis(eye: np.ndarray, target: np.ndarray):
    """返回OpenCV语义轴：right、down、forward，均用世界坐标表示。"""
    forward = normalize(target - eye)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    down = -up
    return right, down, forward


def required_fov(eye, zone_corners, right, down, forward):
    relative = zone_corners - eye[None, :]
    x = relative @ right
    y = relative @ down
    z = relative @ forward
    if np.any(z <= NEAR_M):
        raise RuntimeError("SourceZone lies behind or too close to the virtual camera")
    required_tan_x = float(np.max(np.abs(x) / z))
    required_tan_y = float(np.max(np.abs(y) / z))
    # 单一焦距、方形像素：tan(hfov/2)=aspect*tan(vfov/2)。
    tan_half_h = max(required_tan_x, ASPECT * required_tan_y) * FOV_MARGIN
    tan_half_v = tan_half_h / ASPECT
    horizontal_fov_deg = math.degrees(2.0 * math.atan(tan_half_h))
    vertical_fov_deg = math.degrees(2.0 * math.atan(tan_half_v))
    return tan_half_h, tan_half_v, horizontal_fov_deg, vertical_fov_deg


def make_usd_camera_matrix(eye, right, down, forward) -> Gf.Matrix4d:
    """OpenCV(+X右,+Y下,+Z前)转换为USD Camera(+X右,+Y上,-Z前)。"""
    up = -down
    backward = -forward
    # Gf采用行向量约定；三个局部轴在世界中的方向写入前三行。
    return Gf.Matrix4d(
        (float(right[0]), float(right[1]), float(right[2]), 0.0),
        (float(up[0]), float(up[1]), float(up[2]), 0.0),
        (float(backward[0]), float(backward[1]), float(backward[2]), 0.0),
        (float(eye[0]), float(eye[1]), float(eye[2]), 1.0),
    )


def set_matrix_xform(prim: Usd.Prim, matrix: Gf.Matrix4d) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix)


def define_line(stage, path, points, color, width=0.006):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    curve.CreateWidthsAttr([float(width)] * len(points))
    curve.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])


def define_marker(stage, path, position, color, radius=0.012):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(
        Gf.Vec3d(*map(float, position))
    )


def build_frustum(stage, eye, target, zone_corners, right, down, forward, tan_h, tan_v):
    if stage.GetPrimAtPath(RIG_PATH).IsValid():
        stage.RemovePrim(RIG_PATH)
    UsdGeom.Xform.Define(stage, SENSORS_PATH)
    rig = UsdGeom.Xform.Define(stage, RIG_PATH)
    rig_prim = rig.GetPrim()
    rig_prim.CreateAttribute("dgn2:anchorPrim", Sdf.ValueTypeNames.String, custom=True).Set(
        CAMERA_ANCHOR_PATH
    )
    rig_prim.CreateAttribute("dgn2:targetPrim", Sdf.ValueTypeNames.String, custom=True).Set(
        SOURCE_ZONE_PATH
    )

    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    set_matrix_xform(camera.GetPrim(), make_usd_camera_matrix(eye, right, down, forward))
    vertical_aperture_mm = HORIZONTAL_APERTURE_MM / ASPECT
    focal_length_mm = HORIZONTAL_APERTURE_MM / (2.0 * tan_h)
    camera.CreateHorizontalApertureAttr(float(HORIZONTAL_APERTURE_MM))
    camera.CreateVerticalApertureAttr(float(vertical_aperture_mm))
    camera.CreateFocalLengthAttr(float(focal_length_mm))
    camera.CreateClippingRangeAttr(Gf.Vec2f(float(NEAR_M), float(FAR_M)))

    # 在蓝区中心的光轴深度处画视锥截面，直观看出边界余量。
    center_depth = float(np.dot(target - eye, forward))
    half_width = center_depth * tan_h
    half_height = center_depth * tan_v
    plane_center = eye + center_depth * forward
    plane_corners = np.asarray(
        [
            plane_center + sx * half_width * right + sy * half_height * down
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
    UsdGeom.Xform.Define(stage, FRUSTUM_PATH)
    green = (0.15, 1.0, 0.25)
    for index, corner in enumerate(plane_corners):
        define_line(stage, f"{FRUSTUM_PATH}/Ray_{index}", [eye, corner], green)
    define_line(
        stage,
        f"{FRUSTUM_PATH}/ImagePlane",
        [*plane_corners, plane_corners[0]],
        green,
        width=0.009,
    )
    define_line(stage, f"{FRUSTUM_PATH}/OpticalAxis", [eye, target], (1.0, 0.9, 0.1), 0.008)
    define_line(
        stage,
        f"{FRUSTUM_PATH}/SourceZoneBoundary",
        [*zone_corners, zone_corners[0]],
        (0.1, 0.35, 1.0),
        0.012,
    )
    define_marker(stage, f"{FRUSTUM_PATH}/CameraCenter", eye, (1.0, 1.0, 1.0), 0.014)
    define_marker(stage, f"{FRUSTUM_PATH}/TargetCenter", target, (1.0, 0.9, 0.1), 0.014)
    return focal_length_mm, vertical_aperture_mm, plane_corners


def coverage_audit(eye, zone_corners, right, down, forward, tan_h, tan_v):
    relative = zone_corners - eye[None, :]
    x, y, z = relative @ right, relative @ down, relative @ forward
    normalized_x = x / (z * tan_h)
    normalized_y = y / (z * tan_v)
    inside = (z > NEAR_M) & (z < FAR_M) & (np.abs(normalized_x) <= 1.0) & (
        np.abs(normalized_y) <= 1.0
    )
    return bool(np.all(inside)), normalized_x, normalized_y, z


def install_window(report):
    old = getattr(builtins, WINDOW_KEY, None)
    if old:
        old.visible = False
    window = ui.Window("DGN2 Virtual Depth Camera Coverage", width=620, height=300)
    with window.frame:
        with ui.VStack(spacing=5):
            ui.Label(f"Coverage: {'PASS' if report['coverage_pass'] else 'FAIL'}")
            ui.Label(f"Anchor: {report['anchor_prim']}")
            ui.Label(f"Camera world XYZ: {report['camera_position_world_m']}")
            ui.Label(f"Target world XYZ: {report['target_position_world_m']}")
            ui.Label(
                f"Resolution: {WIDTH}x{HEIGHT} | HFOV={report['horizontal_fov_deg']:.2f} deg | "
                f"VFOV={report['vertical_fov_deg']:.2f} deg"
            )
            ui.Label(
                f"Focal={report['focal_length_mm']:.3f} mm (synthetic) | "
                f"clip={NEAR_M:.2f}-{FAR_M:.2f} m | margin={FOV_MARGIN:.2f}x"
            )
            ui.Label("Green = camera frustum; blue = source-zone boundary; yellow = optical axis")
            ui.Label("This is a functional virtual camera, not a calibrated physical D435i.")
    window.visible = True
    setattr(builtins, WINDOW_KEY, window)


omni.timeline.get_timeline_interface().stop()
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No active stage")
if not LAYOUT_JSON.is_file():
    raise FileNotFoundError(f"Run 02_export_calibrated_layout.py first: {LAYOUT_JSON}")

# 只隐藏视口中的Camera辅助图标/模型，不隐藏Camera Prim，也不影响拍摄。
# 手工等价操作是Viewport眼睛菜单 -> Show By Type -> Cameras取消勾选。
carb.settings.get_settings().set_bool("/app/viewport/show/camera", False)

eye = world_matrix(stage, CAMERA_ANCHOR_PATH)[3, :3].copy()
target, zone_corners = source_zone_corners_world(stage)
right, down, forward = camera_basis(eye, target)
tan_h, tan_v, hfov_deg, vfov_deg = required_fov(
    eye, zone_corners, right, down, forward
)
focal_mm, vertical_aperture_mm, plane_corners = build_frustum(
    stage, eye, target, zone_corners, right, down, forward, tan_h, tan_v
)
passed, normalized_x, normalized_y, depths = coverage_audit(
    eye, zone_corners, right, down, forward, tan_h, tan_v
)

# GroundedSAM使用的OpenCV相机外参：列分别为世界中的right/down/forward。
T_world_camera = np.eye(4, dtype=np.float64)
T_world_camera[:3, :3] = np.column_stack((right, down, forward))
T_world_camera[:3, 3] = eye
fx = WIDTH / (2.0 * tan_h)
fy = HEIGHT / (2.0 * tan_v)
K = np.asarray(
    [[fx, 0.0, WIDTH / 2.0], [0.0, fy, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

report = {
    "schema_version": 1,
    "status": "virtual_depth_camera_frustum_preview",
    "coverage_pass": passed,
    "anchor_prim": CAMERA_ANCHOR_PATH,
    "target_prim": SOURCE_ZONE_PATH,
    "camera_prim": CAMERA_PATH,
    "resolution_wh": [WIDTH, HEIGHT],
    "camera_position_world_m": eye.tolist(),
    "target_position_world_m": target.tolist(),
    "horizontal_fov_deg": hfov_deg,
    "vertical_fov_deg": vfov_deg,
    "fov_margin_multiplier": FOV_MARGIN,
    "near_far_m": [NEAR_M, FAR_M],
    "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM,
    "vertical_aperture_mm": vertical_aperture_mm,
    "focal_length_mm": focal_mm,
    "intrinsics_K": K.tolist(),
    "T_world_camera_opencv": T_world_camera.tolist(),
    "opencv_axes": "+x image-right, +y image-down, +z camera-forward",
    "source_corner_normalized_xy": np.column_stack((normalized_x, normalized_y)).tolist(),
    "source_corner_depth_m": depths.tolist(),
    "interface_outputs_next": [
        "rgb.png",
        "depth_m.npy",
        "intrinsics.npy",
        "T_world_camera.npy",
        "capture_manifest.json",
    ],
}
CAMERA_CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
CAMERA_CONFIG_JSON.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
install_window(report)

print("\n[VIRTUAL DEPTH CAMERA FRUSTUM CREATED]")
print("coverage:", "PASS" if passed else "FAIL")
print("camera:", CAMERA_PATH)
print("anchor:", CAMERA_ANCHOR_PATH)
print("camera world xyz:", np.round(eye, 6).tolist())
print("target world xyz:", np.round(target, 6).tolist())
print(f"HFOV={hfov_deg:.3f} deg, VFOV={vfov_deg:.3f} deg")
print("preview config:", CAMERA_CONFIG_JSON)
if not passed:
    raise RuntimeError("Virtual camera frustum does not cover all four SourceZone corners")
