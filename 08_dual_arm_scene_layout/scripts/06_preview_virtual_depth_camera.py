"""Isaac Sim 5.0 Script Editor：切换到虚拟深度相机的真实第一视角。

运行前必须已经在同一Stage中运行05_create_virtual_depth_camera_frustum.py。
本脚本只切换视口，不拍照、不启动物理。为了人工检查布局，相机预览会明确显示
蓝色抓取区和绿色放置区，同时隐藏视锥与距离标记。正式运行07抓拍时，07会再次
隐藏两个区域，保证这些辅助色块不会污染网络输入。
"""

from __future__ import annotations

import carb.settings
import omni.timeline
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Sdf, UsdGeom


CAMERA_PATH = "/World/Sensors/TopD435iVirtual/Camera"
WIDTH = 1280
HEIGHT = 720

# 预览中隐藏会挡住画面的辅助线，但保留两个区域色块。
HIDE_IN_PREVIEW_PATHS = (
    "/World/Sensors/TopD435iVirtual/Frustum",
    "/World/Markers",
)

SHOW_IN_PREVIEW_PATHS = (
    "/World/Layout/TableAssembly/SourceZone",
    "/World/Layout/TableAssembly/PlacementZone",
)


def configure_preview_visuals(stage) -> None:
    for path in HIDE_IN_PREVIEW_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
    for path in SHOW_IN_PREVIEW_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeVisible()


omni.timeline.get_timeline_interface().stop()
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No active Stage")

camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
    raise RuntimeError(
        "Virtual camera is missing. Run 05_create_virtual_depth_camera_frustum.py first."
    )

configure_preview_visuals(stage)
carb.settings.get_settings().set_bool("/app/viewport/show/camera", False)

viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("No active Isaac Sim viewport")
viewport.camera_path = Sdf.Path(CAMERA_PATH)
viewport.set_texture_resolution((WIDTH, HEIGHT))

print("\n[VIRTUAL DEPTH CAMERA PREVIEW ACTIVE]")
print("camera:", CAMERA_PATH)
print(f"resolution: {WIDTH}x{HEIGHT}")
print("Blue SourceZone and green PlacementZone are visible for layout inspection.")
print("Frustum and distance markers are hidden. Script 07 hides both zones before capture.")
print("Use the viewport camera menu -> Perspective to leave this camera view.")
