"""Isaac Sim 5.0 Script Editor：从虚拟相机抓拍一帧严格对齐的RGB-D。

输出目录：08_dual_arm_scene_layout/captures/latest

输出接口直接匹配GroundedSAM：
  rgb.png                 HxWx3 uint8
  depth_m.npy             HxW float32，沿相机成像平面的米制深度
  intrinsics.npy          3x3 OpenCV内参K
  T_world_camera.npy      4x4 OpenCV相机到世界变换
  depth_preview.png       仅供人眼查看，不参与计算
  capture_manifest.json   完整数据契约和质量检查
  scene_snapshot.json     抓拍时关键实体世界变换

脚本异步执行以保持Isaac Sim界面响应；看到[SINGLE RGBD CAPTURE COMPLETE]才算完成。
"""

from __future__ import annotations

import asyncio
import builtins
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom


PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
OUTPUT_DIR = Path(
    os.environ.get(
        "DGN2_RGBD_OUTPUT_DIR",
        str(PROJECT_ROOT / "08_dual_arm_scene_layout/captures/latest"),
    )
).resolve()
CAMERA_PATH = "/World/Sensors/TopD435iVirtual/Camera"
SOURCE_ZONE_PATH = "/World/Layout/TableAssembly/SourceZone"
TEST_SCENE_PATH = "/World/Layout/TableAssembly/TestScene0000"
TEST_SCENE_REPORT = (
    PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/test_scene0000_import.json"
)
EXPECTED_TEST_OBJECT_COUNT = 6
WIDTH = 1280
HEIGHT = 720
TASK_KEY = "DGN2_SINGLE_RGBD_CAPTURE_TASK"

DEBUG_VISUAL_PATHS = (
    "/World/Sensors/TopD435iVirtual/Frustum",
    "/World/Markers",
    SOURCE_ZONE_PATH,
    "/World/Layout/TableAssembly/PlacementZone",
)

SNAPSHOT_PATHS = {
    "table": "/World/Layout/TableAssembly/Table",
    "source_zone": SOURCE_ZONE_PATH,
    "placement_zone": "/World/Layout/TableAssembly/PlacementZone",
    "dual_arm_mount": "/World/Layout/DualArmMount",
    "virtual_camera": CAMERA_PATH,
}


def hide_debug_visuals(stage: Usd.Stage) -> None:
    for path in DEBUG_VISUAL_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()


def validate_test_scene(stage: Usd.Stage) -> None:
    """避免误把空桌面的抓拍覆盖到captures/latest。"""

    root = stage.GetPrimAtPath(TEST_SCENE_PATH)
    if not root.IsValid():
        raise RuntimeError(
            "Test scene is missing. Run "
            "06b_import_test_scene0000_into_source_zone.py before capture."
        )
    objects = [
        child
        for child in root.GetChildren()
        if child.GetAttribute("dgn2:objectCode").IsValid()
    ]
    if len(objects) != EXPECTED_TEST_OBJECT_COUNT:
        raise RuntimeError(
            f"Incomplete test scene: expected {EXPECTED_TEST_OBJECT_COUNT} objects, "
            f"got {len(objects)}"
        )


def world_matrix(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing required prim: {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return np.asarray(matrix, dtype=np.float64)


def camera_calibration(stage: Usd.Stage):
    """从当前USD Camera读取K及OpenCV语义的T_world_camera。"""
    prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            "Virtual camera is missing. Run 05_create_virtual_depth_camera_frustum.py first."
        )
    camera = UsdGeom.Camera(prim)
    focal_mm = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture_mm = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture_mm = float(camera.GetVerticalApertureAttr().Get())
    clipping = camera.GetClippingRangeAttr().Get()

    fx = focal_mm * WIDTH / horizontal_aperture_mm
    fy = focal_mm * HEIGHT / vertical_aperture_mm
    K = np.asarray(
        [[fx, 0.0, WIDTH / 2.0], [0.0, fy, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # USD Camera局部轴为+X右、+Y上、-Z前；OpenCV为+X右、+Y下、+Z前。
    usd_world = world_matrix(stage, CAMERA_PATH)
    right_world = usd_world[0, :3]
    down_world = -usd_world[1, :3]
    forward_world = -usd_world[2, :3]
    eye_world = usd_world[3, :3]
    T_world_camera = np.eye(4, dtype=np.float64)
    T_world_camera[:3, :3] = np.column_stack(
        (right_world, down_world, forward_world)
    )
    T_world_camera[:3, 3] = eye_world
    return K, T_world_camera, {
        "focal_length_mm": focal_mm,
        "horizontal_aperture_mm": horizontal_aperture_mm,
        "vertical_aperture_mm": vertical_aperture_mm,
        "near_far_m": [float(clipping[0]), float(clipping[1])],
    }


def unpack_annotator_data(value) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    return np.asarray(value)


def make_depth_preview(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("Depth annotator returned no finite positive depth pixels")
    values = depth[valid]
    near = float(np.percentile(values, 1.0))
    far = float(np.percentile(values, 99.0))
    if far <= near:
        far = near + 1.0e-6
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip((far - depth[valid]) / (far - near), 0.0, 1.0)
    preview = np.round(normalized * 255.0).astype(np.uint8)
    return preview, {
        "valid_pixel_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "minimum_depth_m": float(values.min()),
        "maximum_depth_m": float(values.max()),
        "preview_percentile_range_m": [near, far],
    }


def write_scene_snapshot(stage: Usd.Stage) -> dict:
    transforms = {}
    for name, path in SNAPSHOT_PATHS.items():
        matrix = world_matrix(stage, path)
        transforms[name] = {
            "prim_path": path,
            "Gf_local_to_world_row_major": matrix.tolist(),
            "position_world_m": matrix[3, :3].tolist(),
        }
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage_identifier": stage.GetRootLayer().identifier,
        "timeline_state": "stopped",
        "transforms": transforms,
    }
    test_scene = stage.GetPrimAtPath(TEST_SCENE_PATH)
    if test_scene.IsValid():
        scene_objects = []
        for child in test_scene.GetChildren():
            code_attr = child.GetAttribute("dgn2:objectCode")
            if not code_attr.IsValid():
                continue
            matrix = world_matrix(stage, str(child.GetPath()))
            scene_objects.append(
                {
                    "prim_path": str(child.GetPath()),
                    "object_code": str(code_attr.Get()),
                    "class_label": str(child.GetAttribute("dgn2:classLabel").Get()),
                    "segmentation_id": int(
                        child.GetAttribute("dgn2:segmentationId").Get()
                    ),
                    "Gf_local_to_world_row_major": matrix.tolist(),
                    "position_world_m": matrix[3, :3].tolist(),
                }
            )
        payload["test_scene"] = {
            "prim_path": TEST_SCENE_PATH,
            "dataset_split": "test",
            "scene_index": 0,
            "object_count": len(scene_objects),
            "objects": scene_objects,
        }
    (OUTPUT_DIR / "scene_snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


async def capture_once() -> None:
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active Stage")

    validate_test_scene(stage)
    hide_debug_visuals(stage)
    carb.settings.get_settings().set_bool("/app/viewport/show/camera", False)
    K, T_world_camera, camera_model = camera_calibration(stage)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product(CAMERA_PATH, (WIDTH, HEIGHT))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator(
        "distance_to_image_plane"
    )
    rgb_annotator.attach(render_product)
    depth_annotator.attach(render_product)

    try:
        # 给材质、光照和RTX传感器缓存留出更新帧；delta_time=0不会推进物理。
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()
        await rep.orchestrator.step_async(
            rt_subframes=8,
            pause_timeline=True,
            delta_time=0.0,
            wait_for_render=True,
        )
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        rgb_raw = unpack_annotator_data(
            rgb_annotator.get_data(device="cpu", do_array_copy=True)
        )
        depth = unpack_annotator_data(
            depth_annotator.get_data(device="cpu", do_array_copy=True)
        ).astype(np.float32)

        if rgb_raw.ndim != 3 or rgb_raw.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError(f"Unexpected RGB shape: {rgb_raw.shape}")
        rgb = rgb_raw[..., :3].astype(np.uint8, copy=False)
        depth = np.squeeze(depth)
        if depth.shape != (HEIGHT, WIDTH):
            raise RuntimeError(f"Unexpected depth shape: {depth.shape}")

        preview, depth_stats = make_depth_preview(depth)
        Image.fromarray(rgb, mode="RGB").save(OUTPUT_DIR / "rgb.png")
        Image.fromarray(preview, mode="L").save(OUTPUT_DIR / "depth_preview.png")
        np.save(OUTPUT_DIR / "depth_m.npy", depth)
        np.save(OUTPUT_DIR / "intrinsics.npy", K)
        np.save(OUTPUT_DIR / "T_world_camera.npy", T_world_camera)
        snapshot = write_scene_snapshot(stage)

        manifest = {
            "schema_version": 1,
            "status": "single_rgbd_capture_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "camera_prim": CAMERA_PATH,
            "resolution_wh": [WIDTH, HEIGHT],
            "rgb": {"file": "rgb.png", "dtype": "uint8", "shape": list(rgb.shape)},
            "depth": {
                "file": "depth_m.npy",
                "dtype": "float32",
                "shape": list(depth.shape),
                "meaning": "distance_to_image_plane in metres",
                **depth_stats,
            },
            "intrinsics": {"file": "intrinsics.npy", "K": K.tolist()},
            "extrinsics": {
                "file": "T_world_camera.npy",
                "matrix": T_world_camera.tolist(),
                "convention": "OpenCV camera-to-world: +x right, +y down, +z forward",
            },
            "camera_model": camera_model,
            "debug_geometry_hidden": list(DEBUG_VISUAL_PATHS),
            "scene_snapshot": "scene_snapshot.json",
            "stage_identifier": snapshot["stage_identifier"],
            "grounded_sam_compatible": True,
        }
        if "test_scene" in snapshot:
            manifest["test_scene"] = snapshot["test_scene"]
            manifest["test_scene_import_report"] = (
                str(TEST_SCENE_REPORT) if TEST_SCENE_REPORT.is_file() else None
            )
        (OUTPUT_DIR / "capture_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\n[SINGLE RGBD CAPTURE COMPLETE]")
        print("output:", OUTPUT_DIR)
        print("rgb shape:", rgb.shape)
        print("depth shape:", depth.shape)
        print("valid depth fraction:", round(depth_stats["valid_fraction"], 6))
        print("depth range m:", [
            round(depth_stats["minimum_depth_m"], 6),
            round(depth_stats["maximum_depth_m"], 6),
        ])
        print("GroundedSAM interface: PASS")
    finally:
        try:
            rgb_annotator.detach(render_product)
        except Exception:
            pass
        try:
            depth_annotator.detach(render_product)
        except Exception:
            pass
        try:
            render_product.destroy()
        except Exception:
            pass
        try:
            await rep.orchestrator.stop_async()
        except Exception:
            pass
        timeline.stop()


if os.environ.get("DGN2_RGBD_LIBRARY_MODE") != "1":
    old_task = getattr(builtins, TASK_KEY, None)
    if old_task is not None and not old_task.done():
        raise RuntimeError("A single RGB-D capture is already running")
    task = asyncio.ensure_future(capture_once())
    setattr(builtins, TASK_KEY, task)

    def report_completion(finished_task) -> None:
        try:
            finished_task.result()
        except Exception as error:
            print("\n[SINGLE RGBD CAPTURE FAILED]")
            print(type(error).__name__ + ":", error)

    task.add_done_callback(report_completion)
    print("[SINGLE RGBD CAPTURE STARTED] Wait for the COMPLETE message.")
