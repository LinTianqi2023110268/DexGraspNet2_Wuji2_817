"""Isaac Sim 5.0 Script Editor：导出手动摆好的桌面与双臂场景。

必须先运行01_create_manual_layout.py并完成GUI调整。本脚本只记录当前几何
变换，不创建相机、不启动物理。
"""

from __future__ import annotations

import builtins
import json
from datetime import datetime, timezone
from pathlib import Path

import omni.timeline
import omni.usd
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
OUTPUT_USD = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda"
)
OUTPUT_JSON = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
)

PATHS = {
    "table_assembly": "/World/Layout/TableAssembly",
    "table": "/World/Layout/TableAssembly/Table",
    "source_zone": "/World/Layout/TableAssembly/SourceZone",
    "placement_zone": "/World/Layout/TableAssembly/PlacementZone",
    "dual_arm_mount": "/World/Layout/DualArmMount",
    "dual_arm": "/World/Layout/DualArmMount/DualArm",
}


def matrix_row_major(matrix) -> list[list[float]]:
    return [
        [float(matrix[row][column]) for column in range(4)]
        for row in range(4)
    ]


def world_record(stage: Usd.Stage, path: str) -> dict:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing required prim: {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return {
        "prim_path": path,
        "position_world_m": [float(value) for value in translation],
        "Gf_local_to_world_row_major": matrix_row_major(matrix),
        "matrix_convention": "OpenUSD Gf.Matrix4d row-vector convention",
    }


def cube_local_size_m(stage: Usd.Stage, path: str) -> list[float]:
    """读取Property面板中当前Cube尺寸，而不是沿用创建脚本的初始常量。"""
    prim = stage.GetPrimAtPath(path)
    cube = UsdGeom.Cube(prim)
    if not prim.IsValid() or not cube:
        raise RuntimeError(f"Expected Cube prim: {path}")
    base_size = cube.GetSizeAttr().Get()
    base_size = 1.0 if base_size is None else float(base_size)
    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is None:
        scale = (1.0, 1.0, 1.0)
    return [abs(base_size * float(value)) for value in scale]


def current_revolute_joint_positions_deg(stage: Usd.Stage) -> dict[str, float]:
    """保存当前35维关节姿态；JointState优先，Drive Target作为回退。"""
    result = {}
    robot_path = PATHS["dual_arm"]
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(robot_path + "/"):
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        value = None
        state = PhysxSchema.JointStateAPI.Get(prim, "angular")
        if state:
            value = state.GetPositionAttr().Get()
        if value is None:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                value = drive.GetTargetPositionAttr().Get()
        if value is None:
            raise RuntimeError(f"Joint has no current position: {prim.GetPath()}")
        if prim.GetName() in result:
            raise RuntimeError(f"Duplicate revolute joint name: {prim.GetName()}")
        result[prim.GetName()] = float(value)
    if len(result) != 35:
        raise RuntimeError(f"Expected 35 revolute joints, found {len(result)}")
    return dict(sorted(result.items()))


omni.timeline.get_timeline_interface().stop()
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("No open stage")

measure_once = getattr(builtins, "DGN2_LAYOUT_MEASURE_ONCE", None)
measurements = measure_once() if callable(measure_once) else {}
records = {name: world_record(stage, path) for name, path in PATHS.items()}
table_size_m = cube_local_size_m(stage, PATHS["table"])
source_zone_size_m = cube_local_size_m(stage, PATHS["source_zone"])
placement_zone_size_m = cube_local_size_m(stage, PATHS["placement_zone"])
joint_positions_deg = current_revolute_joint_positions_deg(stage)

payload = {
    "schema_version": 1,
    "status": "table_and_dual_arm_layout_calibrated_camera_pending",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "coordinate_contract": {
        "world_up": "+Z",
        "table_top_nominal_z_m": 0.0,
        "matrix_convention": "OpenUSD Gf.Matrix4d row-vector convention",
    },
    "geometry": {
        "table_size_m": table_size_m,
        "source_zone_size_m": source_zone_size_m,
        "placement_zone_size_m": placement_zone_size_m,
        "zone_collision": False,
        "table_collision": True,
    },
    "transforms": records,
    "revolute_joint_positions_deg": joint_positions_deg,
    "measurements_m": {
        key: float(value) for key, value in measurements.items()
    },
    "camera": {
        "status": "deliberately_not_created_in_this_stage",
        "next_step": "add and calibrate top D435i after layout is accepted",
    },
    "next_required_checks": [
        "visually confirm robot base and table do not penetrate",
        "confirm both arms can reach the source and placement zones",
        "freeze the accepted table and DualArmMount transforms",
        "add the top D435i optical frame and camera in the next stage",
    ],
}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
OUTPUT_USD.parent.mkdir(parents=True, exist_ok=True)
if not omni.usd.get_context().save_as_stage(str(OUTPUT_USD)):
    raise RuntimeError(f"Could not save calibrated stage: {OUTPUT_USD}")

print("\n[CALIBRATED LAYOUT EXPORTED]")
print("USD:", OUTPUT_USD)
print("JSON:", OUTPUT_JSON)
print("Camera remains pending by design.")
