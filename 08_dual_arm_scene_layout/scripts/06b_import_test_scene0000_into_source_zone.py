"""Isaac Sim 5.0 Script Editor：把测试集scene_0000原样放入蓝色SourceZone。

输入：
  02_training_dataset/data/scene_datasets/
  wuji2_test60_10upright_10view_v1/scenes/scene_0000/scene_manifest.json

输出到当前打开的Stage：
  /World/Layout/TableAssembly/TestScene0000/Object_XX_...

并写出审计记录：
  08_dual_arm_scene_layout/outputs/test_scene0000_import.json

重要约定：
1. 测试集场景坐标原点是0.50 x 0.30 m桌面中心，z=0是桌面上表面；
2. 当前SourceZone也是0.50 x 0.30 m，因此只做刚体坐标变换，绝不缩放或重排物体；
3. TestScene0000挂在TableAssembly下，所以桌面、蓝区、绿区和物体可整体移动；
4. 此脚本只导入固定视觉网格，不启动物理。真正抓取仿真应另行加载动态资产；
5. scene_0000没有香蕉，严格包含开罐器、记事本、烟灰缸、狗、锤子和笔。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


PROJECT_ROOT = Path("/home/lin/Projects/DexGraspNet2_Wuji2")
TEST_DATASET_ROOT = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/"
    "wuji2_test60_10upright_10view_v1"
)
SCENE_MANIFEST = Path(
    os.environ.get(
        "DGN2_SCENE_MANIFEST",
        str(TEST_DATASET_ROOT / "scenes/scene_0000/scene_manifest.json"),
    )
).resolve()
REPORT_PATH = (
    Path(
        os.environ.get(
            "DGN2_SCENE_IMPORT_REPORT",
            str(PROJECT_ROOT / "08_dual_arm_scene_layout/outputs/test_scene0000_import.json"),
        )
    ).resolve()
)

TABLE_ASSEMBLY_PATH = "/World/Layout/TableAssembly"
SOURCE_ZONE_PATH = f"{TABLE_ASSEMBLY_PATH}/SourceZone"
SCENE_ROOT_PATH = f"{TABLE_ASSEMBLY_PATH}/TestScene0000"
EXPECTED_ZONE_XY_M = np.asarray([0.50, 0.30], dtype=np.float64)
ZONE_SIZE_TOLERANCE_M = 1.0e-4
ZONE_BOUNDS_TOLERANCE_M = 0.004

COLORS = (
    (0.95, 0.38, 0.12),
    (0.20, 0.55, 0.95),
    (0.52, 0.25, 0.82),
    (0.14, 0.72, 0.38),
    (0.95, 0.72, 0.08),
    (0.85, 0.18, 0.42),
)


def gf_matrix_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.asarray([[float(matrix[r][c]) for c in range(4)] for r in range(4)])


def numpy_to_gf_matrix(matrix: np.ndarray) -> Gf.Matrix4d:
    values = np.asarray(matrix, dtype=np.float64)
    return Gf.Matrix4d(*[tuple(map(float, row)) for row in values])


def world_matrix_row(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing required prim: {path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return gf_matrix_to_numpy(matrix)


def rigid_without_scale(matrix_row: np.ndarray) -> np.ndarray:
    """从Gf行向量矩阵剥离Cube尺寸scale，只保留正交旋转和平移。"""

    matrix = np.asarray(matrix_row, dtype=np.float64)
    raw_rotation = matrix[:3, :3]
    row_norms = np.linalg.norm(raw_rotation, axis=1)
    if np.any(row_norms < 1.0e-10):
        raise RuntimeError("SourceZone transform has a zero scale axis")
    normalized = raw_rotation / row_norms[:, None]
    u, _, vt = np.linalg.svd(normalized)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    rigid = np.eye(4, dtype=np.float64)
    rigid[:3, :3] = rotation
    rigid[3, :3] = matrix[3, :3]
    return rigid


def zone_size_m(source_zone_world_row: np.ndarray) -> np.ndarray:
    return np.linalg.norm(source_zone_world_row[:3, :3], axis=1)


def parse_obj(path: Path) -> tuple[np.ndarray, list[int], list[int]]:
    """读取v/f并把任意多边形扇形三角化；无需Isaac外部Python包。"""

    vertices: list[list[float]] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                raw = [token.split("/")[0] for token in line.split()[1:]]
                polygon = []
                for value in raw:
                    index = int(value)
                    polygon.append(index - 1 if index > 0 else len(vertices) + index)
                for offset in range(1, len(polygon) - 1):
                    face_counts.append(3)
                    face_indices.extend((polygon[0], polygon[offset], polygon[offset + 1]))
    if not vertices or not face_indices:
        raise RuntimeError(f"OBJ contains no renderable triangles: {path}")
    return np.asarray(vertices, dtype=np.float64), face_counts, face_indices


def make_material(
    stage: Usd.Stage, path: str, color: tuple[float, float, float]
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*map(float, color))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.58)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def safe_prim_token(text: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return cleaned.strip("_")[:48]


def transform_points_row(points: np.ndarray, matrix_row: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (homogeneous @ matrix_row)[:, :3]


def define_object(
    stage: Usd.Stage,
    scene_object: dict,
    object_order: int,
    source_rigid_world_row: np.ndarray,
    table_assembly_world_row: np.ndarray,
    material: UsdShade.Material,
) -> dict:
    segmentation_id = int(scene_object["segmentation_id"])
    code = str(scene_object["object_code"])
    obj_path = Path(scene_object["asset"]["centered_combined_obj"])
    if not obj_path.is_file():
        raise FileNotFoundError(obj_path)

    vertices, face_counts, face_indices = parse_obj(obj_path)
    test_object_column = np.asarray(
        scene_object["T_world_centered_object"], dtype=np.float64
    )
    # JSON使用列向量；转置后成为OpenUSD/Gf行向量矩阵。
    test_object_row = test_object_column.T
    object_world_row = test_object_row @ source_rigid_world_row
    object_local_row = object_world_row @ np.linalg.inv(table_assembly_world_row)

    name = f"Object_{segmentation_id:02d}_{safe_prim_token(code.split('-', 2)[1])}"
    root_path = f"{SCENE_ROOT_PATH}/{name}"
    root = UsdGeom.Xform.Define(stage, root_path)
    xformable = UsdGeom.Xformable(root.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(numpy_to_gf_matrix(object_local_row))

    prim = root.GetPrim()
    prim.CreateAttribute("dgn2:segmentationId", Sdf.ValueTypeNames.Int).Set(
        segmentation_id
    )
    prim.CreateAttribute("dgn2:objectPoolIndex", Sdf.ValueTypeNames.Int).Set(
        int(scene_object["object_pool_index"])
    )
    prim.CreateAttribute("dgn2:objectCode", Sdf.ValueTypeNames.String).Set(code)
    prim.CreateAttribute("dgn2:classLabel", Sdf.ValueTypeNames.String).Set(
        code.split("-", 2)[1]
    )
    prim.CreateAttribute("dgn2:sourceMesh", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(obj_path))
    )

    mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, vertex)) for vertex in vertices])
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, COLORS[object_order]))])
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    vertices_world = transform_points_row(vertices, object_world_row)
    vertices_source = transform_points_row(vertices, test_object_row)
    source_min = vertices_source.min(axis=0)
    source_max = vertices_source.max(axis=0)
    within_xy = bool(
        source_min[0] >= -0.25 - ZONE_BOUNDS_TOLERANCE_M
        and source_max[0] <= 0.25 + ZONE_BOUNDS_TOLERANCE_M
        and source_min[1] >= -0.15 - ZONE_BOUNDS_TOLERANCE_M
        and source_max[1] <= 0.15 + ZONE_BOUNDS_TOLERANCE_M
    )
    if not within_xy:
        raise RuntimeError(
            f"{code} exceeds SourceZone: min={source_min.tolist()}, max={source_max.tolist()}"
        )
    return {
        "prim_path": root_path,
        "segmentation_id": segmentation_id,
        "object_pool_index": int(scene_object["object_pool_index"]),
        "object_code": code,
        "class_label": code.split("-", 2)[1],
        "source_mesh": str(obj_path),
        "T_test_table_object_column_major": test_object_column.tolist(),
        "Gf_object_local_to_world_row_major": object_world_row.tolist(),
        "source_scene_bounds_min_m": source_min.tolist(),
        "source_scene_bounds_max_m": source_max.tolist(),
        "world_bounds_min_m": vertices_world.min(axis=0).tolist(),
        "world_bounds_max_m": vertices_world.max(axis=0).tolist(),
        "within_source_zone_xy": within_xy,
        "static_visual_only": True,
    }


def main() -> None:
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active Stage")
    if not SCENE_MANIFEST.is_file():
        raise FileNotFoundError(SCENE_MANIFEST)

    table_assembly = stage.GetPrimAtPath(TABLE_ASSEMBLY_PATH)
    source_zone = stage.GetPrimAtPath(SOURCE_ZONE_PATH)
    if not table_assembly.IsValid() or not source_zone.IsValid():
        raise RuntimeError(
            "Open the calibrated layout containing TableAssembly and SourceZone first"
        )

    source_world_row = world_matrix_row(stage, SOURCE_ZONE_PATH)
    actual_size = zone_size_m(source_world_row)
    if not np.allclose(
        actual_size[:2], EXPECTED_ZONE_XY_M, atol=ZONE_SIZE_TOLERANCE_M, rtol=0.0
    ):
        raise RuntimeError(
            f"SourceZone must be 0.50 x 0.30 m, got {actual_size[:2].tolist()}"
        )
    source_rigid_world_row = rigid_without_scale(source_world_row)
    table_assembly_world_row = world_matrix_row(stage, TABLE_ASSEMBLY_PATH)

    manifest = json.loads(SCENE_MANIFEST.read_text(encoding="utf-8"))
    objects = manifest.get("objects", [])
    if len(objects) != 6:
        raise RuntimeError(f"Expected 6 objects in scene_0000, got {len(objects)}")

    # Historical naming note: in the post-physics manifest,
    # ``T_world_centered_object``/``pose_world_object`` retain the original
    # dataset convention and are SourceZone-local.  The unambiguous actual
    # layout-world pose is ``settled_pose_layout_world``.  Validate this
    # relationship instead of guessing from the legacy field name.
    settled_world_pose_input = manifest.get("status") == "dynamic_scene_settled"
    if settled_world_pose_input:
        manifest_world_from_source = np.asarray(
            manifest["world_from_source_zone"], dtype=np.float64
        )
        current_world_from_source = source_rigid_world_row.T
        if not np.allclose(
            manifest_world_from_source,
            current_world_from_source,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Settled manifest and currently opened SourceZone transforms differ; "
                "refusing to rebuild a mismatched camera scene"
            )
        for scene_object in objects:
            source_from_object = np.asarray(
                scene_object["T_world_centered_object"], dtype=np.float64
            )
            recorded_layout_world = np.asarray(
                scene_object["settled_pose_layout_world"], dtype=np.float64
            )
            expected_layout_world = manifest_world_from_source @ source_from_object
            if not np.allclose(
                recorded_layout_world, expected_layout_world, atol=1.0e-6, rtol=0.0
            ):
                raise RuntimeError(
                    f"Settled pose contract mismatch for {scene_object['object_code']}"
                )

    # 仅替换本脚本拥有的节点，不碰桌面、区域、机械臂、相机或标记。
    if stage.GetPrimAtPath(SCENE_ROOT_PATH).IsValid():
        stage.RemovePrim(SCENE_ROOT_PATH)
    scene_root = UsdGeom.Xform.Define(stage, SCENE_ROOT_PATH).GetPrim()
    scene_root.CreateAttribute("dgn2:datasetSplit", Sdf.ValueTypeNames.String).Set(
        "test"
    )
    scene_root.CreateAttribute("dgn2:sceneIndex", Sdf.ValueTypeNames.Int).Set(0)
    scene_root.CreateAttribute("dgn2:sourceManifest", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(SCENE_MANIFEST))
    )
    UsdGeom.Xform.Define(stage, f"{SCENE_ROOT_PATH}/Looks")

    imported = []
    for order, scene_object in enumerate(objects):
        material = make_material(
            stage,
            f"{SCENE_ROOT_PATH}/Looks/Object_{int(scene_object['segmentation_id']):02d}",
            COLORS[order],
        )
        imported.append(
            define_object(
                stage,
                scene_object,
                order,
                source_rigid_world_row,
                table_assembly_world_row,
                material,
            )
        )

    UsdGeom.Imageable(source_zone).MakeVisible()
    placement_zone = stage.GetPrimAtPath(f"{TABLE_ASSEMBLY_PATH}/PlacementZone")
    if placement_zone.IsValid() and placement_zone.IsA(UsdGeom.Imageable):
        UsdGeom.Imageable(placement_zone).MakeVisible()

    report = {
        "schema_version": 1,
        "status": "test_scene_0000_import_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage_identifier": stage.GetRootLayer().identifier,
        "source_manifest": str(SCENE_MANIFEST),
        "dataset_split": "test",
        "scene_index": 0,
        "scene_root_prim": SCENE_ROOT_PATH,
        "parenting_contract": (
            "TestScene0000 is a child of TableAssembly; table, zones and objects move together"
        ),
        "coordinate_mapping": (
            "T_currentWorld_object = T_testTable_object then rigid SourceZone-to-world; "
            "no scale, no re-layout"
        ),
        "source_zone_size_m": actual_size.tolist(),
        "source_zone_rigid_Gf_local_to_world_row_major": (
            source_rigid_world_row.tolist()
        ),
        "physics_mode": "static_visual_only_for_rgbd_capture",
        "input_pose_contract": (
            "validated settled SourceZone-local poses; layout-world audit also present"
            if settled_world_pose_input
            else "legacy dataset SourceZone-local poses"
        ),
        "object_count": len(imported),
        "objects": imported,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n[TEST SCENE 0000 IMPORT COMPLETE]")
    print("parent:", SCENE_ROOT_PATH)
    print("objects:", len(imported))
    for item in imported:
        print(
            f"  seg={item['segmentation_id']:02d}",
            item["class_label"],
            "within-blue-zone=PASS",
        )
    print("table/source/placement remain one TableAssembly: PASS")
    print("report:", REPORT_PATH)
    print("Next: run 06 to inspect, then run 07 to capture RGB-D.")


main()
