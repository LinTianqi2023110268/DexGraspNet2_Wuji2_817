#!/usr/bin/env python3
"""Read-only validation of the consolidated DexGraspNet2-Wuji2 project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from wuji2_dgn2.official_asset import verify_canonical_assets  # noqa: E402

failures: list[str] = []


def require_file(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        failures.append(f"missing file {relative}")
    return path


def require_dir(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_dir():
        failures.append(f"missing directory {relative}")
    return path


for relative in (
    "README.md",
    "01_environment/verify.py",
    "config/project.json",
    "config/wuji2_joint_order.json",
    "02_training_dataset/README.md",
    "03_prediction_network/official_core/src/network/model.py",
    "04_training/scripts/train_wuji2_scratch.py",
    "05_inference/01_predict_single_view.py",
    "05_inference/02_filter_scene_collisions.py",
    "06_leap_to_wuji2_final_pipeline/README.md",
    "06_leap_to_wuji2_final_pipeline/00_shared/isaacsim/common_import.py",
    "06_leap_to_wuji2_final_pipeline/00_shared/isaacsim/common_execute.py",
    "07_wuji2_network_3p3r_sim/README.md",
    "07_wuji2_network_3p3r_sim/00_config/test_5scene.json",
    "07_wuji2_network_3p3r_sim/00_config/select_sim_pose.py",
    "07_wuji2_network_3p3r_sim/02_scripts/01_run_network_and_filter.py",
    "07_wuji2_network_3p3r_sim/02_scripts/02_select_top2_per_object.py",
    "07_wuji2_network_3p3r_sim/02_scripts/03_view_selected_poses.py",
    "07_wuji2_network_3p3r_sim/02_scripts/04_build_selected_sim_3p3r.py",
    "07_wuji2_network_3p3r_sim/02_scripts/05_view_top35_grid.py",
    "07_wuji2_network_3p3r_sim/02_scripts/06_view_current_tip_directions.py",
    "07_wuji2_network_3p3r_sim/03_runtime/import_scene_with_3p3r.py",
    "07_wuji2_network_3p3r_sim/03_runtime/execute_native_grasp.py",
    "07_wuji2_network_3p3r_sim/01_cases/selected_native_case/case.json",
    "07_wuji2_network_3p3r_sim/01_cases/selected_native_case/03_waypoints/native_wuji2_3p3r_waypoints.npz",
    "07_wuji2_network_3p3r_sim/01_cases/selected_native_case/05_isaacsim/01_import.py",
    "07_wuji2_network_3p3r_sim/01_cases/selected_native_case/05_isaacsim/02_execute.py",
):
    require_file(relative)

for relative in (
    "02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_v1",
    "02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1",
    "02_training_dataset/data/scene_datasets/wuji2_test60_10upright_10view_v1",
    "06_leap_to_wuji2_final_pipeline/01_cases",
    "07_wuji2_network_3p3r_sim/01_cases",
):
    require_dir(relative)

# Old top-level simulation/test trees must not return after consolidation.
for obsolete in ("06_isaacsim", "07_test_inference_and_sim"):
    if (ROOT / obsolete).exists():
        failures.append(f"obsolete top-level path still exists: {obsolete}")

joint_order_path = ROOT / "config/wuji2_joint_order.json"
if joint_order_path.is_file():
    joint_order = json.loads(joint_order_path.read_text(encoding="utf-8"))["joint_order"]
    if len(joint_order) != 20 or len(set(joint_order)) != 20:
        failures.append("Wuji2 joint order is not 20 unique names")

try:
    verify_canonical_assets()
except Exception as exc:
    failures.append(f"official Wuji2 asset contract failed: {exc}")

# Five test scenes, two retained poses for every visible object.
selection_path = ROOT / "07_wuji2_network_3p3r_sim/00_config/test_5scene.json"
if selection_path.is_file():
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if len(selection.get("scenes", [])) != 5:
        failures.append("07 test selection must contain exactly five scenes")
    for record in selection.get("scenes", []):
        scene = int(record["scene_index"])
        view = int(record["view_index"])
        prediction_root = ROOT / (
            f"07_wuji2_network_3p3r_sim/01_cases/"
            f"scene_{scene:04d}_view_{view:04d}/02_predictions"
        )
        for name in (
            "raw_predictions.npz", "filtered_predictions.npz",
            "balanced_raw_predictions.npz",
            "balanced_filtered_predictions_all_diagnostics.npz",
            "selected_top2.npz", "selected_top2.json",
        ):
            if not (prediction_root / name).is_file():
                failures.append(f"07 prediction missing: {prediction_root / name}")
        selected = prediction_root / "selected_top2.npz"
        if selected.is_file():
            with np.load(selected, allow_pickle=False) as archive:
                target = np.asarray(archive["target_segmentation_id"], dtype=np.int64)
                ranks = np.asarray(archive["pose_rank_within_object"], dtype=np.int64)
            expected_ids = {int(item["segmentation_id"]) for item in record["objects"]}
            if len(target) != 2 * len(expected_ids):
                failures.append(f"scene {scene} does not contain two poses per object")
            for object_id in expected_ids:
                if sorted(ranks[target == object_id].tolist()) != [0, 1]:
                    failures.append(f"scene {scene} object {object_id} lacks rank 0/1")

# The selected native task must preserve every old action endpoint while adding 6 root DOFs.
task_path = ROOT / (
    "07_wuji2_network_3p3r_sim/01_cases/selected_native_case/"
    "03_waypoints/native_wuji2_3p3r_waypoints.npz"
)
if task_path.is_file():
    with np.load(task_path, allow_pickle=False) as job:
        names = [str(x) for x in job["waypoint_names"].tolist()]
        poses = np.asarray(job["waypoint_pose_world"][0], dtype=np.float64)
        root_dofs = np.asarray(job["waypoint_root_dofs"][0], dtype=np.float64)
        joints = np.asarray(job["waypoint_joint_positions"][0], dtype=np.float64)
        checks = {
            "waypoint names": names == ["pregrasp", "cover_open", "grasp", "squeeze", "lift"],
            "five root poses": poses.shape == (5, 4, 4),
            "five 3P+3R targets": root_dofs.shape == (5, 6),
            "five q20 targets": joints.shape == (5, 20),
            "100 mm tiger-mouth approach": np.isclose(np.linalg.norm(poses[1, :3, 3] - poses[0, :3, 3]), 0.10, atol=1e-6),
            "fixed wrist during close": np.allclose(poses[1], poses[2], atol=1e-7),
            "fixed wrist during squeeze": np.allclose(poses[2], poses[3], atol=1e-7),
            "70 mm world-Z lift": np.allclose(poses[4, :3, 3] - poses[3, :3, 3], [0, 0, 0.07], atol=1e-6),
            "30 mm squeeze metadata": np.isclose(float(job["squeeze_width_m"]), 0.03),
            "local +Y squeeze policy": str(job["squeeze_dense_policy"]) == "linear_q_grasp_to_wuji2_local_plus_y_30mm_keep_z_false",
            "continuous gravity": str(job["gravity_policy"]) == "continuous_-9.81",
            "3P+3R root": str(job["root_control_policy"]) == "leap_isomorphic_3P3R_force_position_K800_D20",
        }
        failures.extend(f"native task contract failed: {name}" for name, passed in checks.items() if not passed)

# The currently confirmed native baseline must remain the successful rank-05
# source462 run, not a stale result from a previously generated candidate.
native_case_path = ROOT / "07_wuji2_network_3p3r_sim/01_cases/selected_native_case/case.json"
native_result_path = ROOT / "07_wuji2_network_3p3r_sim/01_cases/selected_native_case/05_isaacsim/final_result.json"
if native_case_path.is_file() and native_result_path.is_file():
    native_case = json.loads(native_case_path.read_text(encoding="utf-8"))
    native_result = json.loads(native_result_path.read_text(encoding="utf-8"))
    expected_source = int(native_case["source_candidate_index"])
    if int(native_result.get("source_candidate_index", -1)) != expected_source:
        failures.append("native result belongs to a stale source candidate")
    if native_result.get("status") == "native_wuji2_3p3r_execution_complete":
        if not bool(native_result.get("target_specific_success")):
            failures.append("completed native result did not lift the target")

# Current entries/docs may not depend on deleted top-level 06/07 trees.
current_files = [
    ROOT / "README.md",
    ROOT / "docs/PROJECT_STRUCTURE.md",
    ROOT / "05_inference/README.md",
    *sorted((ROOT / "07_wuji2_network_3p3r_sim").rglob("*.py")),
    *sorted((ROOT / "07_wuji2_network_3p3r_sim").rglob("*.md")),
    *sorted((ROOT / "07_wuji2_network_3p3r_sim").rglob("*.json")),
]
for path in current_files:
    text = path.read_text(encoding="utf-8")
    for token in ("07_test_inference_and_sim", "06_isaacsim/common"):
        if token in text:
            failures.append(f"stale dependency in {path.relative_to(ROOT)}: {token}")

if failures:
    print("FAIL")
    for failure in failures:
        print("-", failure)
    raise SystemExit(1)
print("OK: consolidated project, LEAP route 06, and native-Wuji2 3P+3R route 07 are valid")
