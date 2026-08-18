from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(HERE))

from planning.flexible_pose_sampling import (  # noqa: E402
    free_placement_centres_xy,
    halton,
    sample_pregrasp,
    sample_transfer,
)
from planning.flexible_route_search import (  # noqa: E402
    plan_flexible_route,
    screen_exact_cover_batch,
)


class FakeIKClient:
    def solve_ik(self, targets, q_reference_rad, select_chain=False, collision_context=None):
        targets = np.asarray(targets, dtype=np.float64)
        accepted = []
        for i, target in enumerate(targets):
            xyz = target[:3, 3]
            q = np.asarray([xyz[0], xyz[1], xyz[2], 0.05, -0.05, 0.02, -0.02])
            accepted.append([{
                "target_index": i,
                "solution_index": 0,
                "q_rad": q.tolist(),
                "position_error_m": 0.001,
                "orientation_error_rad": 0.01,
                "inner_limit_margin_rad": 0.2,
                "observed_scene_collision_pass": True,
                "unknown_space_exposure": False,
            }])
        return {
            "ik_accepted_solutions": accepted,
            "feasible_solutions": accepted,
            "accepted_per_target": [1] * len(targets),
            "raw_success_per_target": [1] * len(targets),
            "solve_time_s": 0.001,
        }

    def check_joint_path(self, *args, **kwargs):
        return {"path_pass": True, "path_segments": []}


def minimal_layout():
    return {
        "geometry": {
            "placement_zone_size_m": [0.8, 0.3, 0.001],
            "table_size_m": [1.6, 0.4, 0.04],
        },
        "transforms": {
            "placement_zone": {"position_world_m": [0.4, -0.145, 0.46]},
            "table": {"position_world_m": [0.06, -0.148, 0.44]},
            "dual_arm_mount": {"Gf_local_to_world_row_major": np.eye(4).tolist()},
        },
    }


def write_fake_case(root: Path) -> Path:
    case = root / "case"
    (case / "07_arm_execution").mkdir(parents=True)
    (case / "06_isaacsim").mkdir(parents=True)
    (case / "01_input").mkdir(parents=True)
    names = np.asarray(["pregrasp", "cover", "grasp", "squeeze", "lift"])
    wrist = np.repeat(np.eye(4)[None], 5, axis=0)
    wrist[:, :3, 3] = np.asarray([
        [-0.10, 0.0, 0.60],
        [0.00, 0.0, 0.60],
        [0.00, 0.0, 0.60],
        [0.00, 0.0, 0.60],
        [0.00, 0.0, 0.80],
    ])
    flange = wrist.copy()
    np.savez_compressed(
        case / "07_arm_execution/arm_flange_targets.npz",
        waypoint_names=names,
        world_from_right_flange=flange,
        world_from_wuji2_wrist=wrist,
        flange_from_wuji2_wrist=np.eye(4),
        world_from_source_zone=np.eye(4),
    )
    hand_names = np.asarray([f"joint_{i}" for i in range(20)])
    hand_q = np.zeros((1, 5, 20), dtype=np.float64)
    np.savez_compressed(
        case / "06_isaacsim/final_waypoints.npz",
        finger_joint_names=hand_names,
        waypoint_names=names,
        waypoint_joint_positions=hand_q,
        wuji2_semantic_palm_approach_axis_world=np.asarray([1.0, 0.0, 0.0]),
        is_top_grasp=np.asarray(False),
    )
    (case / "case.json").write_text(json.dumps({
        "target_segmentation_id": 7,
        "target_object_code": "fake-object",
        "source_candidate_index": 123,
        "official_score": 10.0,
    }))
    (case / "01_input/scene_fake_manifest.json").write_text(json.dumps({
        "objects": [{"segmentation_id": 7, "pose_world_object": np.eye(4).tolist()}]
    }))
    return case


class FlexiblePlanningTest(unittest.TestCase):
    def test_halton_is_deterministic(self):
        a = halton(8, 6)
        b = halton(8, 6)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.all((a >= 0.0) & (a < 1.0)))

    def test_pregrasp_count_and_nominal_first(self):
        cover = np.eye(4)
        samples = sample_pregrasp(
            cover_wrist_world=cover,
            approach_axis_world=np.asarray([1.0, 0.0, 0.0]),
            count=256,
            distance_range_m=(0.06, 0.16),
            lateral_half_width_m=0.025,
            rotation_half_range_deg_xyz=(10.0, 10.0, 15.0),
            nominal_distance_m=0.10,
        )
        self.assertEqual(samples.poses_world.shape, (256, 4, 4))
        self.assertTrue(np.allclose(samples.poses_world[0, :3, 3], [-0.1, 0.0, 0.0]))
        self.assertTrue(samples.metadata[0]["nominal"])

    def test_transfer_keeps_nominal_first(self):
        lift = np.eye(4)
        lift[:3, 3] = [0.0, 0.0, 0.7]
        samples = sample_transfer(
            lift_wrist_world_nominal=lift,
            place_zone_center_xy_m=np.asarray([0.4, -0.145]),
            place_wrist_nominal_z_m=0.55,
            count=384,
            lambda_range=(0.30, 0.85),
            height_above_place_range_m=(0.12, 0.26),
            lateral_xy_half_width_m=0.05,
            rotation_half_range_deg_xyz=(10.0, 10.0, 20.0),
            nominal_lambda=0.65,
            nominal_height_above_place_m=0.18,
        )
        self.assertEqual(samples.poses_world.shape, (384, 4, 4))
        self.assertTrue(samples.metadata[0]["nominal"])
        self.assertAlmostEqual(samples.metadata[0]["height_above_place_m"], 0.18)

    def test_nominal_placement_spacing(self):
        layout = minimal_layout()
        centres = free_placement_centres_xy(
            layout=layout,
            nominal_object_size_xy_m=(0.12, 0.12),
            edge_margin_m=0.01,
            grid_step_xy_m=(0.025, 0.025),
            occupied_centres_xy_m=[[0.4, -0.145]],
            minimum_center_spacing_m=0.14,
            preferred_world_y_m=-0.05,
        )
        self.assertGreater(len(centres), 0)
        self.assertTrue(np.all(np.linalg.norm(centres - np.asarray([0.4, -0.145]), axis=1) >= 0.14 - 1e-12))

    def test_full_flexible_route_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout_path = root / "08_dual_arm_scene_layout/config"
            layout_path.mkdir(parents=True)
            (layout_path / "manual_layout_calibrated.json").write_text(json.dumps(minimal_layout()))
            case = write_fake_case(root)
            config = json.loads((HERE / "config/closed_loop.json").read_text())
            # Keep unit test fast while preserving all stages.
            config["flexible_ik"]["pregrasp"]["samples"] = 8
            config["flexible_ik"]["lift"]["samples"] = 8
            config["flexible_ik"]["transfer"]["samples"] = 8
            config["flexible_ik"]["place"]["grid_step_xy_m"] = [0.20, 0.10]
            config["flexible_ik"]["place"]["samples_per_xy"] = 2
            config["flexible_ik"]["retreat"]["samples"] = 8
            config["flexible_ik"]["selection"]["beam_width"] = 8
            config["flexible_ik"]["selection"]["solutions_per_pose"] = 1
            measured = {f"joint_{i}": 0.0 for i in range(20)}
            client = FakeIKClient()
            cover = screen_exact_cover_batch(
                client=client,
                case_roots=[case],
                q_current=np.zeros(7),
                measured=measured,
                T_base_from_world=np.eye(4),
                T_world_base=np.eye(4),
                no_planner_collision_check=True,
                block_unknown=False,
                solutions_per_candidate=4,
            )
            self.assertTrue(cover[0]["pass"])
            registry = root / "registry.json"
            registry.write_text(json.dumps({"schema_version": 2, "placements": []}))
            report = plan_flexible_route(
                client=client,
                project_root=root,
                case_root=case,
                cover_solutions=cover[0]["cover_solutions"],
                q_current=np.zeros(7),
                measured=measured,
                placement_registry=registry,
                config=config,
                no_planner_collision_check=True,
                block_unknown=False,
            )
            self.assertEqual(report["status"], "PASS")
            with np.load(report["output_npz"], allow_pickle=False) as z:
                q = z["arm_q_rad"]
                stages = [str(x) for x in z["waypoint_names"].tolist()]
            self.assertEqual(q.shape, (9, 7))
            self.assertEqual(stages[1:4], ["cover", "grasp", "squeeze"])
            self.assertTrue(np.allclose(q[1], q[2]))
            self.assertTrue(np.allclose(q[2], q[3]))
            self.assertTrue(np.allclose(q[6], q[7]))


if __name__ == "__main__":
    unittest.main()
