from __future__ import annotations

"""Simple production route planner.

Contract:
- Exact COVER stays strict and is screened before this module.
- PREGRASP is a relaxed endpoint in the existing retreat region.
- HOME->PREGRASP is one joint-space segment, optionally checked against ROI ESDF
  and self collision.
- PREGRASP->COVER is one joint-space segment; no intermediate Cartesian IK.
- LIFT/TRANSFER/PLACE/RETREAT are relaxed endpoint IK only; no planner ESDF.
- Isaac execution remains the existing joint-space quintic interpolation.
"""

import json
import math
from pathlib import Path

import numpy as np

from .flexible_pose_sampling import (
    PoseSampleSet,
    free_placement_centres_xy,
    placement_zone_bounds,
    sample_lift,
    sample_place_from_centres,
    sample_pregrasp,
    sample_retreat,
    sample_transfer,
)
from .flexible_route_search import (
    BeamState,
    IKNode,
    _ancestry,
    _candidate_geometry,
    _cap_solutions,
    _expand_beam,
    _home_parent,
    _named_state,
    _node_extra_cost,
    _node_from_record,
    _transition_cost,
    _world_from_base,
    _write_plan,
    load_json,
    read_occupied_centres,
)


_PRINTED_TUNING = False


def _route_tuning(config: dict) -> dict:
    """Return the single authoritative route tuning block.

    No code fallback is allowed. If the JSON is missing or incomplete, fail
    loudly so a run can never silently use a different parameter set.
    """
    if "route_tuning" not in config:
        raise KeyError(
            "closed_loop.json is missing required top-level 'route_tuning'"
        )
    tuning = config["route_tuning"]
    required = {
        "exact_cover",
        "pregrasp",
        "pick_path",
        "lift",
        "transfer",
        "place",
        "retreat",
        "selection",
    }
    missing = sorted(required.difference(tuning))
    if missing:
        raise KeyError(f"route_tuning missing required blocks: {missing}")
    return tuning


def _print_tuning_once(config: dict) -> None:
    global _PRINTED_TUNING
    if _PRINTED_TUNING:
        return
    _PRINTED_TUNING = True
    t = _route_tuning(config)
    print("")
    print("[ROUTE TUNING]")
    cover = t["exact_cover"]
    print(
        "  COVER       "
        f"{1000.0*float(cover['ik_position_tolerance_m']):.0f} mm / "
        f"{float(cover['ik_orientation_tolerance_deg']):.0f} deg       LOCKED"
    )
    for stage in ("pregrasp", "lift", "transfer", "place", "retreat"):
        row = t[stage]
        samples = row.get("samples", "grid" if stage == "place" else "NA")
        print(
            f"  {stage.upper():<11} "
            f"{1000.0*float(row['ik_position_tolerance_m']):.0f} mm / "
            f"{float(row['ik_orientation_tolerance_deg']):.0f} deg"
            f" | samples={samples}"
        )
    p = t["pick_path"]
    print(
        "  HOME->PRE   "
        f"ESDF={'ON' if p.get('home_to_pregrasp_esdf_check', True) else 'OFF'} "
        f"SELF={'ON' if p.get('home_to_pregrasp_self_collision_check', True) else 'OFF'} "
        f"step={float(p.get('home_to_pregrasp_joint_step_deg', 3.0)):.1f} deg"
    )
    print(
        "  PRE->COVER  "
        f"ESDF={'ON' if p.get('pregrasp_to_cover_esdf_check', False) else 'OFF'} "
        f"SELF={'ON' if p.get('pregrasp_to_cover_self_collision_check', True) else 'OFF'} "
        f"step={float(p.get('pregrasp_to_cover_joint_step_deg', 5.0)):.1f} deg"
    )
    print("")


def _acceptance_payload(config: dict, stage: str) -> dict:
    row = _route_tuning(config)[stage]
    return {
        "name": stage,
        "position_tolerance_m": float(row["ik_position_tolerance_m"]),
        "orientation_tolerance_rad": math.radians(
            float(row["ik_orientation_tolerance_deg"])
        ),
        "minimum_inner_limit_margin_rad": math.radians(
            float(row.get("minimum_inner_limit_margin_deg", 3.0))
        ),
        "require_raw_success": False,
    }


def _solution_pool(report: dict, target_index: int) -> list[dict]:
    rows = report.get("ik_accepted_solutions") or []
    if target_index < 0 or target_index >= len(rows):
        return []
    return list(rows[target_index])


def _solve_relaxed_pose_set(
    *,
    client,
    stage: str,
    pose_set: PoseSampleSet,
    q_reference: np.ndarray,
    config: dict,
    T_base_from_world: np.ndarray,
    solutions_per_pose: int,
) -> tuple[list[IKNode], dict]:
    """Endpoint IK only; planner ESDF filtering is deliberately absent."""
    poses_world = np.asarray(pose_set.poses_world, dtype=np.float64)
    targets_base = np.stack([T_base_from_world @ pose for pose in poses_world])
    policy = _acceptance_payload(config, stage)

    report = client.solve_ik(
        targets_base,
        q_reference,
        select_chain=False,
        collision_context=None,
        acceptance_policy=policy,
    )

    nodes: list[IKNode] = []
    reachable = 0
    solution_count = 0
    raw_reachable = 0
    raw_counts = report.get("raw_success_per_target") or [0] * len(poses_world)

    for target_index, pose in enumerate(poses_world):
        if int(raw_counts[target_index]) > 0:
            raw_reachable += 1
        records = _solution_pool(report, target_index)
        if records:
            reachable += 1
            solution_count += len(records)
        for row in _cap_solutions(records, q_reference, solutions_per_pose):
            nodes.append(
                IKNode(
                    stage=stage,
                    q_rad=np.asarray(row["q_rad"], dtype=np.float64),
                    target_index=target_index,
                    solution_index=int(row["solution_index"]),
                    target_pose_world=pose.copy(),
                    metadata=dict(pose_set.metadata[target_index]),
                    inner_limit_margin_rad=float(
                        row.get("inner_limit_margin_rad", 0.0)
                    ),
                    intrinsic_penalty=float(
                        pose_set.metadata[target_index].get(
                            "nominal_penalty", 0.0
                        )
                    ),
                )
            )

    summary = {
        "stage": stage,
        "target_count": int(len(poses_world)),
        "raw_success_target_count": int(raw_reachable),
        "reachable_target_count": int(reachable),
        "accepted_solution_count": int(solution_count),
        "node_count": int(len(nodes)),
        "worker_solve_time_s": float(report.get("solve_time_s", 0.0)),
        "acceptance_policy": report.get("acceptance_policy", policy),
        "planner_esdf_collision_check": False,
    }
    return nodes, summary


def _pair_score(
    pre: IKNode,
    cover: IKNode,
    q_current: np.ndarray,
    selection_cfg: dict,
) -> float:
    return (
        _transition_cost(q_current, pre.q_rad, selection_cfg)
        + _transition_cost(pre.q_rad, cover.q_rad, selection_cfg)
        + _node_extra_cost(pre, selection_cfg)
        + _node_extra_cost(cover, selection_cfg)
    )


def _select_pre_cover_pair(
    *,
    client,
    pre_nodes: list[IKNode],
    cover_nodes: list[IKNode],
    q_current: np.ndarray,
    measured: dict,
    geometry: dict,
    T_world_base: np.ndarray,
    config: dict,
    selection_cfg: dict,
) -> tuple[BeamState | None, BeamState | None, dict]:
    """Select endpoint pair; there is no intermediate Cartesian IK."""
    tune = _route_tuning(config)["pick_path"]
    pair_trials = max(1, int(tune.get("pair_trials", 128)))

    pairs = [
        (_pair_score(pre, cover, q_current, selection_cfg), pre, cover)
        for pre in pre_nodes
        for cover in cover_nodes
    ]
    pairs.sort(key=lambda row: row[0])

    counters = {
        "pair_candidates": int(len(pairs)),
        "pair_trials_limit": int(pair_trials),
        "pairs_tested": 0,
        "home_pregrasp_fail": 0,
        "pregrasp_cover_fail": 0,
    }

    pre_named = _named_state(geometry, measured, "pregrasp")
    cover_named = _named_state(geometry, measured, "cover")
    home_cache: dict[tuple[float, ...], dict] = {}
    approach_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], dict] = {}

    for _score, pre_node, cover_node in pairs[:pair_trials]:
        counters["pairs_tested"] += 1

        pre_key = tuple(np.round(pre_node.q_rad, 4).tolist())
        if pre_key not in home_cache:
            home_cache[pre_key] = client.check_joint_path(
                np.stack([q_current, pre_node.q_rad]),
                measured,
                joint_positions_by_node=[pre_named, pre_named],
                T_world_base=T_world_base,
                phases=["pregrasp"],
                margin_m=0.0,
                path_max_joint_step_rad=math.radians(
                    float(tune.get("home_to_pregrasp_joint_step_deg", 3.0))
                ),
                check_observed_map=bool(
                    tune.get("home_to_pregrasp_esdf_check", True)
                ),
                check_self_collision=bool(
                    tune.get("home_to_pregrasp_self_collision_check", True)
                ),
            )
        home_report = home_cache[pre_key]
        if not bool(home_report.get("path_pass")):
            counters["home_pregrasp_fail"] += 1
            continue

        cover_key = tuple(np.round(cover_node.q_rad, 4).tolist())
        approach_key = (pre_key, cover_key)
        if approach_key not in approach_cache:
            approach_cache[approach_key] = client.check_joint_path(
                np.stack([pre_node.q_rad, cover_node.q_rad]),
                measured,
                joint_positions_by_node=[pre_named, cover_named],
                T_world_base=T_world_base,
                phases=["cover"],
                margin_m=0.0,
                path_max_joint_step_rad=math.radians(
                    float(tune.get("pregrasp_to_cover_joint_step_deg", 5.0))
                ),
                check_observed_map=bool(
                    tune.get("pregrasp_to_cover_esdf_check", False)
                ),
                check_self_collision=bool(
                    tune.get("pregrasp_to_cover_self_collision_check", True)
                ),
            )
        approach_report = approach_cache[approach_key]
        if not bool(approach_report.get("path_pass")):
            counters["pregrasp_cover_fail"] += 1
            continue

        home_state = _home_parent(q_current)
        pre_state = BeamState(
            node=pre_node,
            cost=float(_transition_cost(q_current, pre_node.q_rad, selection_cfg))
            + float(_node_extra_cost(pre_node, selection_cfg)),
            parent=home_state,
        )
        cover_state = BeamState(
            node=cover_node,
            cost=float(pre_state.cost)
            + float(_transition_cost(pre_node.q_rad, cover_node.q_rad, selection_cfg))
            + float(_node_extra_cost(cover_node, selection_cfg)),
            parent=pre_state,
        )
        counters.update(
            {
                "status": "PASS",
                "selected_pregrasp_target_index": int(pre_node.target_index),
                "selected_pregrasp_solution_index": int(pre_node.solution_index),
                "selected_cover_solution_index": int(cover_node.solution_index),
                "home_pregrasp_path_report": home_report,
                "pregrasp_cover_path_report": approach_report,
            }
        )
        return pre_state, cover_state, counters

    counters["status"] = "FAIL"
    return None, None, counters


def _save_tuning_snapshot(report: dict, config: dict) -> dict:
    snapshot = json.loads(json.dumps(_route_tuning(config)))
    report["route_tuning_snapshot"] = snapshot
    report_path = report.get("report_json")
    if report_path:
        Path(report_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def plan_flexible_route(
    *,
    client,
    project_root: Path,
    case_root: Path,
    cover_solutions: list[dict],
    q_current: np.ndarray,
    measured: dict,
    placement_registry: Path,
    config: dict,
    no_planner_collision_check: bool,
    block_unknown: bool,
    output_npz: Path | None = None,
) -> dict:
    """Strict COVER + relaxed endpoints + simple joint-space pick path."""
    del no_planner_collision_check, block_unknown
    _print_tuning_once(config)

    if not cover_solutions:
        return {"status": "FAIL", "reason": "no exact COVER IK solution"}

    project_root = Path(project_root).resolve()
    case_root = Path(case_root).resolve()
    geometry = _candidate_geometry(case_root)
    layout = load_json(
        project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
    )
    T_world_base = _world_from_base(project_root)
    T_base_from_world = np.linalg.inv(T_world_base)
    tuning = _route_tuning(config)
    selection_cfg = tuning["selection"]
    beam_width = int(selection_cfg.get("beam_width", 64))
    solutions_per_pose = int(selection_cfg.get("solutions_per_pose", 4))
    summaries: list[dict] = []

    # COVER: already strict-screened; do not reclassify.
    cover_nodes = [
        _node_from_record(
            "cover",
            geometry["cover_flange_world"],
            row,
            {"nominal_penalty": 0.0},
        )
        for row in cover_solutions
    ]

    # PREGRASP
    pre_cfg = tuning["pregrasp"]
    pre_wrist = sample_pregrasp(
        cover_wrist_world=geometry["cover_wrist_world"],
        approach_axis_world=geometry["approach_axis_world"],
        count=int(pre_cfg["samples"]),
        distance_range_m=tuple(pre_cfg["distance_range_m"]),
        lateral_half_width_m=float(pre_cfg["lateral_half_width_m"]),
        rotation_half_range_deg_xyz=tuple(pre_cfg["rotation_half_range_deg_xyz"]),
        nominal_distance_m=float(pre_cfg.get("nominal_distance_m", 0.10)),
    )
    wrist_from_flange = np.linalg.inv(geometry["flange_from_wrist"])
    pre_flange = PoseSampleSet(
        pre_wrist.poses_world @ wrist_from_flange[None], pre_wrist.metadata
    )
    pre_nodes, summary = _solve_relaxed_pose_set(
        client=client,
        stage="pregrasp",
        pose_set=pre_flange,
        q_reference=q_current,
        config=config,
        T_base_from_world=T_base_from_world,
        solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    if not pre_nodes:
        return {
            "status": "FAIL",
            "reason": "PREGRASP relaxed region has no IK",
            "stage_summaries": summaries,
        }

    pre_state, cover_state, pick_summary = _select_pre_cover_pair(
        client=client,
        pre_nodes=pre_nodes,
        cover_nodes=cover_nodes,
        q_current=np.asarray(q_current, dtype=np.float64),
        measured=measured,
        geometry=geometry,
        T_world_base=T_world_base,
        config=config,
        selection_cfg=selection_cfg,
    )
    summaries.append({"stage": "pick_path", **pick_summary})
    if pre_state is None or cover_state is None:
        return {
            "status": "FAIL",
            "reason": "no PREGRASP/COVER pair passed simple joint-space path gates",
            "stage_summaries": summaries,
        }
    summaries.append(
        {
            "stage": "cover",
            "target_count": 1,
            "solution_count": int(len(cover_nodes)),
            "selected_solution_index": int(cover_state.node.solution_index),
            "strict_exact_cover": True,
        }
    )

    # LIFT
    lift_cfg = tuning["lift"]
    lift_axis = (
        -geometry["approach_axis_world"]
        if geometry["is_top_grasp"]
        else np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    lift_wrist = sample_lift(
        cover_wrist_world=geometry["cover_wrist_world"],
        lift_axis_world=lift_axis,
        count=int(lift_cfg["samples"]),
        distance_range_m=tuple(lift_cfg["distance_range_m"]),
        lateral_half_width_m=float(lift_cfg["lateral_half_width_m"]),
        rotation_half_range_deg_xyz=tuple(lift_cfg["rotation_half_range_deg_xyz"]),
        nominal_distance_m=float(lift_cfg.get("nominal_distance_m", 0.20)),
    )
    lift_flange = PoseSampleSet(
        lift_wrist.poses_world @ wrist_from_flange[None], lift_wrist.metadata
    )
    lift_nodes, summary = _solve_relaxed_pose_set(
        client=client,
        stage="lift",
        pose_set=lift_flange,
        q_reference=cover_state.q_rad,
        config=config,
        T_base_from_world=T_base_from_world,
        solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    lift_beam = _expand_beam(
        [cover_state], lift_nodes, beam_width=beam_width, selection_cfg=selection_cfg
    )
    if not lift_beam:
        return {
            "status": "FAIL",
            "reason": "LIFT relaxed region has no IK",
            "stage_summaries": summaries,
        }

    # TRANSFER
    zone_min, zone_max, table_top = placement_zone_bounds(layout)
    zone_center = 0.5 * (zone_min + zone_max)
    place_cfg = tuning["place"]
    nominal_height = float(place_cfg["nominal_object_size_xyz_m"][2])
    place_wrist_nominal_z = float(
        geometry["cover_wrist_world"][2, 3]
        + place_cfg.get("release_wrist_height_delta_m", 0.01)
    )
    transfer_cfg = tuning["transfer"]
    transfer_wrist = sample_transfer(
        lift_wrist_world_nominal=geometry["nominal_lift_wrist_world"],
        place_zone_center_xy_m=zone_center,
        place_wrist_nominal_z_m=place_wrist_nominal_z,
        count=int(transfer_cfg["samples"]),
        lambda_range=tuple(transfer_cfg["lambda_range"]),
        height_above_place_range_m=tuple(
            transfer_cfg["height_above_place_range_m"]
        ),
        lateral_xy_half_width_m=float(transfer_cfg["lateral_xy_half_width_m"]),
        rotation_half_range_deg_xyz=tuple(
            transfer_cfg["rotation_half_range_deg_xyz"]
        ),
        nominal_lambda=float(transfer_cfg.get("nominal_lambda", 0.65)),
        nominal_height_above_place_m=float(
            transfer_cfg.get("nominal_height_above_place_m", 0.18)
        ),
    )
    transfer_flange = PoseSampleSet(
        transfer_wrist.poses_world @ wrist_from_flange[None],
        transfer_wrist.metadata,
    )
    transfer_nodes, summary = _solve_relaxed_pose_set(
        client=client,
        stage="transfer",
        pose_set=transfer_flange,
        q_reference=lift_beam[0].q_rad,
        config=config,
        T_base_from_world=T_base_from_world,
        solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    transfer_beam = _expand_beam(
        lift_beam,
        transfer_nodes,
        beam_width=beam_width,
        selection_cfg=selection_cfg,
    )
    if not transfer_beam:
        return {
            "status": "FAIL",
            "reason": "TRANSFER relaxed region has no IK",
            "stage_summaries": summaries,
        }

    # PLACE
    occupied = read_occupied_centres(placement_registry)
    centres = free_placement_centres_xy(
        layout=layout,
        nominal_object_size_xy_m=tuple(place_cfg["nominal_object_size_xyz_m"][:2]),
        edge_margin_m=float(place_cfg["edge_margin_m"]),
        grid_step_xy_m=tuple(place_cfg["grid_step_xy_m"]),
        occupied_centres_xy_m=occupied,
        minimum_center_spacing_m=float(place_cfg["minimum_center_spacing_m"]),
        preferred_world_y_m=float(place_cfg["preferred_world_y_m"]),
    )
    place_flange = sample_place_from_centres(
        centres_xy_m=centres,
        object_world_initial=geometry["object_world_initial"],
        flange_from_object_grasp=geometry["flange_from_object_grasp"],
        samples_per_xy=int(place_cfg["samples_per_xy"]),
        table_top_world_z_m=table_top,
        nominal_object_height_m=nominal_height,
        z_extra_range_m=tuple(place_cfg["z_extra_range_m"]),
        object_rotation_half_range_deg_xyz=tuple(
            place_cfg["object_rotation_half_range_deg_xyz"]
        ),
    )
    place_nodes, summary = _solve_relaxed_pose_set(
        client=client,
        stage="place",
        pose_set=place_flange,
        q_reference=transfer_beam[0].q_rad,
        config=config,
        T_base_from_world=T_base_from_world,
        solutions_per_pose=solutions_per_pose,
    )
    summary["free_xy_count"] = int(len(centres))
    summaries.append(summary)
    place_beam = _expand_beam(
        transfer_beam, place_nodes, beam_width=beam_width, selection_cfg=selection_cfg
    )
    if not place_beam:
        return {
            "status": "FAIL",
            "reason": "PLACE relaxed region has no IK",
            "stage_summaries": summaries,
        }

    # RETREAT
    retreat_cfg = tuning["retreat"]
    home_q = np.deg2rad(
        np.asarray(
            config.get("home_q_deg", [50, -70, 0, 40, 35, 0, 25]),
            dtype=np.float64,
        )
    )
    best_final: BeamState | None = None
    best_final_cost = math.inf
    best_retreat_summary = None
    parent_trials = min(
        int(selection_cfg.get("retreat_parent_trials", 8)), len(place_beam)
    )

    for trial_index, place_parent in enumerate(place_beam[:parent_trials]):
        place_wrist = place_parent.node.target_pose_world @ geometry["flange_from_wrist"]
        retreat_wrist = sample_retreat(
            place_wrist_world=place_wrist,
            count=int(retreat_cfg["samples"]),
            upward_range_m=tuple(retreat_cfg["upward_range_m"]),
            xy_half_width_m=float(retreat_cfg["xy_half_width_m"]),
            rotation_half_range_deg_xyz=tuple(
                retreat_cfg["rotation_half_range_deg_xyz"]
            ),
            nominal_upward_m=float(retreat_cfg.get("nominal_upward_m", 0.12)),
            start_index=5001
            + trial_index * max(1, int(retreat_cfg["samples"])),
        )
        retreat_flange = PoseSampleSet(
            retreat_wrist.poses_world @ wrist_from_flange[None],
            retreat_wrist.metadata,
        )
        retreat_nodes, retreat_summary = _solve_relaxed_pose_set(
            client=client,
            stage="retreat",
            pose_set=retreat_flange,
            q_reference=place_parent.q_rad,
            config=config,
            T_base_from_world=T_base_from_world,
            solutions_per_pose=solutions_per_pose,
        )
        if not retreat_nodes:
            continue
        retreat_beam = _expand_beam(
            [place_parent],
            retreat_nodes,
            beam_width=beam_width,
            selection_cfg=selection_cfg,
        )
        for state in retreat_beam:
            total = float(state.cost) + float(
                selection_cfg.get("home_return_weight", 0.25)
            ) * _transition_cost(state.q_rad, home_q, selection_cfg)
            if total < best_final_cost:
                best_final = state
                best_final_cost = total
                best_retreat_summary = retreat_summary

    if best_final is None:
        return {
            "status": "FAIL",
            "reason": "RETREAT relaxed region has no IK",
            "stage_summaries": summaries,
        }
    summaries.append(best_retreat_summary or {"stage": "retreat"})

    chain = _ancestry(best_final)
    by_stage = {
        state.node.stage: state
        for state in chain
        if state.node.stage != "q_current"
    }
    required = {"pregrasp", "cover", "lift", "transfer", "place", "retreat"}
    missing = required.difference(by_stage)
    if missing:
        raise RuntimeError(
            f"internal simplified route chain missing stages: {sorted(missing)}"
        )

    output_npz = (
        case_root / "07_arm_execution/flexible_route_plan.npz"
        if output_npz is None
        else Path(output_npz).resolve()
    )
    report = _write_plan(
        geometry=geometry,
        q_current=q_current,
        chosen=by_stage,
        output_npz=output_npz,
        summaries=summaries,
        final_path_report=None,
        placement_registry=placement_registry,
    )
    return _save_tuning_snapshot(report, config)
