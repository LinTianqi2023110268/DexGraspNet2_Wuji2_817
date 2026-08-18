#!/usr/bin/env python3
"""
Filtered offline replay + timing profile for candidate-centric RFS V2.

Purpose
-------
Take the first N PASS candidates from candidate_centric_rfs_v2_filter.json,
preserve DGN2 target-rank order, and replay the expensive post-filter stages:

  build candidate cases
  -> LEAP->Wuji2 retarget (stage-wise timed)
  -> finalize Wuji2 + arm targets
  -> exact COVER IK
  -> Flexible Route for every exact-COVER PASS candidate

This script DOES NOT start Isaac Sim and DOES NOT modify the production
closed-loop implementation.

Run with the project's planner/Isaac-Lab Python.  It launches the existing
network and retarget interpreters configured in closed_loop.json and uses the
existing CuroboWorkerClient for exact IK / route planning.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROL_ROOT = HERE.parents[1]
CLOSED_LOOP_ROOT = HERE.parent
PROJECT_ROOT_DEFAULT = HERE.parents[3]
SCRIPTS = CLOSED_LOOP_ROOT / "scripts"
DEFAULT_CONFIG = CLOSED_LOOP_ROOT / "config/closed_loop.json"
RETARGET_PROFILER = HERE / "05_profile_retarget_stages.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_slug(text: str) -> str:
    import re
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def run_cmd(label: str, cmd: list[str | Path], *, cwd: Path) -> tuple[float, str]:
    cmd_s = [str(x) for x in cmd]
    print(f"\n[{label}]")
    print("$", " ".join(shlex.quote(x) for x in cmd_s), flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(
        cmd_s,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    dt = time.perf_counter() - t0
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-40:])
        raise RuntimeError(f"{label} failed with code {completed.returncode}\n{tail}")
    print(f"[{label}] wall={dt:.3f}s", flush=True)
    return dt, completed.stdout or ""


def find_settled_manifest(capture_root: Path) -> Path:
    direct = [
        capture_root / "settled_scene_manifest.json",
        capture_root / "scene_manifest_settled.json",
    ]
    for p in direct:
        if p.is_file():
            return p
    matches = sorted(capture_root.glob("*settled*manifest*.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find settled scene manifest in {capture_root}. "
            "Expected settled_scene_manifest.json or a unique *settled*manifest*.json."
        )
    raise RuntimeError(f"Ambiguous settled manifests: {[str(p) for p in matches]}")


def load_robot_state(path: Path, right_arm_names: tuple[str, ...]) -> tuple[np.ndarray, dict[str, float]]:
    state = load_json(path)
    measured = {str(k): float(v) for k, v in state["joint_positions_by_name"].items()}
    q = np.asarray([measured[name] for name in right_arm_names], dtype=np.float64)
    return q, measured


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def select_filtered_rows(filter_path: Path, count: int) -> list[dict]:
    payload = load_json(filter_path)
    rows = list(payload.get("rows", []))
    rows.sort(key=lambda r: int(r["target_rank"]))
    passed = [r for r in rows if str(r.get("status", "")).upper() == "PASS"]
    if len(passed) < count:
        raise RuntimeError(f"Requested {count} PASS candidates but filter contains only {len(passed)}")
    return passed[:count]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    p.add_argument("--cycle-root", type=Path, required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--filter-json", type=Path)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path)

    # Match the user's current production invocation by default:
    # --no-planner-collision-check is active, but the mandatory HOME->PREGRASP
    # observed-map gate remains handled by current project code.
    p.add_argument(
        "--full-planner-collision-check",
        action="store_true",
        help="Use full planner observed-map collision checks. Default matches production --no-planner-collision-check.",
    )

    p.add_argument("--expected-rank", type=int, default=447)
    p.add_argument("--expected-candidate-index", type=int, default=6559)
    p.add_argument(
        "--skip-route",
        action="store_true",
        help="Stop after exact COVER timing. Default also runs Flexible Route for exact-COVER PASS cases.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    total_started = time.perf_counter()

    project_root = args.project_root.expanduser().resolve()
    cycle_root = args.cycle_root.expanduser().resolve()
    cfg_path = args.config.expanduser().resolve()
    cfg = load_json(cfg_path)

    query_slug = safe_slug(args.query)
    capture_root = cycle_root / "capture"
    dgn_root = capture_root / "dgn2" / query_slug
    gs_root = capture_root / "grounded_sam" / query_slug

    filter_path = (
        args.filter_json.expanduser().resolve()
        if args.filter_json is not None
        else cycle_root
        / "rfs_prototype/v2_candidate_centric/candidate_centric_rfs_v2_filter.json"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else cycle_root / "rfs_prototype/v2_filtered_offline_replay_top64"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = dgn_root / "official_leap_1024_target_ranked.npz"
    network_input = dgn_root / "network_input.npz"
    mask_path = gs_root / "mask.npy"
    robot_state_path = capture_root / "robot_state.json"
    sim_target_path = cycle_root / "sim_target.json"
    settled_manifest = find_settled_manifest(capture_root)

    required = [
        filter_path,
        prediction,
        network_input,
        mask_path,
        robot_state_path,
        sim_target_path,
        settled_manifest,
        RETARGET_PROFILER,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    network_py = Path(cfg["network_python"]).expanduser().resolve()
    retarget_py = Path(cfg["retarget_python"]).expanduser().resolve()
    for p in (network_py, retarget_py):
        if not p.is_file():
            raise FileNotFoundError(p)

    # Import current project interfaces only after project roots are known.
    sys.path.insert(0, str(CONTROL_ROOT))
    sys.path.insert(0, str(CLOSED_LOOP_ROOT))
    from core.bridge import CuroboWorkerClient
    from core.config import WorkerConfig, RIGHT_ARM_NAMES
    from planning.flexible_route_search import plan_flexible_route, screen_exact_cover_batch

    print("=" * 92)
    print("RFS V2 FILTERED OFFLINE REPLAY + TIMING PROFILE")
    print("NO ISAAC | production files unchanged | first-filter PASS only")
    print("=" * 92)

    selected_rows = select_filtered_rows(filter_path, int(args.count))
    selected_ranks = [int(r["target_rank"]) for r in selected_rows]
    selected_candidates = [int(r["candidate_index"]) for r in selected_rows]

    print(
        f"[1/8] filter: first {len(selected_rows)} PASS candidates | "
        f"rank range={selected_ranks[0]}..{selected_ranks[-1]}"
    )
    print("      first ranks:", selected_ranks[:20])

    if args.expected_rank is not None:
        if args.expected_rank not in selected_ranks:
            raise RuntimeError(
                f"Expected rank {args.expected_rank} is not in first {len(selected_rows)} PASS candidates"
            )
        i = selected_ranks.index(args.expected_rank)
        if (
            args.expected_candidate_index is not None
            and selected_candidates[i] != args.expected_candidate_index
        ):
            raise RuntimeError(
                f"Expected rank {args.expected_rank} candidate={args.expected_candidate_index}, "
                f"but filter has {selected_candidates[i]}"
            )
        print(
            f"      HARD GATE armed: rank={args.expected_rank} "
            f"candidate={selected_candidates[i]} is filtered position {i+1}"
        )

    scratch_root = output_dir / "cases"
    items = []
    for filtered_pos, row in enumerate(selected_rows, start=1):
        rank = int(row["target_rank"])
        cand = int(row["candidate_index"])
        case_id = f"rfs2_replay_r{rank:04d}_cand{cand:04d}"
        case_root = scratch_root / f"rank_{rank:04d}" / case_id
        items.append(
            {
                "filtered_position": filtered_pos,
                "local_target_index": filtered_pos - 1,
                "target_rank": rank,
                "candidate_index": cand,
                "official_score": float(row.get("official_score", float("nan"))),
                "case_id": case_id,
                "case_root": str(case_root),
            }
        )

    items_json = output_dir / "items_top64.json"
    write_json(items_json, items)

    sim_target = load_json(sim_target_path)
    sim_target_id = int(sim_target["segmentation_id"])

    timings: dict[str, float] = {}
    reports: dict[str, str] = {}

    # 2) Build candidate cases in the same network environment as production.
    build_report = output_dir / "batch_build_report.json"
    timings["build_candidate_cases_s"], _ = run_cmd(
        "2/8 build candidate cases",
        [
            network_py,
            SCRIPTS / "batch_build_candidate_cases.py",
            "--project-root",
            project_root,
            "--prediction",
            prediction,
            "--network-input",
            network_input,
            "--capture-root",
            capture_root,
            "--settled-manifest",
            settled_manifest,
            "--sim-target-segmentation-id",
            str(sim_target_id),
            "--items-json",
            items_json,
            "--output",
            build_report,
        ],
        cwd=project_root,
    )
    reports["build"] = str(build_report)

    # 3) Profile 01/02/03 while keeping ONE retarget interpreter alive.
    retarget_report = output_dir / "retarget_stage_profile.json"
    timings["retarget_total_subprocess_s"], _ = run_cmd(
        "3/8 LEAP->Wuji2 stage-wise retarget profile",
        [
            retarget_py,
            RETARGET_PROFILER,
            "--items-json",
            items_json,
            "--output",
            retarget_report,
        ],
        cwd=project_root,
    )
    reports["retarget"] = str(retarget_report)
    retarget_data = load_json(retarget_report)
    stage_totals = retarget_data["stage_totals_s"]

    # 4) Finalize.
    finalize_report = output_dir / "batch_finalize_report.json"
    timings["finalize_s"], _ = run_cmd(
        "4/8 finalize Wuji2 + arm targets",
        [
            network_py,
            SCRIPTS / "batch_finalize_candidate_cases.py",
            "--items-json",
            items_json,
            "--output",
            finalize_report,
        ],
        cwd=project_root,
    )
    reports["finalize"] = str(finalize_report)
    finalize_data = load_json(finalize_report)
    finalized_roots = {
        str(Path(r["case_root"]).resolve())
        for r in finalize_data.get("results", [])
        if r.get("status") == "PASS"
    }
    finalized_items = [
        item for item in items if str(Path(item["case_root"]).resolve()) in finalized_roots
    ]
    print(
        f"      finalize PASS={len(finalized_items)}/{len(items)} "
        f"REJECT={len(items)-len(finalized_items)}"
    )

    # 5) Start cuRobo worker, build current RGB-D map once, exact-COVER screen.
    q_current, measured = load_robot_state(robot_state_path, RIGHT_ARM_NAMES)
    T_world_base = world_from_base(project_root)
    T_base_from_world = np.linalg.inv(T_world_base)

    worker_cfg = WorkerConfig(
        startup_timeout_s=float(cfg.get("worker_startup_timeout_s", 180.0)),
        request_timeout_s=float(cfg.get("worker_request_timeout_s", 600.0)),
    )

    exact_rows = []
    route_rows = []
    map_report = {}
    no_planner_collision_check = not bool(args.full_planner_collision_check)

    print("\n[5/8 cuRobo map + exact COVER]")
    curobo_started = time.perf_counter()
    with CuroboWorkerClient(
        project_root,
        worker_config=worker_cfg,
        seeds=int(cfg.get("gpu_ik_seeds", 48)),
        batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
    ) as curobo:
        map_t0 = time.perf_counter()
        map_report = curobo.build_map(
            capture_root / "depth_m.npy",
            capture_root / "intrinsics.npy",
            capture_root / "T_world_camera.npy",
            mask_path,
        )
        timings["curobo_build_map_s"] = time.perf_counter() - map_t0
        print(f"      RGB-D map wall={timings['curobo_build_map_s']:.3f}s")

        cover_t0 = time.perf_counter()
        exact_rows = screen_exact_cover_batch(
            client=curobo,
            case_roots=[Path(item["case_root"]) for item in finalized_items],
            q_current=q_current,
            measured=measured,
            T_base_from_world=T_base_from_world,
            T_world_base=T_world_base,
            no_planner_collision_check=no_planner_collision_check,
            block_unknown=bool(cfg.get("block_unknown_space", False)),
            solutions_per_candidate=int(
                cfg["flexible_ik"]["selection"]["cover_solutions_per_candidate"]
            ),
        )
        timings["exact_cover_s"] = time.perf_counter() - cover_t0

        pass_cover = [row for row in exact_rows if bool(row.get("pass"))]
        print(
            f"      exact COVER PASS={len(pass_cover)}/{len(exact_rows)} "
            f"| wall={timings['exact_cover_s']:.3f}s"
        )

        # Hard gate for known bottle exact-COVER positive.
        if args.expected_rank is not None:
            by_case = {
                str(Path(item["case_root"]).resolve()): item for item in finalized_items
            }
            expected_case = None
            for item in finalized_items:
                if int(item["target_rank"]) == int(args.expected_rank):
                    expected_case = str(Path(item["case_root"]).resolve())
                    break
            if expected_case is None:
                raise RuntimeError(
                    f"HARD GATE FAIL: expected rank {args.expected_rank} did not finalize"
                )
            row = next((r for r in exact_rows if r["case_root"] == expected_case), None)
            if row is None or not bool(row.get("pass")):
                raise RuntimeError(
                    f"HARD GATE FAIL: expected rank {args.expected_rank} "
                    f"candidate {args.expected_candidate_index} is not Exact COVER PASS"
                )
            print(
                f"      HARD GATE PASS: rank={args.expected_rank} "
                f"candidate={args.expected_candidate_index} Exact COVER accepted"
            )

        # 6) Run Flexible Route for EVERY exact-COVER pass, without Isaac.
        if not args.skip_route:
            print("\n[6/8 Flexible Route for exact-COVER PASS candidates]")
            placement_registry = cycle_root.parent / "placement_registry.json"
            if not placement_registry.is_file():
                # Offline-only empty registry; do not touch the real session.
                placement_registry = output_dir / "offline_empty_placement_registry.json"
                write_json(
                    placement_registry,
                    {
                        "schema_version": 2,
                        "purpose": "offline route replay only",
                        "placements": [],
                    },
                )

            by_case = {
                str(Path(item["case_root"]).resolve()): item for item in finalized_items
            }
            route_total_t0 = time.perf_counter()
            for j, cover_row in enumerate(pass_cover, start=1):
                case_root_s = str(Path(cover_row["case_root"]).resolve())
                item = by_case[case_root_s]
                t0 = time.perf_counter()
                route = plan_flexible_route(
                    client=curobo,
                    project_root=project_root,
                    case_root=Path(case_root_s),
                    cover_solutions=cover_row["cover_solutions"],
                    q_current=q_current,
                    measured=measured,
                    placement_registry=placement_registry,
                    config=cfg,
                    no_planner_collision_check=no_planner_collision_check,
                    block_unknown=bool(cfg.get("block_unknown_space", False)),
                )
                dt = time.perf_counter() - t0
                route_rows.append(
                    {
                        "target_rank": int(item["target_rank"]),
                        "candidate_index": int(item["candidate_index"]),
                        "official_score": float(item["official_score"]),
                        "status": str(route.get("status")),
                        "reason": route.get("reason"),
                        "wall_time_s": dt,
                        "output_npz": route.get("output_npz"),
                        "stage_summaries": route.get("stage_summaries", []),
                    }
                )
                print(
                    f"      route {j:02d}/{len(pass_cover):02d} "
                    f"rank={item['target_rank']:4d} cand={item['candidate_index']:4d} "
                    f"status={route.get('status')} wall={dt:.3f}s "
                    f"reason={route.get('reason')}"
                )
            timings["flexible_route_total_s"] = time.perf_counter() - route_total_t0
        else:
            timings["flexible_route_total_s"] = 0.0

    timings["curobo_context_total_s"] = time.perf_counter() - curobo_started

    # 7) Save exact/route details.
    exact_public = []
    by_case_all = {str(Path(i["case_root"]).resolve()): i for i in finalized_items}
    for row in exact_rows:
        case_key = str(Path(row["case_root"]).resolve())
        item = by_case_all[case_key]
        exact_public.append(
            {
                "target_rank": int(item["target_rank"]),
                "candidate_index": int(item["candidate_index"]),
                "official_score": float(item["official_score"]),
                "pass": bool(row.get("pass")),
                "accepted_solution_count": len(row.get("cover_solutions", [])),
                "reason": row.get("reason"),
            }
        )
    write_json(output_dir / "exact_cover_results.json", {"rows": exact_public})
    write_json(output_dir / "flexible_route_results.json", {"rows": route_rows})

    # 8) Summary.
    total_wall = time.perf_counter() - total_started
    timings["total_wall_s"] = total_wall

    exact_pass_count = sum(1 for r in exact_public if r["pass"])
    route_pass_count = sum(1 for r in route_rows if r["status"] == "PASS")
    route_fail_count = sum(1 for r in route_rows if r["status"] != "PASS")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "architecture": "RFS V2 filtered offline replay + stage timing",
        "does_not_start_isaac": True,
        "does_not_modify_production_pipeline": True,
        "cycle_root": str(cycle_root),
        "query": args.query,
        "filter_json": str(filter_path),
        "selected_count": len(items),
        "selected_target_ranks": selected_ranks,
        "selected_candidate_indices": selected_candidates,
        "expected_hard_gate": {
            "target_rank": args.expected_rank,
            "candidate_index": args.expected_candidate_index,
        },
        "finalize": {
            "pass_count": len(finalized_items),
            "reject_count": len(items) - len(finalized_items),
        },
        "retarget_stage_profile": {
            "candidate_count": retarget_data.get("candidate_count"),
            "stage_totals_s": stage_totals,
            "stage_mean_per_candidate_s": retarget_data.get("stage_mean_per_candidate_s"),
            "retarget_total_wall_s": retarget_data.get("wall_time_s"),
        },
        "exact_cover": {
            "tested_count": len(exact_public),
            "pass_count": exact_pass_count,
        },
        "flexible_route": {
            "tested_count": len(route_rows),
            "pass_count": route_pass_count,
            "fail_count": route_fail_count,
        },
        "planner_collision_mode": {
            "full_planner_collision_check": bool(args.full_planner_collision_check),
            "no_planner_collision_check": no_planner_collision_check,
        },
        "map": map_report,
        "timings_s": timings,
        "outputs": {
            "items": str(items_json),
            "build_report": str(build_report),
            "retarget_profile": str(retarget_report),
            "finalize_report": str(finalize_report),
            "exact_cover_results": str(output_dir / "exact_cover_results.json"),
            "flexible_route_results": str(output_dir / "flexible_route_results.json"),
        },
    }
    report_path = output_dir / "filtered_offline_replay_profile.json"
    write_json(report_path, report)

    print("\n" + "=" * 92)
    print("FILTERED OFFLINE REPLAY DONE")
    print(f"selected filtered PASS          : {len(items)}")
    print(f"finalize PASS                   : {len(finalized_items)}/{len(items)}")
    print(f"exact COVER PASS                : {exact_pass_count}/{len(exact_public)}")
    print(f"Flexible Route PASS             : {route_pass_count}/{len(route_rows)}")
    print("-" * 92)
    print(f"build candidate cases           : {timings['build_candidate_cases_s']:.3f}s")
    print(f"01_retarget_grasp_official      : {stage_totals.get('01_retarget_grasp_official.py', float('nan')):.3f}s")
    print(f"02_align_root6d                 : {stage_totals.get('02_align_root6d.py', float('nan')):.3f}s")
    print(f"03_retarget_squeeze_official    : {stage_totals.get('03_retarget_squeeze_official.py', float('nan')):.3f}s")
    print(f"retarget total                  : {retarget_data.get('wall_time_s', float('nan')):.3f}s")
    print(f"finalize                        : {timings['finalize_s']:.3f}s")
    print(f"cuRobo RGB-D map                : {timings['curobo_build_map_s']:.3f}s")
    print(f"Exact COVER                     : {timings['exact_cover_s']:.3f}s")
    print(f"Flexible Route total            : {timings['flexible_route_total_s']:.3f}s")
    print(f"TOTAL                           : {timings['total_wall_s']:.3f}s")
    print("-" * 92)
    print("report:", report_path)
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
