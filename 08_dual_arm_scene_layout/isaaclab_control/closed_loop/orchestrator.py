#!/usr/bin/env python3
"""One-command persistent semantic dexterous grasp loop.

V2 architecture
---------------
* Isaac Lab/Sim starts once and keeps the same physical world for every capture
  and every grasp cycle.
* cuRobo starts once per planning cycle and is released before Isaac execution.
* legacy approximate GRASP/PREGRASP coarse IK gates are configurable and OFF by
  default.
* after LEAP->Wuji2, exact COVER is the hard grasp-root IK gate.
* PREGRASP/LIFT/TRANSFER/PLACE/RETREAT use large configurable 6D task sets;
  strict IK accuracy is unchanged.
* the q7 route produced by planning is executed directly in the same Isaac
  world: no second runtime IK and no pre-execution FK gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROL_ROOT = HERE.parent
SCRIPTS = HERE / "scripts"
DEFAULT_CONFIG = HERE / "config/closed_loop.json"
sys.path.insert(0, str(CONTROL_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

from core.bridge import CuroboWorkerClient  # noqa: E402
from core.config import WorkerConfig  # noqa: E402
from persistent_isaac import PersistentIsaacClient  # noqa: E402
from planning.flexible_route_search import (  # noqa: E402
    plan_flexible_route,
    screen_exact_cover_batch,
)
from planning.candidate_rfs_v2_runtime import run_candidate_rfs_v2  # noqa: E402
from all_candidate_gpu_prefilter import load_targets  # noqa: E402


VERBOSE = False
DEBUG_LOG: Path | None = None


def load_json(path: Path) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def debug_write(text: str) -> None:
    if DEBUG_LOG is None:
        return
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(text))
        if text and not str(text).endswith("\n"):
            stream.write("\n")


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"


def run(label: str, cmd: list, *, cwd: Path, env=None, capture_json: bool = False):
    command_line = " ".join(shlex.quote(str(value)) for value in cmd)
    debug_write(f"\n{'='*18} {label} {'='*18}\n$ {command_line}\n")
    if VERBOSE:
        print(f"\n{'='*18} {label} {'='*18}")
        print("$", command_line, flush=True)
    completed = subprocess.run(
        [str(value) for value in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    debug_write(completed.stdout or "")
    if VERBOSE and completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        tail = "\n".join((completed.stdout or "").splitlines()[-30:])
        raise RuntimeError(f"{label} failed: {completed.returncode}\n{tail}")
    if not capture_json:
        return None
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except Exception:
            pass
    raise RuntimeError(f"{label} did not emit a final JSON object")


def show_async(template, **kwargs) -> None:
    if not template:
        return
    cmd = [str(value).format(**kwargs) for value in template]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:
        debug_write(f"viewer failed: {exc}")


def prompt_scene(project_root: Path, supplied: str | None) -> Path:
    if supplied:
        folder = Path(supplied).expanduser()
    else:
        print("\n请输入场景文件夹地址（文件夹内需直接包含 scene_manifest.json）")
        print("例如：/home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/.../scenes/scene_0000")
        folder = Path(input("Scene folder > ").strip()).expanduser()
    folder = (project_root / folder).resolve() if not folder.is_absolute() else folder.resolve()
    manifest = folder / "scene_manifest.json"
    if not folder.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f"场景目录必须包含 scene_manifest.json: {folder}")
    print(f"✓ 场景：{folder}")
    return folder


def load_robot_state(path: Path) -> tuple[np.ndarray, dict]:
    state = load_json(path)
    return (
        np.asarray(state["right_arm_q_current_rad"], dtype=np.float64),
        {str(key): float(value) for key, value in state["joint_positions_by_name"].items()},
    )


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def candidate_order(prediction: Path) -> tuple[list[dict], int]:
    with np.load(prediction, allow_pickle=False) as z:
        order = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)
        score = np.asarray(z["score"], dtype=np.float64)
        graspness = np.asarray(z["graspness"], dtype=np.float64)
        log_prob = np.asarray(z["log_prob"], dtype=np.float64)
        total = int(len(score))
    rows = []
    for rank, index in enumerate(order):
        idx = int(index)
        rows.append({
            "target_rank": int(rank),
            "candidate_index": idx,
            "score": float(score[idx]),
            "graspness": float(graspness[idx]),
            "log_prob": float(log_prob[idx]),
        })
    return rows, total


def legacy_coarse_prefilter(
    *,
    client,
    project_root: Path,
    prediction: Path,
    q_current: np.ndarray,
    cfg: dict,
) -> tuple[list[dict], list[int], dict]:
    """Optional compatibility gate; default config bypasses it completely."""
    settings = cfg["coarse_ik_prefilter"]
    candidates, grasp_targets, pregrasp_targets, total = load_targets(
        project_root,
        prediction,
        float(settings.get("legacy_pregrasp_offset_m", 0.10)),
    )
    survivors = list(range(len(candidates)))
    report = {
        "enabled": True,
        "total_proposals": total,
        "target_candidates": len(candidates),
        "grasp_enabled": bool(settings.get("grasp_enabled", False)),
        "pregrasp_enabled": bool(settings.get("pregrasp_enabled", False)),
    }
    if bool(settings.get("grasp_enabled", False)):
        result = client.solve_ik(grasp_targets, q_current, select_chain=False)
        counts = [int(value) for value in result["accepted_per_target"]]
        survivors = [index for index in survivors if counts[index] > 0]
        report["grasp_survivors"] = len(survivors)
    if bool(settings.get("pregrasp_enabled", False)):
        if survivors:
            result = client.solve_ik(pregrasp_targets[survivors], q_current, select_chain=False)
            counts = [int(value) for value in result["accepted_per_target"]]
            survivors = [survivors[local] for local, count in enumerate(counts) if count > 0]
        else:
            survivors = []
        report["pregrasp_survivors"] = len(survivors)
    return candidates, survivors, report


def init_registry(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "purpose": "session-local nominal-size placement centres",
        "placements": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def commit_placement(path: Path, *, cycle: int, execution: dict, selected: dict) -> None:
    registry = load_json(path)
    centre = np.asarray(execution["final_object_position_world_m"][:2], dtype=np.float64)
    registry.setdefault("placements", []).append({
        "cycle": int(cycle),
        "candidate_index": int(selected["candidate_index"]),
        "target_rank": int(selected["target_rank"]),
        "target_segmentation_id": int(execution["target_segmentation_id"]),
        "center_world_xy_m": centre.tolist(),
        "actual_final_object_position_world_m": execution["final_object_position_world_m"],
        "committed_local": datetime.now().isoformat(timespec="seconds"),
    })
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_route_summary(report: dict) -> None:
    for row in report.get("stage_summaries", []):
        stage = str(row.get("stage", "")).upper()
        if not stage:
            continue
        if "target_count" in row:
            print(
                f"    {stage:<10} 目标={row.get('target_count')} | "
                f"可达目标={row.get('reachable_target_count', row.get('solution_count', '—'))} | "
                f"IK节点={row.get('node_count', row.get('beam_count', '—'))}"
            )


def main() -> int:
    global VERBOSE, DEBUG_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-folder")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--sim-execute", action="store_true")
    parser.add_argument("--no-planner-collision-check", action="store_true")
    parser.add_argument(
        "--diagnostic-ignore-static-gate", action="store_true",
        help="兼容旧命令；V2 persistent执行器不再使用旧static gate。",
    )
    parser.add_argument("--isaac-headless", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    VERBOSE = bool(args.verbose)
    if args.planning_only and args.sim_execute:
        raise ValueError("--planning-only and --sim-execute are mutually exclusive")
    if args.diagnostic_ignore_static_gate:
        print("⚠ --diagnostic-ignore-static-gate 在V2中仅为旧命令兼容参数，不再参与筛选。")

    root = args.project_root.expanduser().resolve()
    cfg = load_json(args.config)
    scene_folder = prompt_scene(root, args.scene_folder)
    scene_manifest = scene_folder / "scene_manifest.json"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = resolve(root, cfg["session_root"]) / stamp
    session_root.mkdir(parents=True, exist_ok=False)
    DEBUG_LOG = session_root / "debug.log"
    registry = session_root / "placement_registry.json"
    init_registry(registry)
    (session_root / "session.json").write_text(json.dumps({
        "schema_version": 2,
        "created_local": stamp,
        "architecture": cfg.get("architecture"),
        "source_scene_folder": str(scene_folder),
        "sim_execute": bool(args.sim_execute),
        "planner_collision_checks_disabled": bool(args.no_planner_collision_check),
        "coarse_ik_prefilter": cfg["coarse_ik_prefilter"],
        "flexible_ik": cfg["flexible_ik"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    persistent_config = resolve(root, cfg["persistent_isaac_config"])
    network_py = Path(cfg["network_python"])
    retarget_py = Path(cfg["retarget_python"])
    planner_py = Path(cfg["planner_python"])
    for path in (persistent_config, network_py, retarget_py, planner_py):
        if not path.is_file():
            raise FileNotFoundError(path)

    worker_cfg = WorkerConfig(
        startup_timeout_s=float(cfg.get("worker_startup_timeout_s", 180.0)),
        request_timeout_s=float(cfg.get("worker_request_timeout_s", 600.0)),
    )
    T_world_base = world_from_base(root)
    T_base_from_world = np.linalg.inv(T_world_base)

    print("\n============================================================")
    print("  Wuji2 语义灵巧抓取闭环 V2")
    print("  ✓ Isaac Sim 持续会话：一次启动，全程保留物理场景")
    print("  ✓ cuRobo 按轮启动，规划完成后释放 GPU")
    print("  ✓ COVER 精确 IK；其余阶段采用 6D 可行域批量 IK")
    print("  ✓ 执行前不再重复 IK / FK")
    print("============================================================")

    try:
        with PersistentIsaacClient(
            project_root=root,
            scene_manifest=scene_manifest,
            runtime_config=persistent_config,
            startup_timeout_s=float(cfg.get("isaac_startup_timeout_s", 300.0)),
            request_timeout_s=float(cfg.get("isaac_request_timeout_s", 300.0)),
            headless=bool(args.isaac_headless),
            verbose=VERBOSE,
            log_callback=debug_write,
        ) as isaac:
            print("✓ Isaac 持续场景已连接")

            cycle = 0
            while True:
                cycle += 1
                cycle_started = time.perf_counter()
                cycle_root = session_root / f"cycle_{cycle:03d}"
                capture_root = cycle_root / "capture"
                scratch_root = cycle_root / "scratch/final_planning"
                cycle_root.mkdir(parents=True, exist_ok=False)
                scratch_root.mkdir(parents=True, exist_ok=True)

                print(f"\n================ 第 {cycle:03d} 轮 ================")
                capture = isaac.capture(capture_root)
                rgb = Path(capture["rgb"])
                settled = Path(capture["settled_scene_manifest"])
                robot_state_path = Path(capture["robot_state"])
                show_async(cfg.get("show_rgb_command"), rgb=str(rgb))
                print(
                    f"[1] ✓ RGB-D 拍照完成 | HOME静置={float(capture['hold_s']):.1f}s "
                    f"| 有效深度={100.0*float(capture['valid_depth_fraction']):.1f}%"
                )

                print("\n你要抓什么东西？（例如 dog / red cup；输入“抓取完成”结束）")
                print("规划过程中 Ctrl+C = 取消当前目标并重新选择；输入“抓取完成” = 结束会话")
                query = input("Target > ").strip()
                stop_words = {str(value).lower() for value in cfg.get("stop_words", [])}
                if query.lower() in stop_words:
                    final_snapshot = session_root / "final_scene_manifest.json"
                    isaac.snapshot(final_snapshot)
                    print(f"\n✓ 抓取会话完成，最终场景已保存：{final_snapshot}")
                    return 0
                if not query:
                    print("⚠ 输入为空，本轮不执行规划；场景保持不变。")
                    continue
                target_slug = safe_slug(query)

                # 2) GroundingDINO + SAM
                gs_root = capture_root / "grounded_sam" / target_slug
                backend = cfg.get("grounded_sam_backend")
                if not backend:
                    raise RuntimeError("grounded_sam_backend is not configured")
                command = [str(x).format(project_root=root, rgb=rgb, text=query, output=gs_root) for x in backend]
                started = time.perf_counter()
                print("[2] GroundingDINO + SAM ...")
                run("GroundingDINO(text + RGB) -> SAM", command, cwd=root)
                gs_check = run("validate Grounded-SAM output", [
                    network_py, SCRIPTS / "validate_grounded_sam_output.py",
                    "--rgb", rgb, "--output-root", gs_root, "--query", query,
                ], cwd=root, capture_json=True)
                overlay = Path(gs_check["overlay"])
                show_async(cfg.get("show_overlay_command"), overlay=str(overlay))
                gs_result = load_json(gs_root / "result.json")
                print(
                    f"    ✓ 识别完成 | score={gs_result.get('grounding_score', gs_result.get('score', 'NA'))} "
                    f"| mask={gs_result.get('mask_pixels', gs_result.get('mask_area_px', 'NA'))} "
                    f"| {time.perf_counter()-started:.1f}s"
                )

                # 3) RGB-D -> official 40k input
                dgn_root = capture_root / "dgn2" / target_slug
                print("[3] 构建 DGN2 40k 场景点云 ...")
                run("RGB-D -> full-scene 40k + target membership", [
                    network_py, root / "08_dual_arm_scene_layout/scripts/08_build_target_network_input.py",
                    "--target", target_slug,
                    "--target-segmentation-id", str(int(cfg["dgn2_target_membership_id"])),
                    "--capture-root", capture_root,
                    "--mask", gs_root / "mask.npy",
                ], cwd=root)
                net_meta = load_json(dgn_root / "network_input.json")
                print(f"    ✓ 40k输入完成 | target_points={net_meta.get('sampled_target_point_count', 'NA')}")

                # 4) DGN2
                print("[4] DGN2 生成抓取候选 ...")
                started = time.perf_counter()
                run("Official DGN2 LEAP inference", [
                    network_py, root / "08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py",
                    "--target", target_slug,
                    "--rounds", str(int(cfg["dgn2_rounds"])),
                    "--input-root", dgn_root,
                ], cwd=root)
                prediction = dgn_root / "official_leap_1024_target_ranked.npz"
                candidates_plain, total_proposals = candidate_order(prediction)
                print(
                    f"    ✓ proposals={total_proposals} | 目标候选={len(candidates_plain)} "
                    f"| {time.perf_counter()-started:.1f}s"
                )
                rfs_runtime = run_candidate_rfs_v2(
                    project_root=root,
                    cycle_root=cycle_root,
                    query=query,
                    candidates=candidates_plain,
                    settings=cfg.get("candidate_rfs_v2", {}),
                )
                rfs_priority_indices = list(rfs_runtime.ordered_indices)

                # simulation-only binding after semantic selection
                sim_binding = cycle_root / "sim_target.json"
                bind = run("simulation-only mask -> rigid-body binding", [
                    network_py, SCRIPTS / "resolve_sim_target.py",
                    "--capture-root", capture_root,
                    "--mask", gs_root / "mask.npy",
                    "--settled-manifest", settled,
                    "--output", sim_binding,
                ], cwd=root, capture_json=True)
                sim_target_id = int(bind["segmentation_id"])
                q_current, measured = load_robot_state(robot_state_path)

                try:
                    print("[5] cuRobo 按轮启动（规划完成后释放 GPU）")
                    with CuroboWorkerClient(
                        root,
                        worker_config=worker_cfg,
                        seeds=int(cfg.get("gpu_ik_seeds", 48)),
                        batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
                    ) as curobo:
                        print(
                            f"    ✓ cuRobo 已连接 | seeds={int(cfg.get('gpu_ik_seeds', 48))} "
                            f"| GPU batch={int(cfg.get('gpu_ik_batch_size', 512))}"
                        )

                        home_gate_cfg = cfg.get("home_pregrasp_collision_gate", {})
                        home_gate_enabled = bool(home_gate_cfg.get("enabled", True))
                        need_observed_map = home_gate_enabled or not args.no_planner_collision_check
                        if need_observed_map:
                            started = time.perf_counter()
                            map_report = curobo.build_map(
                                capture_root / "depth_m.npy",
                                capture_root / "intrinsics.npy",
                                capture_root / "T_world_camera.npy",
                                gs_root / "mask.npy",
                            )
                            map_report["home_pregrasp_collision_gate"] = home_gate_enabled
                            map_report["full_planner_collision_check"] = not bool(
                                args.no_planner_collision_check
                            )
                            if args.no_planner_collision_check:
                                print(
                                    f"[5] ✓ HOME→PREGRASP RGB-D/ESDF安全地图完成 | "
                                    f"{time.perf_counter()-started:.2f}s "
                                    f"| 近场/其余规划器碰撞检查仍关闭"
                                )
                            else:
                                print(
                                    f"[5] ✓ RGB-D ESDF地图完成 | "
                                    f"{time.perf_counter()-started:.2f}s "
                                    f"| HOME→PREGRASP门禁 + 全路径碰撞检查"
                                )
                        else:
                            map_report = {
                                "status": "SKIPPED",
                                "reason": "all planner collision checks disabled",
                            }
                            print("[5] ✓ 规划器碰撞检查：全部关闭（Isaac/PhysX仍开启）")

                        # Optional legacy approximate prefilter; OFF by default.
                        coarse_cfg = cfg["coarse_ik_prefilter"]
                        if bool(coarse_cfg.get("grasp_enabled")) or bool(coarse_cfg.get("pregrasp_enabled")):
                            candidates, survivor_indices, coarse_report = legacy_coarse_prefilter(
                                client=curobo,
                                project_root=root,
                                prediction=prediction,
                                q_current=q_current,
                                cfg=cfg,
                            )
                            print(
                                f"[6] 旧粗筛已启用 | target={len(candidates)} -> survivors={len(survivor_indices)}"
                            )
                        else:
                            candidates = candidates_plain
                            survivor_indices = list(range(len(candidates)))
                            coarse_report = {
                                "enabled": False,
                                "grasp_enabled": False,
                                "pregrasp_enabled": False,
                                "survivors": len(survivor_indices),
                            }
                            print(
                                f"[6] ✓ 旧粗 GRASP/PREGRASP IK：关闭 | {len(candidates)} 个目标候选直接进入真实 Wuji2"
                            )

                        allowed_survivors = {int(index) for index in survivor_indices}
                        survivor_indices = [
                            int(index) for index in rfs_priority_indices
                            if int(index) in allowed_survivors
                        ]
                        coarse_report["candidate_rfs_v2"] = rfs_runtime.to_jsonable()
                        print(
                            f"[RFS V2] production order applied | "
                            f"ordered survivors={len(survivor_indices)} | "
                            f"status={rfs_runtime.status}"
                        )

                        max_to_test = int(cfg.get("max_candidates_to_test", 0))
                        if max_to_test > 0:
                            survivor_indices = survivor_indices[:max_to_test]
                        retarget_chunk_size = int(cfg.get("retarget_chunk_size", 64))
                        total_batches = math.ceil(len(survivor_indices) / retarget_chunk_size)
                        selected = None
                        tested_cover = 0
                        retargeted = 0

                        print("[7] Wuji2 重定向 + 精确 COVER + Flexible IK 搜索")
                        for chunk_index, start in enumerate(range(0, len(survivor_indices), retarget_chunk_size), start=1):
                            local_indices = survivor_indices[start:start + retarget_chunk_size]
                            chunk_items = []
                            for local_index in local_indices:
                                item = candidates[local_index]
                                rank = int(item["target_rank"])
                                idx = int(item["candidate_index"])
                                case_id = f"{cfg.get('candidate_case_prefix','closedloop')}_r{rank:04d}_cand{idx:04d}"
                                case_root = scratch_root / f"rank_{rank:04d}" / case_id
                                chunk_items.append({
                                    "local_target_index": int(local_index),
                                    "target_rank": rank,
                                    "candidate_index": idx,
                                    "official_score": float(item.get("score", item.get("official_score", float('nan')))),
                                    "case_id": case_id,
                                    "case_root": str(case_root),
                                })
                            if not chunk_items:
                                continue
                            chunk_dir = scratch_root / f"batch_{chunk_index:03d}"
                            chunk_dir.mkdir(parents=True, exist_ok=True)
                            items_json = chunk_dir / "items.json"
                            items_json.write_text(json.dumps(chunk_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                            batch_started = time.perf_counter()
                            run("batch build candidate cases", [
                                network_py, SCRIPTS / "batch_build_candidate_cases.py",
                                "--project-root", root,
                                "--prediction", prediction,
                                "--network-input", dgn_root / "network_input.npz",
                                "--capture-root", capture_root,
                                "--settled-manifest", settled,
                                "--sim-target-segmentation-id", str(sim_target_id),
                                "--items-json", items_json,
                                "--output", chunk_dir / "batch_build_report.json",
                            ], cwd=root, capture_json=True)
                            run("batch LEAP->Wuji2 retarget", [
                                retarget_py, SCRIPTS / "batch_retarget_cases.py",
                                "--items-json", items_json,
                                "--output", chunk_dir / "batch_retarget_report.json",
                            ], cwd=root, capture_json=True)
                            finalize_report = run("batch finalize Wuji2 + arm targets", [
                                network_py, SCRIPTS / "batch_finalize_candidate_cases.py",
                                "--items-json", items_json,
                                "--output", chunk_dir / "batch_finalize_report.json",
                            ], cwd=root, capture_json=True)
                            retargeted += len(chunk_items)
                            finalized_case_roots = {
                                str(Path(row["case_root"]).resolve())
                                for row in json.loads((chunk_dir / "batch_finalize_report.json").read_text(encoding="utf-8")).get("results", [])
                                if row.get("status") == "PASS"
                            }
                            finalized_items = [
                                item for item in chunk_items
                                if str(Path(item["case_root"]).resolve()) in finalized_case_roots
                            ]
                            finalize_reject_count = int(finalize_report.get("reject_count", len(chunk_items) - len(finalized_items)))
                            if not finalized_items:
                                print(
                                    f"    Batch {chunk_index:02d}/{total_batches:02d} ✓ 重定向={len(chunk_items)} | "
                                    f"finalize PASS=0 REJECT={finalize_reject_count} | "
                                    f"{time.perf_counter()-batch_started:.1f}s"
                                )
                                continue

                            cover_rows = screen_exact_cover_batch(
                                client=curobo,
                                case_roots=[Path(item["case_root"]) for item in finalized_items],
                                q_current=q_current,
                                measured=measured,
                                T_base_from_world=T_base_from_world,
                                T_world_base=T_world_base,
                                no_planner_collision_check=bool(args.no_planner_collision_check),
                                block_unknown=bool(cfg.get("block_unknown_space", False)),
                                solutions_per_candidate=int(cfg["flexible_ik"]["selection"]["cover_solutions_per_candidate"]),
                            )
                            passed_cover = [row for row in cover_rows if row["pass"]]
                            tested_cover += len(cover_rows)
                            print(
                                f"    Batch {chunk_index:02d}/{total_batches:02d} ✓ 重定向={len(chunk_items)} | "
                                f"finalize PASS={len(finalized_items)} REJECT={finalize_reject_count} | "
                                f"精确COVER可达={len(passed_cover)} | {time.perf_counter()-batch_started:.1f}s"
                            )

                            # Preserve official DGN2 order inside the batch.
                            by_case = {str(Path(item["case_root"]).resolve()): item for item in finalized_items}
                            for cover_row in passed_cover:
                                item = by_case[cover_row["case_root"]]
                                route_started = time.perf_counter()
                                route = plan_flexible_route(
                                    client=curobo,
                                    project_root=root,
                                    case_root=Path(item["case_root"]),
                                    cover_solutions=cover_row["cover_solutions"],
                                    q_current=q_current,
                                    measured=measured,
                                    placement_registry=registry,
                                    config=cfg,
                                    no_planner_collision_check=bool(args.no_planner_collision_check),
                                    block_unknown=bool(cfg.get("block_unknown_space", False)),
                                )
                                if route.get("status") == "PASS":
                                    selected = {
                                        "target_rank": int(item["target_rank"]),
                                        "candidate_index": int(item["candidate_index"]),
                                        "official_score": float(item["official_score"]),
                                        "case_root": str(Path(item["case_root"]).resolve()),
                                        "route": route,
                                    }
                                    print(
                                        f"    ✓ Flexible Route PASS | rank={selected['target_rank']} "
                                        f"candidate={selected['candidate_index']} | {time.perf_counter()-route_started:.2f}s"
                                    )
                                    _print_route_summary(route)
                                    break
                                if VERBOSE:
                                    print(
                                        f"    ✗ route rank={item['target_rank']} cand={item['candidate_index']}: "
                                        f"{route.get('reason')}"
                                    )
                            if selected is not None:
                                break

                        planning_result = {
                            "schema_version": 2,
                            "status": "PASS" if selected is not None else "FAIL",
                            "architecture": cfg.get("architecture"),
                            "query": query,
                            "total_proposals": total_proposals,
                            "target_candidates": len(candidates),
                            "retargeted_candidate_count": retargeted,
                            "exact_cover_tested": tested_cover,
                            "coarse_prefilter": coarse_report,
                            "map": map_report,
                            "selected": selected,
                            "planning_wall_s": time.perf_counter() - cycle_started,
                        }
                        planning_path = cycle_root / "planning_result.json"
                        planning_path.write_text(json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                except KeyboardInterrupt:
                    planning_result = {
                        "schema_version": 2,
                        "status": "CANCELLED",
                        "architecture": cfg.get("architecture"),
                        "query": query,
                        "planning_wall_s": time.perf_counter() - cycle_started,
                    }
                    planning_path = cycle_root / "planning_result.json"
                    planning_path.write_text(json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(f"[CANCEL] 已取消当前目标：{query}")
                    print("✓ cuRobo 已释放")
                    print("✓ Isaac 会话继续保留，机械臂仍在 HOME")
                    continue

                if selected is None:
                    print(f"\n✗ 本轮未找到完整可行路线；场景保持原样，可重新描述目标或继续尝试。")
                    print(f"  详细日志：{DEBUG_LOG}")
                    # Persistent Isaac is still paused at the captured state.
                    # The next cycle will only perform the configured 1 s HOME
                    # hold + fresh RGB-D capture; it will NOT reload the scene.
                    continue

                print("\n---------------- 规划结果 ----------------")
                print(f"✓ target rank : {selected['target_rank']}")
                print(f"✓ candidate   : {selected['candidate_index']}")
                print(f"✓ route plan  : {selected['route']['output_npz']}")
                print(f"✓ planning    : {planning_result['planning_wall_s']:.1f}s")
                print("------------------------------------------")

                if args.planning_only or not args.sim_execute:
                    print("✓ Planning-only 完成；未执行物理抓取。")
                    return 0

                print("[8] 同一 Isaac 场景直接执行（不重复加载，不二次 IK）")
                execution_root = cycle_root / "execution"
                execution = isaac.execute(
                    case_root=selected["case_root"],
                    plan_npz=selected["route"]["output_npz"],
                    output_dir=execution_root,
                    target_segmentation_id=sim_target_id,
                )
                execution_status = str(execution.get("status", ""))
                if execution_status == "RECOVERED_FAIL":
                    print("✗ 物理执行失败，但 runtime 已完成恢复；不提交 placement，进入下一轮。")
                    print(f"  failure_stage   : {execution.get('failure_stage')}")
                    print(f"  failure_type    : {execution.get('failure_type')}")
                    print(f"  failure_reason  : {execution.get('failure_reason')}")
                    print(f"  recovery_status : {execution.get('recovery_status')}")
                    print(f"  report          : {execution.get('report')}")
                    continue
                if execution_status != "PASS":
                    print(f"✗ 物理执行失败：{execution.get('report')}")
                    return 3
                commit_placement(registry, cycle=cycle, execution=execution, selected=selected)
                print(
                    f"✓ 物理执行 PASS | 抬升={float(execution['max_object_lift_mm']):.1f}mm "
                    f"| 放置中心在绿色区域={execution['final_object_center_inside_green_zone']}"
                )
                print("✓ 机械臂已回 HOME；下一轮拍照前仅静置 1.0s，场景不会重新加载。")
                print(f"✓ 本轮总耗时={time.perf_counter()-cycle_started:.1f}s")

    except KeyboardInterrupt:
        print("\n[STOP] 用户中断")
        return 130
    except Exception as exc:
        debug_write(traceback_text := f"{type(exc).__name__}: {exc}")
        print(f"\n✗ ERROR: {traceback_text}")
        print(f"详细日志：{DEBUG_LOG}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
