#!/usr/bin/env python3
"""One-command interactive semantic dexterous grasp loop.

Heavy stages remain isolated:
- Isaac Lab/Sim capture and physics: isaaclab22_sim50
- GroundingDINO+SAM: configured vision backend
- Official DexGraspNet2 LEAP inference: graspnet2.0
- LEAP->Wuji2: wuji_retargeting
- cuRobo IK/ESDF: persistent worker in curobo_v2

Physical execution remains disabled unless explicitly enabled outside this
planning-only integration path.
"""
from __future__ import annotations
import argparse, json, math, os, re, selectors, shlex, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE/"scripts"
CONTROL_ROOT = HERE.parent
DEFAULT_CONFIG = HERE/"config/closed_loop.json"
sys.path.insert(0, str(CONTROL_ROOT))
sys.path.insert(0, str(SCRIPTS))

from core.bridge import CuroboWorkerClient  # noqa: E402
from core.config import WorkerConfig  # noqa: E402
from all_candidate_gpu_prefilter import (  # noqa: E402
    load_targets,
    nvidia_memory_mib,
    run_strict_ordered_prefilter,
)
from screen_pick_batches import (  # noqa: E402
    PICK_STAGES,
    HAND_FOR,
    PHASE_FOR,
    load_case_pick_contract,
    retarget_case,
    world_from_base,
)

VERBOSE = False
DEBUG_LOG: Path | None = None

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def debug_write(text: str) -> None:
    if DEBUG_LOG is None:
        return
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as stream:
        stream.write(text)
        if text and not text.endswith("\n"):
            stream.write("\n")

def safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"

def resolve(root: Path, value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root/p).resolve()

def run(label, cmd, *, cwd, env=None, capture_json=False):
    command_line = " ".join(shlex.quote(str(x)) for x in cmd)
    debug_write(f"\n{'='*18} {label} {'='*18}\n$ {command_line}\n")
    if VERBOSE:
        print(f"\n{'='*18} {label} {'='*18}")
        print("$", command_line, flush=True)
    if capture_json:
        cp=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        debug_write(cp.stdout)
        if VERBOSE:
            print(cp.stdout,end="")
        if cp.returncode:
            raise RuntimeError(f"{label} failed: {cp.returncode}")
        lines=[x for x in cp.stdout.splitlines() if x.strip()]
        for line in reversed(lines):
            try: return json.loads(line)
            except Exception: pass
        raise RuntimeError(f"{label} did not emit JSON")
    if VERBOSE:
        cp=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env)
    else:
        cp=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        debug_write(cp.stdout or "")
    if cp.returncode:
        raise RuntimeError(f"{label} failed: {cp.returncode}")

def run_runtime_until_report(label, cmd, *, cwd: Path, report_path: Path, exit_grace_s: float = 20.0) -> dict:
    """Run Isaac runtime and return after PASS/FAIL report is durably visible.

    Isaac/Kit can occasionally hang during application shutdown after the
    runtime has already written report.json and physical_replay_30fps.npz.  The
    closed-loop orchestrator should not block the next perception cycle on that
    shutdown tail.  This function still streams runtime output and only
    terminates the subprocess group after the auditable report and replay exist.
    """
    command_line = " ".join(shlex.quote(str(x)) for x in cmd)
    debug_write(f"\n{'='*18} {label} {'='*18}\n$ {command_line}\n")
    if VERBOSE:
        print(f"\n{'='*18} {label} {'='*18}")
        print("$", command_line, flush=True)
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    report_seen_at = None
    final_report = None
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    while True:
        for key, _ in selector.select(timeout=0.2):
            line = key.fileobj.readline()
            if line:
                debug_write(line)
                if VERBOSE:
                    print(line, end="", flush=True)
        if report_path.is_file():
            try:
                report = load_json(report_path)
                replay = Path(report.get("physical_replay_30fps", ""))
                if report.get("status") in {"PASS", "FAIL"} and replay.is_file():
                    if report_seen_at is None:
                        report_seen_at = time.perf_counter()
                        final_report = report
                    elif time.perf_counter() - report_seen_at >= float(exit_grace_s):
                        if proc.poll() is None:
                            print(
                                f"[RUNTIME WATCHDOG] report/replay ready; terminating lingering Isaac process after {exit_grace_s:.1f}s shutdown grace",
                                flush=True,
                            )
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=10.0)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                                proc.wait(timeout=10.0)
                        return final_report
            except Exception:
                pass
        code = proc.poll()
        if code is not None:
            selector.unregister(proc.stdout)
            remainder = proc.stdout.read()
            if remainder:
                debug_write(remainder)
                if VERBOSE:
                    print(remainder, end="", flush=True)
            if code != 0:
                raise RuntimeError(f"{label} failed: {code}")
            if report_path.is_file():
                return load_json(report_path)
            raise FileNotFoundError(report_path)
        if not line:
            time.sleep(0.1)

def show_async(template, **kwargs):
    if not template: return
    cmd=[str(x).format(**kwargs) for x in template]
    try:
        subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    except Exception as exc:
        print(f"[WARN] viewer failed: {exc}")

def prompt_scene(project_root: Path, supplied: str|None) -> Path:
    if supplied:
        folder=Path(supplied).expanduser()
    else:
        print("\n请输入场景文件夹地址。要求：该文件夹必须直接包含 scene_manifest.json")
        print("示例：")
        print("  /home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/data/scene_datasets/"
              "wuji2_test60_10upright_10view_v1/scenes/scene_0000")
        folder=Path(input("\nScene folder > ").strip()).expanduser()
    if not folder.is_absolute():
        folder=(project_root/folder).resolve()
    else:
        folder=folder.resolve()
    manifest=folder/"scene_manifest.json"
    if not folder.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f"场景目录必须包含 scene_manifest.json: {folder}")
    print(f"[SCENE] {folder}")
    return folder

def write_runtime_config(project_root: Path, cfg: dict, capture_root: Path,
                         target_slug: str, output_dir: Path) -> Path:
    template=load_json(resolve(project_root,cfg["runtime_config_template"]))
    template["capture_root"]=str(capture_root)
    template["target_key"]=target_slug
    template["output_directory"]=str(output_dir)
    path=output_dir/"runtime_config.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(template,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return path

def write_session_placement_policy(project_root: Path, cfg: dict, session_root: Path) -> Path:
    policy = load_json(resolve(project_root, cfg["placement_policy"]))
    registry = session_root / "placement_registry.json"
    policy["occupancy_registry"] = str(registry)
    out = session_root / "placement_policy.json"
    out.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not registry.exists():
        registry.write_text(json.dumps({
            "schema_version": 1,
            "purpose": "Session-local closed-loop diagnostic placement registry",
            "placements": [],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out

def load_robot_state(robot_state: Path) -> tuple[np.ndarray, dict]:
    state = load_json(robot_state)
    q_current = np.asarray(state["right_arm_q_current_rad"], dtype=np.float64)
    measured = {str(k): float(v) for k, v in state["joint_positions_by_name"].items()}
    return q_current, measured

def group_feasible(group: dict, *, block_unknown: bool) -> tuple[bool, bool]:
    unknown = group.get("unknown_space_exposure") or []
    unknown_policy_pass = (not any(bool(x) for x in unknown)) if block_unknown else True
    feasible = bool(
        group.get("ik_pass") is True
        and group.get("observed_scene_collision_pass") is True
        and group.get("path_pass") is True
        and unknown_policy_pass
    )
    return feasible, unknown_policy_pass

def ik_only_prefilter(
    *,
    client: CuroboWorkerClient,
    candidates: list[dict],
    grasp_targets: np.ndarray,
    pregrasp_targets: np.ndarray,
    q_current: np.ndarray,
) -> dict:
    print(f"[IK PREFILTER NO COLLISION] total_target_candidates={len(candidates)}", flush=True)
    grasp_start = time.perf_counter()
    grasp = client.solve_ik(grasp_targets, q_current, select_chain=False)
    grasp_wall_s = time.perf_counter() - grasp_start
    grasp_counts = [int(x) for x in grasp["accepted_per_target"]]
    grasp_survivors = [i for i, count in enumerate(grasp_counts) if count > 0]
    print(
        "[IK PREFILTER NO COLLISION] GRASP "
        f"raw_ik_reachable={sum(int(x) > 0 for x in grasp['raw_success_per_target'])} "
        f"threshold_accepted={len(grasp_survivors)} "
        f"ik_time_s={float(grasp.get('solve_time_s', 0.0)):.3f} wall_s={grasp_wall_s:.3f}",
        flush=True,
    )
    pre_start = time.perf_counter()
    if grasp_survivors:
        pregrasp = client.solve_ik(pregrasp_targets[grasp_survivors], q_current, select_chain=False)
        pre_counts = [int(x) for x in pregrasp["accepted_per_target"]]
        survivors = [grasp_survivors[i] for i, count in enumerate(pre_counts) if count > 0]
    else:
        pregrasp = {"raw_success_per_target": [], "accepted_per_target": [], "solve_time_s": 0.0}
        survivors = []
    pre_wall_s = time.perf_counter() - pre_start
    print(
        "[IK PREFILTER NO COLLISION] PREGRASP "
        f"input={len(grasp_survivors)} "
        f"raw_ik_reachable={sum(int(x) > 0 for x in pregrasp['raw_success_per_target'])} "
        f"threshold_accepted={len(survivors)} "
        f"ik_time_s={float(pregrasp.get('solve_time_s', 0.0)):.3f} wall_s={pre_wall_s:.3f}",
        flush=True,
    )
    print("[IK PREFILTER NO COLLISION] planner collision/path checks skipped", flush=True)
    return {
        "SELF_COLLISION_POLICY": "SKIPPED_BY_NO_PLANNER_COLLISION_CHECK",
        "planner_collision_checks": "DISABLED",
        "candidate_count": len(candidates),
        "survivor_indices": [int(x) for x in survivors],
        "grasp": {
            "raw_ik_reachable": sum(int(x) > 0 for x in grasp["raw_success_per_target"]),
            "threshold_accepted": len(grasp_survivors),
            "ik_time_s": float(grasp.get("solve_time_s", 0.0)),
            "wall_time_s": grasp_wall_s,
        },
        "pregrasp": {
            "input": len(grasp_survivors),
            "raw_ik_reachable": sum(int(x) > 0 for x in pregrasp["raw_success_per_target"]),
            "threshold_accepted": len(survivors),
            "ik_time_s": float(pregrasp.get("solve_time_s", 0.0)),
            "wall_time_s": pre_wall_s,
        },
    }

def solve_pick_gate(
    *,
    client: CuroboWorkerClient,
    case_root: Path,
    measured: dict,
    q_current: np.ndarray,
    T_world_base: np.ndarray,
    T_base_from_world: np.ndarray,
    block_unknown: bool,
    no_planner_collision_check: bool,
) -> dict:
    contract = load_case_pick_contract(case_root, measured, T_base_from_world)
    collision_context = None if no_planner_collision_check else {
        "phases": contract["phases"],
        "joint_positions_by_name": measured,
        "joint_positions_by_target": contract["states"],
        "T_world_base": T_world_base,
        "margin_m": 0.0,
        "include_return_to_reference": False,
    }
    solve = client.solve_ik_groups(
        contract["targets_base"],
        q_current,
        group_sizes=[len(PICK_STAGES)],
        select_chain=True,
        collision_context=collision_context,
    )
    group = solve["groups"][0]
    if no_planner_collision_check:
        unknown_policy_pass = True
        feasible = bool(group.get("ik_pass") is True and group.get("selected") is not None)
    else:
        feasible, unknown_policy_pass = group_feasible(group, block_unknown=block_unknown)
    return {
        "status": "PASS" if feasible else "FAIL",
        "feasible": feasible,
        "unknown_policy_pass": unknown_policy_pass,
        "solve": solve,
        "group": group,
    }

def load_full_route_contract(
    *,
    case_root: Path,
    measured: dict,
    T_base_from_world: np.ndarray,
) -> dict:
    route_path = case_root/"07_arm_execution/full_arm_waypoint_ik.npz"
    hand_path = case_root/"06_isaacsim/final_waypoints.npz"
    with np.load(route_path, allow_pickle=False) as z:
        names = [str(x) for x in z["waypoint_names"].tolist()]
        flange_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
    with np.load(hand_path, allow_pickle=False) as z:
        hand_names = [str(x) for x in z["finger_joint_names"].tolist()]
        hand_stage_names = [str(x) for x in z["waypoint_names"].tolist()]
        hand_q5 = np.asarray(z["waypoint_joint_positions"][0], dtype=np.float64)
    hand_index = {name: i for i, name in enumerate(hand_stage_names)}
    hand_for = {
        **HAND_FOR,
        "transfer": "squeeze",
        "place": "squeeze",
        "release": "pregrasp",
        "retreat": "pregrasp",
    }
    phase_for = {
        **PHASE_FOR,
        "transfer": "lift",
        "place": "lift",
        "release": "lift",
        "retreat": "lift",
    }
    states = []
    phases = []
    for stage in names:
        if stage not in hand_for:
            raise RuntimeError(f"No hand/phase policy for full-route stage: {stage}")
        named = dict(measured)
        qh = hand_q5[hand_index[hand_for[stage]]]
        for joint_name, q in zip(hand_names, qh):
            named[joint_name] = float(q)
        states.append(named)
        phases.append(phase_for[stage])
    return {
        "names": names,
        "targets_base": np.stack([T_base_from_world @ T for T in flange_world]),
        "states": states,
        "phases": phases,
    }

def solve_full_route_gate(
    *,
    client: CuroboWorkerClient,
    case_root: Path,
    measured: dict,
    q_current: np.ndarray,
    T_world_base: np.ndarray,
    T_base_from_world: np.ndarray,
    block_unknown: bool,
    no_planner_collision_check: bool,
) -> dict:
    contract = load_full_route_contract(
        case_root=case_root,
        measured=measured,
        T_base_from_world=T_base_from_world,
    )
    collision_context = None if no_planner_collision_check else {
        "phases": contract["phases"],
        "joint_positions_by_name": measured,
        "joint_positions_by_target": contract["states"],
        "T_world_base": T_world_base,
        "margin_m": 0.0,
        "include_return_to_reference": True,
    }
    solve = client.solve_ik(
        contract["targets_base"],
        q_current,
        select_chain=True,
        collision_context=collision_context,
    )
    unknown = solve.get("unknown_space_exposure") or []
    unknown_policy_pass = (not any(bool(x) for x in unknown)) if block_unknown else True
    if no_planner_collision_check:
        unknown_policy_pass = True
        feasible = bool(solve.get("ik_pass") is True and solve.get("selected") is not None)
    else:
        feasible = bool(
            solve.get("ik_pass") is True
            and solve.get("selected") is not None
            and solve.get("observed_scene_collision_pass") is True
            and solve.get("path_pass") is True
            and unknown_policy_pass
        )
    return {
        "status": "PASS" if feasible else "FAIL",
        "feasible": feasible,
        "unknown_policy_pass": unknown_policy_pass,
        "route_stages": contract["names"] + ["HOME"],
        "home_planned": True,
        "solve": solve,
    }

def summarize_failure(result: dict) -> dict:
    solve = result.get("solve") or {}
    group = result.get("group") or solve
    return {
        "status": result.get("status"),
        "ik_pass": group.get("ik_pass", solve.get("ik_pass")),
        "observed_scene_collision_pass": group.get(
            "observed_scene_collision_pass",
            solve.get("observed_scene_collision_pass"),
        ),
        "path_pass": group.get("path_pass", solve.get("path_pass")),
        "unknown_policy_pass": result.get("unknown_policy_pass"),
        "self_collision_pass_report_only": group.get(
            "self_collision_pass",
            solve.get("self_collision_pass"),
        ),
    }

def main() -> int:
    global VERBOSE, DEBUG_LOG
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    ap.add_argument("--scene-folder")
    ap.add_argument("--planning-only",action="store_true")
    ap.add_argument("--sim-execute",action="store_true")
    ap.add_argument("--no-planner-collision-check",action="store_true")
    ap.add_argument("--diagnostic-ignore-static-gate",action="store_true")
    ap.add_argument("--verbose",action="store_true")
    args=ap.parse_args()
    VERBOSE = bool(args.verbose)
    if args.planning_only and args.sim_execute:
        raise ValueError("--planning-only and --sim-execute are mutually exclusive")
    if args.diagnostic_ignore_static_gate and not args.sim_execute:
        raise ValueError("--diagnostic-ignore-static-gate requires --sim-execute")
    if args.sim_execute:
        print(
            "\n============================================\n"
            "SIMULATION DIAGNOSTIC EXECUTION\n"
            f"PLANNER COLLISION CHECKS: {'DISABLED' if args.no_planner_collision_check else 'ENABLED'}\n"
            f"STATIC STABILITY GATE: {'DIAGNOSTIC OVERRIDE' if args.diagnostic_ignore_static_gate else 'ENFORCED'}\n"
            "ISAAC / PHYSX COLLISIONS: ENABLED\n"
            "REAL ROBOT OUTPUT: DISABLED\n"
            "============================================\n",
            flush=True,
        )
    root=args.project_root.resolve()
    cfg=load_json(args.config.resolve())
    scene_folder=prompt_scene(root,args.scene_folder)
    current_scene_manifest=scene_folder/"scene_manifest.json"

    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root=resolve(root,cfg["session_root"])/stamp
    session_root.mkdir(parents=True,exist_ok=False)
    (session_root/"session.json").write_text(json.dumps({
        "schema_version":1,
        "created_local":stamp,
        "source_scene_folder":str(scene_folder),
        "source_scene_manifest":str(current_scene_manifest),
        "sim_execute": bool(args.sim_execute),
        "no_planner_collision_check": bool(args.no_planner_collision_check),
        "diagnostic_ignore_static_gate": bool(args.diagnostic_ignore_static_gate),
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    DEBUG_LOG = session_root/"debug.log"
    debug_write(f"[SESSION] {session_root}\n")
    session_placement_policy = write_session_placement_policy(root, cfg, session_root)

    network_py=Path(cfg["network_python"])
    retarget_py=Path(cfg["retarget_python"])
    planner_py=Path(cfg["planner_python"])
    for p in (network_py,retarget_py,planner_py):
        if not p.is_file(): raise FileNotFoundError(p)

    cycle=0
    while True:
        wall_started = time.perf_counter()
        cycle+=1
        cycle_root=session_root/f"cycle_{cycle:03d}"
        capture_root=cycle_root/"capture"
        cycle_root.mkdir(parents=True,exist_ok=False)
        print(f"\n================ CYCLE {cycle:03d} ================", flush=True)

        capture_launcher=resolve(root,cfg["capture_launcher"])
        stage_started=time.perf_counter()
        print("[1] Capture", flush=True)
        run("ISAAC LAB / SIM: settle + aligned RGB-D capture",[
            capture_launcher,
            "--scene-manifest",current_scene_manifest,
            "--output",capture_root,
        ],cwd=root)
        rgb=capture_root/"rgb.png"
        settled=capture_root/"settled_scene_manifest.json"
        robot_state=capture_root/"robot_state.json"
        for p in (rgb, settled):
            if not p.is_file(): raise FileNotFoundError(p)
        show_async(cfg.get("show_rgb_command"),rgb=str(rgb))
        print(f"    PASS | {time.perf_counter()-stage_started:.1f} s", flush=True)

        print("\n你要抓什么东西？")
        print("直接输入自然语言，例如：dog / ashtray / red cup")
        print("终端不会读取场景物体列表来替你做语义判断；输入 q 结束。")
        query=input("Target > ").strip()
        if query.lower() in {"q","quit","exit","结束","退出"}:
            print("[DONE] closed loop stopped by user")
            return 0
        if not query:
            print("[WARN] empty query; recapture is kept, ask again next cycle")
            cycle-=1
            continue
        target_slug=safe_slug(query)

        gs_root=capture_root/"grounded_sam"/target_slug
        backend=cfg.get("grounded_sam_backend")
        if not backend:
            raise RuntimeError(
                "closed_loop.json grounded_sam_backend is null. "
                "Codex must wire the local GroundingDINO+SAM backend first."
            )
        command=[str(x).format(project_root=root,rgb=rgb,text=query,output=gs_root) for x in backend]
        stage_started=time.perf_counter()
        print("[2] GroundingDINO + SAM", flush=True)
        run("GroundingDINO(text + RGB) -> SAM",command,cwd=root)
        gs_check=run("validate Grounded-SAM output",[
            network_py,SCRIPTS/"validate_grounded_sam_output.py",
            "--rgb",rgb,"--output-root",gs_root,"--query",query
        ],cwd=root,capture_json=True)
        overlay=Path(gs_check["overlay"])
        show_async(cfg.get("show_overlay_command"),overlay=str(overlay))
        gs_result=load_json(gs_root/"result.json") if (gs_root/"result.json").is_file() else {}
        mask_pixels=gs_result.get("mask_pixels", gs_result.get("mask_area_px", "NA"))
        score=gs_result.get("grounding_score", gs_result.get("score", "NA"))
        print(f"    PASS | score={score} | mask={mask_pixels} | {time.perf_counter()-stage_started:.1f} s", flush=True)

        dgn_root=capture_root/"dgn2"/target_slug
        stage_started=time.perf_counter()
        print("[3] RGB-D -> 40k", flush=True)
        run("RGB-D -> full-scene 40k + target membership",[
            network_py,root/"08_dual_arm_scene_layout/scripts/08_build_target_network_input.py",
            "--target",target_slug,
            "--target-segmentation-id",str(int(cfg["dgn2_target_membership_id"])),
            "--capture-root",capture_root,
            "--mask",gs_root/"mask.npy",
        ],cwd=root)
        net_meta = load_json(dgn_root/"network_input.json") if (dgn_root/"network_input.json").is_file() else {}
        target_points = net_meta.get("target_points", net_meta.get("target_point_count", "NA"))
        print(f"    PASS | target_points={target_points} | {time.perf_counter()-stage_started:.1f} s", flush=True)
        stage_started=time.perf_counter()
        print("[4] DGN2", flush=True)
        run("Official DGN2 LEAP inference",[
            network_py,root/"08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py",
            "--target",target_slug,
            "--rounds",str(int(cfg["dgn2_rounds"])),
            "--input-root",dgn_root,
        ],cwd=root)
        prediction=dgn_root/"official_leap_1024_target_ranked.npz"
        dgn_meta = load_json(dgn_root/"official_leap_1024_target_ranked.json") if (dgn_root/"official_leap_1024_target_ranked.json").is_file() else {}
        print(
            f"    PASS | proposals={dgn_meta.get('total_proposals', 'NA')} "
            f"| target={dgn_meta.get('target_proposals', 'NA')} | {time.perf_counter()-stage_started:.1f} s",
            flush=True,
        )

        sim_binding=cycle_root/"sim_target.json"
        bind=run("simulation-only mask -> rigid-body binding",[
            network_py,SCRIPTS/"resolve_sim_target.py",
            "--capture-root",capture_root,
            "--mask",gs_root/"mask.npy",
            "--settled-manifest",settled,
            "--output",sim_binding,
        ],cwd=root,capture_json=True)
        sim_target_id=int(bind["segmentation_id"])

        if not robot_state.is_file():
            raise FileNotFoundError(
                f"{robot_state}\nCapture must save measured 35-joint state and q_current after settle."
            )
        q_current, measured = load_robot_state(robot_state)
        T_world_base = world_from_base(root)
        T_base_from_world = np.linalg.inv(T_world_base)
        planning_result_path = cycle_root/"planning_result.json"
        scratch_root = cycle_root/"scratch/final_planning"
        scratch_root.mkdir(parents=True, exist_ok=True)

        worker_start_count = 1
        map_build_count = 0
        retargeted_count = 0
        exact_pick_feasible_count = 0
        full_route_candidates_tested = 0
        selected = None
        tested = []
        mem_before = nvidia_memory_mib()

        worker_cfg = WorkerConfig(
            startup_timeout_s=float(cfg.get("worker_startup_timeout_s", 180.0)),
            request_timeout_s=float(cfg.get("worker_request_timeout_s", 600.0)),
        )
        with CuroboWorkerClient(
            root,
            worker_config=worker_cfg,
            seeds=int(cfg.get("gpu_ik_seeds", 48)),
            batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
        ) as client:
            if args.no_planner_collision_check:
                map_report = {
                    "status": "SKIPPED_BY_NO_PLANNER_COLLISION_CHECK",
                    "planner_collision_checks": "DISABLED",
                }
                map_wall_s = 0.0
                map_build_count = 0
            else:
                map_started = time.perf_counter()
                map_report = client.build_map(
                    capture_root/"depth_m.npy",
                    capture_root/"intrinsics.npy",
                    capture_root/"T_world_camera.npy",
                    gs_root/"mask.npy",
                )
                map_wall_s = time.perf_counter() - map_started
                map_build_count = 1

            candidates, grasp_targets, pregrasp_targets, total_proposals = load_targets(
                root,
                prediction,
                float(cfg.get("pregrasp_offset_m", 0.10)),
            )
            prefilter_started = time.perf_counter()
            print("[5] Coarse IK", flush=True)
            if args.no_planner_collision_check:
                ordered_prefilter = ik_only_prefilter(
                    client=client,
                    candidates=candidates,
                    grasp_targets=grasp_targets,
                    pregrasp_targets=pregrasp_targets,
                    q_current=q_current,
                )
            else:
                ordered_prefilter = run_strict_ordered_prefilter(
                    client=client,
                    candidates=candidates,
                    grasp_targets=grasp_targets,
                    pregrasp_targets=pregrasp_targets,
                    q_current=q_current,
                    measured=measured,
                    T_world_base=T_world_base,
                    path_batch_size=int(cfg.get("path_collision_progress_batch_size", cfg.get("gpu_ik_batch_size", 512))),
                    path_max_joint_step_rad=math.radians(float(cfg.get("approach_path_max_joint_step_deg", 3.0))),
                    progress=True,
                )
            prefilter_wall_s = time.perf_counter() - prefilter_started
            survivor_indices = [int(x) for x in ordered_prefilter["survivor_indices"]]
            survivor_source = (
                "ik_only_grasp_pregrasp_survivors_no_planner_collision_check"
                if args.no_planner_collision_check
                else "strict_ordered_grasp_pregrasp_scene_and_approach_path_survivors"
            )

            print(
                f"[PREFILTER] total={total_proposals} target={len(candidates)} "
                f"survivors={len(survivor_indices)} source={survivor_source}",
                flush=True,
            )
            print("[6] Wuji2 + Exact IK Search", flush=True)

            retarget_chunk_size = int(cfg.get("retarget_chunk_size", 64))
            for chunk_index, start in enumerate(range(0, len(survivor_indices), retarget_chunk_size), start=1):
                chunk_indices = survivor_indices[start:start + retarget_chunk_size]
                chunk_items = []
                for local_index in chunk_indices:
                    item = candidates[local_index]
                    rank = int(item["target_rank"])
                    idx = int(item["candidate_index"])
                    case_id = f"{cfg.get('candidate_case_prefix','closedloop')}_r{rank:03d}_cand{idx:04d}"
                    case_root = scratch_root/f"rank_{rank:04d}"/case_id
                    chunk_items.append({
                        "local_target_index": int(local_index),
                        "target_rank": rank,
                        "candidate_index": idx,
                        "official_score": float(item["score"]),
                        "case_id": case_id,
                        "case_root": str(case_root),
                    })
                if not chunk_items:
                    continue
                chunk_dir = scratch_root/f"batch_{chunk_index:03d}_rank_{chunk_items[0]['target_rank']:04d}_{chunk_items[-1]['target_rank']:04d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                items_json = chunk_dir/"items.json"
                items_json.write_text(json.dumps(chunk_items, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                chunk_started = time.perf_counter()
                print(
                    f"\n[BATCH {chunk_index}] "
                    f"official rank range = {chunk_items[0]['target_rank']}..{chunk_items[-1]['target_rank']} "
                    f"candidate count={len(chunk_items)}",
                    flush=True,
                )
                build_report = run("batch build scratch candidate cases", [
                    network_py, SCRIPTS/"batch_build_candidate_cases.py",
                    "--project-root", root,
                    "--prediction", prediction,
                    "--network-input", dgn_root/"network_input.npz",
                    "--capture-root", capture_root,
                    "--settled-manifest", settled,
                    "--sim-target-segmentation-id", str(sim_target_id),
                    "--items-json", items_json,
                    "--output", chunk_dir/"batch_build_report.json",
                ], cwd=root, capture_json=True)
                retarget_report = run("batch LEAP->Wuji2 retarget 01/02/03", [
                    retarget_py, SCRIPTS/"batch_retarget_cases.py",
                    "--items-json", items_json,
                    "--output", chunk_dir/"batch_retarget_report.json",
                ], cwd=root, capture_json=True)
                finalize_report = run("batch final Wuji2 waypoints + flange targets", [
                    network_py, SCRIPTS/"batch_finalize_candidate_cases.py",
                    "--items-json", items_json,
                    "--output", chunk_dir/"batch_finalize_report.json",
                ], cwd=root, capture_json=True)
                retargeted_count += len(chunk_items)
                retarget_wall_s = time.perf_counter() - chunk_started
                print(
                    f"[BATCH RETARGET] input={len(chunk_items)} "
                    f"success={len(chunk_items)} "
                    f"time_s={retarget_wall_s:.3f} "
                    f"candidate_per_s={len(chunk_items) / max(retarget_wall_s, 1.0e-9):.3f}",
                    flush=True,
                )
                print(
                    f"[BATCH FINALIZE] input={len(chunk_items)} "
                    f"success={len(chunk_items)} "
                    f"time_s={float(finalize_report.get('wall_time_s', 0.0)):.3f}",
                    flush=True,
                )

                contracts = [
                    load_case_pick_contract(Path(item["case_root"]), measured, T_base_from_world)
                    for item in chunk_items
                ]
                targets = np.concatenate([contract["targets_base"] for contract in contracts], axis=0)
                states = [state for contract in contracts for state in contract["states"]]
                phases = [phase for contract in contracts for phase in contract["phases"]]
                collision_context = None if args.no_planner_collision_check else {
                    "phases": phases,
                    "joint_positions_by_name": measured,
                    "joint_positions_by_target": states,
                    "T_world_base": T_world_base,
                    "margin_m": 0.0,
                    "include_return_to_reference": False,
                }
                exact_started = time.perf_counter()
                exact = client.solve_ik_groups(
                    targets,
                    q_current,
                    group_sizes=[len(PICK_STAGES)] * len(chunk_items),
                    select_chain=True,
                    collision_context=collision_context,
                )
                exact_wall_s = time.perf_counter() - exact_started
                pass_rows = []
                print(
                    f"[GROUPED EXACT IK] candidate groups={len(chunk_items)} "
                    f"total poses={len(targets)} gpu solve time={float(exact.get('solve_time_s', 0.0)):.3f}s "
                    f"wall={exact_wall_s:.3f}s",
                    flush=True,
                )
                for item, group in zip(chunk_items, exact["groups"]):
                    if args.no_planner_collision_check:
                        feasible = bool(group.get("ik_pass") is True and group.get("selected") is not None)
                        unknown_policy_pass = True
                    else:
                        feasible, unknown_policy_pass = group_feasible(
                            group, block_unknown=bool(cfg.get("block_unknown_space"))
                        )
                    row = {
                        "target_rank": item["target_rank"],
                        "candidate_index": item["candidate_index"],
                        "official_score": item["official_score"],
                        "case_root": item["case_root"],
                        "pick": summarize_failure({
                            "status": "PASS" if feasible else "FAIL",
                            "unknown_policy_pass": unknown_policy_pass,
                            "group": group,
                        }),
                        "batch": {
                            "chunk_index": chunk_index,
                            "build_wall_s": build_report.get("wall_time_s"),
                            "retarget_wall_s": retarget_report.get("wall_time_s"),
                            "finalize_wall_s": finalize_report.get("wall_time_s"),
                            "exact_gpu_solve_time_s": exact.get("solve_time_s"),
                            "exact_wall_s": exact_wall_s,
                        },
                    }
                    if feasible:
                        pass_rows.append(row)
                    else:
                        tested.append(row)
                print(
                    "[GROUPED EXACT IK] exact 5-stage pass="
                    f"{len(pass_rows)} pass candidates="
                    f"{[(r['target_rank'], r['candidate_index']) for r in pass_rows[:8]]}",
                    flush=True,
                )
                if not pass_rows:
                    continue
                exact_pick_feasible_count += len(pass_rows)
                for row in pass_rows:
                    print("[7] Full Route IK", flush=True)
                    print(
                        f"[FULL ROUTE TRY] rank={row['target_rank']} "
                        f"candidate={row['candidate_index']}",
                        flush=True,
                    )
                    run("placement allocation + full Cartesian route", [
                        planner_py,
                        SCRIPTS/"build_cartesian_route.py",
                        "--case-root", row["case_root"],
                        "--placement-policy", session_placement_policy,
                    ], cwd=root, capture_json=True)
                    full_route_candidates_tested += 1
                    full_result = solve_full_route_gate(
                        client=client,
                        case_root=Path(row["case_root"]),
                        measured=measured,
                        q_current=q_current,
                        T_world_base=T_world_base,
                        T_base_from_world=T_base_from_world,
                        block_unknown=bool(cfg.get("block_unknown_space")),
                        no_planner_collision_check=bool(args.no_planner_collision_check),
                    )
                    row["full_route"] = summarize_failure(full_result)
                    row["final_route_stages"] = full_result["route_stages"]
                    row["home_planned"] = full_result["home_planned"]
                    tested.append(row)
                    print(
                        f"[FULL ROUTE TRY] rank={row['target_rank']} "
                        f"candidate={row['candidate_index']} status={full_result['status']}",
                        flush=True,
                    )
                    if full_result["feasible"]:
                        selected = {
                            **row,
                            "full_route_status": "PASS",
                        }
                        print(
                            f"[SELECTED] rank={selected['target_rank']} "
                            f"candidate={selected['candidate_index']} "
                            f"score={selected['official_score']}",
                            flush=True,
                        )
                        break
                if selected is not None:
                    break

        mem_after = nvidia_memory_mib()
        total_wall_s = time.perf_counter() - wall_started
        final_route_stages = [] if selected is None else selected["final_route_stages"]
        result = {
            "schema_version": 1,
            "status": "PASS" if selected is not None else "FAIL",
            "one_command_completed": selected is not None,
            "planning_only": not bool(args.sim_execute),
            "execution_enabled": bool(args.sim_execute),
            "sim_execute": bool(args.sim_execute),
            "no_planner_collision_check": bool(args.no_planner_collision_check),
            "diagnostic_ignore_static_gate": bool(args.diagnostic_ignore_static_gate),
            "self_collision_policy": "REPORT_ONLY_UNRESOLVED",
            "SELF_COLLISION_POLICY": "REPORT_ONLY_UNRESOLVED",
            "selected": selected,
            "selected_candidate": None if selected is None else selected["candidate_index"],
            "selected_target_rank": None if selected is None else selected["target_rank"],
            "selected_official_score": None if selected is None else selected["official_score"],
            "total_proposals": total_proposals,
            "target_proposals": len(candidates),
            "coarse_survivors": len(survivor_indices),
            "coarse_survivor_source": survivor_source,
            "retargeted_candidate_count": retargeted_count,
            "exact_pick_feasible_count": exact_pick_feasible_count,
            "full_route_candidates_tested": full_route_candidates_tested,
            "final_route_stages": final_route_stages,
            "HOME_planned": bool(selected is not None and selected.get("home_planned")),
            "worker_start_count": worker_start_count,
            "map_build_count": map_build_count,
            "placement_registry": str(session_root / "placement_registry.json"),
            "map_wall_s": map_wall_s,
            "prefilter_wall_s": prefilter_wall_s,
            "total_wall_time_s": total_wall_s,
            "peak_vram_mib": max(x for x in (mem_before, mem_after) if x is not None) if (mem_before is not None or mem_after is not None) else None,
            "map": map_report,
            "strict_coarse_prefilter": ordered_prefilter,
            "grasp_prefilter_summary": ordered_prefilter["grasp"],
            "pregrasp_prefilter_summary": ordered_prefilter.get("pregrasp"),
            "approach_prefilter_summary": ordered_prefilter.get("approach_path"),
            "tested_candidates": tested,
            "remaining_blockers": [] if selected is not None else [
                "No candidate passed hard IK / threshold / observed-scene ESDF / continuous path gates."
            ],
        }
        planning_result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2)+"\n",
            encoding="utf-8",
        )

        print("\n[ONE-COMMAND PLANNING RESULT]")
        print(json.dumps({
            "status": result["status"],
            "selected_candidate": result["selected_candidate"],
            "selected_target_rank": result["selected_target_rank"],
            "selected_official_score": result["selected_official_score"],
            "worker_start_count": worker_start_count,
            "map_build_count": map_build_count,
            "planning_result": str(planning_result_path),
        }, ensure_ascii=False, indent=2), flush=True)

        if args.planning_only or not args.sim_execute:
            return 0
        if selected is None:
            print("[NO SIM EXEC] no selected candidate")
            return 1

        output_dir=cycle_root/"execution"
        runtime_config=write_runtime_config(root,cfg,capture_root,target_slug,output_dir)
        runtime_launcher = resolve(root,cfg["runtime_launcher"])
        sim_cmd = [
            "bash",
            runtime_launcher,
            "--case-root",selected["case_root"],
            "--config",runtime_config,
        ]
        if args.no_planner_collision_check:
            sim_cmd.append("--no-planner-collision-check")
        if args.diagnostic_ignore_static_gate:
            sim_cmd.append("--diagnostic-ignore-static-gate")
        report_path = output_dir/"report.json"
        print("[8] Isaac Sim", flush=True)
        sim_wall_started = time.perf_counter()
        report=run_runtime_until_report(
            "Isaac Lab + Isaac Sim diagnostic pick/place/home",
            sim_cmd,
            cwd=root,
            report_path=report_path,
            exit_grace_s=float(cfg.get("runtime_exit_grace_s", 20.0)),
        )
        replay=Path(report["physical_replay_30fps"])
        next_manifest=cycle_root/"next_scene_manifest.json"
        if report.get("status")!="PASS":
            print(json.dumps({
                "cycle": cycle,
                "execution_status": report.get("status"),
                "report": str(report_path),
                "home_reached": "RETURN_HOME" in Path(report.get("trace_csv","")).read_text(encoding="utf-8", errors="ignore") if report.get("trace_csv") else None,
            }, ensure_ascii=False, indent=2), flush=True)
            raise RuntimeError(f"Diagnostic simulation cycle failed; loop stops: {report_path}")
        print(
            f"    PASS | simulation={float(report.get('action_duration_s', 0.0)):.2f} s "
            f"| wall={float(report.get('action_wall_duration_s', time.perf_counter()-sim_wall_started)):.2f} s "
            f"| lift={float(report.get('max_object_lift_mm', 0.0)):.1f} mm",
            flush=True,
        )
        run("persist final object poses for next RGB-D cycle",[
            network_py,SCRIPTS/"build_next_scene_manifest.py",
            "--settled-manifest",settled,
            "--physical-replay",replay,
            "--output",next_manifest,
        ],cwd=root,capture_json=True)
        current_scene_manifest=next_manifest
        print(
            f"\n================ RESULT =================\n"
            f"SUCCESS\n"
            f"rank={selected['target_rank']}\n"
            f"candidate={selected['candidate_index']}\n"
            f"total_cycle_wall={time.perf_counter()-wall_started:.1f} s\n"
            f"next_manifest={next_manifest}\n"
            f"placement_registry={session_root / 'placement_registry.json'}\n"
            f"debug_log={DEBUG_LOG}\n"
            f"=========================================\n",
            flush=True,
        )
        continue

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] interrupted")
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERROR stage=closed_loop\nreason={type(exc).__name__}: {exc}\n"
            f"debug_log={DEBUG_LOG}",
            file=sys.stderr,
        )
        raise
