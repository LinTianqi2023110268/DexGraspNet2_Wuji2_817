#!/usr/bin/env python3
"""Read-only wiring audit for the flexible persistent closed-loop overlay.

Run after copying the overlay into the real project:

    python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/check_v2_wiring.py \
      --project-root ~/Projects/DexGraspNet2_Wuji2

No Isaac Sim, cuRobo, DGN2 or retarget environment is started.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import py_compile
import sys


def fail(message: str) -> None:
    print(f"✗ {message}")
    raise SystemExit(2)


def check_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"{label} 缺失：{path}")
    print(f"✓ {label}: {path.relative_to(PROJECT_ROOT)}")


def check_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"JSON读取失败 {path}: {exc}")


parser = argparse.ArgumentParser()
parser.add_argument("--project-root", type=Path, required=True)
args = parser.parse_args()
PROJECT_ROOT = args.project_root.expanduser().resolve()
if not PROJECT_ROOT.is_dir():
    fail(f"project root不存在：{PROJECT_ROOT}")

files = {
    "总入口": PROJECT_ROOT / "run_closed_loop.sh",
    "V2总控": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py",
    "Flexible采样": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/planning/flexible_pose_sampling.py",
    "Flexible路线": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/planning/flexible_route_search.py",
    "Isaac客户端": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/client.py",
    "Isaac持续Worker": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/worker.py",
    "Isaac启动器": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/launchers/run_persistent_isaac_worker.sh",
    "闭环配置": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/closed_loop.json",
    "Isaac配置": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/runtime/config/persistent_closed_loop.json",
    "批量case构建": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_build_candidate_cases.py",
    "批量重定向": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_retarget_cases.py",
    "批量finalize": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_finalize_candidate_cases.py",
    "仿真目标绑定": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/resolve_sim_target.py",
    "GroundedSAM后端": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/grounded_sam_backend.py",
    "GroundedSAM校验": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/validate_grounded_sam_output.py",
    "40k输入": PROJECT_ROOT / "08_dual_arm_scene_layout/scripts/08_build_target_network_input.py",
    "DGN2推理": PROJECT_ROOT / "08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py",
    "Wuji2 waypoint": PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline/02_scripts/05_build_isaacsim_validation.py",
    "机械臂flange目标": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/tools/03_build_arm_execution_targets.py",
    "场景标定": PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json",
    "cuRobo client": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/core/bridge/worker_client.py",
    "cuRobo worker": PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/core/bridge/curobo_worker.py",
}
for label, path in files.items():
    check_file(path, label)

cfg = check_json(files["闭环配置"])
isaac_cfg = check_json(files["Isaac配置"])
required_flexible = {"pregrasp", "lift", "transfer", "place", "retreat", "selection"}
missing = required_flexible - set(cfg.get("flexible_ik", {}))
if missing:
    fail(f"flexible_ik缺少字段：{sorted(missing)}")
if bool(cfg["coarse_ik_prefilter"].get("grasp_enabled")) or bool(cfg["coarse_ik_prefilter"].get("pregrasp_enabled")):
    print("⚠ 粗GRASP/PREGRASP IK当前不是默认关闭状态")
else:
    print("✓ 旧粗GRASP/PREGRASP IK默认关闭")
if float(isaac_cfg.get("post_home_hold_s", -1)) != 1.0:
    fail("post_home_hold_s应为1.0s")
print("✓ HOME后静置=1.0s")

for key in ("gpu_ik_seeds", "gpu_ik_batch_size"):
    if key not in cfg:
        fail(f"缺少可调参数 {key}")
print(f"✓ GPU IK参数可调：seeds={cfg['gpu_ik_seeds']} batch={cfg['gpu_ik_batch_size']}")

place = cfg["flexible_ik"]["place"]
if [float(x) for x in place.get("nominal_object_size_xyz_m", [])] != [0.12, 0.12, 0.12]:
    fail("PLACE名义物体尺寸应为0.12m立方估计")
if float(place.get("minimum_center_spacing_m", -1.0)) != 0.14:
    fail("PLACE中心最小间距应为0.14m")
print("✓ PLACE策略：名义120mm物体 + 中心间距140mm")

if "抓取完成" not in cfg.get("stop_words", []):
    fail("stop_words缺少‘抓取完成’")
print("✓ 交互结束词包含‘抓取完成’")

for path in [
    files["V2总控"], files["Flexible采样"], files["Flexible路线"],
    files["Isaac客户端"], files["Isaac持续Worker"],
]:
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        fail(f"Python语法检查失败 {path}: {exc}")
print("✓ V2 Python文件语法检查通过")

worker_text = files["Isaac持续Worker"].read_text(encoding="utf-8")
if "CuroboWorkerClient" in worker_text or ".solve_ik(" in worker_text:
    fail("persistent Isaac worker中不应出现二次cuRobo IK调用")
if "forward_kinematics" in worker_text or "rebase_pick_waypoints" in worker_text:
    fail("persistent Isaac worker中不应恢复FK/rebase预检查")
print("✓ Isaac执行端：无二次IK / 无FK-rebase预检查")

orch_text = files["V2总控"].read_text(encoding="utf-8")
if "run_capture_cycle.sh" in orch_text or "run_full_pick_place_closed_loop.sh" in orch_text:
    fail("V2总控仍引用旧one-shot Isaac launcher")
print("✓ V2总控未引用旧one-shot Capture/Execution launcher")

print("\n================ WIRING AUDIT PASS ================")
print("该检查只覆盖静态路径/配置/语法；Isaac/curobo/retarget运行仍需真实环境实验。")
