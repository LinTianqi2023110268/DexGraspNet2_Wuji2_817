# Closed-loop V2: Persistent Isaac + Flexible GPU IK

本目录是 2026-08-17 基于 GitHub 快照
`7d84aa7b4d4bf8ed6194b65243810bccd48cdaa2` 的 V2 闭环改造。

## 核心原则

1. **Isaac Sim 只启动一次。** 机器人、相机、桌面和全部动态物体只加载一次；每轮抓取后保留真实物理场景。
2. **cuRobo 也只启动一个持续 Worker。** 不混装 Python/CUDA 环境。
3. **旧的 DGN2 前置粗 GRASP / 固定 10 cm PREGRASP IK 默认关闭。** 它们仍可在 JSON 中重新打开做消融实验。
4. **COVER 是唯一严格抓取根位姿。** GRASP 与 SQUEEZE 的机械臂 q7 直接复用 COVER；Wuji2 手指继续执行原有 GRASP / 41 点 SQUEEZE 轨迹。
5. **非接触阶段放宽“任务目标集合”，不降低 IK 数值精度。** PREGRASP、LIFT、TRANSFER、PLACE、RETREAT 各自在 6D 合法区域生成大量目标，交给 GPU 批量 IK。
6. **PLACE 不再使用第一个固定网格点作为硬门槛。** 绿色区域内全部合法中心都可参与搜索；默认未知物体按 120 mm × 120 mm × 120 mm 名义尺寸处理，已放置物体中心间距默认至少 140 mm。
7. **同一 Isaac 场景直接执行规划 q7。** 不再在 Isaac 执行前重新做 9 点 IK，也不做额外 FK 预检查。
8. **每轮回 HOME 后，只在下一次 CAPTURE 中统一静置 1.0 s。** 相机最后若干渲染帧包含在这 1.0 s 内，不额外增加隐藏静置。

## 默认动作链

```text
HOME
  -> FORM_PREGRASP（手先张开）
  -> PREGRASP（柔性 6D 目标）
  -> COVER（精确抓取根位姿）
  -> GRASP（臂保持 COVER，手闭合）
  -> SQUEEZE（臂保持 COVER，手走 41 点收紧轨迹）
  -> LIFT（柔性 6D 目标）
  -> TRANSFER（柔性走廊目标）
  -> PLACE（绿色区域柔性目标）
  -> RELEASE（臂复用 PLACE，手张开）
  -> RETREAT（柔性 6D 目标）
  -> HOME
  -> 下一轮 CAPTURE 内静置 1.0 s
```

## 用户入口

推荐仿真实验命令：

```bash
cd ~/Projects/DexGraspNet2_Wuji2
./run_closed_loop.sh --sim-execute --no-planner-collision-check
```

如需要显示 Isaac GUI，不加 `--isaac-headless`；需要看所有子进程日志时加 `--verbose`。

交互终止词包含：`q`、`quit`、`exit`、`结束`、`退出`、`完成`、`抓取完成`。

## 参数在哪里改

主要只改：

```text
08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/closed_loop.json
```

其中：
- `gpu_ik_seeds`：每个 6D 目标的 cuRobo IK 初始种子数；
- `gpu_ik_batch_size`：GPU 每批目标数；
- `flexible_ik.*.samples`：PREGRASP/LIFT/TRANSFER/RETREAT 采样数；
- `flexible_ik.place.samples_per_xy`：绿色区域每个 XY 中心生成多少个 6D 变体；
- `flexible_ik.selection.beam_width`：逐阶段保留多少条 q7 链；
- `coarse_ik_prefilter.*_enabled`：是否恢复旧粗筛。

Isaac 动作时间、HOME 后 1 s、ft04 参数在：

```text
08_dual_arm_scene_layout/isaaclab_control/runtime/config/persistent_closed_loop.json
```

## 重要：不要删掉原项目中未覆盖的依赖

V2 仍然复用原项目的：
- `closed_loop/scripts/grounded_sam_backend.py`
- `validate_grounded_sam_output.py`
- `resolve_sim_target.py`
- `batch_build_candidate_cases.py`
- `batch_retarget_cases.py`
- `batch_finalize_candidate_cases.py`
- `all_candidate_gpu_prefilter.py`（仅供可选旧粗筛）
- `08_build_target_network_input.py`
- `09_predict_official_leap_target.py`
- LEAP->Wuji2 01/02/03/05 脚本
- `03_build_arm_execution_targets.py`
- `core/bridge` 现有 cuRobo Worker

因此本补丁应当**合并覆盖**到原项目，不能先删除整个 `closed_loop/`、`core/` 或 `runtime/` 目录。

## 实验边界

该 V2 仍然只输出/执行 Isaac 仿真控制，不接真实机器人。`--no-planner-collision-check` 只关闭规划器 ESDF/路径否决；Isaac/PhysX 物理碰撞仍然开启。
