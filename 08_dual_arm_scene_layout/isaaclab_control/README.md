# Isaac Lab dual-arm control

这里是 Route C 的长期控制目录。根目录只保留两个正式入口；实验、诊断和被否决功能均已分层。

## 正式入口

完整抓放（dog candidate3800，24.85 s 仿真动作）：

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
./08_dual_arm_scene_layout/isaaclab_control/launch_full_pick_place_25s_visible.sh
```

稳定实时回放（747帧，不运行物理、IK或网络）：

```bash
./08_dual_arm_scene_layout/isaaclab_control/launch_replay_25s_visible.sh
```

正式证据：`outputs/full_pick_place_25s_dog_candidate3800/`。物理动作时长 24.85 s；本机计算墙钟时间约 273 s，两者不可混写。

## 冻结控制合同

- Stage：`../scenes/manual_layout_calibrated_mass_fixed.usda`；
- 35-DOF dual-arm + official Wuji2 articulation；
- 右臂：Implicit Force Drive，ft04 NF `[50,40,45,40,55,50,40]`，阻尼比 1；
- Wuji2：官方 drive，不覆盖；
- 右法兰与 Wuji2 固连，由右臂 7 关节 IK 实现 wrist 6D；
- 不使用 `set_world_pose()` 瞬移手根。

## 目录

- `runtime/`：正式运行脚本、配置和内部 launcher；
- `report_demo/`：缓存感知结果的交互汇报演示；
- `diagnostics/`：仍有诊断价值的静置、质量树和短运动检查；
- `history/`：旧参数扫描、失败路线和阶段验证，只供追溯；
- `rejected/`：明确否决的逐帧 PNG/MP4 路线；
- `tools/`：RGB-D、候选筛选、IK和碰撞审计等离线工具；
- `outputs/`：正式结果和历史证据，索引见 `outputs/README.md`。

`report_demo/launch_interactive_demo.sh` 使用缓存 RGB-D、GroundedSAM 和 DGN2 结果；它不是每次现场重跑感知网络。

任何新控制任务都必须保持：唯一 AppLauncher、35 DOF 自检、右臂 J1–J7 映射自检、固定基座、全关节初始状态/目标同步、失败闭锁。不要在已打开的 Script Editor 中启动 Isaac Lab，也不要启动第二个 Kit。
