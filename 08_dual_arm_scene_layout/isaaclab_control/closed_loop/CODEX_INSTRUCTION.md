# Codex final audit only — V2 flexible persistent closed loop

**这是最终串联检查任务，不是重新设计任务。**

用户已经决定 V2 架构。不要把代码改回旧版固定 5/9 关键点硬筛，不要恢复每轮重启 Isaac，不要重新增加执行前第二遍 IK/FK。

## 固定设计，不得擅自改动

- Isaac Sim / Isaac Lab：一个持续进程贯穿整个会话。
- cuRobo：继续使用现有 `curobo_v2` 持续 Worker。
- GroundingDINO+SAM、DGN2、retarget 环境继续隔离。
- 旧 DGN2 前置粗 GRASP / 10 cm PREGRASP IK 默认关闭，仅保留配置开关。
- COVER 是严格抓取根位姿；GRASP/SQUEEZE 机械臂复用 COVER q7。
- PREGRASP/LIFT/TRANSFER/PLACE/RETREAT 通过大量 6D 合法目标 + GPU IK 搜索。
- IK 数值接受精度仍使用项目现有 5 mm / 5° / 关节余量合同；不要用放大 IK 误差代替任务空间采样。
- PLACE 使用名义 0.12 m 物体尺寸和绿色区域，默认中心间距 0.14 m；不恢复 `free[0]` 唯一放置点硬筛。
- 执行使用规划阶段输出的 q7；同一 Isaac 场景内不做第二遍 cuRobo IK，不做 FK 预检查。
- HOME 后下一次 CAPTURE 统一静置 1.0 s。
- `抓取完成` 必须结束整个交互会话。
- 真实机器人输出保持禁用；本阶段只验证 Isaac 物理执行。

## 你只需要检查

1. import / 文件路径 / conda 环境是否正确；
2. NPZ / JSON 字段名是否在生产者和消费者之间一致；
3. `arm_r_joint_1..7` 和 Wuji2 20 关节顺序没有变化；
4. `T_world_flange = T_world_SourceZone @ T_SourceZone_r_wrist @ inverse(T_flange_r_wrist)` 坐标合同没有被破坏；
5. persistent Isaac Worker 的 stdin/stdout JSON 协议能启动、ping、capture、execute、snapshot、shutdown；
6. CAPTURE 写出的 RGB-D、`settled_scene_manifest.json`、`robot_state.json` 对应同一暂停瞬间；
7. 规划期间 `SimulationContext` 保持 pause，且没有 `SimulationContext.step()` 导致场景漂移；
8. `flexible_route_plan.npz` 的 9 个 stage、9×7 `arm_q_rad` 能被 persistent Isaac Worker 直接执行；
9. COVER/GRASP/SQUEEZE q7 相同，PLACE/RELEASE q7 相同；
10. SQUEEZE 仍使用原有 `squeeze_dense_q20_path`；
11. 下一轮不会重新加载 USD/机械臂/物体；只 HOME 后静置 1.0 s 再拍照；
12. 默认终端为摘要输出，`--verbose` 才打印详细日志。

## 先运行只读检查

```bash
cd ~/Projects/DexGraspNet2_Wuji2
python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/check_v2_wiring.py \
  --project-root ~/Projects/DexGraspNet2_Wuji2

PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/closed_loop \
python -m unittest -v \
  08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_flexible_planning.py
```

然后只做最小启动检查。若发现明确 API/字段/路径错误，只修复该错误并说明文件和行；**不要扩大改动范围**。

最终只汇报：

```text
V2_FINAL_WIRING_AUDIT
- static checks
- imports/paths
- config fields
- persistent Isaac protocol
- cuRobo protocol
- NPZ/JSON contracts
- first real launch blocker (if any)
- exact files changed by Codex (ideally none)
```
