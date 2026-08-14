# Execution Contracts

## 通用门槛

```text
network proposal
-> target membership and score
-> scene/table/PREGRASP/path geometry
-> arm reachability and joint-path audit (route C)
-> real PhysX execution
```

前一层 PASS 不能代替后一层 PASS。只有机器可读物理结果明确成功，案例才可进入 verified。

## Route A

- 输入：原生 Wuji2 q20、官方 Wuji2 wrist 根、场景对象。
- PREGRASP：Wuji2 虎口反向 100 mm，预张开 37.5 mm。
- GRASP：网络 q20。
- SQUEEZE：五个指尖链接局部 `+Y` 30 mm，`keep_z=False`。
- LIFT：世界 `+Z` 70 mm。
- 手根：3P+3R Force position drive，K800/D20；禁止 `set_world_pose()` 瞬移。

## Route B

- 输入：官方 LEAP q16 和 LEAP root。
- q20：LEAP FK 21 点 → 官方 wuji-retargeting。
- root：对应指尖 Kabsch，有界对齐。
- SQUEEZE：从官方 LEAP SQUEEZE 再次迁移，不使用 Route A local-axis 规则。
- 独立手验证同样使用 3P+3R 根驱动。

## Route C

- right arm + Wuji2 是 35-DOF articulation；Wuji2 与右法兰固定连接。
- 当前机械臂控制基线：质量修复组合 USD、Implicit Force Drive ft04、阻尼比 1。
- IK 只控制右臂 J1–J7；Wuji2 20 关节服从官方 drive。
- AppLauncher、SimulationContext 和 Articulation 管理唯一 Isaac Sim 进程。
- 完整状态：PREGRASP → COVER → GRASP → SQUEEZE → LIFT → TRANSFER → PLACE → RELEASE → RETREAT → RETURN。

## 正式执行与回放

完整抓放入口重新运行物理，动作时间 24.85 s，但本机现实耗时约 273 s。

稳定回放只写回已记录的 747 帧状态，不运行网络、IK 或物理。它用于复看同一次成功结果，不证明新一轮抓取。

逐帧 PNG/MP4 捕获已被否决。不要通过隔帧、插值或滤波修改原轨迹来换取视频。

## 允许与禁止修改

允许：项目配置中的场景选择、候选、时间、可视化和明确标出的规划容差。

禁止：官方 Wuji2/dual-arm 质量、惯量、关节限制和官方 drive；NVIDIA/Isaac 版本；三条路线之间的手根和 SQUEEZE 合同。

