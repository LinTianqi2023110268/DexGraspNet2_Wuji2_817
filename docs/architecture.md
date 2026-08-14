# Architecture

## 总体数据流

```text
RGB-D + camera calibration
  -> full single-view scene cloud (40,000 points)
  -> text target mask (GroundingDINO + SAM)
  -> DexGraspNet 2.0 proposals (1,024)
  -> target membership + score + collision/path filters
  -> Route A native Wuji2 q20 OR Route B LEAP q16 -> Wuji2 q20/root
  -> right-arm flange targets and path audit
  -> Isaac Lab pick, transfer, place, release, return
```

## Route A：原生 Wuji2

`02_training_dataset` 产生场景、单视角点云和标签；`04_training` 训练 20 关节模型；`05_inference` 预测与过滤；`07_wuji2_network_3p3r_sim` 生成原生动作和独立手物理验证。

正式动作合同由 `07` 管理，不消费 LEAP SQUEEZE。

## Route B：官方 LEAP → Wuji2

`03_prediction_network/official_core` 输出 LEAP；`06_leap_to_wuji2_final_pipeline` 用官方 wuji-retargeting 求 q20，再用四指尖 Kabsch 求 Wuji2 手根 6D，并迁移 LEAP SQUEEZE。

`06/04_verified_baseline` 是不可修改的成功基线区；`06/01_cases/active` 最多一个可再生成工作案例，其余进入 `99_archive`。Route-C dog 来源由兼容链接保持旧读取合同。

## Route C：双臂完整系统

`08_dual_arm_scene_layout` 保存标定桌面、蓝/绿区域、顶部 RGB-D、dual-arm + official Wuji2 组合资产接口和 Isaac Lab 控制。控制目录根部只有正式抓放和正式回放两个入口；实现、诊断、历史、拒绝路线分别位于 `runtime/`、`diagnostics/`、`history/`、`rejected/`。

右手与右法兰固连。抓取给出 `T_world_wuji_wrist` 后，机械臂目标为：

```text
T_world_flange = T_world_wuji_wrist @ inverse(T_flange_wuji_wrist)
```

独立手验证使用 3P+3R 虚拟根；装到机械臂后由右臂 7 关节 IK 实现手腕 6D，不再瞬移手根。

## 权威边界

- upstream/vendor：官方 Wuji2、dual-arm、wuji-retargeting、官方 DexGraspNet2，禁止无差别修改。
- core：`src/wuji2_dgn2` 与配置合同。
- datasets：`02_training_dataset/data` 原位保护。
- models：`03` 官方 LEAP checkpoint 与 `04` Wuji2 checkpoint。
- cases：可生成工作材料；status 必须读 JSON。
- verified：只指向三条路线各一个正式业务案例。
- experiments：诊断和失败历史，不和运行入口并列。
- archive：迁移快照与待处理清单；未经确认不删除。

## 环境边界

网络使用 `graspnet2.0`；Isaac Sim/Lab 使用 `wuji2_factory`；retarget 使用项目内 Python 3.11 环境。Isaac Lab `AppLauncher` 必须拥有唯一 Kit 进程。
