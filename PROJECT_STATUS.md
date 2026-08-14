# Project Status

更新日期：2026-08-14。状态以机器结果、冻结 manifest 和当前用户决定为准，不以目录名推断。

## ACTIVE

- 路线 A：Wuji2 20 关节数据、训练、推理和原生 3P+3R 仿真。
- 路线 B：官方 LEAP 输出经 wuji-retargeting 迁移到 Wuji2。
- 路线 C：双臂场景、相机接口、右臂 IK、完整抓放和稳定回放。
- 默认训练模型：force-adjusted 100 场景、50000 步 checkpoint。
- 下一开发目标：把路线 C 从缓存感知改成现场一次性 RGB-D / GroundedSAM / DGN2 循环。

## VERIFIED

### 业务抓取基线

| 路线 | 案例 | 物理结果 | 权威证据 |
|---|---|---|---|
| A | Ashtray scene0000/view0001/source462 | 目标抬升 60.69 mm，只有 seg14 达到抬升判据 | `07_wuji2_network_3p3r_sim/04_verified_baseline/scene0000_view0001_ashtray_source0462/05_isaacsim/final_result.json` |
| B | Battery candidate49 | `VERIFIED_PHYSICAL` + `PRE_EXISTING_INTEGRITY_EXCEPTION`；目标抬升 204.14 mm，PASS | `verified/B_verified_leap_to_wuji2/checksum_audit.md` |
| C | dog candidate3800 | 完整抓、搬、放、松手、回程 PASS；动作 24.85 s；现实计算约 273 s | `08_dual_arm_scene_layout/isaaclab_control/outputs/full_pick_place_25s_dog_candidate3800/report.json` |

17.78 s dog 结果只到 LIFT，是阶段验证，不再作为第二个完整业务案例。

### 控制诊断基线

- 质量修复：J6 下游质量约 7.341 kg → 1.341 kg；J6 重力补偿约 -14.27 → -1.35 N·m。
- ft04：Implicit Force Drive，J1..J7 natural frequency `[50,40,45,40,55,50,40]`，阻尼比 1。
- TGS 审计：raw PhysX qdot 非零但位置有限差分速度、关节 span、法兰和 wrist span 近零；raw qdot 不再单独作为静置失败判据。
- 20 mm IK：回程位置误差 2.58 mm、姿态误差 0.60°，PASS。

Battery 的物理结论仍然有效，但旧冻结目录不是 bitwise-clean：原 SHA 8/10，且漂移发生在本次整理前。旧 checksum 不重写；未来如需干净基线，必须新建目录重新生成并重新验证。

## EXPERIMENTAL

- 现场 GroundedSAM + DGN2 循环尚未接入完整抓放状态机。
- 06 的可再生成 case 已移入 archive；08 的正式、诊断、历史和拒绝路线已分层。
- `.agents/skills` 两份本地技能可作约束参考，但缺少它们声明的 `reference.md` 和 `evaluations.md`，不是完整可发布技能包。

## ARCHIVED / ARCHIVE_CANDIDATE

- 06 非代表 case 已进入 `99_archive/regenerable_cases/`，少量失败代表进入 `failed_cases/`。
- 08 的 grouped PD、acceleration/force 扫描、explicit PD、J6 扫描和旧 short-motion 已归入 `history/` 或 `diagnostics/`。
- 详细列表：`archive/migration_snapshots/pre_reorg_20260814/case_inventory.tsv`、`06_case_reduction_plan.tsv`、`08_reorg_plan.tsv`。

## REJECTED

- 逐帧写状态 + PNG 捕获 + MP4 自动导出。该路线改变了回放观感并产生抖动。
- `export_replay_mp4.sh` 及相关视频输出只作为待清理证据，不是正式入口。
- 原则：正确的 24.85 s 连续实时回放优先于自动视频。

## 标签状态

- q_opt：`training_joint_field=pre_force_joint_positions_rad`。
- force-adjusted：`training_joint_field=joint_positions_rad`。
- 当前默认完成模型是 force-adjusted；不能再写成“force-adjusted 不作网络监督”。

## 验收边界

- `tools_validate.py` 当前 PASS，但它主要覆盖项目结构和路线 06/07，不代表路线 08 的所有历史实验都有效。
- 几何碰撞过滤、IK 可达和物理抓取是三个不同门槛。
- cache replay 不运行网络、IK 或 PhysX；缓存汇报也不等于每次现场重跑感知。
