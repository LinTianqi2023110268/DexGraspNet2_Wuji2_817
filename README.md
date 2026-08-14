# DexGraspNet2-Wuji2

本项目把 DexGraspNet 2.0、Wuji2 Hand 2、Isaac Sim 5.0 / Isaac Lab 2.2 和双臂机械臂连接成一条可审计的抓取链。最终目标是：单张 RGB-D 场景图经文本分割和抓取预测后，由右机械臂上的 Wuji2 手抓起指定物体并放到绿色区域。

当前已验证的是这条目标链的关键子集：正式数据集与 20 关节训练、官方 LEAP 到 Wuji2 迁移、原生 Wuji2 独立手抓取，以及使用缓存视觉/网络结果的双臂完整抓放。系统尚未做到每轮都现场重新运行 GroundingDINO、SAM 和 DexGraspNet 2.0。

## 三条严格分离的路线

| 路线 | 目的 | 唯一正式入口 | 当前代表案例 |
|---|---|---|---|
| A：原生 Wuji2 网络 | Wuji2 数据、20 关节训练、原生 q20 推理与 3P+3R 独立手验证 | `python 07_wuji2_network_3p3r_sim/02_scripts/04_build_selected_sim_3p3r.py` | Ashtray `scene0000/view0001/source462`，抬升 60.69 mm，PASS |
| B：官方 LEAP → Wuji2 | 官方 LEAP q16 经 wuji-retargeting 变成 Wuji2 q20，并单独求手根 6D | `/usr/bin/python3 06_leap_to_wuji2_final_pipeline/run_new_test_case.py --rebuild` | Battery candidate49，`VERIFIED_PHYSICAL` + `PRE_EXISTING_INTEGRITY_EXCEPTION` |
| C：双臂完整系统 | 右臂 IK、抓取、搬运、放置、松手和回程 | `./08_dual_arm_scene_layout/isaaclab_control/launch_full_pick_place_25s_visible.sh` | dog candidate3800，完整动作 24.85 s，PASS |

路线 A 与 B 的 PREGRASP、手根和 SQUEEZE 合同不同，禁止混用。路线 C 可以消费 A 或 B 的结果；当前正式完整抓放案例来自路线 B。

## 项目导航

| 目录 | 状态 | 职责 |
|---|---|---|
| `01_environment/` | ACTIVE / UPSTREAM | 环境说明、官方 Wuji2/双臂资产、wuji-retargeting |
| `02_training_dataset/` | VERIFIED | 约 312 GB 正式数据，原位保护，禁止移动或复制 |
| `03_prediction_network/` | UPSTREAM | 官方 DexGraspNet2 核心与 LEAP checkpoint |
| `04_training/` | VERIFIED | Wuji2 数据加载、50000 步训练结果与 checkpoint |
| `05_inference/` | ACTIVE | 单视角推理、排序、碰撞过滤和任务生成 |
| `06_leap_to_wuji2_final_pipeline/` | ACTIVE / VERIFIED | 路线 B |
| `07_wuji2_network_3p3r_sim/` | ACTIVE / VERIFIED | 路线 A |
| `08_dual_arm_scene_layout/` | ACTIVE / EXPERIMENTAL | 路线 C、相机接口、控制诊断和完整抓放 |
| `verified/` | VERIFIED INDEX | 三个正式代表基线的统一索引，不复制大结果 |
| `experiments/` | EXPERIMENT INDEX | 诊断、失败历史与待归档项目索引 |
| `archive/` | ARCHIVED METADATA | 整理前快照、迁移清单和回滚材料 |
| `src/wuji2_dgn2/` | ACTIVE | 公共坐标、资产、碰撞和可视化实现 |
| `trimesh/` | ACTIVE | 几何可视化源码；输出与源码分开 |
| `docs/` | ACTIVE | 架构、数据、执行与坐标合同 |

状态总表见 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)，完整结构见 [`docs/architecture.md`](docs/architecture.md)，本轮终态整理证据见 [`REORG_PRACTICAL_FINAL_REPORT.md`](REORG_PRACTICAL_FINAL_REPORT.md)。

## 正式数据与默认模型

三套正式场景数据均在 `02_training_dataset/data/scene_datasets/` 原位保存：

- `wuji2_train60_100seminal_256view_v1`：q_opt，字段 `pre_force_joint_positions_rad`；
- `wuji2_train60_100seminal_256view_force_adjusted_legacy_v1`：力调整后监督，字段 `joint_positions_rad`；
- `wuji2_test60_10upright_10view_v1`：独立 q_opt 测试集，10 场景 × 10 视角。

默认 Wuji2 模型是：

```text
04_training/experiments/
wuji2_dexgraspnet2_train60_100seminal_256view_force_adjusted_legacy_v1_scratch/
checkpoints/wuji2_scratch_050000.pth
```

它由力调整后标签从头训练 50000 步。测试集 q20 是 q_opt 口径，不能直接和该模型混算同口径 joint loss。详细规则见 [`docs/dataset_contracts.md`](docs/dataset_contracts.md)。

## 三个正式物理基线

统一索引位于 [`verified/INDEX.md`](verified/INDEX.md) 和 `verified/index.json`。

1. 路线 A：`07_wuji2_network_3p3r_sim/04_verified_baseline/scene0000_view0001_ashtray_source0462/`。
2. 路线 B：`06_leap_to_wuji2_final_pipeline/04_verified_baseline/scene0001_view0001_official_rank0/`；物理结果 VERIFIED，但旧 SHA 仅 8/10，属于整理前完整性例外。禁止重写旧 checksum，详见 `verified/B_verified_leap_to_wuji2/`。
3. 路线 C：`08_dual_arm_scene_layout/isaaclab_control/outputs/full_pick_place_25s_dog_candidate3800/`；24.85 s 是仿真动作时间，当前机器实际计算约 273 s。

ft04 静置、TGS 速度审计和 20 mm IK PASS 是控制诊断基线，不是第四个业务抓取案例。

## 双臂唯一正式入口与稳定回放

真实物理完整抓放：

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
./08_dual_arm_scene_layout/isaaclab_control/launch_full_pick_place_25s_visible.sh
```

稳定实时回放同一条已记录物理轨迹：

```bash
./08_dual_arm_scene_layout/isaaclab_control/launch_replay_25s_visible.sh
```

回放读取 747 帧 `physical_replay_30fps.npz`，不重新运行物理、IK 或网络。逐帧 PNG / MP4 自动导出已被否决并归档到 `isaaclab_control/rejected/mp4_export/`，不是正式功能。

## 环境

- `graspnet2.0`：Python 3.8.20、PyTorch 2.0.1+cu117；数据、训练、推理和 Trimesh。
- `wuji2_factory`：Isaac Sim 5.0 / Isaac Lab 2.2；只运行仿真和机械臂控制。
- `01_environment/conda/wuji_retargeting`：Python 3.11；路线 B 的官方重定向。

不要合并这些环境，不要自动升级 NVIDIA 驱动、CUDA、Isaac Sim 或 Isaac Lab。Isaac Lab 由 `AppLauncher` 启动唯一 Kit 进程，不能同时打开第二个 Isaac Sim。

只读验收：

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
/home/lin/miniconda3/envs/graspnet2.0/bin/python tools_validate.py
```

## GroundedSAM 外部依赖

GroundingDINO + SAM 的代码和权重当前在项目外：

```text
/home/lin/Projects/分类抓取开源项/03_检测加分割_GroundedSAM
```

本项目只保存接口合同和已生成缓存。它不是自包含依赖，也不能把缓存演示写成每次实时重新推理。机器可读登记见 `config/external_dependencies.json`。

## 当前下一项开发任务

在不改变三个物理成功基线的前提下，把路线 C 的缓存视觉/网络入口替换成真正的一次性循环：现场 RGB-D → 文本目标 → GroundingDINO + SAM → 完整场景 40000 点 → DGN2 1024 候选 → 目标候选筛选 → 路线 B 或 A → 右臂抓放。自动 MP4 不在当前任务范围。

## 禁止事项

- 不移动、复制或重建 `02_training_dataset` 的正式大数据；
- 不修改官方 Wuji2 USD/URDF、dual-arm 供应商资产或官方 DexGraspNet2 upstream；
- 不混用 q_opt 与 force-adjusted 标签；
- 不把几何过滤 PASS 当作物理抓取 PASS；
- 不把缓存回放当作实时推理；
- 未经用户确认不删除 `archive_candidate` 或 `delete_candidates.tsv` 中的项目。
