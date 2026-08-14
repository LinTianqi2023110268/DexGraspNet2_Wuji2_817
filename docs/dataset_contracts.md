# Dataset Contracts

## 原位数据根

所有正式数据保持在 `02_training_dataset/data`，本轮整理不移动、不复制、不重新生成。

| 数据根 | 规模 | q20 字段 | 角色 |
|---|---|---|---|
| `scene_datasets/wuji2_train60_100seminal_256view_v1` | 100 场景 × 256 视角 | `pre_force_joint_positions_rad` | q_opt 训练基线 |
| `scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1` | 同几何/视角 | `joint_positions_rad` | 当前默认模型的 force-adjusted 监督 |
| `scene_datasets/wuji2_test60_10upright_10view_v1` | 10 场景 × 10 视角 | `pre_force_joint_positions_rad` | 独立 q_opt 测试集 |

`single_object_dataset` 是 Wuji2 1.0 单物体优化与六方向物理验证来源。`assets/wuji2_factory` 保存对象网格和工厂资产。

## 场景数据阶段

1. 单物体抓取变换到场景。
2. 场景/桌面碰撞及 paper keep mask。
3. Wuji2 增强掌心路径及 safe keep mask。
4. 完整表面参考点与 graspness。
5. 单视角可见点标签与训练索引。

每个训练样本使用一个单视角 40000 点。完整表面参考点用于标签投影，不能代替网络输入。

## 训练/评估保护

- `training_joint_field` 必须跟随数据集配置进入 run_config 和 checkpoint provenance。
- q_opt 与 force-adjusted 不得在同一训练/损失统计中混用。
- 当前独立测试集有 100 个相机输入，其中 99 个有有效可见抓取标签。
- 测试集可用于当前 force-adjusted 模型的候选生成和物理验证；q20 joint loss 不能直接宣称同口径精度。
- 数据生成完成后优先运行只读检查，不因整理重跑 100 × 256 相机或 Stage03/04。

## 默认模型

`04_training/experiments/wuji2_dexgraspnet2_train60_100seminal_256view_force_adjusted_legacy_v1_scratch/checkpoints/wuji2_scratch_050000.pth`

50000 步模型从头训练；官方 LEAP checkpoint 未加载。旧 40 场景 checkpoint 只作历史回归。

