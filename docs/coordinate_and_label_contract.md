# 坐标与标签合同

## 单视角点云

- 网络输入：`point_clouds.shape = (40000, 3)`。
- 相机坐标使用 OpenCV：`+X` 向图像右、`+Y` 向图像下、`+Z` 向相机前方。
- 外参满足 `p_world = R_world_camera @ p_camera + t_world_camera`。
- 分割数组必须和点云使用同一个采样/打乱 permutation，保持点与成员标签一一对应。

GroundedSAM 只给目标成员掩码。网络仍接收完整场景 40000 点，不能把孤立目标点云当成正式输入。

## Wuji2 根坐标

- 历史训练标签根：`r_base_link`。
- 旧优化器根：`r_wrist`。
- 语义掌心：自定义物理掌心中心与朝向，不是手根。
- 官方 Isaac Sim 运行根：官方 Wuji Hand 2 USD 的 `r_wrist`。

旧标签根到官方 USD 根的固定变换由 `config/wuji2_official_asset.json` 和 `src/wuji2_dgn2/official_asset.py` 管理。禁止在脚本里另写第二套常量。

## 关节监督合同

网络 q20 顺序只由 `config/wuji2_joint_order.json` 定义。训练关节语义由数据集 manifest/config 中的 `training_joint_field` 决定：

| 数据集 | 字段 | 语义 |
|---|---|---|
| `wuji2_train60_100seminal_256view_v1` | `pre_force_joint_positions_rad` | q_opt：能量优化后、力调整前 |
| `wuji2_train60_100seminal_256view_force_adjusted_legacy_v1` | `joint_positions_rad` | Wuji2 1.0 力调整后的命令目标 |
| `wuji2_test60_10upright_10view_v1` | `pre_force_joint_positions_rad` | 独立 q_opt 测试标签 |

两种监督都合法，但禁止混用。当前默认 50000 步模型学习 force-adjusted q20；独立测试集 q20 是 q_opt，所以只可做推理/物理测试，不能把 joint loss 当成同口径定量指标。

## 路线 A 与路线 B 的执行合同

路线 A（原生 Wuji2）：虎口后退/接近 100 mm；SQUEEZE 为五个指尖各自链接局部 `+Y` 30 mm，`keep_z=False`。

路线 B（官方 LEAP → Wuji2）：LEAP q16 经 21 点重定向得到 Wuji2 q20；Wuji2 手根 6D 由对应指尖 Kabsch 单独求；SQUEEZE 从官方 LEAP SQUEEZE 再迁移。它不是路线 A 的 local `+Y` SQUEEZE。

## 候选与结果

候选至少保存 rotation、translation、qpos、joint_order、seed、graspness、log_prob 和最终 score。默认排序 `score = log_prob + 5 * graspness`。

进入仿真的基本阶段为 PREGRASP、COVER、GRASP、SQUEEZE、LIFT；双臂完整路线还包含 TRANSFER、PLACE、RELEASE、RETREAT 和 RETURN。任务生成、几何过滤、IK PASS 都不能代替真实物理 PASS。

