# Trimesh三维检查

本目录只保存当前工程仍可运行的三维可视化入口。Trimesh用于检查坐标、点云、
姿态和碰撞关系，不模拟重力、摩擦、接触力或关节驱动。

## 当前脚本

| 脚本 | 用途 |
|---|---|
| `view_force_adjusted_training_ab.py` | 比较q_opt与力调整后20关节训练标签 |
| `view_predictions_trimesh.py` | 显示网络推理候选、种子点、热力图和手姿态 |
| `build_completed_scene_dataset_ppt.py` | 生成数据集统计图片及PPT材料 |

## 查看原生Wuji2网络候选

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
conda activate graspnet2.0

python 07_wuji2_network_3p3r_sim/02_scripts/03_view_selected_poses.py \
  --scene 0 --mode pose --object-id 14 --pose-rank 0 --show
```

`--mode overview`显示场景全部候选，`--mode object`比较指定物体两条位姿。
LEAP迁移路线的四阶段对比图由08目录内的可视化入口生成。

## 其他监控窗口

这些程序不属于Trimesh，仍放在所属阶段：

- 场景筛选：`02_training_dataset/code/live_scene_filter_monitor.py`；
- Stage03/04标签：`02_training_dataset/code/live_label_generation_monitor.py`；
- 训练损失：`04_training/scripts/monitor_wuji2_training_loss.py`。

`03_prediction_network/official_core`中的上游可视化脚本属于官方源码快照，不迁移。
