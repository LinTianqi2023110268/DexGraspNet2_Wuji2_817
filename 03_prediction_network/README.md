# 03 预测网络

`official_core`是官方DexGraspNet2网络源码快照。04训练和05推理都从这里导入同一套网络定义，避免训练/推理结构分叉。

## 输入

```text
point_clouds  (B,40000,3)
coors        稀疏体素坐标
feats        初始点/体素特征，当前为常数1
```

体素大小为5 mm。稀疏卷积使用`coors`决定哪些空间格子互为邻居，对这些位置携带的`feats`进行卷积，最终把局部几何上下文编码成每点512维特征。

## 网络流程

```text
40000×3单视角点云
→ 5 mm体素化
→ 3D稀疏ResUNet
→ 每点512维特征
→ Linear(512,3)
   ├── 2维objectness
   └── 1维graspness
→ 高分点 + FPS
→ 1024个种子点与对应512维条件特征
→ 12维姿态扩散（9维旋转表示+3维平移偏移）
→ SVD恢复合法3×3旋转
→ Joint MLP
→ 20维Wuji2 qpos
```

## 训练损失

- `loss_objectness`：所有40000点的物体/背景交叉熵；
- `loss_graspness`：只在真实物体点上计算Smooth L1；
- `loss_diffusion`：真实抓取姿态加噪后的扩散速度预测误差；
- `loss_joint`：20关节预测与当前数据集配置指定的真实qpos标签之间的Smooth L1。
  q_opt基线使用`pre_force_joint_positions_rad`；严格A/B版本使用力调整后的
  `joint_positions_rad`。

总损失权重沿用配置：objectness 1、graspness 1、diffusion 10、joint 1。

## 推理输出

一个视角默认生成1024条候选。每条候选包含：

- 3×3旋转；
- 3维平移；
- 20维关节值；
- 种子点；
- graspness；
- diffusion `log_prob`；
- 联合分数。

## Wuji2改动边界

网络主体保持官方实现。明确修改的是：

- `joint_num: 16 → 20`；
- 训练标签换成Wuji2的20关节目标；具体使用q_opt还是力调整后目标由数据集配置
  决定，网络结构不因此改变；
- 手根和相机/世界坐标转换遵守Wuji2合同。

官方LEAP checkpoint不能直接作为Wuji2最终权重使用。
