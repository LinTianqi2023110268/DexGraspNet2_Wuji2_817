# 交互式汇报演示

本目录只负责把已经验证过的完整链路清楚地展示出来：

```text
选择目标
  → 顶部相机 RGB / depth
  → GroundingDINO + SAM 目标分割
  → 40,000 点单视角网络输入
  → 官方 DexGraspNet 2.0 候选评分与逐级筛选
  → 用户确认
  → Isaac Lab 2.2 启动 Isaac Sim 5.0 动力学抓取
```

当前可完整执行的目标是 `dog`（segmentation 33，候选 3800）。数据源并不是
随意拼出来的场景，而是正式测试集第一场景 `scene_0000` 在双臂大桌面中开启重力
重新落稳、再由顶部相机拍摄后的 `live_dynamic_scene0000`。

## 20 秒口径

已实测物理动作段为 **17.78 s**，最大抬升 **123.20 mm**，结果 PASS。
动作计时从 `FORM → TOP_RETRACT → ... → LIFT_HOLD` 开始，不包括：

- Isaac Sim / Isaac Lab 启动耗时；
- USD、网络缓存和模型资源加载耗时；
- 开始动作前的 3 s 重力静置。

这三个准备步骤仍会在终端中单独显示，不能把它们伪装进 20 秒动作指标。

## 一次性生成汇报图

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
/home/lin/miniconda3/envs/graspnet2.0/bin/python \
  08_dual_arm_scene_layout/isaaclab_control/report_demo/scripts/01_build_report_assets.py
```

输出位于 `assets/generated/`，包括目标点云、官方候选评分和完整门控漏斗。

## 交互式入口

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
./08_dual_arm_scene_layout/isaaclab_control/report_demo/launch_interactive_demo.sh
```

终端固定在屏幕右侧，并让用户输入目标名称。选择 `dog` 后，RGB/深度、分割和
点云共三个同尺寸窗口会保留在屏幕左侧固定位置；上一幅图不会被替换，各窗口也
不会彼此重叠。候选评分和各级筛选数量只在终端用文字显示，不再单独弹图。最后
最后直接按回车（或输入 `y`）会统一关闭三个汇报窗并启动 Isaac Lab 可视化仿真；
输入 `n` 才取消，而且取消时保留三个图窗供继续检查。

`ashtray` 与 `hammer` 也会显示在目标菜单中，但当前尚未通过完整机械臂可执行门，
程序会明确显示失败阶段，不会把“网络有输出”冒充“可以抓取”。

## 缓存与实时的边界

当前汇报入口回放的是同一时间戳下已经落盘的 RGB-D、GroundedSAM 与官方网络结果，
因此适合稳定汇报；物理抓取会重新运行。它不是“每次现场重新执行神经网络”。后续
若要实现连续抓取循环，应把相机、视觉环境和网络环境改为受控的多进程状态机，不能
在同一个 Isaac Lab Python 解释器中混装所有依赖。
