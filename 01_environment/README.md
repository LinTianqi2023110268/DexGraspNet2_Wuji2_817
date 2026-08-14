# 01 环境

本工程的训练和仿真仍故意使用两个主环境，不合并。姿态重定向另外使用一个项目内
隔离环境，避免升级既有环境中的Pinocchio。

## `graspnet2.0`

- Python 3.8.20；
- PyTorch 2.0.1+cu117；
- 负责标签处理、MinkowskiEngine、训练、预测和Trimesh；
- RTX 4070已验证可运行CUDA 11.7版PyTorch。

## `wuji2_factory`

- 负责Isaac Sim 5.0与Isaac Lab 2.2；
- 只运行场景物理生成和抓取仿真；
- 不承担网络训练。

## 项目内`wuji_retargeting`

```text
/home/lin/Projects/DexGraspNet2_Wuji2/01_environment/conda/wuji_retargeting
```

- Python 3.11.15；
- 只负责LEAP FK 21点→官方Wuji2 20关节姿态重定向和对应Trimesh；
- 使用Pinocchio 3.8.0、NLopt 2.11.0和官方`wuji-retargeting`；
- 不训练网络、不启动Isaac Sim；
- 不改`graspnet2.0`和`wuji2_factory`。

官方包要求Python 3.10以上，且当前代码使用Pinocchio 3的接口，所以不能直接放进
Python 3.8的`graspnet2.0`，也不应升级已有`wuji2_factory`的Pinocchio。
实际版本合同记录在`wuji_retargeting_environment.txt`；其中`cmeel-urdfdom`和
`cmeel-tinyxml2`是经过导入测试的ABI配套版本，不能随意单独升级。

## 检查命令

```bash
cd DexGraspNet2_Wuji2
/home/lin/miniconda3/envs/graspnet2.0/bin/python 01_environment/verify.py
/home/lin/miniconda3/envs/graspnet2.0/bin/python tools_validate.py
```

`verify.py`会分别调用两个环境的Python，打印Python、PyTorch、PyTorch编译CUDA和CUDA是否可用。它不会安装依赖、更新显卡驱动或启动Isaac Sim。

Codex的隔离终端有时看不到主机的`/dev/nvidia*`，会显示
`torch.cuda.is_available() == False`；这不能单独证明显卡驱动离线。应以你自己的
主机终端`nvidia-smi`和`graspnet2.0`中的PyTorch检查为准；如果Isaac Sim正在
运行，再把其控制台和GPU进程作为辅助证据。本文档不假设当前存在后台任务。

## 显卡版本说明

`nvidia-smi`显示的CUDA版本是驱动能够支持的最高CUDA接口；`torch.version.cuda`显示PyTorch自身编译使用的CUDA版本。驱动显示13.0而PyTorch显示11.7并不构成冲突，只要`torch.cuda.is_available()`为`True`即可。

## 安全约束

- 不自动更新NVIDIA驱动；
- 不在后台数据生成时启动第二个Isaac Sim；
- 不用`cuda`或`nvidia`作为检查命令，正确命令是`nvidia-smi`和PyTorch检查；
- 不把`graspnet2.0`与`wuji2_factory`强行合并。

## Wuji2官方模型（唯一运行基准）

Wuji2手在Isaac Sim中的唯一基准资产是：

```text
01_environment/vendor/wuji-description/
  hand2/hand2_beta1/body/usd/right/wujihand2.usd
```

必须保留完整的`usd/right/`目录，不能只复制入口文件。固定的上游提交、
文件哈希、根坐标转换和配套URDF见`config/wuji2_official_asset.json`。

- Isaac Sim加载官方USD，不再运行URDF转USD；
- 离线FK必须使用同一提交内的官方配套URDF；
- 旧训练标签仍保存`T_world_r_base_link`，导入官方USD前由
  `src/wuji2_dgn2/official_asset.py`显式转换为官方`r_wrist`；
- 已完成数据集的SDF/碰撞标签仍是旧几何的不可变历史输入；只有重新生成
  Stage02/03并重新物理验证，才能称为官方几何标签，禁止仅改路径或字段名；
- 旧标签的关节值按20个同名关节迁移，不按Isaac内部索引硬拷贝；
- 官方USD没有物理摩擦材质，实验仍需显式设置并记录手、物体、桌面的
  静摩擦、动摩擦和恢复系数。

## 双臂机械臂实体模型

双臂机械臂和Wuji2手位于同一个上游描述仓库根目录，但分别保持完整独立模型包，
禁止把机械臂文件混入`hand2`：

```text
01_environment/vendor/wuji-description/
├── glove/
├── hand/
├── hand2/                          原有Wuji2模型，保持不动
└── dual_arm/
    ├── urdf/dual_arm.urdf
    ├── meshes/
    └── README.md
```

`08_dual_arm_scene_layout`只负责场景、相机、测距和抓放流程，不保存机器人实体。
