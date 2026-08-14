# 工程结构与数据边界

## 正式阶段与三条技术路线

| 目录 | 唯一职责 | 主要输入 | 主要输出 |
|---|---|---|---|
| `01_environment` | 检查运行环境 | Conda、PyTorch、CUDA、项目内源码路径 | 只读检查结果 |
| `02_training_dataset` | 生成和验收Wuji2训练标签 | 单物体抓取库、物体网格、Isaac Sim相机数据 | 场景、单视角、graspness与姿态标签 |
| `03_prediction_network` | 保存网络结构 | 官方DexGraspNet2核心源码 | 512维特征、种子点、扩散姿态与20关节网络定义 |
| `04_training` | 训练和验证网络 | 完整且验收通过的数据集 | Wuji2网络checkpoint和损失记录 |
| `05_inference` | 单视角预测及几何过滤 | 40000点、checkpoint、场景清单 | 1024候选、排序、碰撞过滤和任务NPZ |
| `06_leap_to_wuji2_final_pipeline` | LEAP→Wuji2迁移与仿真 | 官方LEAP预测、wuji-retargeting | 迁移q20、手根6D、3P+3R仿真结果 |
| `07_wuji2_network_3p3r_sim` | 原生Wuji2预测与仿真 | 5个单视角、Wuji2 checkpoint | 每物体top2、任意过滤排名、Trimesh、原生动作+3P+3R仿真 |
| `08_dual_arm_scene_layout` | 双臂完整系统 | 标定场景、缓存感知结果、路线A/B抓取姿态 | 右臂IK、完整抓放、机器报告和稳定回放 |

## 公共目录

- `config/project.json`：所有项目内数据相对路径和运行环境路径的唯一登记处。
- `src/wuji2_dgn2`：训练与推理共同使用的坐标、碰撞、虎口接近和可视化实现。
- `trimesh`：本工程自写的Trimesh、Matplotlib和Tk可视化Python入口唯一目录。
- `docs`：坐标合同、标签合同和工程边界。
- `verified`：三条路线各一个正式业务基线的统一索引，不复制大结果。
- `experiments`：控制诊断、失败历史和待归档索引。
- `archive`：整理前快照和迁移清单；未经确认不删除候选。
- `tools_validate.py`：只读检查工程结构，不启动训练或仿真。

## 当前数据

1. `single_object_dataset`：项目内Wuji2 1.0单物体优化结果和六方向物理验证来源。
2. `wuji2_train60_100seminal_256view_v1`：q_opt监督的100场景×256视角基线。
3. `wuji2_train60_100seminal_256view_force_adjusted_legacy_v1`：相同输入、只把20关节监督换为力调整后位姿的严格A/B版本。
4. `wuji2_test60_10upright_10view_v1`：独立10场景×10视角、q_opt关节标签测试集；
   可供力调整模型做推理和物理测试，但不能混算其joint loss。
5. `assets/wuji2_factory`：项目内Wuji2手、仿真USD/URDF和完整DexGraspNet物体网格库。

正式代码、manifest和配置均保存项目根相对路径；Conda与Isaac Sim安装路径是唯一允许保留的机器绝对路径。

## 禁止混用

- q_opt基线使用`pre_force_joint_positions_rad`监督。
- 力调整A/B数据集显式使用`joint_positions_rad`监督；两者不能在同一次训练中混用。
- checkpoint与定量评估集的`training_joint_field`必须一致；推理不使用真实qpos，
  但joint loss使用，二者不要混淆。
- `semantic_palm_6d`不是`r_base_link`手根位姿。
- 虎口方向负责PREGRASP接近；当前已确认的SQUEEZE方向是五个指尖链接各自
  局部`+Y`绿色轴30 mm、`keep_z=False`，两者不是同一方向。
- 几何过滤通过不等于Isaac Sim抓取成功。

## 正式大数据验收条件

两个100场景训练版本均已完成Stage01-04；独立10场景测试集也已生成：

1. `run_manifest.json`的`status`为`complete`；
2. 100个`scene_manifest.json`完整；
3. 25600个视角均包含采样像素、深度与分割数据；
4. 后续场景抓取、graspness和单视角标签生成完成；
5. `check_wuji2_dataset.py`通过；
6. 测试集来自独立场景族，而不是从这100个训练场景中拆分。
