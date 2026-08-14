# LEAP → Wuji2 已验证抓取流程

本目录是当前唯一有效的 LEAP → Wuji2 位姿迁移、SQUEEZE 和 Isaac Sim 5.0
验证流程。2026-08-11，案例 `scene0001_view0001_official_rank0` 已真实抓起目标
物体，并被标记为首个已验证成功基线。

## 1. 目录

```text
06_leap_to_wuji2_final_pipeline/
├── active_case.json          当前案例
├── 00_shared/                所有案例共用的配置、模型和Sim执行器
├── 01_cases/active/          最多一个临时开发案例
├── 02_scripts/               从网络结果到Sim任务的单一处理流程
├── 03_template/              新案例模板
├── 04_verified_baseline/     冻结的成功脚本、任务、参数和结果
├── 99_archive/               可再生成案例、少量失败代表和Route-C来源
└── run_new_test_case.py      一键建立新案例
```

不要从 `99_archive` 运行脚本。当前所有新案例只生成一种 Sim 方法，不再生成旧的
`set_world_pose()` 手根分支。

## 2. 当前唯一方法

```text
官方DexGraspNet 2.0输出的LEAP GRASP
  → 官方wuji-retargeting得到Wuji2 q20
  → 四个真实对应指尖等权求Wuji2手根6D
  → 官方wuji-retargeting得到Wuji2 SQUEEZE
  → 小拇指复制无名指屈曲链，MCP侧摆向外偏10°
  → 41点SQUEEZE关节路径
  → 大拇指PREGRASP沿已确认指腹内收法向的反方向张开
  → 指尖目标IK得到大拇指4关节值
  → 官方Wuji2 Hand2 Beta1 USD
  → LEAP同构3P+3R手根位置驱动
  → COVER → GRASP → SQUEEZE → LIFT
```

关键约束：

- Wuji2 运行资产：官方 `wujihand2.usd`；
- 手根：3个移动关节＋3个旋转关节，force型位置驱动；
- 手根有效刚度/阻尼：`K=800`、`D=20`；
- 手指20关节：服从官方USD驱动、限位和最大力；
- 大拇指局部 `+Y` 是收紧方向，局部 `-Y` 是预张开方向；
- `keep_z=false`，保留指尖方向的完整三维分量；
- 大拇指预张开目标：30 mm，20步IK；
- 小拇指：MCP屈曲、PIP、DIP与无名指一致，MCP侧摆在无名指基础上
  沿官方右手URDF的外偏正方向增加10°；
- SQUEEZE之前及过程中零重力，随后恢复世界重力 `-9.81 m/s²`；
- 成功标准：目标物体质心上升至少30 mm。

方向配置见 `00_shared/config/wuji2_native_width_mapper.json`。大拇指张开幅度见
`02_scripts/05_build_isaacsim_validation.py` 中的
`PREGRASP_THUMB_EXTRA_OPEN_M`。
小拇指策略见 `00_shared/config/pinky_ring_coupling.json`；它只在官方四指
重定向完成后执行，不改变官方优化器和已经验证的四指解。

## 3. 已验证成功基线

案例：`scene0001_view0001_official_rank0`

| 项目 | 数值 |
|---|---:|
| 场景/视角 | scene_0001 / view_0001 |
| 网络候选 | 49 |
| 目标分割ID | 60 |
| 目标物体 | Battery |
| 网络分数 | 56.1419 |
| 大拇指张开目标 | 30 mm |
| 大拇指沿批准方向实际移动 | 28.45 mm |
| 目标物体抬升 | 204.14 mm |
| 横向位移 | 16.09 mm |
| 目标抓取结果 | PASS |

冻结副本在 `04_verified_baseline/scene0001_view0001_official_rank0/`。其中保存：

- Isaac Sim两阶段脚本；
- `final_waypoints.npz/json`任务文件；
- `final_result.json`物理结果；
- 方向和迁移参数快照；
- 公共Sim执行器快照；
- SHA-256校验清单。

## 4. 案例状态

机器可读入口见 `case_registry.json`。长期正式案例只有 Battery candidate49；
`01_cases/active/` 最多一个临时案例。其余案例均在 `99_archive/`，大多数可由
一键脚本重新生成。Route-C dog 来源案例保存在 `route_c_provenance/`，旧读取路径
由兼容链接维持。

## 5. 运行当前案例

在 Isaac Sim 5.0 Script Editor 中依次运行：

```text
04_verified_baseline/scene0001_view0001_official_rank0/scripts/01_import.py
04_verified_baseline/scene0001_view0001_official_rank0/scripts/02_execute.py
```

必须等待01输出 `[01 IMPORT COMPLETE]` 后再运行02。正式结果在冻结目录的
`result/final_result.json`；不要覆盖它。

## 6. 查看Trimesh

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
01_environment/conda/wuji_retargeting/bin/python \
  06_leap_to_wuji2_final_pipeline/02_scripts/04_visualize_final.py --show
```

蓝色：LEAP GRASP；青色：LEAP SQUEEZE；紫色：Wuji2 GRASP；绿色：Wuji2
SQUEEZE。

## 7. 建立新案例

编辑 `run_new_test_case.py` 顶部的：

```python
SCENE_INDEX = 1
VIEW_INDEX = 1
CASE_ID = "scene0001_view0001_official_rank0"
```

然后运行：

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
/usr/bin/python3 06_leap_to_wuji2_final_pipeline/run_new_test_case.py --rebuild
```

流程会生成点云、1024条LEAP候选、rank0动作、Wuji2 GRASP/SQUEEZE迁移、
Trimesh和最终Sim脚本。新案例默认继承本次成功方法。

环境分工由一键脚本自动处理：网络推理和Sim任务生成使用 `graspnet2.0`，
官方 `wuji-retargeting` 姿态迁移使用项目内的 `wuji_retargeting` 环境。

## 8. 每阶段输入输出

| 阶段 | 输入 | 输出 |
|---|---|---|
| 网络推理 | 单视角40000点 | `official_leap_1024.npz` |
| LEAP动作 | rank0候选 | `leap_official_waypoints.npz` |
| GRASP迁移 | LEAP q16、迁移YAML | `grasp_official.npz` |
| 手根对齐 | 四个对应指尖 | `root_alignment.npz` |
| SQUEEZE迁移 | LEAP SQUEEZE、迁移YAML | `squeeze_official.npz` |
| 可视化 | LEAP/Wuji2两阶段姿态 | `four_hand_final.glb` |
| Sim任务 | 场景、q20路径、手根6D | `final_waypoints.npz`、01/02脚本 |
| 物理结果 | Isaac Sim完整执行 | `final_result.json` |

`.npz` 是供程序读取的多数组容器，不要手工解压；旁边的 JSON 是人可读审计文件。
