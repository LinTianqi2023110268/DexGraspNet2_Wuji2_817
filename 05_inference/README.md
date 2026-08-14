# 05 单视角预测

本目录把一个已经生成好的单视角点云变成可检查、可过滤、可送入仿真准备阶段的Wuji2抓取候选。

默认checkpoint由`config/project.json:sources.wuji2_checkpoint`指定，当前为
力调整后100场景数据训练至50000步的Wuji2 20关节模型。旧40场景checkpoint仅
保留为历史回归文件，不是默认推理权重。

## 固定顺序

1. `01_predict_single_view.py`：40000点 → 1024候选；
2. `02_filter_scene_collisions.py`：PREGRASP、最终手和增强路径碰撞过滤；
3. `../trimesh/view_predictions_trimesh.py`：点云、热力图、种子点和完整手姿态；
4. `04_build_isaacsim_job.py`：生成五阶段NPZ/JSON，不启动Isaac Sim。

## 01 网络预测

输入必须包含该场景视角的点云、分割和`T_world_camera`。输出候选同时保存相机坐标和世界坐标表示。

默认联合分数：

\[
score=log\_prob+5\times graspness
\]

`log_prob`来自训练好的姿态扩散概率模型，不是独立的第三个全连接评分器。

## 02 碰撞过滤

当前同时保存三组概念：

- 官方仿真端点零穿透规则；
- 论文标签使用的2.5 mm严格间隙规则；
- 明确标记的Wuji2增强规则：掌心路径、最终桌面和非目标物体碰撞。

目标物体接触在最终抓取阶段是允许的。增强路径是掌心中心线检查，不等价于完整手扫掠体证明。

PREGRASP统一采用Wuji2虎口方向：从语义掌心指向当前`q_grasp`下拇指尖/食指尖中点，根位姿反向后退100 mm。

## 03 Trimesh

Trimesh只做几何解释，不模拟接触力、重力、摩擦或关节驱动。建议先看前5条候选，不要一次显示1024只手。
所有自写可视化Python入口统一保存在工程根目录的`trimesh/`中。

## 04 Isaac任务生成

输出阶段：

```text
PREGRASP：虎口反向后退100 mm，手指打开
COVER：到达抓取根位姿，仍保持打开
GRASP：网络20关节q_grasp
SQUEEZE：五指表面内法向30 mm、keep_z、20步IK
LIFT：保持q_squeeze，世界+Z抬升
```

任务状态写为`ready_for_isaacsim_not_yet_physically_validated`。这个名字很重要：生成任务不等于已经抓起来。

## 输出目录

- 网络与过滤输出：`05_inference/outputs/`；
- 新生成的Isaac任务：`05_inference/outputs/isaacsim_jobs/`。

物理验证现在分为两条独立路线：LEAP迁移进入
`06_leap_to_wuji2_final_pipeline/`，原生Wuji2网络输出进入
`07_wuji2_network_3p3r_sim/`。

05生成的NPZ/JSON是“待审核任务”。要把一个新候选变成Isaac Sim案例，必须：

1. 检查目标ID、场景清单、官方`r_wrist`根、20关节顺序和任务资产；
2. 明确它属于06 LEAP迁移路线还是07原生Wuji2路线，不得混用SQUEEZE；
3. 用对应路线的生成脚本创建scene/source/candidate合同；
4. 运行`tools_validate.py`并在Isaac Sim中完成实际物理验证；
5. 只有验证成功且人工确认后，才能纳入对应路线的`04_verified_baseline/`。

因此05与06/07之间是有意设置的审核门，而不是生成后直接宣称物理成功。

临时smoke结果不保留在正式工程中。
