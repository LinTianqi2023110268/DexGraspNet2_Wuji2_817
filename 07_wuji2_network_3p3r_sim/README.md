# 07 原生 Wuji2 网络输出 + 3P+3R 仿真

本目录只服务于“Wuji2 网络直接输出 q20”的路线。它不进行 LEAP→Wuji2
重定向；LEAP 路线仍完整保存在 `06_leap_to_wuji2_final_pipeline/`。

## 固定合同

本路线只把旧版手根逐物理步 `set_world_pose()` 改为已经验证过的 3 个移动
关节 + 3 个转动关节位置驱动。下面内容保持旧原生 Wuji2 流程不变：

1. WidthMapper 预张开 37.5 mm；
2. 沿虎口方向后退/接近 100 mm；
3. 手根固定，20 关节闭合到网络输出；
4. 五个指尖沿各自指尖链接的局部 `+Y`（Trimesh绿色箭头）
   SQUEEZE 30 mm，`keep_z=False`，不再删除世界竖直分量；
5. 世界 +Z 抬升 70 mm；
6. 120 Hz、minimum-jerk、连续重力 -9.81 m/s²；
7. 手/物体/桌面摩擦分别为 0.2/0.5/1.0，物体质量覆盖为 0.1 kg。

手根 3P+3R 使用位置目标、刚度 800、阻尼 20；执行器不会再对手调用
`set_world_pose()`。源 Wuji2 USD 不会被修改。

## 目录

- `00_config/`：5 个测试场景清单和用户选择。
- `01_cases/scene_*/02_predictions/`：网络输出与过滤结果快照。
- `01_cases/selected_native_case/`：`TEMPORARY_REGENERABLE`，当前选择覆盖式生成工作区，不是正式证据。
- `02_scripts/`：选视角、推理、选每物体 top2、Trimesh、生成仿真任务。
- `03_runtime/`：Isaac Sim 5.0 共用 3P+3R 导入器与执行器。
- `04_verified_baseline/`：唯一正式物理成功证据；当前为 Ashtray source462。

## 更换场景、物体、位姿

只编辑 `00_config/select_sim_pose.py` 的三个整数。`FILTERED_RANK`使用
Top35可视化中的红色编号，即从1开始：

```python
SCENE_INDEX = 0
OBJECT_SEGMENTATION_ID = 14
FILTERED_RANK = 5
```

然后在 `graspnet2.0` 环境生成任务：

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
conda activate graspnet2.0
python 07_wuji2_network_3p3r_sim/02_scripts/04_build_selected_sim_3p3r.py
```

生成结果固定覆盖 `01_cases/selected_native_case/`，避免每次试验留下散乱脚本。

## Isaac Sim 手动运行

打开 Isaac Sim 5.0 的 Script Editor，在同一个会话中依次打开并运行：

1. `01_cases/selected_native_case/05_isaacsim/01_import.py`
2. 等控制台出现 `[01 IMPORT COMPLETE]`
3. `01_cases/selected_native_case/05_isaacsim/02_execute.py`

02 会自动 Play，完成后暂停，并把结果写到
`01_cases/selected_native_case/05_isaacsim/final_result.json`。

Stage 01 会显示 `[01 1/7]` 到 `[01 7/7 OK]` 的导入进度。只出现桌面或
少数物体而没有 `[01 IMPORT COMPLETE]`，表示导入失败，不能继续运行 02。
若 Script Editor 仍打开已经移动或删除的旧脚本标签，请先关闭这些旧标签；
Isaac Sim 会持续尝试读取失效路径，造成大量无意义日志和界面卡顿。

## 重新执行网络流程

脚本的输入输出都直接位于本目录，不依赖旧顶层 06/07：

```bash
python 07_wuji2_network_3p3r_sim/02_scripts/00_select_visible_views.py
python 07_wuji2_network_3p3r_sim/02_scripts/01_run_network_and_filter.py
python 07_wuji2_network_3p3r_sim/02_scripts/02_select_top2_per_object.py
python 07_wuji2_network_3p3r_sim/02_scripts/03_view_selected_poses.py --scene 0 --object 14 --pose-rank 0 --show
```

查看同一物体过滤后按评分排序的前35条抓取（第1页5×5，第2页5×2）：

```bash
python 07_wuji2_network_3p3r_sim/02_scripts/05_view_top35_grid.py \
  --scene 0 --object-id 14 --show page1

python 07_wuji2_network_3p3r_sim/02_scripts/05_view_top35_grid.py \
  --scene 0 --object-id 14 --show page2
```

每格左上角红色数字为排名。黄色是目标物体、青色是该排名的Wuji2
抓取姿态、灰色是同一场景的桌面和其他物体。对应的候选编号、评分及
碰撞间隙记录在同目录的`top35_index.json`中。

推理/过滤脚本仍调用项目公共 `05_inference/` 能力；它们不调用 08 的 LEAP
重定向代码。
