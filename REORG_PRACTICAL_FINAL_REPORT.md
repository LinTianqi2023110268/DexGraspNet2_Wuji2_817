# 实用终态整理报告

日期：2026-08-14。范围：`/home/lin/Projects/DexGraspNet2_Wuji2`。

## 结论

本轮未移动/复制/生成 `02_training_dataset`，未启动 Isaac Sim、训练、推理或数据生成，未修改官方 Wuji2/dual-arm/DexGraspNet2 资产、ft04、mass-fixed USD、物理参数或驱动环境。

项目保留原 01–08 顶层顺序，三条正式业务路线各收敛到一个代表案例：

- Route A：Ashtray source462，SHA **15/15 PASS**；
- Route B：Battery candidate49，物理状态 `VERIFIED_PHYSICAL`，完整性状态 `PRE_EXISTING_INTEGRITY_EXCEPTION`，旧 SHA **8/10**；
- Route C：dog candidate3800，完整 24.85 s 抓放 PASS，更新后的正式 SHA **19/19 PASS**。

## Route B 收敛

- `01_cases/active/`：1 个临时可再生成案例；
- `99_archive/regenerable_cases/`：38 个可再生成案例；
- `99_archive/failed_cases/`：2 个少量失败代表；
- `99_archive/route_c_provenance/`：1 个 dog candidate3800 来源案例；
- 旧 dog 来源路径由符号链接保持兼容；
- `case_registry.json` 只登记正式代表、当前活动案例、Route-C 来源和归档索引。

Battery 冻结目录和旧 `SHA256SUMS` 未修改。当前十个受审计文件的实际值另存于 `verified/B_verified_leap_to_wuji2/observed_sha256_20260814.txt`。

## Route A 收敛

`selected_native_case` 已明确标注 `TEMPORARY_REGENERABLE`；正式成功只认 `04_verified_baseline/scene0000_view0001_ashtray_source0462/`。

## Route C / Isaac Lab 收敛

`isaaclab_control/` 根目录现有 10 个条目：两个正式 launcher、README，以及 `runtime/`、`diagnostics/`、`history/`、`rejected/`、`report_demo/`、`outputs/`、`tools/`。

- `runtime/`：正式执行与连续实时回放；
- `diagnostics/`：静置、质量树、RGB-D、短运动；
- `history/`：旧 PD/NF/explicit/J6 研究和阶段验证；
- `rejected/mp4_export/`：失败方案说明与源码快照；
- 错误 MP4、PNG 中间缓存和导出日志已清除；
- 正式 replay 已去除 MP4 分支，只保留未经插值/滤波的连续墙钟回放。

## 静态验收

- 活动 Python 源码 AST：73 个通过；
- 活动 JSON：142 个通过；
- 08 下全部 Shell：`bash -n` 通过；
- 非环境符号链接：无断链；
- `tools_validate.py`：PASS；
- Route A：15/15；Route B：预期 8/10；Route C：19/19。

## 清理内容

只删除了明确可再生成的 Python `__pycache__`、被否决的视频/PNG缓存和空视频目录。正式 baseline 内的历史字节码（属于其原 SHA 合同）未删除。

## 长期入口

项目导航：`README.md`、`PROJECT_STATUS.md`、`verified/INDEX.md`。

Route C 正式入口：

```bash
./08_dual_arm_scene_layout/isaaclab_control/launch_full_pick_place_25s_visible.sh
./08_dual_arm_scene_layout/isaaclab_control/launch_replay_25s_visible.sh
```

历史实验不再与正式入口平级。若未来需要新的 bitwise-clean Battery baseline，必须新建目录重新生成并重新物理验证，不能覆盖旧 Battery。
