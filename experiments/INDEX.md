# Experiment Index

正式业务案例只在 `../verified/` 登记。本目录索引历史诊断和失败；实际控制研究已归入 `../08_dual_arm_scene_layout/isaaclab_control/history/` 与 `diagnostics/`。

## VERIFIED_DIAGNOSTIC

- mass tree audit：修复 6 个非预期默认 1 kg frame。
- ft04 + TGS velocity reality audit。
- `short_motion_ft04_z20_fixed` 20 mm 往返 IK PASS。

## HISTORY

- acceleration natural-frequency scan；
- force natural-frequency scan/fine tune 的非 ft04 轮次；
- grouped PD；
- explicit PD / gravity feed-forward 尝试；
- J6 gravity margin；
- 旧 short-motion。

## FAILED_HISTORY

只保留少量有独立原因的代表：抓取失败、IK/路径失败、控制失败。Route B 同质可再生成案例已集中到 `06_leap_to_wuji2_final_pipeline/99_archive/regenerable_cases/`。

## REJECTED

逐帧 PNG/MP4 视频导出。稳定实时回放保留，自动视频不是正式功能。

机器清单：

- `../archive/migration_snapshots/pre_reorg_20260814/case_inventory.tsv`
- `../archive/migration_snapshots/pre_reorg_20260814/06_case_reduction_plan.tsv`
- `../archive/migration_snapshots/pre_reorg_20260814/08_reorg_plan.tsv`
- `../archive/migration_snapshots/pre_reorg_20260814/delete_candidates.tsv`
