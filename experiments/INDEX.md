# Experiment Index

正式业务案例只在 `../verified/` 和长期 evidence 目录登记。本目录仅保留历史分类说明；实际可复用诊断工具在 `../08_dual_arm_scene_layout/isaaclab_control/diagnostics/`，阶段结论在 `../08_dual_arm_scene_layout/isaaclab_control/core/worklog/`。

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

只保留少量有独立原因的代表结论；可重新生成的 payload 已从工作树清理，必要时通过 Git 历史恢复。

## REJECTED

逐帧 PNG/MP4 视频导出已被否决；自动视频不是正式功能，结论保留在 worklog 中。

机器清单：

- 清理前完整历史由 Git 快照保存。
- 当前权威状态见根目录 `README.md`、`PROJECT_STATUS.md`、`verified/`、`08_dual_arm_scene_layout/isaaclab_control/evidence/` 和 core worklog。
