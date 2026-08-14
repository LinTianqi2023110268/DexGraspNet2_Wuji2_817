# Verified native-Wuji2 baseline: Ashtray source462

状态：`VERIFIED_SUCCESS_BASELINE`。

- scene/view：`scene_0000/view_0001`；
- target：Ashtray，segmentation 14；
- network source candidate：462；
- 碰撞过滤后排名：5；
- score：20.9556465；
- 物理结果：目标抬升 60.6928 mm，且只有目标达到抬升成功判据。

本目录由 `01_cases/selected_native_case` 在 2026-08-14 冻结。原 `case.json` 的 `ready_for_manual_isaacsim_validation` 是仿真前状态快照；最终状态以新增的 `verified_manifest.json` 和 `05_isaacsim/final_result.json` 为准。

执行合同：Wuji2 虎口 100 mm 接近、37.5 mm 预张开、网络 q20、五指局部 `+Y` 30 mm SQUEEZE、世界 `+Z` 70 mm LIFT、3P+3R K800/D20 根驱动。

不要用 LEAP 路线的 SQUEEZE 替换本案例合同。

