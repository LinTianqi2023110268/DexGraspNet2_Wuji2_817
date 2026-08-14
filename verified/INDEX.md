# Verified Baselines

这里只建立统一索引。大结果保持在所属路线原位，避免复制 checkpoint、轨迹和场景数据。

## A — Native Wuji2

- Ashtray，scene0000/view0001，segment14，source462，过滤排名 5。
- 目标抬升 60.69 mm，`target_specific_success=true`。
- 冻结目录：`../07_wuji2_network_3p3r_sim/04_verified_baseline/scene0000_view0001_ashtray_source0462/`。

## B — Official LEAP to Wuji2

- Battery，candidate49，score 56.1419。
- root RMS 7.99 mm，SQUEEZE RMS 8.75 mm，目标抬升 204.14 mm。
- 原位冻结：`../06_leap_to_wuji2_final_pipeline/04_verified_baseline/scene0001_view0001_official_rank0/`。
- 状态：`VERIFIED_PHYSICAL` + `PRE_EXISTING_INTEGRITY_EXCEPTION`。
- 物理结果仍 VERIFIED，但目录不是 bitwise-clean。原 SHA256SUMS 当前 8/10 通过；2 项整理前即已漂移，详见 `B_verified_leap_to_wuji2/checksum_audit.md`。
- 旧 checksum 禁止重写；未来需在新目录重新生成、重新物理验证，才能得到全新 bitwise-clean baseline。

## C — Dual-arm full pick-place

- dog，candidate3800，完整抓、搬、放、松手和回程。
- 仿真动作 24.85 s；本机现实计算约 273 s；PASS。
- 原位冻结：`../08_dual_arm_scene_layout/isaaclab_control/outputs/full_pick_place_25s_dog_candidate3800/`。
- `physical_replay_30fps.npz` 是 747 帧稳定回放，不是重新仿真。

## Control diagnostics

- ft04/TGS：`../08_dual_arm_scene_layout/isaaclab_control/outputs/force_nf_finetune/ft04_j1_50_j3_45_j5_55_j6_50/`。
- 20 mm IK PASS：`../08_dual_arm_scene_layout/isaaclab_control/outputs/short_motion_ft04_z20_fixed/`。
