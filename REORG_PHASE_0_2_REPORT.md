# Reorganization Phase 0–2 Report

Date: 2026-08-14  
Project: `/home/lin/Projects/DexGraspNet2_Wuji2`

## Outcome

Phase 0–2 is complete. The project now has a rollback inventory, one coherent documentation layer, machine-readable verified/experiment indexes, and one explicitly identified representative physical baseline for each route A/B/C.

This was a conservative reorganization. No formal dataset was moved or copied, no historical case was deleted, no top-level 05–08 route was renamed or moved, and no Isaac Sim, Isaac Lab, training, inference, or dataset-generation job was started.

One pre-existing integrity exception was discovered: the immutable Battery baseline's original `SHA256SUMS` currently validates 8 of 10 entries. The two files drifted before this reorganization. The baseline was not edited and its checksum list was not rewritten.

## Phase 0 — Read-only snapshot

Rollback and audit material is stored at:

```text
archive/migration_snapshots/pre_reorg_20260814/
```

It contains the requested tree, directory-size report, full manifest, absolute-path audit, critical hashes, entry-point inventory, case inventory, move plan, deletion candidates, route-06 reduction plan, route-08 reorganization plan, snapshot metadata, and a compressed backup of the pre-reorganization text/configuration layer.

Pre-reorganization project inventory, excluding the snapshot itself:

- regular files: 298,860;
- bytes: 338,124,520,312;
- snapshot size after completion: 127,090,436 bytes;
- pre-reorganization text backup SHA256: `00b1d9021104b0c6e52fc68be82c3d7e275ee5d1921ca19d4591efb0493d38f3`.

The manifest received a supplementary read-only pass for 1,382 pre-existing symlink names after the first pass was found to resolve their target paths in the displayed names.

## Phase 1 — Unified documentation and indexes

The following current documents were created or rewritten:

- `README.md`: project goal, routes A/B/C, authoritative data/model locations, official entry points, three verified cases, environment split, and current next task.
- `PROJECT_STATUS.md`: ACTIVE / VERIFIED / EXPERIMENTAL / ARCHIVED / REJECTED state.
- `docs/architecture.md`: components and route boundaries.
- `docs/dataset_contracts.md`: q_opt versus force-adjusted labels and dataset contracts.
- `docs/execution_contracts.md`: runtime, physical-validation, replay, and cache boundaries.
- `docs/coordinate_and_label_contract.md`: coordinate frames and label semantics.
- `config/external_dependencies.json`: external repositories and environments, including GroundedSAM.
- `verified/INDEX.md` and `verified/index.json`: authoritative A/B/C baseline pointers.
- `experiments/INDEX.md` and `experiments/index.json`: diagnostics, history, and rejected paths.

Resolved documentation conflicts include:

- dual-arm IK is no longer described as locked;
- automatic per-frame PNG/MP4 export is marked REJECTED, not a formal feature;
- q_opt uses `pre_force_joint_positions_rad`, while force-adjusted supervision uses `joint_positions_rad`;
- the default completed Wuji2 model is the force-adjusted 100-scene, 50,000-step model;
- dog candidate3800 is the only formal complete dual-arm business case;
- 24.85 s is simulated action time, while the recorded local wall time is about 272.98 s;
- cached perception/network replay is not described as live re-inference.

## Phase 2 — Minimal verified baselines

### Route A — Native Wuji2

Frozen copy:

```text
07_wuji2_network_3p3r_sim/04_verified_baseline/
scene0000_view0001_ashtray_source0462/
```

Identity and result:

- scene0000 / view0001;
- Ashtray, segmentation 14;
- source candidate 462, filtered rank 5;
- `target_specific_success=true`;
- target lift: 0.060692802 m (60.69 mm);
- 15/15 entries in its `SHA256SUMS` pass.

The authoritative physical result is `05_isaacsim/final_result.json`. A stale status field retained inside the copied pre-execution `case.json` is not authoritative and is called out by the verified manifest.

### Route B — Official LEAP to Wuji2

Immutable original baseline, referenced in place:

```text
06_leap_to_wuji2_final_pipeline/04_verified_baseline/
scene0001_view0001_official_rank0/
```

Identity and result:

- Battery;
- candidate 49, score 56.14188385;
- root RMS 7.99 mm;
- SQUEEZE RMS 8.75 mm;
- target lift: 0.204140544 m (204.14 mm);
- manifest status: `VERIFIED_SUCCESS_BASELINE`.

Checksum audit: **8 PASS / 2 FAIL**. The mismatches are:

- `task/final_waypoints.json`;
- `runtime/common_import.py`.

Both files have local modification time 2026-08-11 23:14:38 +08:00, later than the original checksum file at 18:58:03 and earlier than this reorganization. The directory was kept immutable as required. Evidence is in `verified/B_verified_leap_to_wuji2/checksum_audit.md`.

### Route C — Dual-arm full pick-place

Frozen in place:

```text
08_dual_arm_scene_layout/isaaclab_control/outputs/
full_pick_place_25s_dog_candidate3800/
```

Identity and result:

- dog, candidate3800;
- complete grasp, lift, transfer, placement, release, and return;
- report status: PASS;
- simulated action duration: 24.85 s;
- measured action wall duration: 272.978849 s;
- maximum object lift: 180.852771 mm;
- final object footprint inside the green placement zone: true;
- `physical_replay_30fps.npz`: present, stable 747-frame replay artifact;
- 19/19 root-relative entries in its `SHA256SUMS` pass.

The checksum set covers the result, trace, replay, formal config and launchers, control/replay scripts, source case, retargeted waypoints, arm IK result, path collision audit, placement plan, GroundedSAM result, and DGN2 ranked/filtered outputs. The rejected `videos/` experiment is intentionally excluded.

### Control diagnostics

These are diagnostic baselines, not additional business grasp cases:

- ft04 and TGS velocity audit: `08_dual_arm_scene_layout/isaaclab_control/outputs/force_nf_finetune/ft04_j1_50_j3_45_j5_55_j6_50/velocity_reality_audit.json`;
- 20 mm short-motion IK: `08_dual_arm_scene_layout/isaaclab_control/outputs/short_motion_ft04_z20_fixed/report.json`;
- the 20 mm report is PASS with 2.5845 mm return-position error and 0.6000° return-orientation error.

## Required acceptance results

1. **Project total size:** basically unchanged. Excluding the rollback snapshot, the project grew by 11,901,635 bytes (about 11.35 MiB), primarily the frozen Route-A baseline and new documentation. The separate rollback snapshot occupies 127,090,436 bytes.
2. **`02_training_dataset`:** unchanged in place: 264,432 regular files and 333,447,049,476 bytes before and after. No formal dataset was copied, moved, regenerated, or edited.
3. **`tools_validate.py`:** PASS under `/home/lin/miniconda3/envs/graspnet2.0/bin/python`.
4. **Three formal cases:** clearly indexed as A/Ashtray source462, B/Battery candidate49, and C/dog candidate3800.
5. **Route-A Ashtray:** formally frozen with manifest, README, inputs/results, and passing SHA256 list.
6. **Battery SHA256:** not fully passing; 8/10 pass because of documented pre-existing drift. The immutable baseline was left untouched.
7. **dog 24.85 s result/replay:** complete in place; report is PASS and replay exists.
8. **ft04 / 20 mm IK:** both have stable, machine-readable evidence paths in `verified/index.json` and `experiments/index.json`.
9. **Formal checkpoints:** official LEAP, final Wuji2 50,000-step, and legacy 40-scene checkpoints exist and load on CPU. The completed force-adjusted training directory contains 10 intermediate checkpoint files.
10. **GroundedSAM:** registered as an external dependency at `/home/lin/Projects/分类抓取开源项/03_检测加分割_GroundedSAM`; no repository or weights were copied into this project.
11. **README conflicts:** current root/status/route-control documentation now agrees on route boundaries, label fields, dog full-pick-place status, timing, cache semantics, and rejected video export.
12. **Deletion candidates:** 407 entries totaling 109,924,277 bytes (104.83 MiB). Separately, 38 case directories totaling 511,875,696 bytes (488.16 MiB) are `ARCHIVE_CANDIDATE`, not deletion-approved. Nothing was deleted.
13. **Regenerable cases:** all 39 non-representative route-B cases are marked regenerable by the retained `run_new_test_case.py` path from formal datasets/caches; one is also required as Route-C dog provenance. Historical native-Wuji2 selections can be rebuilt through the retained selected-case builder. Repeated route-C failures can later be reduced to metadata/summary once the user approves, while dog provenance and the three diagnostic baselines must remain.

## Validation performed

- JSON parsing for all new machine-readable indexes/manifests/config.
- project structural validator: PASS.
- exact before/after count and byte comparison for `02_training_dataset`: unchanged.
- Route-A checksum: 15/15 PASS.
- Route-B checksum: 8/10 PASS with a documented pre-existing exception.
- Route-C checksum: 19/19 PASS when evaluated from project root as specified by its paths.
- official LEAP and Wuji checkpoints loaded with CPU-only `torch.load`; no CUDA job was started.
- official Wuji asset hashes were checked by `01_environment/verify.py`.

The environment verifier's CUDA availability line is not used as a GPU-health judgment because this was a read-only CPU/sandbox validation. No driver, CUDA, Isaac Sim, or Isaac Lab change was made.

## Deferred work

Phase 3–7 was not started. In particular, no 05/06/07/08 top-level move, mass rename, archive move, deletion, checkpoint cleanup, Conda migration, GroundedSAM copy, or 312 GB dataset move was performed.

The next safe decision is user review of this report and the TSV plans. Only after explicit approval should Phase 3 begin.
