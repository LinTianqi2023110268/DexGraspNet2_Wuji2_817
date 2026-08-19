# Route-C V2 session summary

## Current phase

Closed-loop one-command integration in progress; physical execution remains locked.

## Completed

- Confirmed the project root is `/home/lin/Projects/DexGraspNet2_Wuji2`.
- Captured the initial Git status and diff; tracked diff is empty.
- Read `CODEX_INSTRUCTION.md`, `INTEGRATION_GUIDE.md`, and `README.md` completely.
- Read both available Isaac Lab / Isaac Sim skill instruction files completely.
- Added the minimal repository-level `AGENTS.md` requested for durable rules.
- Completed the `curobo_v2` probe; all required cuRobo V2 imports succeed.
- Formal host-channel environment probe sees the RTX 4070 and CUDA successfully.
- Core pure tests pass: 15/15 after adding runtime-math, collision-payload, and worker-environment regressions.
- Persistent worker ping passes from `isaaclab22_sim50` to `curobo_v2` with exact right-arm joint order.
- candidate3800 exact five-stage GPU IK passes with 31–34 accepted solutions per target and continuous chained selection.
- Existing generated robot YAML loads on GPU and returns 233 spheres for all 35 active joints.
- Pre-integration Top-20 five-stage coarse GPU regression passes 8/20; candidate3800 and candidate34 both pass.
- Formal RGB-D + GroundedSAM mask builds separate scene/target ESDF layers after one minimal cuRobo 0.8.x RGB-placeholder compatibility fix.
- Worker now supports per-solution phase-aware observed-collision hard filtering before q-reference continuity selection and preserves the audited solution sets.
- candidate3800 collision-aware exact solve passes observed collision at all five stages, but every selected state has single-view unknown exposure.
- Formal Isaac Lab dry-run passes the complete q_current -> worker -> GPU IK -> ESDF planning chain for nine stages.
- Archived the former runtime SciPy IK and old reachability/complete-mesh collision tool cohort under history.
- Post-integration Top-20 regression is unchanged at 8/20 PASS; candidate3800 and candidate34 remain PASS.
- Final `git diff --check` and Python compile/import audit pass; no vendor, DGN2, URDF, or USD file was modified.
- Began the closed-loop integration pass after explicit user approval.
- Re-read `AGENTS.md`, the closed-loop task instruction, and the Isaac Lab control skill body.
- Audited the copied `closed_loop/` patch and confirmed the main missing interfaces: live Grounded-SAM backend, capture-time measured `robot_state.json`, cuRobo self-collision, and continuous path gate.
- Found the local isolated Grounded-SAM project at `/home/lin/Projects/分类抓取开源项/03_检测加分割_GroundedSAM`; its environment and DINO/SAM weights pass the bundled audit.
- Stopped the slow per-candidate full pipeline after user correction; confirmed the architecture issue in source: per candidate retarget/materialization, per candidate `CuroboWorkerClient`, and per candidate ESDF rebuild.
- Added known-good candidate3800 five-stage regression against the current worker; PREGRASP/COVER/GRASP/SQUEEZE/LIFT raw success is restored and positive.
- Added grouped batched IK worker/client support so one request can solve multiple independent candidate waypoint groups while preserving q_current-based branch continuity per group.
- Added `batch_pick_candidate_gate.py` as the chunked pick-stage screening primitive: one worker, one map build, DGN2 score order, `chunk_size * 5` poses per grouped GPU request.
- Added explicit worker self-collision and joint-path regression hooks and validated positive/negative semantics.
- Added scratch case-root support for reviewed LEAP→Wuji2 scripts via `DGN2_CASE_ROOT`, so rejected candidates do not need permanent `01_cases/active` directories.
- Fixed pick-stage path screening so it checks only `q_current→PREGRASP→COVER→GRASP→SQUEEZE→LIFT`; it no longer appends `LIFT→HOME` during pick screening.
- Added cycle-scoped `screen_pick_batches.py`: one cuRobo worker, one map build, lazy chunk materialization, and one grouped solve per chunk.
- Replaced the closed-loop orchestrator's old per-candidate gate loop with one call to the cycle-scoped lazy batch screener.
- Ran same-frame Top-16 dog validation on `closed_loop_sessions/20260816_190311/cycle_001`.
- Diagnosed candidate1422 collision filtering in detail; every threshold-accepted pick-stage IK solution is rejected by self-collision, and GRASP/SQUEEZE are additionally rejected by target ESDF.
- Implemented and benchmarked all-candidate GPU coarse prefilter using DGN2 GRASP root poses only, without LEAP→Wuji2 retargeting or scratch cases.
- Switched closed-loop planning-only feasibility to `SELF_COLLISION_POLICY=REPORT_ONLY_UNRESOLVED`; self-collision is still computed/reported but no longer a hard planning rejection.
- Added low-cost approximate PREGRASP/approach-path coarse prefilter code, but its GPU benchmark could not be completed in this turn because the available execution channel could not expose CUDA and escalation was rejected.
- Integrated the one-command planning-only orchestrator so `./run_closed_loop.sh --planning-only` uses existing modules for capture, GroundedSAM, 40k RGB-D input, DGN2, all-candidate GPU coarse prefilter, survivor-only LEAP→Wuji2 retargeting, exact pick gate, placement allocation, full-route gate, and HOME planning.
- The integrated orchestrator keeps one `CuroboWorkerClient` and one Mapper/TSDF/ESDF build per closed-loop cycle and no longer calls the CLI scripts that would restart worker/map inside the cycle.
- Removed the obsolete physical-execution branch from the planning-only orchestrator; `execution_enabled` remains false and no physical motion is launched.
- Fixed final-planning scratch case path construction so `case_root.name == case_id` for `build_candidate_case.py`.
- Reordered the all-candidate coarse prefilter into a strict cheap-to-expensive funnel: GRASP IK/threshold/scene, PREGRASP IK/threshold/scene only for GRASP scene survivors, then q_current→PREGRASP and PREGRASP→GRASP continuous observed-scene path only for previous-stage survivors.
- Restored the existing multi-cycle execution wiring for explicit Isaac Sim diagnostic execution: selected full-route candidate -> existing runtime launcher -> existing full pick/place/HOME runtime -> existing replay -> existing next-scene-manifest builder -> continue the original `while True` loop.
- Added diagnostic flags `--sim-execute --no-planner-collision-check --diagnostic-ignore-static-gate`; planner collision checks are skipped without disabling Isaac/PhysX collision/contact.
- Added a session-local placement policy/registry under each closed-loop session so diagnostic cycles do not consume or depend on the global placement registry.
- Replaced per-survivor retarget/exact-IK with score-ordered batch retarget chunks and grouped exact IK. Each chunk uses one build-case process, one persistent retarget process for 01/02/03, one finalize/flange process, then one `solve_ik_groups` request for `N*5` exact pick poses.
- Retarget chunk size is now 64. Runtime launcher is invoked as `bash run_full_pick_place_closed_loop.sh ...`, so executable permission on the launcher file is no longer required.

## Current conclusion

- `core/` and several diagnostic/result paths are currently untracked user work and must be preserved.
- The `isaaclab22-manipulator-control` skill's referenced `reference.md` and `evaluations.md` files are absent; work continues from the complete skill body and repository source.
- `rg` is unavailable, so repository audits use `find` and `grep`.
- Installed versions are API-compatible. CUDA is unavailable only in the restricted execution channel; the host channel sees the RTX 4070 correctly.
- Production runtime no longer imports SciPy/Pinocchio IK or consumes the old path-collision report as a gate.
- `closed_loop/config/closed_loop.json` currently has `execution_enabled=false`; this must remain false until a fresh 10 s static gate passes.
- The current cuRobo worker is not raw-success broken on the known-good case: old raw `[32,36,36,36,31]`, current raw `[33,37,37,37,31]`.
- The new batched primitive is implemented and validated at the IK contract level. Same-frame observed-ESDF chunk validation still requires matching capture + mask + scratch candidate cases; mismatched archived candidate/capture data must not be used to claim observed collision PASS.
- Same-frame Top-16 validation used one worker and one map build as required. It found no pick-stage feasible candidate in the first 16 official-score candidates; rank14/candidate1422 had positive raw IK but all IK-accepted solutions were rejected by collision filtering.
- candidate1422 and the measured baseline both expose a tiny recurring self-collision penetration (`0.000251 m`), so current self-collision sphere semantics are likely over-rejecting and must not be treated as final physical truth.
- All-candidate GPU prefilter is fast: `7454` target proposals, batch size `512`, `15` batches, `4435` raw reachable, `4347` threshold accepted, `4269` scene-ESDF pass, but `0` self-collision pass and therefore `0` coarse survivors under the current self-collision gate.
- Under report-only self-collision policy, the already measured GRASP coarse survivors without self-collision are `4269`. PREGRASP/approach survivors are not yet measured because the GPU rerun was blocked by execution-channel CUDA visibility.
- One-command code is integrated and syntax-valid, but real end-to-end validation from this Codex command channel is blocked: Isaac/Kit and `curobo_v2` cannot see the NVIDIA driver/GPU here, while the user's own terminal shows the RTX 4070 via `nvidia-smi`.
- The first user-side one-command run reached GroundedSAM/DGN2/coarse prefilter and found `2748` coarse survivors before hitting the scratch path contract bug at candidate330; that path bug is fixed locally.
- Simulation diagnostic execution wiring is in place but intentionally not run from the restricted Codex channel.
- Batch retarget regression for candidate330 is numerically identical to the old scratch output; GPU exact IK timing/classification must be measured in the user's GPU-visible terminal.
- User-side batch result showed exact IK is no longer the bottleneck (`160` poses solved in `0.027 s` GPU time); retarget/finalize dominate.

## Current blockers

Physical/action smoke is blocked by the unchanged ft04 static stability thresholds: right arm and Wuji2 both FAIL. GPU-dependent commands require a GPU-visible host execution channel; the current Codex execution channel reports `nvidia-smi` failure and `torch.cuda.is_available() == False` in `curobo_v2`. Link-scoped intentional finger/target contact remains a limitation unless implemented later. Self-collision model semantics remain unresolved and are now report-only for planning.

## Next step

Run `./run_closed_loop.sh --planning-only` from a GPU-visible terminal using `scene_0000` and target text `dog`. The expected flow is capture → DINO/SAM → 40k → DGN2 → all-candidate prefilter → survivor retarget → exact pick gate → placement/full-route/HOME planning. Do not add, commit, push, upload, or run physical motion while the static gate is FAIL.

## Modified files

- New/updated core production implementation: `bridge/curobo_worker.py`, `bridge/worker_client.py`, `perception_collision/rgbd_mapper.py`, `runtime_math.py`, tests, generated robot YAML/asset aliases, READMEs, and worklogs.
- Runtime replacement: `runtime/scripts/10_run_full_pick_place.py`, runtime launchers, visible launcher, and runtime README.
- Environment-entry fixes: five active diagnostics launchers now activate `isaaclab22_sim50` and expose local Isaac Lab sources.
- Documentation: `isaaclab_control/README.md`, `tools/README.md`, `history/INDEX.md`, root `AGENTS.md`, and archive README.
- Archived intact: `runtime_rebase_ik.py` and tools `04`-`09`, `12`-`14` under `history/legacy_route_c_cpu_mesh/`.
- Generated evidence: post/pre-integration Top-20 reports, current ft04 report, and `route_c_v2_planning.json`.
- Preserved pre-existing untracked diagnostics/results/manifest files; they were not edited or deleted.

## Test results

- A. Environment probe: PASS on host channel (CUDA true, RTX 4070 Laptop GPU); restricted-channel false result retained only as execution-channel diagnosis.
- B. Core unit tests: PASS (15/15) in `curobo_v2`; the base-interpreter attempt failed only because base has no NumPy.
- C. Robot model: PASS; existing YAML loads on GPU (35 joints, 233 spheres).
- D. Persistent worker: PASS from `isaaclab22_sim50`.
- E. candidate3800 exact GPU IK: PASS; accepted counts `32/34/34/34/31`, solve time `1.989 s`.
- F. candidate3800 coarse five-stage: PASS.
- G. candidate34 coarse five-stage: PASS.
- H. Top-20 coarse five-stage: 8/20 PASS (Top-10 4/10); limitation: pure coarse IK only.
- I. Post-integration Top-20 regression: unchanged at 8/20 PASS; candidate3800 and candidate34 remain PASS.
- RGB-D Mapper: PASS; 407,537 valid pixels, separate scene/target ESDF grids.
- J. Isaac Lab dry-run: planning PASS after real settle; q_current read and nine continuous stages selected; headless Kit shutdown hang remains.
- K. Physical smoke: NOT RUN because the current 10-second static gate FAILS (right arm FAIL, Wuji2 FAIL, flange/wrist PASS).
- Final checks: `git diff --check` PASS; related Python compilation PASS; production-path grep found no active legacy IK/collision imports.
- Known-good 5-stage worker regression: PASS; current raw `[33,37,37,37,31]`, current accepted `[32,34,34,34,31]`, worker starts `1`, map builds `0`.
- Grouped batched IK regression: PASS; `group_count=1`, `pose_count=5`, raw `[33,37,37,37,31]`.
- Self-collision semantics: PASS; safe state `self_collision_pass=true`, folded all-zero right-arm state `self_collision_pass=false`.
- Continuous-path semantics: PASS; safe same-state path `path_pass=true`, deliberate path into folded collision state `path_pass=false`.
- Closed-loop logic tests: PASS `4/4` in `isaaclab22_sim50`.
- Top-16 same-frame batch pick validation: command PASS / screening result FAIL; worker starts `1`, map builds `1`, chunk size `16`, materialized candidates `16`, grouped solve poses `80`, selected candidate `null`.
- candidate1422 collision diagnosis: command PASS; per-stage raw/threshold/self/scene/target/survivors are `pregrasp 36/33/33/0/0/0`, `cover 37/35/35/0/0/0`, `grasp 37/35/35/0/35/0`, `squeeze 37/35/35/0/35/0`, `lift 34/33/33/0/0/0`.
- All-candidate GPU prefilter: command PASS; total proposals `8192`, target proposals `7454`, GPU batch size `512`, batch count `15`, raw reachable `4435`, threshold accepted `4347`, scene pass `4269`, self pass `0`, coarse survivors `0`, total wall `18.242 s`, peak VRAM `1608 MiB`.
- Self report-only patch: py_compile PASS; `git diff --check` PASS. GPU rerun for PREGRASP/approach report-only benchmark was blocked because this channel reported CUDA unavailable and escalation was rejected.
- One-command planning-only integration: `orchestrator.py` py_compile PASS; `git diff --check` PASS. Real run reached Isaac capture but was interrupted after this command channel reported NVML/CUDA/Vulkan GPU initialization failures; escalated rerun was rejected by execution policy.
- Scratch path + strict prefilter patch: `orchestrator.py` and `all_candidate_gpu_prefilter.py` py_compile PASS; `git diff --check` PASS. No GPU/Isaac rerun was attempted in the restricted channel.
- Simulation diagnostic wiring: `orchestrator.py` and `runtime/scripts/10_run_full_pick_place.py` py_compile PASS; `git diff --check` PASS.
- Batch retarget + grouped exact IK wiring: py_compile PASS; `git diff --check` PASS. Candidate330 retarget/flange numerical regression PASS with max abs diff `0.0`; only path metadata differs.
- Retarget chunk 64 + bash launcher patch: py_compile PASS; `git diff --check` PASS.

## Current environment versions

- Python `3.11.15`
- PyTorch `2.13.0+cu130`
- cuRobo `0.8.0.post1.dev42`
- CUDA available: `True` on the host execution channel (`False` only in the restricted sandbox channel)
- Isaac runtime conda: Python `3.11.15`, NumPy `1.26.4`, Gymnasium `0.29.0`; Isaac Sim `5.0`, local Isaac Lab `2.2.x` source tree


## 2026-08-17 16:18:00 +0800 - Final worktree cleanup
- Current phase: final local worktree cleanup after closed-loop diagnostic success.
- Completed: preserved compact candidate5989 evidence; removed generated outputs/scratch/history payloads; moved calibration out of outputs; pruned intermediate checkpoints; updated README/PROJECT_STATUS/.gitignore.
- Current conclusion: runtime outputs are ignored/regenerated; compact evidence is under `08_dual_arm_scene_layout/isaaclab_control/evidence/`.
- Current blockers: user review required before git add/commit/push.
- Modified files: docs, .gitignore, worklog, calibration path references, verified indices/readmes.
- Test results: pending static validation commands.
- Environment: no expensive GPU/Isaac/network/training workloads run.

## 2026-08-18 21:40:00 +0800 - DualArmMount y=0.16 final layout calibration
- Current phase: final mechanical arm/table relative position calibration.
- Completed: moved only `/World/Layout/DualArmMount` to `[0, 0.16, 0.8]` in both calibrated stages; preserved `RotateXYZ=[0,0,-90]` and `Scale=[1,1,1]`; regenerated `/World/Sensors/TopD435iVirtual/Camera` and `Frustum` from current `arm_base_link_d435i_2` anchor to SourceZone center; updated `config/manual_layout_calibrated.json`; refreshed RobotRoot distance markers.
- Current conclusion: production persistent Isaac still loads `manual_layout_calibrated_mass_fixed.usda`; orchestrator/cuRobo still read `manual_layout_calibrated.json`; Stage and JSON are synchronized.
- Current blockers: exact PhysX penetration was not run from this static edit pass; USD AABB robot/table overlap is not conclusive and should be checked visually/with Isaac static validation before long runs.
- Modified files: `scenes/manual_layout_calibrated.usda`, `scenes/manual_layout_calibrated_mass_fixed.usda`, `config/manual_layout_calibrated.json`, plus generated output metadata under `08_dual_arm_scene_layout/outputs/`.
- Test results: USD open and JSON sync PASS; camera coverage PASS; `05/06/07` py_compile PASS; `git diff --check` PASS.
- Environment: `isaaclab22_sim50` Python with Isaac USD extension paths; no Isaac app, DGN2, RFS, cuRobo planning, retarget, grasp execution, or full closed-loop was run.

## 2026-08-19 - Simplified planning plumbing pass
- Current phase: prepare production code for simplified endpoint/joint-space planning without touching core IK/RFS mathematics.
- Completed: GroundingDINO fixed workspace ROI crop; ESDF ROI depth invalidation outside `[170,0,970,700]`; persistent capture debug-prim hide/restore; per-64-candidate cuRobo worker lifecycle; per-batch GPU memory print hooks; `flexible_route_failures.jsonl`; RFS normal zero-pass status separation.
- Current conclusion: Exact COVER core remains unchanged; `CuroboGpuIK` still computes accepted with global config thresholds and exposes raw success/residual/margin for every returned seed. Stage-specific tolerance support still needs the planned core patch.
- Current blockers: `flexible_route_search.py` still contains existing broad task-space route sampling and final observed-map path check semantics; RFS core still contains 5120 support-pose/layer-graph algorithm. Both are intentionally not rewritten here.
- Modified files: `closed_loop/orchestrator.py`, `closed_loop/scripts/grounded_sam_backend.py`, `closed_loop/persistent_isaac/worker.py`, `closed_loop/planning/candidate_rfs_v2_runtime.py`, worklogs.
- Test results: py_compile PASS; `isaaclab22_sim50` closed-loop CPU unit tests PASS 9/9; `git diff --check` PASS.
- Environment: no full closed-loop, Isaac app, DGN2, RFS backend, cuRobo GPU worker, retarget, or physical execution was run.

## 2026-08-19 - PlacementZone 5 cm X-edge gap layout update
- Current phase: static layout parameter update only.
- Completed: moved only formal `PlacementZone` X from `0.40069047181642775` to `0.27617723418526796` in `manual_layout_calibrated.json`, `manual_layout_calibrated.usda`, and `manual_layout_calibrated_mass_fixed.usda`; synchronized PlacementZoneCenter marker and SourceToPlacement measurement; changed draft generator to derive the placement center from `ZONE_EDGE_GAP_M=0.05`; synchronized `manual_layout_draft.usda` to X `0.30`.
- Current conclusion: SourceZone right edge and PlacementZone left edge gap is `0.049999999999999989 m`, satisfying the required `<1e-9` invariant. SourceZone, Table, DualArmMount, Camera, zone sizes, robot joint pose, and planning tuning were not changed.
- Current blockers: none for this static layout update.
- Modified files: `08_dual_arm_scene_layout/config/manual_layout_calibrated.json`, `08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda`, `08_dual_arm_scene_layout/scenes/manual_layout_calibrated_mass_fixed.usda`, `08_dual_arm_scene_layout/scenes/manual_layout_draft.usda`, `08_dual_arm_scene_layout/scripts/01_create_manual_layout.py`, `08_dual_arm_scene_layout/scripts/check_placement_zone_gap.py`, worklog files.
- Test results: invariant script PASS; modified Python py_compile PASS; `isaaclab22_sim50` closed-loop CPU unit tests PASS 9/9; `git diff --check` PASS; old formal X grep over active paths returned no matches.
- Environment: no Isaac Sim, DGN2, RFS, cuRobo, retarget, full closed-loop, or physical simulation was run.

## 2026-08-19 - Experimental planner collision bypass execution flag
- Current phase: diagnostic/experimental execution plumbing only.
- Completed: added `--experimental-bypass-planner-collision` to `closed_loop/orchestrator.py`; it is allowed only with `--sim-execute` and does not change existing `--diagnostic-disable-*` semantics.
- Current conclusion: in experimental execution mode, RFS observed ESDF, Exact COVER observed ESDF, HOME->PRE observed ESDF/self-collision, and PRE->COVER observed ESDF/self-collision are bypassed through the existing per-gate plumbing while Isaac/PhysX execution remains enabled.
- Current blockers: not runtime-validated in this channel by request.
- Modified files: `closed_loop/orchestrator.py`, worklog files.
- Test results: orchestrator py_compile PASS; `isaaclab22_sim50` closed-loop CPU unit tests PASS 9/9; `git diff --check` PASS.
- Environment: no Isaac Sim, DGN2, RFS, cuRobo, retarget, full closed-loop, or physical simulation was run.

## 2026-08-19 - Standalone cuRobo RobotSegmenter capture adapter
- Current phase: local engineering adapter only; no baseline planner integration.
- Completed: added `isaaclab_control/perception/robot_segmentation/` with a standalone RobotDepthCleaner and CLI for existing capture folders.
- Current conclusion: `RobotSegmenter` import works in `curobo_v2`; the adapter maps `T_world_camera` to `T_base_camera = inv(T_world_base) @ T_world_camera`, adds the `[1,H,W]` batch depth shape, maps all 35 measured robot joints by name, and writes planning outputs under `capture/planning/`.
- Current blockers: none for standalone use; planner integration intentionally not performed.
- Modified files: new `perception/robot_segmentation` package and worklogs.
- Test results: py_compile PASS; CLI help PASS; standalone RobotSegmenter run PASS on `20260819_174407/cycle_001/capture`; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, baseline planner, or execution was run.

## 2026-08-19 - Route B current-to-PREGRASP MotionPlanner adapter

- Current phase: Route B phase-1 standalone backend integration.
- Completed: added `isaaclab_control/curobo_motion_planning_routeB/` with config, adapter, README, import test, and `test_current_to_pregrasp.py`; adapter builds ESDF from `capture/planning/filtered_depth.npy`, converts camera pose into `arm_base_link` frame, creates `MotionPlanner`, and calls `plan_cspace(q_current, q_pregrasp)`.
- Current conclusion: Route A remains untouched and is still the default via `use_legacy_keypoint_route: true`; Route B import/API wiring reaches cuRobo MotionPlanner.
- Current blocker: standalone test on `20260819_174407/cycle_001` returns `Start or End state in collision` from MotionPlanner with the filtered ESDF; no IK/retarget/Route A logic was changed to mask this.
- Modified files: new `08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/` package plus worklogs.
- Test results: cuRobo MotionPlanner/Mapper import PASS; Route B py_compile PASS; Route B import unittest PASS; `git diff --check` PASS; standalone Route B smoke FAIL at MotionPlanner endpoint collision.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-19 - Route B formal collision policy wiring

- Current phase: enforce official Route B collision policy.
- Completed: Route B config now records `collision.environment_collision=true` and `collision.self_collision=false`; adapter explicitly disables self-collision on IK metrics/optimizer rollouts, TrajOpt metrics/optimizer rollouts, and Graph metrics rollout before constructing `MotionPlanner`; graph seed is disabled by default for phase-1 C-space TrajOpt planning to avoid the separate PRM start/end feasibility path.
- Current conclusion: environment ESDF collision remains enabled (`scene_collision_cfg_present=true`); self-collision is confirmed disabled across all Route B rollout configs (`all_self_collision_rollouts_disabled=true`); Route A remains untouched/default.
- Current blocker: same `current -> PREGRASP` command no longer emits `Start or End state in collision`, but MotionPlanner still returns `success=false` with zero trajectory points. This is now a TrajOpt planning failure under ESDF ON / self OFF, not the prior self-collision pair 2576 endpoint rejection.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, `config_example.yaml`, `test_current_to_pregrasp.py`, `test_collision_audit.py`, and worklogs.
- Test results: Route B py_compile PASS; import unittest PASS; `git diff --check` PASS; standalone MotionPlanner run FAIL with new status `success=false`, `trajectory_points=0`, no start/end collision message.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-19 - Route B TrajOpt feasibility constraint audit

- Current phase: diagnose Route B `current -> PREGRASP` TrajOpt success=false without changing planning parameters.
- Completed: added standalone `test_trajopt_feasibility_audit.py`; recomputed cuRobo raw/interpolated metrics from the returned top trajectory; independently checked project ESDF clearance, joint position bounds, acceleration, and jerk.
- Current conclusion: raw and interpolated cuRobo metrics are both infeasible because `scene_collision` is strongly positive and `cspace` is tiny positive. Worst cuRobo scene-collision entry is timestep `[0,76,112]`, sphere `112`, link `arm_r_link_7`, value `169.164154`. Project ESDF post-check over the same returned trajectory reports no environment collision and min clearance `0.078266 m`; acceleration and jerk are within YAML limits.
- Current blocker: mismatch between cuRobo TrajOpt `scene_collision` metric feasibility and the project-side ESDF sphere query for the same trajectory.
- Modified files: new `curobo_motion_planning_routeB/test_trajopt_feasibility_audit.py`, worklogs.
- Test results: py_compile PASS; Route B import unittest PASS; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B cuRobo scene_collision semantics audit

- Current phase: one-to-one diagnose cuRobo `scene_collision` vs project ESDF query for the same returned TrajOpt trajectory samples.
- Completed: added standalone `test_scene_collision_semantics_audit.py`; compared samples `t=76/sphere112`, `t=61/sphere184`, and `t=64/sphere185`; recorded cuRobo cost config, activation rule, VoxelData metadata, project ESDF query, cuRobo raw checker cost, and rollout constraint.
- Current conclusion: cuRobo raw checker and rollout constraint agree exactly, so the mismatch is not TrajOpt postprocessing. For `t=76/sphere112`, project query reports signed distance `0.143382 m` and clearance `0.112177 m`, while cuRobo raw checker implies signed distance `-0.002628 m` and penetration `0.033833 m`. Activation distance is `0`, weight is `5000`, and `constraint = weight * max(radius - signed_distance, 0)`.
- Current blocker: cuRobo SceneCollision/VoxelData query semantics or scene representation does not match the project `query_spheres` interpretation of the same `VoxelGrid`.
- Modified files: new `curobo_motion_planning_routeB/test_scene_collision_semantics_audit.py`, worklogs.
- Test results: py_compile PASS; Route B import unittest PASS; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B ESDF filtered-depth ground-truth audit

- Current phase: decide project ESDF query vs cuRobo VoxelData query using original `filtered_depth` surface points.
- Completed: added standalone `test_esdf_ground_truth_audit.py`; sampled 500 valid filtered-depth surface points in `arm_base_link`; compared project `query_esdf_distance`, a CPU reproduction of cuRobo VoxelData sampling semantics, cuRobo radius-0 raw collision probe, and 20 direct voxel-center reads.
- Current conclusion: project query is correct for the current VoxelGrid. It uses `grid_sample` order `zyx`, `align_corners=True`, `[nx,ny,nz]`, and gives surface median/p90 abs SDF `0.002678/0.003372 m`; voxel-center direct-read median error is `1.49e-08 m`. cuRobo VoxelData semantic query gives surface median/p90 abs SDF `0.168801/0.278765 m` and voxel-center median error `0.071903 m`.
- Current blocker: `CUROBO_SCENE_REPRESENTATION_WRONG`. `VoxelData.params` stores dims as floats `[36.0, 55.0000038, 26.9999981]`; cuRobo Warp kernel truncates with `wp.int32()` to `[36,55,26]`, while feature tensor shape is `[36,55,27]`, corrupting Z-fastest indexing.
- Modified files: new `curobo_motion_planning_routeB/test_esdf_ground_truth_audit.py`, worklogs.
- Test results: py_compile PASS; Route B import unittest PASS; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B VoxelData dimension contract fix

- Current phase: fix Route B cuRobo VoxelData discrete voxel count contract after ground-truth audit.
- Completed: `RouteBMotionPlannerAdapter` now normalizes shared cuRobo `VoxelData.params[...,0:3]` from authoritative `scene_grid.feature_tensor.shape` after `MotionPlanner(cfg)` and before `warmup/plan_cspace`; report plumbing records `voxel_shape_contract`.
- Current conclusion: scene-collision false positive is fixed. Params changed from `[[[36.0,55.0000038,26.9999981,0.02]]]` to `[[[36.0,55.0,27.0,0.02]]]`; feature shape `[36,55,27]`, feature count `53460`; dims/inv_pose/features unchanged. Ground-truth cuRobo surface median/p90 abs SDF improved to `0.002678/0.003372 m`, matching project query; voxel center median error is `0`.
- Current blocker: `plan_cspace` still returns success=false, but feasibility audit now reports only `cspace` constraint, max `5.80343e-08`; environment collision is false and min clearance is `0.075243 m`.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, `test_current_to_pregrasp.py`, `test_scene_collision_semantics_audit.py`, worklogs.
- Test results: py_compile PASS; ESDF ground-truth audit PASS; scene_collision semantics audit PASS; Route B import unittest PASS; `git diff --check` PASS; current-to-PREGRASP standalone still FAIL due only to cspace residual.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B cspace numerical-bound residual fix

- Current phase: finish Route B phase-1 `current -> PREGRASP` standalone MotionPlanner feasibility.
- Completed: added Route B-only pre-planning joint-state numerical-bound sanitization in `RouteBMotionPlannerAdapter`; raw `robot_state.json` and raw route plan are not modified. The sanitizer uses the MotionPlanner rollout's actual position bounds, fixes only violations within `1e-5 rad`, moves corrected values `1e-6 rad` inside bounds, and raises on larger violations.
- Current conclusion: the previous `cspace` residual was caused by tiny static left-arm bound residuals, not by right-arm motion or environment collision. Corrected DOF are `arm_l_joint_2` and `arm_l_joint_4` for both `q_current_planning` and `q_pregrasp_planning`; max original violation `4.818e-06 rad`.
- Current blocker: none for standalone Route B `current -> PREGRASP`; formal closed-loop Route B wiring is still a future step.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, `test_trajopt_feasibility_audit.py`, `test_scene_collision_semantics_audit.py`, `test_current_to_pregrasp.py`, and worklogs.
- Test results: py_compile PASS; feasibility audit PASS with `scene_collision=0`, `cspace=0`, no failed constraints, min clearance `0.075290 m`; standalone `test_current_to_pregrasp.py` PASS with `success=True`, `trajectory_point_count=41`, planning time `1.863 s`; returned trajectory postcheck PASS with environment collision false and joint-limit violations `0`; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B true right-arm-only current-to-PREGRASP

- Current phase: integrate ChatGPT-provided `right_arm_only_core` into local Route B as a standalone test path.
- Completed: added `test_current_to_pregrasp_right_arm_only.py`; reused existing Route B `MotionPlannerCfg`/SceneCfg parameters, collision policy `environment=true/self=false`, VoxelData feature-shape fix, and postchecks. The 35DOF adapter is used only to build the scene and sanitized full q contract; it does not generate a 35DOF trajectory.
- Current conclusion: true right-arm-only cuRobo MotionPlanner succeeds for `current -> PREGRASP`. Planner action_dim is `7`; active joints are exactly `arm_r_joint_1..7`; locked joint count is `28`; raw optimizer solution shape is `[1,1,16,7]`; dense trajectory is `[41,7]`.
- Current blocker: none for standalone right-arm-only phase-1 test. Production closed-loop Route B wiring remains a future step.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, new `test_current_to_pregrasp_right_arm_only.py`, and worklogs.
- Test results: right-arm-only core contract unittest PASS; Route B import unittest PASS; py_compile PASS; `git diff --check` PASS; generated `trajectory_right_arm.npz` and `report_right_arm.json`; postcheck PASS with environment collision false, min clearance `0.075289 m`, `scene_collision=0`, `cspace=0`, joint limits PASS, velocity/acceleration/jerk finite PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.
