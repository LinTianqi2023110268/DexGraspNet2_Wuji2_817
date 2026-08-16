# Codex integration instruction — use supplied code, do not redesign

Work in `/home/lin/Projects/DexGraspNet2_Wuji2`.

A reviewed replacement package is supplied at:
`08_dual_arm_scene_layout/isaaclab_control/core/`.

Your job is **integration, cleanup, regression, and GitHub update only**.  Do not redesign the
IK solver, mapper, score, robot model, scene, or grasp policy.

1. Start with read-only audit: `git status`, `git diff`, current `runtime/scripts/10_run_full_pick_place.py`,
   `runtime/scripts/runtime_rebase_ik.py`, `tools/08_solve_full_arm_waypoints.py`, existing cuRobo
   diagnostics, and current RGB-D/GroundedSAM paths.  Preserve unrelated user work.

2. Run the exact commands in `core/INTEGRATION_GUIDE.md`.  First prove:
   - pure tests PASS;
   - `curobo_v2` probe imports current `Mapper` and `InverseKinematics`;
   - the one-time cuRobo collision-sphere robot model is generated under `core/generated/`;
   - persistent worker can be pinged from `isaaclab22_sim50`.
   Fix only concrete API/path mismatches found against the installed cuRobo V2 source.  Do not
   replace the architecture with old nvblox or a KD-tree collision implementation.

3. Production IK replacement:
   - use `core/bridge/CuroboWorkerClient` from Isaac Lab runtime;
   - worker runs persistently in `curobo_v2`; do not install cuRobo into `isaaclab22_sim50`,
     `wuji2_factory`, or base;
   - use the existing right-arm URDF, base/tool frames and joint order;
   - keep 5 mm / 5 deg / 3 deg acceptance and 0.01 rad inner-limit shrink;
   - default 48 seeds, batch 64, CUDA graph on;
   - online first reference is measured Isaac Lab q_current; offline fallback is
     `[50,-70,0,40,35,0,25] deg`;
   - preserve all accepted IK solutions, dedupe, and use the supplied continuity selector;
   - select PREGRASP from q_current, COVER from q_PREGRASP, GRASP from q_COVER, SQUEEZE from
     q_GRASP, LIFT from q_SQUEEZE.  Do not select only by largest margin or smallest pose error.

4. Keep `GPT_cuRoboV2_GPU全场景IK对比.py` (or equivalent validated GPU scanner) in diagnostics as
   regression evidence.  Keep old SciPy/Pinocchio IK only under history/diagnostics.  Once new
   production regression passes, runtime must no longer import `runtime_rebase_ik.solve_right_arm_targets`
   or use SciPy `least_squares` for production IK.

5. Observed-scene collision replacement:
   - use current capture `depth_m.npy + intrinsics.npy + T_world_camera.npy`;
   - use GroundedSAM target mask to split non-target scene and target layers;
   - use supplied cuRobo V2 native `Mapper -> TSDF -> ESDF` implementation;
   - current camera view is assumed not to contain robot/hand, so do not add robot-image masking now;
   - use the supplied `CuroboRobotSphereModel`/worker `check_robot_state` path for robot-vs-ESDF
     checks; feed the measured Isaac Lab named joint state and calibrated `T_world_base` rather than
     inventing a second collision geometry representation;
   - non-target observed scene is always blocking;
   - target is blocking for PREGRASP/APPROACH/COVER and intentional-contact layer for
     GRASP/SQUEEZE/LIFT;
   - preserve `unknown` separately; no-point/occluded/out-of-FOV must not be labelled definitely free;
   - do not use complete simulator mesh/hidden object backside as production collision truth.

6. Do not blindly delete legacy collision files.  First identify every writer/consumer of
   `official_leap_target_collision_filtered.npz` and any complete-mesh/table collision checker.
   After the RGB-D collision path passes smoke tests, move obsolete production-only code to a clearly
   named `history/legacy_mesh_collision/` or diagnostics location.  Historical cached results may stay
   for provenance, but production decisions must not depend on them.

7. Keep statuses separate in runtime/reporting:
   `ik_pass`, `observed_scene_collision_pass`, `unknown_space_safe_enough` (or equivalent explicit
   unknown status), `path_pass`, `physics_grasp_pass`.  IK PASS is not collision/path/physics PASS.

8. Do not change: DGN2 score/ranking, GroundingDINO/SAM semantics, LEAP->Wuji2 retargeting, Wuji2
   USD/URDF parameters, arm URDF limits, initial posture, 5mm/5deg/3deg thresholds, scene calibration,
   camera calibration, object poses, Isaac Lab drive tuning, or existing successful physics results.

9. Regression order before cleanup:
   candidate3800 -> candidate34 if present -> Top-20 -> historical full coarse GPU IK regression.
   Run collision-map smoke test separately.  Do not trigger physical motion unless static gates pass.

10. After all tests pass, update README minimally to document:
    - cuRobo V2 production GPU IK and two-conda persistent worker architecture;
    - q_current/initial-pose continuity selection;
    - RGB-D native Mapper/TSDF/ESDF observed-scene collision;
    - target phase semantics and single-view unknown-space caveat;
    - legacy CPU/full-mesh code retained only for diagnostics/history.

11. Finish with `git diff --check`, changed-file `py_compile`, report exactly what moved/changed and
    why, then commit and push to the existing GitHub repository.  Do not touch the two vendor repos
    unless an actual required change is proven.
