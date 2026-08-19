# Route B right-arm-only core v1

Purpose: make `current -> PREGRASP` a **true 7DOF cuRobo plan**, not a 35DOF
plan whose output is sliced afterward.

## What this module owns

- Canonical active joints:
  `arm_r_joint_1 ... arm_r_joint_7`.
- Builds the 28-joint lock contract from `q_current_planning`.
- Verifies PREGRASP does not request motion in locked DOFs.
- Rebuilds cuRobo kinematics with official `kinematics.lock_joints`.
- Requires `planner.action_dim == 7`.
- Requires `result.solution.shape[-1] == 7`.
- Calls `MotionPlanner.plan_cspace()` using the pre-existing q_PREGRASP.
- Extracts cuRobo's **existing dense/interpolated trajectory** without
  quintic re-interpolation or resampling.
- Saves `trajectory_right_arm.npz`.
- Saves `report_right_arm.json`.
- Rejects endpoint mismatch or any exposed locked-DOF motion.

## What the local Route B adapter still owns

The local repository has newer, unpushed fixes that are not available in this
package. Therefore Codex should only wire these already-existing local pieces:

1. `planner_factory(locked_robot_cfg)`
   - create a new MotionPlanner using the **same current Route B settings**.
   - same scene, same TrajOpt config, same `max_attempts=2`,
     same graph behavior, same interpolation settings.

2. `planner_fixup(planner)`
   - call the already-tested Route B collision-policy hook:
     environment collision ON, self collision OFF.
   - call the already-tested VoxelData exact-shape normalization hook.

3. `trajectory_postcheck(...)`
   - reuse the existing project ESDF / cspace / joint-limit / dynamics checks.
   - do not invent a second collision implementation.

Do not modify Route A, PREGRASP/COVER IK, DGN2, retargeting, Isaac executor,
ESDF values, voxel pose, limits, or planner parameters.

## Expected use

```python
from right_arm_only_core import plan_right_arm_only

result = plan_right_arm_only(
    robot_source=ORIGINAL_ROBOT_CONFIG_DICT_OR_ROBOTCFG,
    full_joint_names=full_joint_names,
    q_current_planning=q_current_planning,
    q_pregrasp_planning=q_pregrasp_planning,
    planner_factory=build_locked_planner_using_existing_routeB_settings,
    planner_fixup=apply_existing_routeB_fixups,
    trajectory_postcheck=reuse_existing_routeB_postcheck,
    output_dir=capture_dir / "curobo_test_result",
)
```

Success artifacts:

- `trajectory_right_arm.npz`
- `report_right_arm.json`

`trajectory_right_arm.npz` fields:

- `joint_names`
- `q_rad` `[N,7]`
- `qd_rad_s` `[N,7]`
- `qdd_rad_s2` `[N,7]`
- `jerk_rad_s3` `[N,7]`
- `time_s`
- `dt_s`
- `dt_source`

## Local smoke test

The pure contract test does not require cuRobo:

```bash
python -m unittest right_arm_only_core.test_contract
```

The actual planning test must run in the project's `curobo_v2` environment.
