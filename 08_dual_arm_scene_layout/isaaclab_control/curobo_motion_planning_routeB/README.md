# cuRobo Motion Planning Route B

Route B is an independent backend for testing cuRobo V2 `MotionPlanner`
against the current project data.  It does not replace the existing Route A.

Current phase:

```text
robot-filtered depth -> Mapper/ESDF -> MotionPlanner -> q_current to PREGRASP
```

Route A remains the default:

```yaml
use_legacy_keypoint_route: true
```

## Standalone smoke command

Run inside the `curobo_v2` environment:

```bash
conda run -n curobo_v2 python \
  08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py \
  --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture \
  --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
```

The test writes:

```text
<capture>/curobo_test_result/trajectory.npz
<capture>/curobo_test_result/report.json
```
