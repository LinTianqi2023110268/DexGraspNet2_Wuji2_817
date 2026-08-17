# Closed-loop semantic dexterous grasp patch

This patch turns the existing project into one user-facing loop without mixing
the three incompatible Python/CUDA environments in one interpreter.

## Intended user experience

```text
./run_closed_loop.sh
  ↓
Scene folder > /.../scenes/scene_0000
  ↓
Isaac Lab/Sim settles the scene and captures aligned RGB-D
  ↓
rgb.png opens
  ↓
Target > dog
  ↓
GroundingDINO(text + RGB only)
  ↓
SAM mask + overlay
  ↓
full SourceZone RGB-D -> 40,000 scene points + target membership
  ↓
official DexGraspNet2 LEAP proposals
  ↓
proposals sorted by untouched official score
  ↓
for candidate chunks in descending score:
    LEAP -> Wuji2
    cuRobo batched PREGRASP/COVER/GRASP/SQUEEZE/LIFT gate
    first pick-stage feasible candidate then gets placement + full Cartesian route
    full route must pass cuRobo endpoint IK + ESDF + self collision + continuous path gate
    stop at FIRST fully feasible full-route candidate
  ↓
Isaac Lab/Sim executes pick -> place -> release -> retreat -> HOME
  ↓
successful placement is committed to placement_registry.json
  ↓
final object poses are persisted as next scene state
  ↓
new RGB-D capture -> Target > ...
```

The **first feasible item in descending official DGN2 score order** is the
highest-scoring feasible grasp.  IK margin is not allowed to silently replace
the network score as the primary semantic/grasp ranking.

## Important semantic boundary

GroundingDINO must receive only:
- the captured RGB image,
- the user's text query,
- model/checkpoint parameters.

It must NOT receive the simulator scene manifest, class labels, USD/URDF,
segmentation IDs, or 3-D object poses.  `resolve_sim_target.py` runs only after
GroundingDINO+SAM and merely binds the already-selected visual mask to a
simulator rigid body for contact/lift instrumentation.

## What is already reused from the repository

- dynamic gravity settle + aligned RGB-D capture:
  `isaaclab_control/tools/11_settle_and_capture_dynamic_scene.py`
- live RGB-D -> official 40k DGN2 input:
  `08_dual_arm_scene_layout/scripts/08_build_target_network_input.py`
- untouched official LEAP DGN2 target inference:
  `08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py`
- reviewed LEAP->Wuji2 scripts:
  `06_leap_to_wuji2_final_pipeline/02_scripts/01,02,03,05`
- flange conversion:
  `isaaclab_control/tools/03_build_arm_execution_targets.py`
- placement registry / footprint-aware slot allocation:
  `runtime/scripts/placement_allocator.py`
- Route-C V2 cuRobo worker + observed RGB-D ESDF:
  `isaaclab_control/core/`
- physical full pick/place/home:
  `runtime/scripts/10_run_full_pick_place.py`

## Deliberately not reused as a production decision

`08_dual_arm_scene_layout/scripts/10_filter_target_pregrasp_collision.py` and
the archived SciPy/Pinocchio/complete-mesh Route-C tools are historical
evidence only.  The new loop is designed to make final feasibility a cuRobo
Wuji2 + observed-ESDF decision.

## Candidate screening performance contract

The old prototype gate started a new cuRobo process and rebuilt the same RGB-D
ESDF for every DGN2 candidate.  The production screening primitive is now
`closed_loop/scripts/batch_pick_candidate_gate.py`:

- one persistent cuRobo worker per screening call;
- one Mapper/TSDF/ESDF build for the current RGB-D frame;
- candidates remain in untouched official DGN2 score order;
- each chunk sends `candidate_screen_chunk_size * 5` flange poses to one grouped
  GPU IK solve;
- branch continuity is selected independently per candidate from measured
  `q_current`;
- rejected candidates should live in the closed-loop session scratch area, not
  permanently under `06_leap_to_wuji2_final_pipeline/01_cases/active`.

## Current motion lock

`closed_loop.json` ships with:

```json
"execution_enabled": false
```

and `route_candidate_gate.py` requires explicit `self_collision_pass=true` and
`path_pass=true`.  Current GitHub Route-C V2 does not yet provide those two
fields, and the project's static stability gate is currently not cleared.
Codex must complete those items before enabling motion.

## After Codex finishes integration

The only command should be:

```bash
cd ~/Projects/DexGraspNet2_Wuji2
./run_closed_loop.sh
```
