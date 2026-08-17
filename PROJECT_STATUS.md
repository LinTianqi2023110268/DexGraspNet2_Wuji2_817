# Project Status

Updated: 2026-08-17.

## Active production path

The current user-facing entry is `./run_closed_loop.sh`.

`08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py` integrates the current live-perception and planning stack: Isaac capture, robot state, GroundingDINO/SAM with user text, RGB-D to 40k points, DexGraspNet2 proposal ordering, GPU IK prefilter, LEAP-to-Wuji2 retarget, exact cuRobo IK, placement, full route planning, and Isaac Sim diagnostic execution.

## Latest evidence

Compact closed-loop golden evidence:

`08_dual_arm_scene_layout/isaaclab_control/evidence/closed_loop_20260817_candidate5989/`

- source session: `20260817_102537`
- selected official rank: `685`
- selected candidate: `5989`
- result: Isaac/PhysX simulated grasp/place PASS
- planner collision rejection: disabled only for diagnostic simulation
- static stability gate: diagnostic override enabled
- Isaac/PhysX collision/contact: enabled
- real robot output: disabled

Diagnostic collision bypass/static override is not a real-robot safety mode.

## Retained baselines

- Route A: native Wuji2 network evidence under `07_wuji2_network_3p3r_sim/04_verified_baseline/`.
- Route B: LEAP-to-Wuji2 evidence under `06_leap_to_wuji2_final_pipeline/04_verified_baseline/`.
- Route C: current compact closed-loop evidence under `08_dual_arm_scene_layout/isaaclab_control/evidence/`.

Historical reorganization notes are in `docs/history/20260814_reorganization.md`; detailed command and phase logs are in `08_dual_arm_scene_layout/isaaclab_control/core/worklog/`.
