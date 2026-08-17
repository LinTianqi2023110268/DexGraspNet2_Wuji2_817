# DexGraspNet2-Wuji2

This repository integrates DexGraspNet 2.0, Wuji2 Hand 2, cuRobo, Isaac Sim 5.0 / Isaac Lab 2.2, and a dual-arm scene into an auditable grasp pipeline.

## Current user entry

The current top-level entry point is:

```bash
./run_closed_loop.sh
```

`08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py` wires the existing modules for live RGB-D capture, robot state, user text, GroundingDINO + SAM, RGB-D to 40k points, DexGraspNet2 candidate generation, all-candidate GPU IK prefilter, LEAP-to-Wuji2 retargeting, exact cuRobo IK, placement allocation, full route planning, and Isaac Sim diagnostic execution.

## Latest closed-loop evidence

Compact golden evidence is retained at:

`08_dual_arm_scene_layout/isaaclab_control/evidence/closed_loop_20260817_candidate5989/`

It corresponds to session `20260817_102537`, selected official rank `685`, candidate `5989`, and an Isaac/PhysX simulated grasp/place PASS.

The successful diagnostic run used planner collision rejection disabled and static gate override enabled. Isaac Sim / PhysX collision/contact remained enabled. This diagnostic mode is not a real-robot safety mode and must not be described as formal deployment safety.

## Repository map

- `01_environment/`: environment contracts and vendor assets/submodules.
- `02_training_dataset/`: protected training dataset; do not move or copy.
- `03_prediction_network/`: official DexGraspNet2 core/checkpoint area.
- `04_training/`: Wuji2 training code and retained final checkpoint.
- `05_inference/`: inference utilities; generated outputs are ignored.
- `06_leap_to_wuji2_final_pipeline/`: official LEAP-to-Wuji2 retarget pipeline.
- `07_wuji2_network_3p3r_sim/`: native Wuji2 network simulation path.
- `08_dual_arm_scene_layout/`: closed-loop scene, perception, planning, runtime, evidence, diagnostics.
- `verified/`: compact index of verified/historical baselines.
- `docs/`: architecture and history notes.

## Environment rules

- Run cuRobo only in `curobo_v2`.
- Run Isaac Lab only in `isaaclab22_sim50`.
- Do not modify official Wuji2 URDF/USD to hide integration errors.
- Do not relax IK or collision acceptance thresholds without explicit approval.
- Do not commit generated captures, outputs, raw logs, videos, or scratch cases.
