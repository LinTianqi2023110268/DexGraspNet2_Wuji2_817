# Right-Arm Workspace / Table Layout Calibration

This is the main code for deciding **where the DualArmMount center should be
placed relative to the fixed table**.

It answers the user's question:

> Where should the mechanical-arm center block (`DualArmMount`) be placed so
> the right arm has the best practical reachability over the SourceZone,
> while still retaining useful PlacementZone reachability?

## What is optimized

Only the `DualArmMount` translation is scanned:

- `x`
- `y`
- `z`

The current calibrated mount rotation is frozen in the first pass.

The production layout file is **not modified**.

## Why this is more meaningful than a sphere-radius test

The scanner does not use a simple spherical reach estimate.

It extracts **real finalized Wuji2 COVER grasp templates** from previous
candidate cases:

`target object -> right arm flange`

Then it translates those real grasp relationships across SourceZone and
PlacementZone grid points and runs the project's real `CuroboGpuIK` with:

- 5 mm position tolerance
- 5 deg orientation tolerance
- 3 deg inner-joint margin
- 48 seeds
- tool frame `arm_r_link_tf`

So it directly evaluates exact 6-D task reachability.

## Search size

Default coarse search:

- x: -0.16 ... 0.00 m, step 0.02
- y: +0.10 ... +0.22 m, step 0.02
- z: +0.70 ... +0.80 m, step 0.025

= 315 layouts.

Then 5×5×5 local refinement around the coarse best:

= 125 layouts.

Total ≈ 440 mount centers.

## Recommended template groups for the first run

Use two groups:

1. **validated bottle** — only previously Exact-COVER-PASS cases;
2. **pencil stress** — finalized pencil cases from the physically hard
   far-side target, to include different grasp orientation/offset families.

The script balances and downsamples them into a diverse library of 12 task
templates, so the much larger pencil case count cannot dominate the library.

## Run environment

Run the scanner in `curobo_v2`.

It does **not** start Isaac.

Example:

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2

/home/lin/miniconda3/bin/conda run --no-capture-output -n curobo_v2 \
python \
08_dual_arm_scene_layout/isaaclab_control/tools/06_optimize_dual_arm_mount_workspace.py \
--template-group \
'validated_bottle|08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260818_164056/cycle_001/rfs_prototype/v2_filtered_offline_replay_top64/cases|08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260818_164056/cycle_001/rfs_prototype/v2_filtered_offline_replay_top64/exact_cover_results.json' \
--template-group \
'pencil_stress|08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260818_200104/cycle_001/scratch/final_planning'
```

## Main outputs

Default output directory:

```text
08_dual_arm_scene_layout/isaaclab_control/outputs/right_arm_workspace_layout_calibration/
```

Important files:

- `layout_calibration_report.json`
- `layout_scan_all.csv`
- `top20_layouts.json`
- `task_template_library.json`
- `best_source_reachability_heatmap.png`
- `best_placement_reachability_heatmap.png`

## How to choose the final mount position

The report gives two recommendations:

1. `best_score_layout`
   - pure best reachability score;
2. `plateau_nearest_current_layout`
   - among layouts within 0.5 percentage points of the best score, choose the
     one requiring the smallest physical move, while preferring a basic
     center-to-table-edge clearance indicator.

Do **not** freeze the layout yet.

Next step after the scan:

1. take the top 3–5 layouts;
2. run a static Isaac/PhysX penetration check;
3. if multiple remain valid, choose the best score / nearest plateau layout;
4. only then update `manual_layout_calibrated.json`;
5. rerun RFS + full closed loop.

## Codex scope

Codex may:
- fix import/path/API compatibility;
- fix case discovery for the current output tree;
- fix small plotting/output issues;
- run py_compile and `--help`;
- run the real GPU scan on the user's machine.

Codex must not:
- change the 5 mm / 5 deg / 3 deg acceptance contract;
- replace exact IK with a sphere approximation;
- silently modify the production layout;
- start Isaac during the scan;
- change the Source/Placement weights without reporting it.
